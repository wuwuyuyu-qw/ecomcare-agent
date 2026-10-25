# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
import asyncio
import logging
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from copy import deepcopy
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.config import settings
from app.core import summarizer
from app.core.observability import attach_observability
from app.core import budget, memory
from app.core.execution import UsageCollector, current_execution, execution_scope, remaining_seconds, utc_now, record_stream_event
from app.db import repository
from app.graph.build import build_graph
from app.graph import task_context

logger = logging.getLogger(__name__)

# Model tokens are withheld until evidence_guard has approved the complete text.
# A retracted answer cannot undo facts already shown to the customer.
ANSWER_NODES: set[str] = set()
# 确定性节点把答复写在 state['answer'],整块作为 delta 吐出
DETERMINISTIC_ANSWER_NODES = {"script_reply", "complaint_reply", "fallback_reply",
                              "logistics_extract", "logistics_investigate", "evidence_guard", "task_entry", "compound_prepare"}
# 产出编号引用的检索节点(知识路 retrieve_knowledge + 退款政策路 retrieve_policy)
CITATION_NODES = {"retrieve_knowledge", "retrieve_policy", "logistics_investigate"}

_graph = None
_cm = None  # AsyncSqliteSaver context manager,持有以免被 GC


class ConversationNotFound(Exception):
    pass


class ConversationBusy(Exception):
    pass


class InvalidResume(Exception):
    pass


class TurnExecutionTimeout(Exception):
    pass


_active_conversations: set[int] = set()


async def assert_conversation_owner(conversation_id: int, user_id: str):
    conv = await repository.get_conversation(conversation_id)
    if conv is None or not user_id or conv.user_id != user_id:
        # The same result avoids disclosing another customer's conversation IDs.
        raise ConversationNotFound(conversation_id)
    return conv


@asynccontextmanager
async def _conversation_guard(cid: int):
    # Process-local serialization for the documented one-worker local demo.
    # No await between check/add; distributed workers require a database lease.
    if cid in _active_conversations:
        raise ConversationBusy(cid)
    _active_conversations.add(cid)
    try:
        yield
    finally:
        _active_conversations.discard(cid)


def _config(cid: int) -> dict:
    scope = current_execution()
    config = {"configurable": {"thread_id": str(cid)},
              "metadata": {"langfuse_session_id": str(cid)}}
    if scope:
        config["metadata"].update(task_id=scope.task_id, request_id=scope.request_id)
        config["callbacks"] = [UsageCollector(scope)]
    return config


async def init_graph() -> None:
    """lifespan 起:开持久 checkpointer + setup + 编译图。"""
    global _graph, _cm
    _cm = AsyncSqliteSaver.from_conn_string(settings.checkpointer_db_path)
    checkpointer = await _cm.__aenter__()
    await checkpointer.setup()
    # ch09:编译时挂一次 Langfuse 回调,全图自动 trace(未配置时原样返回,零依赖)
    _graph = attach_observability(build_graph(checkpointer=checkpointer))
    logger.info("图已编译,checkpointer=%s", settings.checkpointer_db_path)


async def close_graph() -> None:
    global _graph, _cm
    if _cm is not None:
        await _cm.__aexit__(None, None, None)
        _cm = None
    _graph = None


def get_graph():
    if _graph is None:
        raise RuntimeError("图未初始化,应在 FastAPI lifespan 调 init_graph()")
    return _graph


async def get_turn_snapshot(conversation_id: int) -> dict:
    """👎 快照尽力回捞(ch09):读该会话 checkpointer 终态,返回最近一轮的用户问题与召回快照。
    调用方拿去比对被踩的 question,对得上才用快照——踩历史消息时快照可能已是别轮的,不硬塞。"""
    config = {"configurable": {"thread_id": str(conversation_id)}}
    st = await get_graph().aget_state(config)
    values = getattr(st, "values", None) or {}
    question = ""
    for m in reversed(values.get("messages", [])):
        if getattr(m, "type", "") == "human":
            question = m.content if isinstance(m.content, str) else ""
            break
    return {"question": question, "snapshot": values.get("retrieved_snapshot") or []}


def dedup_actions(actions: list) -> list:
    """跨多个 ReAct 轮次累积的 suggested_actions 可能重复(如 agent 多次提议
    create_ticket);按 type 去重,保留每种 type 首次出现的那条。"""
    seen = set()
    out = []
    for action in actions:
        t = action.get("type")
        if t in seen:
            continue
        seen.add(t)
        out.append(action)
    return out


async def _ensure_conversation(user_id: str,
                               conversation_id: int | None) -> tuple[int, str, int, int]:
    """返回 (cid, summary, summary_upto_msg_id, layer1_from_msg_id):
    入口一次库读把摘要和两个分层锚点一并带出。"""
    if conversation_id is None:
        return await repository.create_conversation(user_id), "", 0, 0
    conv = await assert_conversation_owner(conversation_id, user_id)
    return (conversation_id, conv.summary or "",
            conv.summary_upto_msg_id or 0, conv.layer1_from_msg_id or 0)


def _graph_input(user_id: str, message: str, cid: int, msg_id: int,
                 summary: str, summary_upto: int, layer1_from: int = 0) -> dict:
    # 显式重置每轮的输出通道:checkpointer 按 thread_id 持久化 State,这些无 reducer 的标量
    # 会跨轮残留(如上一轮闲聊/投诉的 answer、建工单可选项串到本轮),须在入口清零。
    # messages 有 add_messages reducer(追加=跨轮历史),不清;steps/tokens_used 是本轮预算。
    # ch07:用户消息带 db-{msg_id} 锚点(滑窗与摘要边界对齐);summary 两字段每轮从库刷新。
    scope = current_execution()
    return {"messages": [HumanMessage(message, id=f"db-{msg_id}")], "user_id": user_id,
            "task_context": deepcopy(scope.initial_context) if scope else task_context.begin_turn(
                None, user_id, cid, message, {"now": utc_now()}),
            "execution_summary": {},
            "answer_validation": {},
            "policy_assessment": {},
            "compound_plan": {}, "compound_completion": {}, "request_triage": {},
            "conversation_id": cid, "steps": 0, "tokens_used": 0,
            "answer": "", "suggested_actions": [], "evidence": "",
            "evidence_strong": False, "citations": [],
            "evidence_confidence": 0.0, "fallback_source": "", "retrieved_snapshot": [],
            "resolved_query": "", "intent_confidence": 0.0,
            "order_id": "", "order_data": {},
            "logistics_claim_type": "", "logistics_claim": "",
            "logistics_needs_policy": False, "logistics_order_id": "",
            "logistics_decision": {},
            "summary": summary, "summary_upto_msg_id": summary_upto,
            "layer1_from_msg_id": layer1_from,
            "trace": None}  # merge 通道:None=重置哨兵(见 merge_dict),清上一轮 trace 残留


def _interrupt_payload(state: dict):
    """从终态提取中断负载(整个 __interrupt__[0].value:含 type + orders/preview 等);
    无中断返回 None。ch08 起中断不止订单选择器,负载透传、消费方按 type 分发。"""
    intr = state.get("__interrupt__")
    if not intr:
        return None
    return intr[0].value


async def _final_state(cid: int) -> dict:
    """流式跑完后从 checkpointer 取终态。astream 不返回终态,分层判断要靠它。"""
    snap = await get_graph().aget_state({"configurable": {"thread_id": str(cid)}})
    return snap.values if snap else {}


async def settle_layers(cid: int, state: dict | None,
                        summary_upto: int, layer1_from: int) -> None:
    """轮末结算分层:层 1 超预算就把起点往后挪一批。

    不调模型，但更新分层锚点仍需数据库往返。挪一次挪到位、两次之间不动——边界稳住,渲染才逐字节
    稳定,前缀缓存才保得住;每轮微调等于每轮重写中间内容,缓存天天作废。

    state 传 None 时自己去 checkpointer 取(流式路径拿不到终态)。整段吞异常:
    这是轮末的锦上添花,不能影响已经发出去的回复。"""
    try:
        if state is None:
            state = await _final_state(cid)
        msgs = (state or {}).get("messages") or []
        if not msgs:
            return
        l1_budget = int(memory.window_budget() * settings.layer1_ratio)
        _, layer1_tok = memory.layer_tokens(msgs, summary_upto, layer1_from)
        if layer1_tok <= l1_budget:
            return
        new_from = memory.next_layer1_from(msgs, summary_upto, l1_budget)
        if new_from > layer1_from:
            await repository.update_layer1_from(cid, new_from)
            logger.info("层1 降级 conv=%s %s→%s(层1 %s > 预算 %s)",
                        cid, layer1_from, new_from, layer1_tok, l1_budget)
    except Exception:
        logger.exception("分层结算失败 conv=%s(不影响本轮回复)", cid)


async def _post_turn_maintenance(cid: int, state: dict | None = None,
                                 summary_upto: int | None = None, layer1_from: int | None = None) -> None:
    """Optional housekeeping cannot turn a completed answer into a timeout.

    Keep a margin before the request deadline. This timeout ends inside the
    helper, and never crosses a streamed yield. Model summary work is scheduled
    separately with a detached usage ledger.
    """
    remaining = remaining_seconds()
    if remaining is not None and remaining < 0.2:
        return
    allowance = min(0.5, remaining / 2) if remaining is not None else 0.5
    try:
        async with asyncio.timeout(allowance):
            if state is None:
                state = await _final_state(cid)
            if summary_upto is None:
                summary_upto = state.get("summary_upto_msg_id", 0)
            if layer1_from is None:
                layer1_from = state.get("layer1_from_msg_id", 0)
            await settle_layers(cid, state, summary_upto, layer1_from)
            await summarizer.maybe_schedule_summary(cid)
    except TimeoutError:
        logger.info("post_turn_maintenance_skipped conv=%s reason=timeout", cid)
    except Exception:
        logger.exception("post_turn_maintenance_failed conv=%s", cid)


async def _run_turn_impl(user_id, message, conversation_id) -> dict:
    """非流式:落 user 消息 → ainvoke → 返回终态(供 /api/agent、eval)。遇 interrupt 带 orders。"""
    cid, summary, upto, layer1 = await _ensure_conversation(user_id, conversation_id)
    msg_id = await repository.append_message(cid, "user", content=message)
    # ch09:langfuse_session_id 让同一会话多轮在 Langfuse 界面串成一组(未挂回调时无害)
    config = _config(cid)
    final = await get_graph().ainvoke(
        _graph_input(user_id, message, cid, msg_id, summary, upto, layer1), config)
    return {"conversation_id": cid, "state": final, "interrupt": _interrupt_payload(final)}


async def _resume_turn_impl(conversation_id: int, resume_value, *, user_id: str) -> dict:
    """非流式续跑(供 /api/agent、eval):Command(resume) 回填后跑到下一个中断或结束。"""
    await assert_conversation_owner(conversation_id, user_id)
    config = _config(conversation_id)
    final = await get_graph().ainvoke(Command(resume=resume_value), config)
    return {"conversation_id": conversation_id, "state": final,
            "interrupt": _interrupt_payload(final)}


async def _stream_events(cid: int, stream_source) -> AsyncIterator[dict]:
    """astream 多模式 → 事件 dict。stream_source 为 _graph_input(...)(新一轮)或 Command(resume=...)(续跑)。
    遇 interrupt(updates 块含 __interrupt__,Task 1 冒烟钉死)→ 吐 interrupt 事件并结束本次流
    (前端点选后走 /api/actions/resume 续流)。"""
    config = _config(cid)
    actions: list = []
    terminal_context: dict = {}
    source = get_graph().astream(
        stream_source, config, stream_mode=["messages", "updates"],
    )
    async with aclosing(source):
        async for mode, chunk in source:
            if mode == "messages":
                msg, meta = chunk
                if meta.get("langgraph_node") in ANSWER_NODES:
                    text = msg.content if isinstance(msg.content, str) else ""
                    if text:
                        yield {"type": "delta", "text": text}
            elif mode == "updates":
                if "__interrupt__" in chunk:
                    payload = chunk["__interrupt__"][0].value
                    # 带上 conversation_id:中断时不发 done 帧,前端要靠这个才知道续跑用哪个会话。
                    # Active readonly graph only pauses for owned-order selection.
                    ev = {"type": "interrupt", "kind": payload.get("type", ""), "conversation_id": cid}
                    for k in ("orders", "preview"):
                        if payload.get(k) is not None:
                            ev[k] = payload[k]
                    yield ev
                    return
                for node, upd in chunk.items():
                    if not isinstance(upd, dict):
                        continue
                    if node == "log" and isinstance(upd.get("task_context"), dict):
                        terminal_context = upd["task_context"]
                    if node in {"logistics_extract", "logistics_investigate", "task_entry"} and upd.get("logistics_decision"):
                        yield {"type": "logistics_decision", **upd["logistics_decision"]}
                    if node in DETERMINISTIC_ANSWER_NODES and upd.get("answer"):
                        yield {"type": "delta", "text": upd["answer"]}
                    if node in CITATION_NODES and upd.get("citations"):
                        yield {"type": "citations", "items": upd["citations"]}
                    if node == "agent_tools":
                        for m in upd.get("messages", []):
                            name = getattr(m, "name", None)
                            if name:
                                yield {"type": "tool", "name": name[:128],
                                       "ok": getattr(m, "status", "") != "error"}
                    if upd.get("suggested_actions"):
                        actions.extend(upd["suggested_actions"])
    if actions:
        yield {"type": "actions", "items": dedup_actions(actions)}
    yield {"type": "done", "conversation_id": cid, "task_context": terminal_context}


async def _stream_turn_impl(user_id, message, conversation_id) -> AsyncIterator[dict]:
    """流式:落 user 消息 → astream 多模式 → 事件 dict(供 /api/chat 转 SSE)。"""
    cid, summary, upto, layer1 = await _ensure_conversation(user_id, conversation_id)
    msg_id = await repository.append_message(cid, "user", content=message)
    async with aclosing(_stream_events(
            cid, _graph_input(user_id, message, cid, msg_id, summary, upto, layer1))) as source:
        async for ev in source:
            yield ev


async def _stream_resume_impl(conversation_id: int, resume_value, *, user_id: str) -> AsyncIterator[dict]:
    """前端点选订单后续跑同一会话的暂停流(Command(resume) 续 astream,事件与 stream_turn 同构)。"""
    await assert_conversation_owner(conversation_id, user_id)
    async with aclosing(_stream_events(conversation_id, Command(resume=resume_value))) as source:
        async for ev in source:
            yield ev


def _unanswered_tool_messages(state: dict, reason: str) -> list:
    answered = {m.tool_call_id for m in state.get("messages", []) if isinstance(m, ToolMessage)}
    return [ToolMessage(content=f"本轮已停止：{reason}，此调用未完成，不可作为证据。",
                        tool_call_id=tc["id"], name=tc["name"], status="error")
            for m in state.get("messages", []) if isinstance(m, AIMessage)
            for tc in m.tool_calls if tc["id"] not in answered]


async def _prepare_scope(cid: int, user_id: str, message: str | None, scope, *, resume_value=None) -> None:
    snapshot = await get_graph().aget_state({"configurable": {"thread_id": str(cid)}})
    state = getattr(snapshot, "values", None) or {}
    previous = state.get("task_context") or {}
    # SQL ownership does not establish the identity of a separately restored
    # checkpoint. Reject stale ID reuse before any history reaches the model.
    if state and (state.get("user_id") != user_id or state.get("conversation_id") != cid):
        raise ConversationNotFound(cid)
    if message is None:
        # A resume cannot create a fresh anonymous task or revive an ended task.
        waiting = any(getattr(task, "interrupts", ()) for task in getattr(snapshot, "tasks", ()))
        if not getattr(snapshot, "next", None) or not waiting:
            raise ConversationNotFound(cid)
        if state.get("user_id") != user_id:
            raise ConversationNotFound(cid)
        payloads = [item.value for task in getattr(snapshot, "tasks", ())
                    for item in getattr(task, "interrupts", ())]
        if (len(payloads) != 1 or not isinstance(payloads[0], dict)
                or payloads[0].get("type") != "select_order"):
            raise InvalidResume("当前暂停步骤不支持此恢复操作")
        from app.tools.business import owns_order
        if (not isinstance(resume_value, str) or not 1 <= len(resume_value) <= 64
                or not owns_order(user_id, resume_value)):
            raise InvalidResume("请从当前账号自己的订单中选择")
        scope.initial_context = deepcopy(previous)
        scope.initial_context.update(request_id=scope.request_id, updated_at=utc_now())
        scope.task_id = previous.get("task_id") or scope.task_id
    else:
        if getattr(snapshot, "next", None):
            # A new message abandons an old preview. It never acts as confirmation.
            await get_graph().aupdate_state(
                {"configurable": {"thread_id": str(cid)}},
                {"messages": _unanswered_tool_messages(state, "用户发起了新请求"),
                 "suggested_actions": [], "task_context": task_context.finish(previous, "superseded")},
                as_node="log")
        scope.initial_context = task_context.begin_turn(previous, user_id, cid, message,
            {"task_id": scope.task_id, "request_id": scope.request_id, "now": utc_now()})
        scope.task_id = scope.initial_context["task_id"] or scope.task_id
    limits = scope.initial_context.setdefault("budgets", {})
    limits.update(max_steps=settings.max_agent_steps, max_tool_calls=settings.max_task_tool_calls,
                  max_identical_tool_calls=settings.max_identical_tool_calls)


def _summary(scope, state: dict | None = None, *, interrupted=False) -> dict:
    if not scope.termination_reason:
        scope.termination_reason = "awaiting_input" if interrupted else (
            ((state or {}).get("task_context") or {}).get("termination_reason") or "completed")
    return scope.summary(settings.model_prices, settings.model_price_currency)


async def _abort_turn(cid: int | None, scope, reason: str) -> dict:
    scope.termination_reason = reason
    answer = "本轮已超时，查询尚未完成，请重试或联系人工客服。" if reason == "deadline_exceeded" else "本轮已停止，未完成的查询不能作为判断依据。"
    state = {"answer": answer, "suggested_actions": [],
             "task_context": task_context.finish(scope.initial_context, reason),
             "execution_summary": _summary(scope)}
    if cid is None or not scope.initial_context:
        return state  # No owned graph execution started; do not touch a checkpoint.
    try:
        async with asyncio.timeout(2):
            previous = await _final_state(cid)
            state["task_context"] = task_context.finish(previous.get("task_context") or scope.initial_context, reason)
            state["messages"] = _unanswered_tool_messages(previous, reason)
            await get_graph().aupdate_state({"configurable": {"thread_id": str(cid)}}, state, as_node="log")
            await repository.append_message(cid, "assistant", content=answer)
    except Exception:
        logger.exception("终止状态写入未完成 conv=%s request=%s", cid, scope.request_id)
    return state


async def _bounded_run(user_id: str, conversation_id: int | None, *, message=None,
                       resume_value=None) -> dict:
    cid = conversation_id
    with execution_scope(settings.turn_timeout_seconds) as scope:
        async with AsyncExitStack() as stack:
            execution_finished = False
            try:
                async with asyncio.timeout_at(scope.deadline):
                    cid, *_ = await _ensure_conversation(user_id, conversation_id)
                    await stack.enter_async_context(_conversation_guard(cid))
                    await _prepare_scope(cid, user_id, message, scope, resume_value=resume_value)
                    out = await (_run_turn_impl(user_id, message, cid) if message is not None else
                                 _resume_turn_impl(cid, resume_value, user_id=user_id))
                    execution_finished = True
                    out["state"]["execution_summary"] = _summary(scope, out["state"],
                                                                  interrupted=bool(out.get("interrupt")))
                if not out.get("interrupt"):
                    await _post_turn_maintenance(cid, out["state"])
                return out
            except TimeoutError as exc:
                if execution_finished:
                    logger.warning("post_completion_cleanup_timeout conv=%s", cid)
                    return out
                await _abort_turn(cid, scope, "deadline_exceeded")
                raise TurnExecutionTimeout(cid) from exc
            except asyncio.CancelledError:
                if not execution_finished:
                    from app.core.execution import finish_cancellation
                    await finish_cancellation(_abort_turn(cid, scope, "client_cancelled"))
                raise
            except (ConversationNotFound, ConversationBusy, InvalidResume):
                scope.termination_reason = "invalid_resume" if message is None else "request_rejected"
                raise
            except Exception:
                if not execution_finished:
                    await _abort_turn(cid, scope, "execution_failed")
                raise
            finally:
                logger.info("execution_summary conv=%s metrics=%s", cid, _summary(scope))

async def run_turn(user_id, message, conversation_id) -> dict:
    return await _bounded_run(user_id, conversation_id, message=message)


async def resume_turn(conversation_id: int, resume_value, *, user_id: str) -> dict:
    return await _bounded_run(user_id, conversation_id, resume_value=resume_value)


async def _bounded_stream(user_id: str, conversation_id: int | None, *, message=None,
                          resume_value=None) -> AsyncIterator[dict]:
    cid = conversation_id
    with execution_scope(settings.turn_timeout_seconds) as scope:
        async with AsyncExitStack() as stack:
            interrupted = False
            execution_finished = False
            terminal_context = {}
            try:
                async with asyncio.timeout_at(scope.deadline):
                    cid, *_ = await _ensure_conversation(user_id, conversation_id)
                    await stack.enter_async_context(_conversation_guard(cid))
                    await _prepare_scope(cid, user_id, message, scope, resume_value=resume_value)
                source = (_stream_turn_impl(user_id, message, cid) if message is not None else
                          _stream_resume_impl(cid, resume_value, user_id=user_id))
                async with aclosing(source):
                    while True:
                        try:
                            # The timeout must end before yielding to the SSE
                            # consumer. Keeping it across yield can cancel the
                            # HTTP send task during backpressure and misclassify
                            # deadline expiry as a client cancellation.
                            async with asyncio.timeout_at(scope.deadline):
                                event = await anext(source)
                        except StopAsyncIteration:
                            break
                        if event["type"] != "done":
                            record_stream_event(event["type"])
                        if event["type"] == "done":
                            # The graph iterator has finished and committed its
                            # terminal update. Later housekeeping/backpressure
                            # must never revoke this completed business result.
                            execution_finished = True
                            terminal_context = event.get("task_context") or {}
                            summary = _summary(scope, {"task_context": event.get("task_context") or {}})
                            break
                        if event["type"] == "interrupt":
                            interrupted = True
                            execution_finished = True
                            summary = _summary(scope, interrupted=True)
                        yield event
                        if interrupted:
                            break
                if not execution_finished:
                    raise RuntimeError("图事件流未产生完成或暂停状态")
                if not interrupted:
                    await _post_turn_maintenance(cid)
                yield {"type": "execution_summary", **summary}
                if not interrupted:
                    yield {"type": "done", "conversation_id": cid, "task_id": scope.task_id,
                           "request_id": scope.request_id, "termination_reason": scope.termination_reason,
                           "task_context": terminal_context}
            except TimeoutError:
                if execution_finished:
                    # Generator closing can itself time out after completion.
                    # Keep the committed/paused state and its original reason.
                    logger.warning("post_completion_cleanup_timeout conv=%s", cid)
                    yield {"type": "execution_summary", **_summary(scope, interrupted=interrupted)}
                    if not interrupted:
                        yield {"type": "done", "conversation_id": cid, "task_id": scope.task_id,
                               "request_id": scope.request_id, "termination_reason": scope.termination_reason,
                               "task_context": terminal_context}
                    return
                state = await _abort_turn(cid, scope, "deadline_exceeded")
                yield {"type": "termination", "reason": "deadline_exceeded", "message": state["answer"],
                       "task_id": scope.task_id, "request_id": scope.request_id, "conversation_id": cid}
                yield {"type": "execution_summary", **_summary(scope)}
            except (asyncio.CancelledError, GeneratorExit):
                if not execution_finished and not interrupted:
                    from app.core.execution import finish_cancellation
                    await finish_cancellation(_abort_turn(cid, scope, "client_cancelled"))
                raise
            except (ConversationNotFound, ConversationBusy, InvalidResume):
                scope.termination_reason = "invalid_resume" if message is None else "request_rejected"
                raise
            except Exception:
                if not execution_finished:
                    await _abort_turn(cid, scope, "execution_failed")
                raise
            finally:
                logger.info("execution_summary conv=%s metrics=%s", cid, _summary(scope))

async def stream_turn(user_id, message, conversation_id) -> AsyncIterator[dict]:
    async with aclosing(_bounded_stream(user_id, conversation_id, message=message)) as source:
        async for event in source:
            yield event


async def stream_resume(conversation_id: int, resume_value, *, user_id: str) -> AsyncIterator[dict]:
    async with aclosing(_bounded_stream(user_id, conversation_id, resume_value=resume_value)) as source:
        async for event in source:
            yield event
