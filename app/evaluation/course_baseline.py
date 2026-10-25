"""A course-inspired ReAct reference with shared read-only safety.

No current graph nodes, TaskContext, logistics decision function or answer guard
are used. This local execution transcript is deliberately not an HTTP-v3 record.
Importing this module does not load settings or call providers/databases.
"""
from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

PROTOCOL = "course-baseline-transcript-v1"
READ_TOOLS = frozenset({"query_order", "query_product", "query_faq", "query_logistics",
                        "query_warranty", "query_return_status"})
WRITE_TOOLS = frozenset({"create_ticket", "submit_refund"})
_SECRET = re.compile(
    r"(?i)(?:bearer\s+\S+|(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{8,}|"
    r"[\"']?(?:password|passwd|api[_ -]?key|access[_ -]?token|authorization|secret)[\"']?\s*[:=]|"
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})")
READONLY_SUFFIX = ("\n\n本实验只提供只读查询。禁止创建工单、退款或任何写入，"
                   "不能宣称已经提交或办理。需要处理时建议人工核实。")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_course_prompt(root: Path, ref: str = "90e4b24") -> tuple[str, dict]:
    """Read a Git blob as data; never import or execute historical Python."""
    if not re.fullmatch(r"[0-9a-f]{7,40}", ref):
        raise ValueError("course reference must be a commit hash")
    # Public release intentionally starts a new Git history. Preserve this
    # evaluation reference as attributed data, without importing old objects.
    snapshot = root / "provenance" / "course-agent-system.json"
    if snapshot.exists():
        value = json.loads(snapshot.read_text(encoding="utf-8"))
        prompt = value.get("agent_system")
        commit_id = value.get("course_commit", "")
        if (not isinstance(prompt, str) or not prompt
                or not re.fullmatch(r"[0-9a-f]{40}", commit_id)
                or not commit_id.startswith(ref)
                or value.get("source_path") != "app/core/prompts.py"
                or not re.fullmatch(r"[0-9a-f]{64}", value.get("source_blob_sha256", ""))
                or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != value.get("course_prompt_sha256")):
            raise ValueError("frozen course prompt provenance invalid")
        return prompt + READONLY_SUFFIX, {
            "course_commit": commit_id, "source_path": value["source_path"],
            "source_blob_sha256": value["source_blob_sha256"],
            "course_prompt_sha256": value["course_prompt_sha256"],
            "effective_prompt_sha256": hashlib.sha256((prompt + READONLY_SUFFIX).encode("utf-8")).hexdigest(),
            "source_kind": "attributed_snapshot_without_original_git_history",
            "adaptation": "original course AGENT_SYSTEM plus an explicit shared read-only safety suffix",
        }
    result = subprocess.run(["git", "show", f"{ref}:app/core/prompts.py"], cwd=root,
                            capture_output=True, check=False, timeout=10)
    commit = subprocess.run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=root,
                            capture_output=True, check=False, timeout=10)
    if result.returncode or commit.returncode:
        raise ValueError("course prompt Git source unavailable")
    source = result.stdout.decode("utf-8")
    prompt = None
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "AGENT_SYSTEM"
                                                for t in node.targets):
            prompt = ast.literal_eval(node.value)
            break
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("course AGENT_SYSTEM literal unavailable")
    return prompt + READONLY_SUFFIX, {
        "course_commit": commit.stdout.decode("ascii").strip(),
        "source_path": "app/core/prompts.py",
        "source_blob_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "course_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "effective_prompt_sha256": hashlib.sha256((prompt + READONLY_SUFFIX).encode("utf-8")).hexdigest(),
        "adaptation": "original course AGENT_SYSTEM plus an explicit shared read-only safety suffix",
    }


class SafeRecorder:
    """Whole-observation screening, without prefixes or hidden reasoning."""
    def __init__(self, private_strings=()):
        self.private_strings = tuple(s for s in private_strings if isinstance(s, str) and s)

    def record(self, value: Any, source: str) -> dict:
        base = {"source": source, "complete": False, "projection": None, "sha256": None}
        try:
            raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            encoded = raw.encode("utf-8")
            if len(encoded) > 1_000_000:
                return {**base, "reason": "observation_exceeds_byte_limit"}
            if _SECRET.search(raw) or any(s in raw for s in self.private_strings):
                return {**base, "reason": "credential_like_observation_withheld"}
        except (TypeError, ValueError, UnicodeError, OverflowError):
            return {**base, "reason": "observation_not_safe_json"}
        return {**base, "complete": True, "projection": json.loads(raw),
                "sha256": hashlib.sha256(encoded).hexdigest(), "reason": "complete"}


def message_projection(message) -> dict:
    """Only visible content and parsed calls; omit additional_kwargs/reasoning."""
    content = message.content
    if isinstance(content, list):
        content = "".join(block.get("text", "") for block in content
                          if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
                          and isinstance(block.get("text"), str))
    out = {"role": message.type, "content": content if isinstance(content, str) else None}
    if message.type == "ai":
        out["tool_calls"] = [{k: call.get(k) for k in ("id", "name", "args")}
                             for call in (getattr(message, "tool_calls", None) or [])]
    if message.type == "tool":
        out.update(tool_call_id=message.tool_call_id, name=message.name)
    return out


@dataclass
class BaselineSession:
    user_id: str
    conversation_id: int
    messages: list[Any] = field(default_factory=list, repr=False)
    authorized_tracking: set[str] = field(default_factory=set, repr=False)
    owned_orders: set[str] = field(default_factory=set, repr=False)


class CourseBaseline:
    def __init__(self, system_prompt: str, *, temperature: float = 0.3,
                 recorder: SafeRecorder | None = None):
        self.system_prompt, self.temperature = system_prompt, temperature
        self.recorder = recorder or SafeRecorder()

    async def new_session(self, authenticated_user_id: str) -> BaselineSession:
        from app.db import repository
        if not isinstance(authenticated_user_id, str) or not authenticated_user_id.strip():
            raise ValueError("authenticated identity required")
        cid = await repository.create_conversation(authenticated_user_id)
        return BaselineSession(authenticated_user_id, cid)

    async def run_turn(self, session: BaselineSession, message: str) -> dict:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        from app.config import settings
        from app.core.execution import execution_scope, UsageCollector, record_tool_event
        from app.core.llm import get_chat_model
        from app.db import repository
        from app.tools import engine, registry

        conversation = await repository.get_conversation(session.conversation_id)
        if conversation is None or conversation.user_id != session.user_id:
            raise ValueError("baseline session owner mismatch")
        if not isinstance(message, str) or not message.strip() or len(message) > 20_000:
            raise ValueError("nonempty synthetic message under recording limit required")
        sent, transcript, tool_results, answer = now(), [], [], None
        tool_manifest = None
        session.messages.append(HumanMessage(message))
        await repository.append_message(session.conversation_id, "user", content=message)
        reason, counts = "unknown", {}
        with execution_scope(settings.turn_timeout_seconds) as scope:
            collector = UsageCollector(scope)
            try:
                async with asyncio.timeout(settings.turn_timeout_seconds):
                    specs_list = [s for s in await registry.get_all_specs()
                                  if s.permission == "read" and s.name in READ_TOOLS
                                  and s.name not in WRITE_TOOLS]
                    specs = {s.name: s for s in specs_list}
                    tool_manifest = self.recorder.record([
                        {"name": s.name, "description": s.description, "json_schema": s.json_schema,
                         "source": s.source, "server": s.mcp_server, "permission": s.permission}
                        for s in specs_list], "actual_registered_readonly_tool_manifest")
                    # Callback binding also covers chat calls inside query_faq;
                    # it never mutates the global registry or candidate tools.
                    execution_specs = {s.name: replace(s, tool=s.tool.with_config({"callbacks": [collector]}))
                                       for s in specs_list}
                    # Preserve the course's model-controlled ReAct loop. RAG is
                    # available through query_faq, without current graph routing.
                    model = get_chat_model(streaming=True, temperature=self.temperature)
                    if specs_list:
                        model = model.bind_tools([s.tool for s in specs_list])
                    for _step in range(settings.max_agent_steps):
                        model_input = [SystemMessage(self.system_prompt), *session.messages]
                        ai = await model.ainvoke(model_input, config={"callbacks": [collector],
                            "metadata": {"langgraph_node": "course_baseline_react"}})
                        if not isinstance(ai, AIMessage):
                            raise ValueError("model did not return an AI message")
                        transcript.append({"input": [message_projection(m) for m in model_input],
                                           "output": message_projection(ai)})
                        session.messages.append(ai)
                        calls = ai.tool_calls or []
                        if not calls:
                            answer = message_projection(ai)["content"]
                            reason = "model_finished"
                            break
                        for call in calls:
                            name, args = call.get("name"), call.get("args")
                            signature = digest({"name": name, "args": args})
                            counts[signature] = counts.get(signature, 0) + 1
                            blocked = ("tool_budget_exceeded" if scope.tool_calls >= settings.max_task_tool_calls else
                                       "repeated_call_limit" if counts[signature] > settings.max_identical_tool_calls else
                                       "read_only_tool_denied" if name not in specs or name in WRITE_TOOLS else
                                       "order_tracking_not_authorized" if name == "query_logistics" and
                                       (not isinstance(args, dict) or args.get("tracking_no") not in session.authorized_tracking)
                                       else "order_not_authorized" if name in {"query_warranty", "query_return_status"} and
                                       (not isinstance(args, dict) or args.get("order_id") not in session.owned_orders)
                                       else None)
                            if blocked:
                                tm = ToolMessage(content="工具未执行：" + blocked, tool_call_id=call["id"],
                                                 name=name or "unknown", status="error")
                                record_tool_event(name or "unknown", ok=False, error_code=blocked)
                                result = {"tool": name, "ok": False, "business_status": "forbidden",
                                          "error_code": blocked, "data": None}
                            else:
                                run = await engine.execute_tool_call(call, session.conversation_id, execution_specs,
                                    user_id=session.user_id, confirmed=False, deadline=scope.deadline)
                                tm = run.tool_message
                                result = {"tool": run.name, "ok": run.ok, "business_status": run.business_status,
                                          "error_code": run.error_code, "data": run.data, "trace": run.trace}
                                if run.ok and name == "query_order" and isinstance(run.data, dict):
                                    if run.business_status == "ok":
                                        session.owned_orders.add(str(run.data.get("order_id")))
                                    tracking = run.data.get("tracking_no")
                                    if isinstance(tracking, str) and tracking:
                                        session.authorized_tracking.add(tracking)
                            result.update(tool_message=message_projection(tm), observed_at=now(), user_id=session.user_id,
                                          conversation_id=session.conversation_id, request_id=scope.request_id,
                                          tool_call_id=call["id"], arguments=args)
                            tool_results.append(result)
                            session.messages.append(tm)
                        if scope.tool_calls >= settings.max_task_tool_calls:
                            reason = "tool_budget_exceeded"
                            break
                    else:
                        reason = "agent_step_limit"
            except TimeoutError:
                reason = "deadline_exceeded"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reason = "baseline_execution_error"
                # Exception text may contain provider credentials or response data.
                transcript.append({"error_type": type(exc).__name__})
            scope.termination_reason = reason
            execution = scope.summary(settings.model_prices, settings.model_price_currency)
        # Pair pending tool calls before a subsequent turn, without inventing a
        # successful result or feeding an unmatched assistant call to the model.
        last_ai_index = next((i for i in range(len(session.messages) - 1, -1, -1)
                              if isinstance(session.messages[i], AIMessage)), None)
        if last_ai_index is not None:
            last_ai = session.messages[last_ai_index]
            answered = {m.tool_call_id for m in session.messages[last_ai_index + 1:] if isinstance(m, ToolMessage)}
            for call in last_ai.tool_calls or []:
                if call["id"] not in answered:
                    session.messages.append(ToolMessage(content="本轮已停止：" + reason,
                        tool_call_id=call["id"], name=call.get("name") or "unknown", status="error"))
        received = now()
        if answer is not None:
            await repository.append_message(session.conversation_id, "assistant", content=answer)
        observation = {"decision": None, "decision_source": None, "termination_reason": None,
                       "facts": {}, "evidence_sources": [], "actions": [],
                       "fact_observation_complete": False, "action_observation_complete": False,
                       "measurement": "course_baseline_missing_explicit_task_state"}
        recording = self.recorder.record({"user_id": session.user_id,
            "conversation_id": session.conversation_id, "request_sent_at": sent,
            "response_received_at": received, "messages": transcript,
            "tool_results": tool_results, "answer": answer, "execution_summary": execution},
            "actual_local_model_and_shared_executor")
        input_recording = self.recorder.record(message, "actual_local_baseline_input")
        answer_recording = self.recorder.record(answer, "actual_local_baseline_answer")
        return {"conversation_id": session.conversation_id, "user_message": input_recording["projection"],
                "input_recording": input_recording, "final_answer": answer_recording["projection"],
                "answer_recording": answer_recording, "baseline_recording": recording,
                "tool_manifest_recording": tool_manifest,
                "observed": observation, "runtime_termination_reason": reason,
                "execution_summary": execution, "latency_ms": execution["elapsed_ms"],
                "cost": execution["cost"], "currency": execution["currency"],
                "cost_kind": execution["cost_kind"],
                "usage_complete": execution["usage_coverage"] == 1.0,
                "scoring_note": "Task decision, business termination, facts and side-effect proof remain unknown; no text inference."}
