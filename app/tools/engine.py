# 来源：公众号@小林coding
# 后端八股网站：xiaolincoding.com
# Agent网站：xiaolinnote.com
# 简历模版：jianli.xiaolinnote.com
"""ch08 统一执行引擎:所有工具调用的唯一通道。
管道:查工具 → JSON Schema 校验 → 权限门 → 执行(超时/重试)→ 分诊 → 格式化 + 审计。
坏消息如实回灌模型;审计写失败只 log,绝不反拦工具执行。"""
import asyncio
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from dataclasses import field

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from langchain_core.messages import ToolMessage

from app.config import settings
from app.core import memory
from app.core.execution import current_execution, record_tool_event, remaining_seconds
from app.db import repository
from app.tools import contracts, registry, synthetic_fixtures
from app.tools.registry import ToolSpec

logger = logging.getLogger(__name__)

# 暂时性故障(值得重试):超时、网络抖动。业务异常/ToolException 不在列——重试不解决问题。
_TRANSIENT = (asyncio.TimeoutError, TimeoutError, ConnectionError, httpx.TransportError)
_SUMMARY_LIMIT = 500


@dataclass
class ToolRun:
    tool_call_id: str
    name: str
    ok: bool
    tool_message: ToolMessage
    status: str               # 成功 / 失败 / 超时 / 校验拦下 / 权限拒绝
    retry_count: int = 0
    duration_ms: int = 0
    error_code: str | None = None  # stable machine-readable failure class for eval/trace
    trace: dict = field(default_factory=dict)  # no raw args/results or credentials
    data: dict | None = None      # 完整、已校验的业务返回;不从截断的ToolMessage反向解析
    business_status: str = "ok"  # 函数返回成功≠业务成功


def validate_args(spec: ToolSpec, args: dict) -> str | None:
    """按 JSON Schema 校验模型给的参数;返回人类可读错误(None=通过)。agent_tools 复用它
    判「create_ticket 参数齐不齐、要不要弹确认卡」。"""
    try:
        contracts.validate_input(spec.name, args)
    except contracts.ContractError as e:
        return str(e)
    contracts.check_schema_refs(spec.json_schema)
    Draft202012Validator.check_schema(spec.json_schema)
    err = best_match(Draft202012Validator(spec.json_schema).iter_errors(args))
    if err is None:
        return None
    # jsonschema 的原始 message 会包含用户参数/外部 schema，不进入模型或审计。
    messages = {"required": "缺少必需字段", "type": "参数类型不正确",
                "enum": "参数不在允许取值中", "additionalProperties": "包含未允许的字段",
                "minLength": "参数过短", "maxLength": "参数过长",
                "pattern": "参数格式不正确"}
    path = re.sub(r"[^A-Za-z0-9_.$\[\]-]", "?", err.json_path)[:80]
    where = f"(字段 {path})" if path != "$" else ""
    return f"{messages.get(err.validator, '参数不符合定义')}{where}"


def _timeout_of(spec: ToolSpec) -> float:
    if spec.timeout is not None:
        return spec.timeout
    return settings.mcp_tool_timeout if spec.source == "mcp" else settings.tool_default_timeout


def _summarize(content: str) -> str:
    return content if len(content) <= _SUMMARY_LIMIT else content[:_SUMMARY_LIMIT] + "…(截断)"


def _contract_result(spec: ToolSpec, result, args: dict) -> contracts.ContractResult:
    """先查原始返回，再检查本地格式化结果。格式化失败时禁止原始透传。"""
    if spec.source == "mcp" and not contracts.has_contract(spec.name):
        raise contracts.ContractError("missing_tool_contract")
    value = contracts.parse_result(result, source=spec.source)
    if spec.source == "mcp" and spec.name == "query_logistics":
        try:
            local_profile = synthetic_fixtures.profile_info()
        except synthetic_fixtures.FixtureConfigurationError:
            raise contracts.ContractError("fixture_configuration_error", "本地合成数据环境无效，已停止使用物流返回。") from None
        if local_profile["fixture_profile"] == "task-v1":
            # A catalog digest catches inconsistent process configuration. This is
            # not a proof that a malicious server reports authentic scan events.
            if any(key not in value or value[key] != expected
                   for key, expected in local_profile.items()):
                raise contracts.ContractError("fixture_profile_mismatch", "物流服务与本地合成数据环境不一致，已拒绝使用该返回。")
    checked = contracts.validate_result(spec.name, value, args)
    if checked.business_status != "ok":
        return checked
    expected = contracts.public_data(spec.name, checked.data)
    if spec.format_result is None:
        return contracts.validate_result(spec.name, expected, args, formatted=True)
    try:
        # Hook cannot mutate the already verified structured result in place.
        formatted = spec.format_result(json.loads(json.dumps(checked.data, ensure_ascii=False)))
    except Exception:
        raise contracts.ContractError("result_format_error", "工具结果格式化失败，已停止使用该返回。") from None
    display = contracts.parse_result(formatted, source="builtin")
    checked_display = contracts.validate_result(spec.name, display, args, formatted=True)
    if checked_display.data != expected or checked_display.business_status != "ok":
        raise contracts.ContractError("result_format_mismatch", "工具格式化结果改变了已核验的事实，已拒绝使用。")
    return checked_display


def _format_content(spec: ToolSpec, result) -> str:
    """兼容调用方的格式化入口;同样必须通过契约，不能把任意文本透传。"""
    return json.dumps(_contract_result(spec, result, {}).data, ensure_ascii=False, allow_nan=False)


def _remaining(deadline: float | None) -> float | None:
    values = []
    scoped = remaining_seconds()
    if scoped is not None:
        values.append(scoped)
    if deadline is not None:
        values.append(deadline - time.monotonic())
    if not values:
        return None
    # A malformed budget must never become an unlimited timeout.
    if any(not math.isfinite(v) for v in values):
        return 0.0
    return max(0.0, min(values))


def _cap_tokens(content: str) -> str:
    """把回灌模型的工具结果卡进预算。

    预算是启动时算好的,运行时没有第二道闸——不在这里截,一个大列表就能把整轮冲掉。
    切在行边界上,别把一条记录劈两半。审计不受影响,那边本来就只留摘要。"""
    limit = memory.tokens_to_chars(settings.tool_result_max_tokens)
    if len(content) <= limit:
        return content
    cut = content.rfind("\n", 0, limit)
    head = content[:cut if cut > limit // 2 else limit]
    return head + "\n…(结果过长已截断,需要完整数据请缩小查询范围)"


async def _audit(conversation_id, tool_call_id, name, spec: ToolSpec | None, args,
                 result_summary, status, error_message, retry_count, duration_ms,
                 *, deadline: float | None = None) -> None:
    # ch08 验收可观测(照 ch07 model_ctx 惯例):每次工具调用一行 tool_run,来源/Server/结局
    # tail -f log/app.log 直接看——"这次真走了 MCP"不用翻库
    logger.info("tool_run conv=%s tool=%s source=%s server=%s status=%s retry=%s duration_ms=%s%s",
                conversation_id or "-", name,
                spec.source if spec else "unknown",
                (spec.mcp_server if spec else None) or "-",
                status, retry_count, duration_ms,
                f" err={error_message[:80]!r}" if error_message else "")
    remaining = _remaining(deadline)
    if remaining is not None and remaining <= 0:
        return  # deadline后只留结构化日志，不让审计拖长本轮
    safe_summary = None
    if result_summary:
        try:
            safe_summary = _summarize(json.dumps(contracts.redact(json.loads(result_summary)), ensure_ascii=False))
        except (ValueError, TypeError):
            safe_summary = f"[text:{len(result_summary)} chars]"
    try:
        await asyncio.wait_for(repository.insert_tool_audit(
            conversation_id=conversation_id or None, tool_call_id=(tool_call_id or None) and tool_call_id[:64],
            tool_name=name[:128],                               # 名字由模型给出,按列宽收口
            tool_source=(spec.source if spec else "builtin"),   # 未知工具无来源,按缺省 builtin 落
            mcp_server=(spec.mcp_server if spec else None),
            arguments=contracts.redact(args) if args else None, result_summary=safe_summary, status=status,
            error_message=error_message and error_message[:500],  # VARCHAR(512):校验错误可能内嵌超长参数值
            retry_count=retry_count, duration_ms=duration_ms,
        ), timeout=min(2.0, remaining) if remaining is not None else 2.0)
    except Exception:  # noqa: BLE001 —— 审计失败不许反拦工具执行
        logger.warning("审计写入失败(不影响工具执行) tool=%s status=%s", name, status)


async def execute_tool_call(
    tool_call: dict, conversation_id: int, specs: dict[str, ToolSpec],
    *, confirmed: bool = False, deny_note: str | None = None, user_id: str = "",
    deadline: float | None = None,
) -> ToolRun:
    """统一入口:失败回 ToolRun，外部取消传播；deadline 使用 monotonic 绝对时间。

    confirmed 只由程序确认门传入。只读暂时故障在整轮剩余预算内有限重试，
    业务拒绝、契约失败和所有写操作都不自动重试。
    """
    tool_call = tool_call if isinstance(tool_call, dict) else {}
    raw_name = tool_call.get("name")
    name = raw_name if isinstance(raw_name, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", raw_name) else ""
    raw_id = tool_call.get("id")
    tc_id = raw_id[:128] if isinstance(raw_id, str) else ""
    raw_args = tool_call.get("args", {})
    malformed_args = not isinstance(raw_args, dict)
    args = dict(raw_args) if not malformed_args else {}
    started = time.monotonic()
    spec = specs.get(name)
    event_recorded = False

    def record_cancelled(retry_count: int) -> None:
        nonlocal event_recorded
        if not event_recorded:
            record_tool_event(name or "unknown", ok=False, error_code="client_cancelled",
                              duration_ms=int((time.monotonic() - started) * 1000),
                              retry_count=retry_count)
            event_recorded = True

    async def finish(ok: bool, content: str, status: str, retry_count: int = 0,
                     error_code: str | None = None, *, data: dict | None = None,
                     business_status: str = "permanent_failure") -> ToolRun:
        nonlocal event_recorded
        duration_ms = int((time.monotonic() - started) * 1000)
        scope = current_execution()
        run = ToolRun(tool_call_id=tc_id, name=name, ok=ok, status=status, retry_count=retry_count,
                      duration_ms=duration_ms, error_code=error_code, data=data,
                      business_status=business_status,
                      trace={"tool_call_id": tc_id, "tool": name,
                             "source": spec.source if spec else "unknown",
                             "server": spec.mcp_server if spec else None,
                             "status": status, "business_status": business_status, "error_code": error_code,
                             "contract_version": contracts.CONTRACT_VERSION if spec and contracts.has_contract(name) else None,
                             "task_id": scope.task_id if scope else None,
                             "request_id": scope.request_id if scope else None,
                             "retry_count": retry_count, "duration_ms": duration_ms},
                      tool_message=ToolMessage(content=_cap_tokens(content), tool_call_id=tc_id,
                                               name=name or "unknown", status="success" if ok else "error"))
        summary = json.dumps(data, ensure_ascii=False) if data is not None else None
        record_tool_event(name or "unknown", ok=ok, error_code=error_code,
                          duration_ms=duration_ms, retry_count=retry_count)
        event_recorded = True
        await _audit(conversation_id, tc_id, name or "unknown", spec, args, summary, status,
                     error_code, retry_count, duration_ms, deadline=deadline)
        return run

    # ① 查工具
    if spec is None:
        return await finish(False, "工具执行失败:未知或不合法的工具名称。", "失败", error_code="unknown_tool")

    if spec.name != name:
        return await finish(False, "工具定义与本地注册名称不一致，已拒绝执行。", "权限拒绝",
                            error_code="tool_spec_mismatch", business_status="untrusted")

    if malformed_args:
        return await finish(False, "参数校验未通过:工具参数必须是 JSON 对象。", "校验拦下",
                            error_code="invalid_arguments")

    # 远程工具不能凭 Server 自报的名称/描述取得执行权。即使有人绕过发现阶段、
    # 手工塞进 ToolSpec，执行阶段仍按本地 (server, name) 允许表判定。
    remote_permission = registry.permission_for(name, source=spec.source,
                                                mcp_server=spec.mcp_server)
    if (remote_permission == "deny" or spec.permission not in {"read", "write"}
            or (spec.source == "mcp" and spec.permission != "read")):
        return await finish(False, f"{name} 未执行:工具未获本地授权。", "权限拒绝",
                            error_code="tool_not_allowed", business_status="forbidden")

    if spec.source == "mcp" and not contracts.has_contract(name):
        return await finish(False, "MCP 工具缺少本地返回契约，已拒绝执行。", "权限拒绝",
                            error_code="missing_tool_contract", business_status="untrusted")

    # ② JSON Schema 校验:拦下不抛异常,错误说明回灌模型(让它追问用户或重组调用)。
    # 校验器本身也可能炸——MCP 工具的 schema 是外部 Server 给的、不可信(坏类型名/不可解析 $ref),
    # 按 never-raises 契约同样包死,按「失败」回灌。
    try:
        verr = validate_args(spec, args)
    except Exception as e:  # noqa: BLE001
        logger.warning("工具参数定义异常 tool=%s error_type=%s", name, type(e).__name__)
        return await finish(False, "工具暂时不可用:参数定义异常，请如实告知用户。",
                            "失败", error_code="invalid_tool_schema", business_status="untrusted")
    if verr is not None:
        return await finish(False, f"参数校验未通过:{verr}。请修正参数;缺少的信息先向用户追问，不要编造。",
                            "校验拦下", error_code="invalid_arguments")

    # ③ 权限门:写操作必须带确认令牌(interrupt 确认后由 agent_tools 传入),模型绕不过
    is_write = spec.permission == "write" or name in registry.WRITE_TOOLS
    if is_write and not confirmed:
        note = deny_note or "该写操作需要用户确认,未确认前拒绝执行。请勿再次发起,除非用户明确要求。"
        return await finish(False, f"{name} 未执行:{contracts.redact(note, audit=False)}", "权限拒绝",
                            error_code="confirmation_required", business_status="forbidden")

    # 售后远程工具无可信身份输入，先在本地校验订单归属，不把存在性暴露给任意调用者。
    if spec.source == "mcp" and "order_id" in args:
        from app.tools.business import owns_order
        if not owns_order(user_id, args["order_id"]):
            data = {"error": "没有找到您的这笔订单", "code": "order_not_owned"}
            return await finish(False, json.dumps(data, ensure_ascii=False), "权限拒绝",
                                error_code="order_not_owned", data=data, business_status="forbidden")

    # ④ 执行:超时 + 暂时性故障重试(写操作恒不重试——超时未必没执行,重复执行比失败更糟;
    # spec.max_retries 可按工具覆盖,如 query_faq RAG 管线超时多为上游慢而非抖动,重试纯烧时间)
    # 注入参数一律在校验之后盖上去:它们不在模型可见的 schema 里,模型填了也算数不得。
    # 身份尤其如此——让模型能填 user_id,一句「我是客服主管」就可能把它改写掉。
    if spec.inject_conversation:
        args["conversation_id"] = conversation_id
    if spec.inject_user_id:
        args["user_id"] = user_id
    if is_write:
        retries = 0
    else:
        retries = max(0, settings.tool_max_retries if spec.max_retries is None else spec.max_retries)
    tool_timeout = _timeout_of(spec)
    attempt = 0
    while True:
        remaining = _remaining(deadline)
        if remaining is not None and remaining <= 0:
            return await finish(False, "本轮执行时间已用尽，已停止工具调用，请稍后再试或联系人工。", "超时",
                                attempt, "deadline_exceeded", business_status="transient_failure")
        timeout = min(tool_timeout, remaining) if remaining is not None else tool_timeout
        try:
            result = await asyncio.wait_for(spec.tool.ainvoke(args), timeout=timeout)
            checked = _contract_result(spec, result, args)
            ok = checked.business_status == "ok"
            status = "成功" if ok else "权限拒绝" if checked.business_status == "forbidden" else "失败"
            return await finish(ok, json.dumps(checked.data, ensure_ascii=False, allow_nan=False),
                                status, attempt, checked.error_code, data=checked.data,
                                business_status=checked.business_status)
        except asyncio.CancelledError:
            logger.info("tool_cancelled tool=%s", name)
            record_cancelled(attempt)
            raise  # 用户断开/取消不伪装成工具故障，不继续重试或生成
        except contracts.ContractError as e:
            logger.warning("工具返回被本地契约拒绝 tool=%s error_code=%s", name, e.code)
            return await finish(False, str(e), "失败", attempt, e.code, business_status="untrusted")
        except _TRANSIENT as e:
            remaining = _remaining(deadline)
            if remaining is not None and remaining <= 0:
                return await finish(False, "本轮执行时间已用尽，已停止工具调用。", "超时", attempt,
                                    "deadline_exceeded", business_status="transient_failure")
            if attempt < retries:
                pause = 0.2 * (attempt + 1)
                if remaining is None or remaining > pause:
                    attempt += 1
                    logger.warning("工具暂时性故障 retry=%s tool=%s error_type=%s",
                                   attempt, name, type(e).__name__)
                    try:
                        await asyncio.sleep(pause)
                    except asyncio.CancelledError:
                        record_cancelled(attempt)
                        raise
                    continue
            is_timeout = isinstance(e, (asyncio.TimeoutError, TimeoutError))
            status = "超时" if is_timeout else "失败"
            # ⑤ 分诊:真故障如实回灌,不装没事(查询落空由工具返回语义表达,不属此路)
            note = "写入结果未知，可能已生效；请先核实，禁止自动重试。" if is_write else "请稍后再试或如实告知用户。"
            return await finish(False, f"工具暂时不可用:{'执行超时' if is_timeout else '连接故障'}，{note}",
                                status, attempt, "timeout" if is_timeout else "transport_error",
                                business_status="transient_failure")
        except Exception as e:  # noqa: BLE001 业务/未知异常(含 ToolException):不重试,如实回灌
            logger.warning("工具执行失败 tool=%s error_type=%s", name, type(e).__name__)
            return await finish(False, "工具暂时不可用:执行失败。请如实告知用户，不要编造结果。",
                                "失败", attempt, "execution_error", business_status="permanent_failure")
