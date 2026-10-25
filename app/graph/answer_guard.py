"""Pure, conservative final-answer checks for the local readonly Agent demo.

This is NOT semantic claim-entailment verification. It proves only that required
fresh facts exist in the current task scope, detects obvious order mix-ups and
some explicit side-effect completion/promises. It cannot prove that every date,
amount, qualification or policy in natural language is supported. Policy scope,
contradiction checking and evaluated semantic claim support remain separate work.

No model/DB calls, no wall-clock reads: the caller supplies fresh updated_at in
TaskContext. Labelled order numbers and qualified bare numbers avoid interpreting
ordinary years, prices or duration as orders. The regex checks are heuristics;
reported evidence_ids are provenance, never a declaration of semantic proof.
"""
from __future__ import annotations

import re
from datetime import datetime
from app.core.policy_evidence import render_assessment

MAX_FACT_AGE_SECONDS = 900
_CONTROL_REASONS = {
    "cancelled", "client_cancelled", "deadline_exceeded", "execution_failed", "superseded",
    "forbidden", "permission_denied", "tool_budget_exceeded", "tool_budget_exhausted",
    "repeated_call_limit", "repeated_tool_call", "step_budget_exceeded", "context_budget_exceeded",
    "write_service_not_ready", "task_cancelled", "tool_evidence_unavailable",
}
_ORDER_LABEL = re.compile(
    r"(?:订单(?:号)?|order(?:_id)?)\s*(?:[号为是:#：]\s*)*[（(]?\s*([0-9]{4,})", re.I)
_BARE_NUMBER = re.compile(r"(?<![A-Za-z0-9])([0-9]{4,64})(?![A-Za-z0-9])")
_NUMBER_UNIT = re.compile(r"\s*(?:年|月|日|元|块钱|小时|分钟|秒|天|米|公里|件|个|%|％|[-/]\d{1,2}[-/]\d{1,2})")
_PRICE_PREFIX = re.compile(r"(?:价格|售价|金额|价款|费用|人民币|赔偿|补偿|退款金额|[¥￥])\s*(?:为|是|约|大约|最高为)?\s*[:：]?\s*$")
_ORDER_NUMBER_SUFFIX = re.compile(r"\s*(?:号订单|(?:的\s*)?(?:订单|物流|快递|包裹)|已(?:发货|签收|付款)|为(?:待付款|已付款|已发货|已签收))")
_CLAUSE_SPLIT = re.compile(r"[。！？!?；;\n]")
_SIDE_EFFECT_PATTERNS = [
    re.compile(r"(?:已经|现已|成功|刚刚|刚才|已).{0,12}(?:创建|建立|提交|生成|建好|建了).{0,8}(?:工单|退款申请|退款单)"),
    re.compile(r"(?:工单|退款申请|退款单).{0,12}(?:已经|已|成功|完成).{0,8}(?:创建|建立|提交|生成|办好|处理)"),
    re.compile(r"(?:已经|现已|成功|已).{0,8}(?:为您|为你|帮您|帮你)?.{0,4}(?:退款|退还|退回)(?:了|到账|成功|完成|\s*\d)"),
    re.compile(r"(?:我|我们|系统|这边).{0,6}(?:已经|现已|成功|已).{0,8}(?:申请|办理|执行|完成).{0,5}退款"),
    re.compile(r"(?:已经|现已|成功|已)(?:完成|执行|办理|提交)(?:了|为您|为你|帮您|帮你)?退款"),
    re.compile(r"(?:已经|现已|成功|已)(?:为您|为你|帮您|帮你|给您|给你).{0,5}(?:退款|退还|退回)(?:了|成功|完成|到账|$)"),
    re.compile(r"(?:退款|退款金额|款项).{0,8}(?:已经|已|成功).{0,6}(?:到账|退还|退回|完成)"),
    re.compile(r"(?:我|我们|系统|这边).{0,6}(?:将|会|马上|立即|立刻).{0,8}(?:创建|提交|生成|办理).{0,8}(?:工单|退款)"),
    re.compile(r"(?:保证|承诺).{0,8}(?:退款|赔付|赔偿|补偿|退还)"),
    re.compile(r"(?:ticket\s+(?:has\s+been|was|is)\s+(?:created|submitted)|refund\s+(?:has\s+been|was|is)\s+(?:processed|issued)|I\s+(?:have\s+|will\s+)(?:created?|submitted?|issued?).{0,12}(?:ticket|refund))", re.I),
]
_NEGATION = re.compile(r"未|没有|尚未|并未|不能|不会|不曾|不支持|无法|拒绝|暂停|停止|取消|无需|不要|勿|not\b|never\b|cannot\b|can't\b", re.I)


def _result(answer: str, *, allowed: bool, reason: str = "", ids: list[str] | None = None,
            termination: str | None = None) -> dict:
    return {"answer": answer, "allowed": allowed, "reason": reason,
            "evidence_ids": list(dict.fromkeys(ids or [])), "termination_reason": termination}


def _scope(context: dict) -> dict:
    return {k: context.get(k) for k in ("user_id", "conversation_id", "task_id", "active_order_id")}


def _fresh_fact(context: dict, name: str, sources: set[str]) -> dict | None:
    facts = context.get("verified_facts") or {}
    if not isinstance(facts, dict):
        return None
    fact = facts.get(name)
    if (not isinstance(fact, dict) or fact.get("scope") != _scope(context)
            or not isinstance(fact.get("source"), str) or fact["source"] not in sources
            or not isinstance(fact.get("evidence_id"), str)
            or not fact["evidence_id"]):
        return None
    try:
        now = datetime.fromisoformat(context["updated_at"].replace("Z", "+00:00"))
        observed = datetime.fromisoformat(fact["observed_at"].replace("Z", "+00:00"))
        age = (now - observed).total_seconds()
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    if age < 0 or age > MAX_FACT_AGE_SECONDS:
        return None
    return fact


def _historical_refund_clause(clause: str, context: dict) -> bool:
    """A narrow read-status statement is not a claim that this Agent refunded.

    This exception requires fresh owned-order and return-status evidence. It
    does not cover bank arrival, amounts, future promises or new applications.
    """
    active = str(context.get("active_order_id") or "")
    owned = _fresh_fact(context, "order_owned", {"query_order"})
    order = _fresh_fact(context, "order_id", {"query_order"})
    returned = _fresh_fact(context, "return_status", {"query_return_status"})
    if (not active or not context.get("user_id") or not context.get("task_id")
            or not owned or owned.get("value") is not True
            or not order or order.get("value") != active
            or not returned or returned.get("value") != "已退款"):
        return False
    pattern = (r"\s*(?:查询结果显示|查询记录显示|售后查询显示|售后记录显示|退货状态显示)[，,:：\s]*"
               r"(?:(?:这笔|当前)(?:订单)?(?:的)?|订单\s*" + re.escape(active) + r"\s*(?:的)?)?"
               r"(?:退款(?:状态)?(?:为|是|显示)?已退款|退款(?:已经|已)(?:完成|退回)|退货状态(?:为|是)?已退款)\s*")
    return re.fullmatch(pattern, clause) is not None


def _side_effect_claim(answer: str, context: dict) -> bool:
    for clause in _CLAUSE_SPLIT.split(answer):
        if _historical_refund_clause(clause, context):
            continue
        for pattern in _SIDE_EFFECT_PATTERNS:
            for match in pattern.finditer(clause):
                # A negative explanation such as '尚未创建工单' or '已拒绝提交'
                # is safe. Look near this match so an earlier '没有收到' does
                # not conceal a later independent '已经创建工单' assertion.
                nearby = clause[max(0, match.start() - 5):match.end()]
                if not _NEGATION.search(nearby):
                    return True
    return False


def _mentioned_orders(answer: str, context: dict) -> set[str]:
    labelled = set(_ORDER_LABEL.findall(answer))
    known = set(str(v) for v in context.get("explicit_order_ids", []))
    known.add(str(context.get("active_order_id") or ""))
    out = set(labelled)
    for match in _BARE_NUMBER.finditer(answer):
        number = match.group(1)
        if number in labelled:
            continue
        if _NUMBER_UNIT.match(answer[match.end():]) or _PRICE_PREFIX.search(answer[max(0, match.start() - 12):match.start()]):
            continue
        # An isolated four-digit year remains a year unless the current user
        # explicitly supplied it as an order. A labelled '订单2002' is kept.
        if 1900 <= int(number) <= 2100 and number not in known:
            continue
        # Unqualified numbers are not orders just because they have four
        # digits. Keep user-supplied identifiers, or an order/status context.
        if number in known or _ORDER_NUMBER_SUFFIX.match(answer[match.end():]):
            out.add(number)
    return out


def _compound_observed_text(state: dict, context: dict) -> tuple[str, list[str]]:
    """Keep independently verified requested facts when policy remains unknown.

    Render only current, fresh, owned-order tool observations. No candidate
    model sentence is retained, and this does not assert policy eligibility,
    receipt of money or a new backend action.
    """
    plan = state.get("compound_plan") or {}
    if state.get("route") != "compound" or plan.get("protocol") != "bounded-compound-v1":
        return "", []
    owned = _fresh_fact(context, "order_owned", {"query_order"})
    order = _fresh_fact(context, "order_id", {"query_order"})
    if not owned or owned.get("value") is not True or not order or order.get("value") != context.get("active_order_id"):
        return "", []
    fields = {
        "order": (("order_status", "订单状态", "query_order"),),
        "logistics": (("logistics_status", "物流状态", "query_logistics"),
                      ("logistics_current_city", "物流当前城市", "query_logistics"),
                      ("logistics_stalled_hours", "距最近物流更新的小时数", "query_logistics")),
        "warranty": (("warranty", "查询记录中的在保状态", "query_warranty"),
                     ("warranty_until", "查询记录中的保修截止日期", "query_warranty")),
        "return_status": (("return_status", "既有退货记录状态", "query_return_status"),
                          ("updated_at", "既有退货记录更新时间", "query_return_status")),
    }
    lines, ids = [], []
    for check in dict.fromkeys(plan.get("checks", [])):
        for name, label, source in fields.get(check, ()):
            fact = _fresh_fact(context, name, {source})
            value = fact.get("value") if fact else None
            if value is not None and isinstance(value, (str, int, float, bool)):
                lines.append(f"- {label}：{str(value)[:200]}")
                ids.append(fact["evidence_id"])
    if not lines:
        return "", []
    return (f"订单{context['active_order_id']}的已核验查询记录（本次仅查询）：\n"
            + "\n".join(lines) + "\n\n", ids)


def guard_answer(state: dict, answer: str) -> dict:
    """Return a safe final text and diagnostic provenance; never execute an action."""
    text = answer if isinstance(answer, str) else ""
    context = state.get("task_context") or {}
    if not isinstance(context, dict):
        context = {}
    terminal = context.get("termination_reason", "")

    if _side_effect_claim(text, context):
        return _result("当前演示尚未创建工单或执行退款。可以继续核验问题并提供人工交接建议。",
                       allowed=False, reason="readonly_action_claim", termination="unsupported_action_claim")

    # A server-produced controlled failure already has a safe answer. Do not
    # replace a cancellation/permission/deadline explanation with a generic
    # evidence warning, or accidentally reclassify it as successful completion.
    if terminal in _CONTROL_REASONS and text.strip():
        return _result(text, allowed=True, reason="controlled_termination", termination=terminal)

    route, intent = state.get("route", ""), state.get("intent", "")
    order_required = intent == "订单" or route in {"refund_flow", "compound"}
    # The deterministic logistics investigator has its own contract guard.
    # This branch concerns main-agent answers that happen to involve logistics.
    logistics_required = (route != "logistics" and (
        intent == "物流" or (route in {"business", "compound"} and context.get("topic") == "logistics")))
    if route == "compound" and (state.get("compound_plan") or {}).get("protocol") == "bounded-compound-v1":
        # Topic keywords also occur in negated requests ("不查物流，只查在保").
        # The validated current-turn checklist, rather than the coarse topic,
        # defines the evidence required by this bounded investigation.
        logistics_required = "logistics" in state["compound_plan"].get("checks", [])
    active = str(context.get("active_order_id") or "")
    selected: list[str] = []

    if active and route in {"business", "refund_flow"} and _mentioned_orders(text, context) - {active}:
        return _result("答复中的订单与当前查询对象不一致，已停止显示该结论。请确认要查询的订单。",
                       allowed=False, reason="answer_order_mismatch", termination="unverified_answer")

    if order_required or logistics_required:
        if not active:
            return _result("请提供要查询的订单号，或从您自己的订单中选择一笔。", allowed=False,
                           reason="missing_order_id", termination="clarification_required")
        if (not context.get("user_id") or not context.get("task_id")
                or (state.get("user_id") is not None and state["user_id"] != context.get("user_id"))
                or (state.get("conversation_id") is not None and state["conversation_id"] != context.get("conversation_id"))):
            return _result("当前订单身份或任务作用域无法核验，请重新选择自己的订单。", allowed=False,
                           reason="invalid_task_scope", termination="unverified_answer")
        owned = _fresh_fact(context, "order_owned", {"query_order"})
        order = _fresh_fact(context, "order_id", {"query_order"})
        if not owned or owned.get("value") is not True or not order or order.get("value") != active:
            return _result("这笔订单尚未完成当前身份与订单信息核验，暂时不能据此判断；请重试或联系人工客服。",
                           allowed=False, reason="current_order_evidence_required", termination="unverified_answer")
        selected.extend([owned["evidence_id"], order["evidence_id"]])
        if _mentioned_orders(text, context) - {active}:
            return _result("答复中的订单与当前查询对象不一致，已停止显示该结论。请确认要查询的订单。",
                           allowed=False, reason="answer_order_mismatch", ids=selected,
                           termination="unverified_answer")
        if logistics_required:
            status = (_fresh_fact(context, "logistics_status", {"query_logistics"})
                      or _fresh_fact(context, "logistics_status_code", {"query_logistics"}))
            if not status:
                return _result("当前订单的物流信息尚未核验，不能确认物流状态；请稍后重试或联系人工客服。",
                               allowed=False, reason="current_logistics_evidence_required", ids=selected,
                               termination="unverified_answer")
            selected.append(status["evidence_id"])

        if route == "compound" and (state.get("compound_plan") or {}).get("protocol") == "bounded-compound-v1":
            other_checks = {"warranty": ("在保", "query_warranty", ("warranty", "warranty_until")),
                            "return_status": ("既有退货进度", "query_return_status", ("return_status", "updated_at"))}
            for check in state["compound_plan"].get("checks", []):
                if check not in other_checks:
                    continue
                label, source, names = other_checks[check]
                facts = [_fresh_fact(context, name, {source}) for name in names]
                if any(fact is None for fact in facts):
                    partial, partial_ids = _compound_observed_text(state, context)
                    return _result(partial + f"本次尚未取得{label}的完整可靠查询记录，不能确认该项结果；请重试或联系人工客服核实。",
                                   allowed=False, reason="requested_check_evidence_required", ids=selected + partial_ids,
                                   termination="unverified_answer")
                selected.extend(fact["evidence_id"] for fact in facts)

    if route in {"knowledge", "refund_flow"}:
        citations = state.get("citations") or []
        failed = (state.get("fallback_source") in {"retrieval_low_conf", "self_check"}
                  or not isinstance(citations, list) or not citations or not state.get("evidence")
                  or state.get("evidence_strong") is not True)
        if failed:
            return _result("当前检索证据不足，暂时不能确认政策或退款资格。请补充条件或联系人工客服核实。",
                           allowed=False, reason="policy_evidence_insufficient", ids=selected,
                           termination="insufficient_evidence")
        # Citation IDs identify source records; they are not assertion proof.
        selected.extend(f"knowledge:{c['id']}" for c in citations
                        if isinstance(c, dict) and isinstance(c.get("id"), (str, int))
                        and not isinstance(c.get("id"), bool))

    # Critical policy conclusions use a controlled renderer. A citation number
    # on an arbitrary model sentence cannot prove eligibility, amount or dates.
    # Logistics is independently rendered by its investigator; its policy
    # subtask does not erase an otherwise supported anomaly draft.
    if route != "logistics" and state.get("policy_application_required") is True:
        assessment = state.get("policy_assessment") or {}
        observed_text, observed_ids = _compound_observed_text(state, context)
        selected.extend(observed_ids)
        policy_text = observed_text + render_assessment(assessment)
        if assessment.get("outcome") == "applicable" and assessment.get("supported_claims"):
            return _result(policy_text, allowed=True,
                           reason="structured_policy_conditions", ids=selected)
        return _result(policy_text, allowed=False,
                       reason="policy_scope_unverified", ids=selected,
                       termination="policy_evidence_unverified")

    if not text.strip():
        return _result("当前未形成可核验的答复，请补充问题或联系人工客服。", allowed=False,
                       reason="empty_answer", ids=selected, termination="unverified_answer")
    return _result(text, allowed=True, reason="required_evidence_present", ids=selected)
