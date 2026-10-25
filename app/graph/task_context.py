"""Task facts are structured data, not a conversation summary or model reasoning.

All transitions in this module are pure: callers supply IDs and observation time.
The JSON-only result can be persisted by the graph checkpointer. Only successful,
validated tool results may become verified facts; user statements stay claims.
"""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import re
from typing import Any


SCHEMA_VERSION = 1
_ORDER_RE = re.compile(r"(?<![A-Za-z0-9])(\d{4,})(?![A-Za-z0-9])")
_LABELLED_ORDER_RE = re.compile(r"(?:订单(?:号)?|单号|order(?:\s+id)?)\s*[:：#]?\s*(\d{4,})", re.I)
_CANCEL_RE = re.compile(r"(?:不查了|不用(?:再)?查|(?:停止|取消)(?:这次|本次|当前|这个|此次)?(?:的)?(?:查询|核查|调查)|取消刚才|取消这个请求|不办了|不用了|算了)")
_CORRECT_RE = re.compile(r"(?:不是|说错|写错|改成|换成|更正|实际是|应该是|是指|另一笔)")
_CORRECTION_TARGET_RE = re.compile(
    r"(?:改成|换成|更正(?:为)?|实际是|应该是|是指|[，,]\s*是)\s*(?:订单(?:号)?\s*[:：#]?\s*)?(\d{4,})")
_NON_ORDER_NUMBER_RE = re.compile(r"(?:手机|电话|价格|金额|费用|日期|地址)")
_ORDER_REFERENCE_RE = re.compile(r"(?:订单(?:号)?|单号|那笔|这笔|这单|那个号|编号|另一笔)")
_LOGISTICS_TERMS = ("物流", "快递", "包裹", "签收", "没收到", "未收到", "运输", "停滞")
_PRODUCT_TERMS = ("商品", "库存", "规格", "产品", "猫粮", "猫砂", "猫爬架")
_MAX_CLAIMS = 12
_MAX_ACTIONS = 16
_MAX_EVIDENCE = 40


def _copy(context: dict | None) -> dict:
    return deepcopy(context or {})


def _non_order_subject(prefix: str) -> bool:
    non_order = list(_NON_ORDER_NUMBER_RE.finditer(prefix))
    order_refs = list(_ORDER_REFERENCE_RE.finditer(prefix))
    return bool(non_order) and (
        not order_refs or non_order[-1].start() > order_refs[-1].start())


def explicit_order_ids(message: str) -> list[str]:
    """Accept explicit order references; conservatively exclude dates and phones."""
    labelled = _LABELLED_ORDER_RE.findall(message)
    # Bare demo IDs also work (e.g. a selector sends only "3001"). Longer
    # numbers require an order label to avoid phone and logistics-number leakage.
    numbers = []
    for match in _ORDER_RE.finditer(message):
        oid = match.group(1)
        prefix = message[max(0, match.start(1) - 32):match.start(1)]
        if (len(oid) <= 6 and not _non_order_subject(prefix) and not re.match(
                r"\s*(?:年|月|日|小时|分钟|秒|天|元|块钱|[-/]\d{1,2}[-/]\d{1,2})",
                message[match.end(1):])):
            numbers.append(oid)
    return list(dict.fromkeys(oid for oid in labelled + numbers if len(oid) <= 64))


def _correction_targets(message: str) -> tuple[list[str], set[str]]:
    """Recognize order corrections without promoting a changed phone/price.

    The nearest explicit subject in the short local prefix determines whether
    a number change refers to an order. This is a conservative parser, not an
    unrestricted entity extractor; ownership is still enforced by tools.
    """
    targets, rejected = [], set()
    for match in _CORRECTION_TARGET_RE.finditer(message):
        target = match.group(1)
        prefix = message[max(0, match.start() - 32):match.start(1)]
        invalid_shape = len(target) > 64 or bool(re.match(
            r"\s*(?:年|月|日|元|块钱|小时|分钟|秒|天|[-/]\d{1,2}[-/]\d{1,2})",
            message[match.end(1):]))
        if _non_order_subject(prefix) or invalid_shape:
            rejected.add(target)
        else:
            targets.append(target)
    return list(dict.fromkeys(targets)), rejected


def _topic(message: str) -> str:
    if any(term in message for term in _LOGISTICS_TERMS):
        return "logistics"
    if "订单" in message:
        return "order"
    if any(term in message for term in ("退款", "退货", "售后")):
        return "refund"
    if any(term in message for term in _PRODUCT_TERMS):
        return "product"
    if any(term in message for term in ("政策", "规则", "运费", "赔偿", "补偿")):
        return "policy"
    if any(term in message for term in ("换个问题", "换个话题", "聊点别的")):
        return "other"
    return ""


def _cancel_requested(message: str) -> bool:
    message = message.strip()
    for match in _CANCEL_RE.finditer(message):
        prefix = re.split(r"[。！？!?；;，,\n]", message[:match.start()])[-1][-16:]
        if re.search(r"不要|别|勿|不想|不打算|没有说|怎么|如何|如果|假如|假设|是否|上次|之前", prefix):
            continue
        if message[max(0, match.start() - 1):match.start()] in {"“", "‘", "'", '"'}:
            continue
        # A negated named subtask is not cancellation of the entire request:
        # "不用查在保，只查物流" must reach request triage. This parser handles
        # explicit global cancellation only; requested check selection stays
        # with triage and still cannot bypass the execution/ownership guards.
        suffix = message[match.end():].lstrip()
        if re.match(r"(?:询)?(?:在保|保修|质保|维修|物流|快递|运输|退货|售后)", suffix):
            continue
        return True
    return False


def begin_turn(previous: dict | None, user_id: str, conversation_id: int,
               message: str, ids: dict | None = None) -> dict:
    """Start a request, retain scoped task facts, and invalidate stale actions.

    ``ids`` contains request_id, task_id and now (UTC ISO string). A resume does
    not call begin_turn: its existing budget and order-selection pause survive.
    An ordinary new message resets the per-request budget. Scope change, explicit
    order correction, unrelated topic and cancellation erase old tool evidence.
    """
    ids = ids or {}
    old = _copy(previous)
    same_scope = (old.get("user_id") == user_id
                  and old.get("conversation_id") == conversation_id
                  and old.get("schema_version") == SCHEMA_VERSION)
    if not same_scope:
        old = {}
    candidates = explicit_order_ids(message)
    cancelled = _cancel_requested(message)
    topic = _topic(message)
    old_topic = old.get("topic", "")
    # Logistics and an order/status or policy follow-up form one task. Product
    # and refund are separate tasks; their stale logistics evidence is removed.
    related = {topic, old_topic} <= {"", "logistics", "order", "policy"}
    topic_changed = bool(topic and old_topic and topic != old_topic and not related)
    raw_targets, rejected_targets = _correction_targets(message)
    candidates = [oid for oid in candidates if oid not in rejected_targets]
    # A bare follow-up can correct the active identifier without repeating its
    # label. Equal length avoids treating a changed phone/date/price as an order.
    same_identifier_shape = bool(old.get("active_order_id")) and any(
        len(target) == len(str(old["active_order_id"])) for target in raw_targets)
    unrelated_number = bool(_NON_ORDER_NUMBER_RE.search(message))
    correction = bool(_CORRECT_RE.search(message)) and bool(
        candidates or _ORDER_REFERENCE_RE.search(message)
        or (same_identifier_shape and not unrelated_number)) and not (
            rejected_targets and not raw_targets)
    targets = raw_targets if correction else []
    # An explicit correction may contain long IDs without repeating '订单号'
    # before each number. Keep these targets only inside an order correction;
    # ordinary bare phone numbers and prices still do not select an order.
    candidates = list(dict.fromkeys(candidates + targets))
    selected = targets[0] if len(targets) == 1 and targets[0] in candidates else candidates[0] if len(candidates) == 1 else ""
    ambiguous = len(candidates) > 1 and not selected
    changed_order = (selected and old.get("active_order_id")
                     and old["active_order_id"] != selected)
    invalidation = ("cancelled" if cancelled else "topic_changed" if topic_changed
                    else "ambiguous_order" if ambiguous else "order_changed" if changed_order
                    else "order_correction_without_id" if correction and not candidates else "")
    if selected:
        order_id = selected
        origin = "current_message_correction" if targets else "current_message"
    elif not ambiguous and not invalidation:
        order_id = old.get("active_order_id", "")
        origin = "session_active_order" if order_id else "missing"
    else:
        order_id, origin = "", "ambiguous" if ambiguous else "missing"
    if cancelled:
        order_id = ""
    keep_facts = bool(old and not invalidation)
    limits = old.get("budgets") or {}
    now = str(ids.get("now") or "")
    new_task = not old or bool(invalidation)
    task_topic = topic or old_topic
    goal = old.get("goal", message) if selected and message.strip() == selected and keep_facts else message
    context = {
        "schema_version": SCHEMA_VERSION,
        "task_id": str(ids.get("task_id") or old.get("task_id") or "") if new_task else old.get("task_id", ""),
        "request_id": str(ids.get("request_id") or ""),
        "user_id": user_id, "conversation_id": conversation_id,
        "started_at": now if new_task else old.get("started_at", now),
        "updated_at": now,
        "goal": goal[:500], "topic": task_topic,
        "active_order_id": order_id, "order_origin": origin,
        "explicit_order_ids": candidates,
        "missing_information": ["order_id"] if not order_id and not cancelled and task_topic in {"logistics", "order", "refund"} else [],
        "user_claims": (old.get("user_claims", []) if keep_facts else [])[-(_MAX_CLAIMS - 1):],
        "verified_facts": old.get("verified_facts", {}) if keep_facts else {},
        "evidence": old.get("evidence", []) if keep_facts else [],
        # A new message cannot implicitly confirm a preview from the previous turn.
        "pending_actions": [], "actions": [],
        "decision": None, "decision_source": None,
        "budgets": {"steps": 0, "tool_calls": 0, "repeated_signatures": {},
                    "max_steps": int(limits.get("max_steps", 8)),
                    "max_tool_calls": int(limits.get("max_tool_calls", 8))},
        "termination_reason": "cancelled" if cancelled else "",
        "invalidation_reason": invalidation, "cancelled": cancelled,
    }
    if message.strip():
        context["user_claims"].append({
            "text": message[:500], "source": "user_message", "observed_at": now,
            "request_id": context["request_id"], "scope": _scope(context),
        })
    return context


def _scope(context: dict) -> dict:
    return {key: context.get(key) for key in
            ("user_id", "conversation_id", "task_id", "active_order_id")}


def with_goal(context: dict, goal: str, topic: str = "logistics") -> dict:
    out = _copy(context)
    out.update(goal=goal[:500], topic=topic)
    return out


def set_missing(context: dict, fields: list[str]) -> dict:
    out = _copy(context)
    out["missing_information"] = list(dict.fromkeys(fields))
    return out


def _scoped_fact(context: dict, name: str) -> dict | None:
    fact = (context.get("verified_facts") or {}).get(name)
    if not isinstance(fact, dict) or fact.get("scope") != _scope(context):
        return None
    return fact


def _expired(context: dict, fact: dict, max_seconds: int = 900) -> bool:
    """Missing/invalid timestamps cannot authorize a dependent logistics call."""
    try:
        observed = datetime.fromisoformat(fact["observed_at"].replace("Z", "+00:00"))
        current = datetime.fromisoformat(context["updated_at"].replace("Z", "+00:00"))
        if observed.tzinfo is None or current.tzinfo is None:
            return True
        age = (current - observed).total_seconds()
        return age < 0 or age > max_seconds
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return True


def current_facts(context: dict) -> dict:
    """Only fresh facts in the active identity/order scope enter model context."""
    return {name: deepcopy(fact)
            for name in (context.get("verified_facts") or {})
            if (fact := _scoped_fact(context, name)) is not None
            and not _expired(context, fact)}


def guard_tool_call(context: dict, name: str, args: dict) -> str | None:
    """A model's guessed tracking number can never authorize query_logistics."""
    if context.get("cancelled"):
        return "task_cancelled"
    if not context.get("user_id"):
        return "missing_identity"
    if name in {"query_order", "query_warranty", "query_return_status"}:
        order_id = str(args.get("order_id") or "")
        if not order_id or not context.get("active_order_id"):
            return "missing_order_id"
        if order_id != context["active_order_id"]:
            return "order_scope_mismatch"
        if name != "query_order":
            owned = _scoped_fact(context, "order_owned")
            if not owned or owned.get("value") is not True or _expired(context, owned):
                return "order_evidence_required"
    if name == "query_logistics":
        owned = _scoped_fact(context, "order_owned")
        tracking = _scoped_fact(context, "tracking_no")
        if not owned or owned.get("value") is not True or not tracking:
            return "order_evidence_required"
        if _expired(context, owned) or _expired(context, tracking):
            return "order_evidence_expired"
        if args.get("tracking_no") != tracking.get("value"):
            return "tracking_scope_mismatch"
    return None


def tool_signature(name: str, args: dict) -> str:
    payload = json.dumps({"name": name, "args": args}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def consume_tool(context: dict, name: str, args: dict) -> dict:
    out = _copy(context)
    budget = out.setdefault("budgets", {})
    budget["tool_calls"] = int(budget.get("tool_calls", 0)) + 1
    counts = budget.setdefault("repeated_signatures", {})
    signature = tool_signature(name, args)
    counts[signature] = int(counts.get(signature, 0)) + 1
    return out


def consume_step(context: dict) -> dict:
    out = _copy(context)
    budget = out.setdefault("budgets", {})
    budget["steps"] = int(budget.get("steps", 0)) + 1
    return out


def record_action(context: dict, action: dict, *, pending: bool = False) -> dict:
    out = _copy(context)
    item = deepcopy(action)
    item.setdefault("request_id", out.get("request_id", ""))
    item.setdefault("observed_at", out.get("updated_at", ""))
    item.setdefault("scope", _scope(out))
    key = "pending_actions" if pending else "actions"
    out[key] = (out.get(key, []) + [item])[-_MAX_ACTIONS:]
    return out


def _add_fact(context: dict, name: str, value: Any, source: str) -> None:
    scope = _scope(context)
    eid = hashlib.sha256(json.dumps([context.get("request_id"), scope, source, name, value],
                                    ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    fact = {"value": value, "source": source, "observed_at": context.get("updated_at", ""),
            "scope": scope, "evidence_id": eid}
    context.setdefault("verified_facts", {})[name] = fact
    evidence = {"id": eid, "field": name, **deepcopy(fact)}
    context["evidence"] = [e for e in context.get("evidence", []) if e.get("field") != name]
    context["evidence"] = (context["evidence"] + [evidence])[-_MAX_EVIDENCE:]


def _drop_sources(context: dict, sources: set[str]) -> None:
    context["verified_facts"] = {k: v for k, v in context.get("verified_facts", {}).items()
                                 if v.get("source") not in sources}
    context["evidence"] = [e for e in context.get("evidence", []) if e.get("source") not in sources]


def record_tool_result(context: dict, name: str, args: dict, data: dict | None,
                       business_status: str) -> dict:
    """Call only after engine result-contract validation and identity injection.

    The engine validates types/enums. This second boundary validates task scope.
    A query_order `ok` means its server-injected identity passed owns_order; an
    error dictionary returned normally is not a successful ownership check.
    """
    out = record_action(context, {"kind": "tool_result", "tool": name,
                                  "business_status": business_status})
    if name == "query_order":
        # A failed refresh also revokes its dependent facts. A later successful
        # ownership refresh must not accidentally revive old after-sales data.
        _drop_sources(out, {"query_order", "query_logistics", "query_warranty", "query_return_status", "query_faq"})
        out.pop("policy_assessment", None)
        if (business_status != "ok" or not isinstance(data, dict)
                or guard_tool_call(out, name, args)
                or data.get("order_id") != args.get("order_id")
                or not isinstance(data.get("tracking_no"), str) or not data["tracking_no"]):
            return out
        _add_fact(out, "order_owned", True, name)
        for key in ("order_id", "tracking_no", "status", "amount", "product", "created_at",
                    "destination_region", "fulfillment_type", "delivered_at"):
            if key in data:
                _add_fact(out, "order_status" if key == "status" else key, data[key], name)
        out["missing_information"] = [f for f in out.get("missing_information", []) if f != "order_id"]
    elif name == "query_logistics":
        _drop_sources(out, {"query_logistics"})
        if (business_status != "ok" or not isinstance(data, dict)
                or guard_tool_call(out, name, args)
                or data.get("tracking_no") != args.get("tracking_no")):
            return out
        for key in ("status", "status_code", "current_city", "stalled_hours", "trace"):
            if key in data:
                _add_fact(out, "logistics_" + key, data[key], name)
    elif name in {"query_warranty", "query_return_status"}:
        _drop_sources(out, {name})
        if (business_status != "ok" or not isinstance(data, dict)
                or guard_tool_call(out, name, args)
                or data.get("order_id") != args.get("order_id")):
            return out
        for key in ("warranty", "warranty_until") if name == "query_warranty" else ("return_status", "updated_at"):
            if key in data:
                _add_fact(out, key, data[key], name)
    elif name == "query_faq":
        # Failed refresh must not leave a previous policy available for reuse.
        _drop_sources(out, {"query_faq"})
        out.pop("policy_assessment", None)
        if business_status != "ok" or not isinstance(data, dict) or data.get("sufficient") is not True:
            return out
        # Retrieval is provenance, not proof that a policy applies to this order.
        for citation in (data.get("citations") or [])[:4]:
            if isinstance(citation, dict):
                _add_fact(out, f"policy_chunk_{citation.get('id')}",
                          {k: citation[k] for k in ("id", "section_path", "question", "answer", "content_type", "policy_source") if k in citation}, name)
    return out


def set_decision(context: dict, decision: str, source: str) -> dict:
    """Persist a decision actually made by a node, never infer one from text."""
    if decision not in {"clarify", "deny", "cancelled", "no_ticket", "offer_ticket", "human_review"}:
        raise ValueError("unsupported task decision")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("task decision requires its program source")
    out = _copy(context)
    out.update(decision=decision, decision_source=source)
    return out


def finish(context: dict, reason: str, *, decision: str | None = None,
           decision_source: str = "") -> dict:
    out = _copy(context)
    if decision is not None:
        out = set_decision(out, decision, decision_source)
    out["termination_reason"] = reason
    if reason == "cancelled":
        out.update(cancelled=True, active_order_id="", verified_facts={}, evidence=[], pending_actions=[])
    elif reason in {"client_cancelled", "deadline_exceeded", "tool_budget_exhausted", "repeated_tool_call",
                    "tool_budget_exceeded", "repeated_call_limit", "step_budget_exceeded",
                    "context_budget_exceeded", "execution_failed", "forbidden", "superseded",
                    "permission_denied", "tool_evidence_unavailable", "unverified_answer",
                    "unsupported_action_claim", "insufficient_evidence", "draft_ready"}:
        out["pending_actions"] = []
    return out
