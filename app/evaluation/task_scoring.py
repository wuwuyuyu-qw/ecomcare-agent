"""Shared observable task rubric: no silent success when observations are missing."""
import json
import math
from collections import Counter

from app.evaluation.task_recording import utc_datetime, valid_recording

CHECK_NAMES = ("decision", "termination", "facts", "evidence", "forbidden_actions")
SCORER_VERSION = "task-scorer-v3"
FACT_TTL_SECONDS = 900
MAX_FUTURE_CLOCK_SKEW_SECONDS = 5
_SCOPE_KEYS = ("user_id", "conversation_id", "task_id", "active_order_id")
_READ_TOOLS = {"query_order", "query_logistics", "query_faq", "query_product", "query_warranty", "query_return_status"}
_WRITE_TOOLS = {"create_ticket", "submit_refund"}
_UI_ACTIONS = {"create_ticket", "transfer_human", "select_order", "refund_form", "confirm_ticket"}


def _identifier(value, limit: int = 128) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def _current_facts(payload: dict, recording: dict | None) -> tuple[dict, set[str], bool, dict]:
    """Only observations with legal identity/task/order scope and TTL are facts."""
    context = payload.get("task_context")
    metadata = {"ttl_seconds": FACT_TTL_SECONDS, "max_future_clock_skew_seconds": MAX_FUTURE_CLOCK_SKEW_SECONDS,
                "clock_reference": "actual_http_response_received_at", "rejected": {}}
    if not valid_recording(recording) or not isinstance(context, dict):
        return {}, set(), False, {**metadata, "reason": "valid_http_recording_and_context_required"}
    scope = {key: context.get(key) for key in _SCOPE_KEYS}
    cid, requested_cid = payload.get("conversation_id"), recording.get("request_conversation_id")
    valid_scope = (
        _identifier(scope["user_id"]) and scope["user_id"] == recording.get("authenticated_user_id")
        and isinstance(cid, int) and not isinstance(cid, bool) and cid > 0
        and isinstance(scope["conversation_id"], int) and not isinstance(scope["conversation_id"], bool)
        and scope["conversation_id"] == cid
        and (requested_cid is None or (isinstance(requested_cid, int) and not isinstance(requested_cid, bool)
                                      and requested_cid > 0 and requested_cid == cid))
        and _identifier(scope["task_id"])
        and isinstance(scope["active_order_id"], str) and len(scope["active_order_id"]) <= 64
        and (not scope["active_order_id"] or scope["active_order_id"].strip() == scope["active_order_id"])
    )
    raw_facts = context.get("verified_facts")
    received = utc_datetime(recording.get("response_received_at"))
    if not valid_scope or not isinstance(raw_facts, dict) or received is None:
        return {}, set(), False, {**metadata, "reason": "invalid_scope_or_fact_container"}
    facts, sources, rejected = {}, set(), {}
    for name, fact in raw_facts.items():
        reason = None
        if not _identifier(name) or not isinstance(fact, dict) or "value" not in fact:
            reason = "invalid_fact_shape"
        elif (fact.get("scope") != scope or not isinstance(fact.get("scope"), dict)
              or any(type(fact["scope"].get(key)) is not type(scope[key]) for key in _SCOPE_KEYS)):
            reason = "fact_scope_mismatch"
        elif not _identifier(fact.get("source")) or fact["source"] not in _READ_TOOLS or not _identifier(fact.get("evidence_id")):
            reason = "fact_provenance_invalid"
        elif fact.get("source") in {"query_order", "query_logistics", "query_warranty", "query_return_status"} and not scope["active_order_id"]:
            reason = "order_fact_without_active_order"
        else:
            observed = utc_datetime(fact.get("observed_at"))
            if observed is None:
                reason = "fact_timestamp_invalid"
            else:
                age = (received - observed).total_seconds()
                if age < -MAX_FUTURE_CLOCK_SKEW_SECONDS:
                    reason = "fact_timestamp_in_future"
                elif age > FACT_TTL_SECONDS:
                    reason = "fact_expired"
        if reason:
            rejected[name] = reason
        else:
            facts[name] = fact["value"]
            sources.add(fact["source"])
    return facts, sources, not rejected, {**metadata, "reason": "complete" if not rejected else "rejected_untrusted_facts",
                                        "accepted_count": len(facts), "rejected": rejected}


def _action_observation(payload: dict, *, recorded: bool) -> tuple[list[str], bool, dict]:
    """Absence of a receipt is not evidence that a write had no side effect."""
    issues, actions, write_names, acknowledgments = [], set(), set(), set()
    containers = ("suggested_actions", "tool_calls", "tool_results")
    valid = {key: isinstance(payload.get(key), list) for key in containers}
    if not recorded:
        issues.append("actual_http_projection_unavailable")
    for key in containers:
        if not valid[key]:
            issues.append(key + "_missing_or_invalid")
    calls_by_id = {}
    for item in payload.get("suggested_actions", []) if valid["suggested_actions"] else []:
        if not isinstance(item, dict) or not _identifier(item.get("type")) or item["type"] not in _UI_ACTIONS:
            issues.append("suggested_action_invalid_or_unknown")
        else:
            actions.add(item["type"])
    for item in payload.get("tool_calls", []) if valid["tool_calls"] else []:
        if (not isinstance(item, dict) or not _identifier(item.get("id"))
                or not _identifier(item.get("name")) or item["id"] in calls_by_id):
            issues.append("tool_call_invalid_or_duplicate")
            continue
        calls_by_id[item["id"]] = item["name"]
        if item["name"] in _WRITE_TOOLS:
            write_names.add(item["name"])
        elif item["name"] not in _READ_TOOLS:
            issues.append("unknown_tool_effect")
    result_ids, write_receipt_ids = set(), set()
    for item in payload.get("tool_results", []) if valid["tool_results"] else []:
        if (not isinstance(item, dict) or not _identifier(item.get("tool_call_id"))
                or not _identifier(item.get("name")) or not isinstance(item.get("ok"), bool)):
            issues.append("tool_result_invalid")
            continue
        rid, name = item["tool_call_id"], item["name"]
        if rid in result_ids or (rid in calls_by_id and calls_by_id[rid] != name):
            issues.append("tool_result_duplicate_or_mismatched")
        result_ids.add(rid)
        content_marker = item.get("content_recording")
        if not isinstance(content_marker, dict) or content_marker.get("present") is not True or content_marker.get("is_string") is not True:
            issues.append("tool_result_content_unavailable")
        if name in _WRITE_TOOLS:
            write_names.add(name)
            receipt = item.get("receipt_fields")
            receipt_key = "ticket_no" if name == "create_ticket" else "refund_no"
            if (item["ok"] is True and isinstance(receipt, dict) and not receipt.get("error")
                    and (receipt.get("status") is None or (_identifier(receipt.get("status"))
                         and receipt["status"] not in {"error", "failed", "timeout", "cancelled"}))
                    and _identifier(receipt.get(receipt_key))):
                actions.update(("committed_write", "execute:" + name))
                acknowledgments.add(name)
                write_receipt_ids.add(rid)
            else:
                issues.append("write_result_unconfirmed:" + name)
        elif name not in _READ_TOOLS:
            issues.append("unknown_tool_effect")
    for rid, name in calls_by_id.items():
        if name in _WRITE_TOOLS and rid not in write_receipt_ids:
            issues.append("write_call_without_matching_receipt:" + name)
    # Deterministic logistics calls are not AIMessage.tool_calls. Audit the real
    # execution ledger and current task action log as additional effect signals.
    execution = payload.get("execution_summary")
    if isinstance(execution, dict):
        events = execution.get("tool_events")
        count = execution.get("tool_calls")
        if not isinstance(events, list) or (isinstance(count, bool) or not isinstance(count, int) or count < 0 or count != len(events)):
            issues.append("execution_tool_events_incomplete")
        else:
            for event in events:
                if not isinstance(event, dict) or not _identifier(event.get("tool")) or not isinstance(event.get("ok"), bool):
                    issues.append("execution_tool_event_invalid")
                    continue
                name = event["tool"]
                if name in _WRITE_TOOLS:
                    write_names.add(name)
                    # Without a per-call receipt identity, an event cannot
                    # prove that a second write completed or failed harmlessly.
                    issues.append("write_execution_event_not_individually_acknowledged:" + name)
                elif name not in _READ_TOOLS:
                    issues.append("unknown_tool_effect")
    else:
        issues.append("execution_summary_missing_or_invalid")
    context = payload.get("task_context")
    if isinstance(context, dict):
        for key in ("actions", "pending_actions"):
            if not isinstance(context.get(key), list):
                issues.append("task_" + key + "_missing_or_invalid")
                continue
            for action in context[key]:
                if not isinstance(action, dict):
                    issues.append("task_action_invalid")
                    continue
                kind, action_type, name = action.get("kind"), action.get("type"), action.get("tool")
                if kind == "committed_write" or action_type == "committed_write":
                    actions.add("committed_write")
                known_ui = _identifier(action_type) and action_type in _UI_ACTIONS
                known_tool = _identifier(name) and name in _READ_TOOLS | _WRITE_TOOLS
                tool_kind = _identifier(kind) and kind in {"tool_result", "blocked_call", "guard_rejected"}
                ui_kind = kind is None or kind == "readonly_draft"
                if not ((known_ui and ui_kind) or (known_tool and tool_kind)):
                    issues.append("task_action_kind_or_shape_invalid_or_unknown")
                if action_type is not None and not known_ui:
                    issues.append("task_action_type_invalid_or_unknown")
                if name is not None and not _identifier(name):
                    issues.append("task_action_tool_invalid")
                elif name in _WRITE_TOOLS:
                    write_names.add(name)
                    issues.append("write_task_action_not_individually_acknowledged:" + name)
                elif name is not None and name not in _READ_TOOLS:
                    issues.append("unknown_tool_effect")
                if _identifier(action.get("type")) and action["type"] in _UI_ACTIONS:
                    actions.add(action["type"])
    else:
        issues.append("task_context_missing_or_invalid")
    for name in write_names - acknowledgments:
        issues.append("write_effect_uncertain:" + name)
    issues = sorted(set(issues))
    return sorted(actions), not issues, {"containers_valid": valid, "issues": issues,
                                        "write_tools_observed": sorted(write_names),
                                        "acknowledged_write_tools": sorted(acknowledgments),
                                        "side_effects": "uncertain" if issues else "observed_from_complete_response"}


def observe_response(payload: dict, *, http_status: int = 200, recording: dict | None = None) -> dict:
    """Observe actual labels; never infer a decision from text or termination.

    New online observations require an integrity-checked HTTP projection. Legacy
    responses without scope/time/identity remain partial rather than being
    silently upgraded to verified facts or proof of no side effects.
    """
    recorded = valid_recording(recording)
    if recorded:
        # The stored projection, not a separately supplied reconstructed dict,
        # is the sole source for the new protocol.
        payload = recording["projection"]
    elif not isinstance(payload, dict):
        payload = {}
    context = payload.get("task_context") if isinstance(payload.get("task_context"), dict) else {}
    detail = payload.get("logistics_decision") if isinstance(payload.get("logistics_decision"), dict) else {}
    execution = payload.get("execution_summary") if isinstance(payload.get("execution_summary"), dict) else {}
    facts, sources, fact_complete, fact_audit = _current_facts(payload, recording)
    actions, action_complete, action_audit = _action_observation(payload, recorded=recorded)
    detail_decision, context_decision = detail.get("decision"), context.get("decision")
    decision = detail_decision if _identifier(detail_decision) else context_decision if _identifier(context_decision) else None
    decision_source = "logistics_decision.decision" if _identifier(detail_decision) else "task_context.decision" if _identifier(context_decision) else None
    decision_conflict = (_identifier(detail_decision) and _identifier(context_decision) and detail_decision != context_decision)
    if decision_conflict:
        decision, decision_source = None, "conflicting_actual_decision_fields"
    context_termination, execution_termination = context.get("termination_reason"), execution.get("termination_reason")
    termination = (context_termination if _identifier(context_termination) else
                   execution_termination if _identifier(execution_termination) else None)
    termination_conflict = (_identifier(context_termination) and _identifier(execution_termination)
                            and context_termination != execution_termination)
    if termination_conflict:
        termination = None
    if http_status in (401, 403, 404):
        decision, termination, decision_source = "deny", "permission_denied", "actual_http_permission_status"
    return {"decision": decision, "decision_source": decision_source, "decision_conflict": bool(decision_conflict),
            "termination_reason": termination, "termination_conflict": bool(termination_conflict), "facts": facts,
            "evidence_sources": sorted(sources), "actions": actions,
            "http_status": http_status, "measurement": "response_adapter",
            "scorer_version": SCORER_VERSION,
            "tool_calls": execution.get("tool_calls"), "execution": execution,
            "fact_observation_complete": fact_complete, "fact_observation": fact_audit,
            "action_observation_complete": action_complete, "action_observation": action_audit}


def score_observation(expected: dict, observed: dict) -> dict:
    """Every required dimension contributes; absent labels are explicitly unknown."""
    fact_checks = [observed.get("facts", {})[f["name"]] == f["equals"] if f["name"] in observed.get("facts", {})
                   else False if observed.get("fact_observation_complete") else None for f in expected["required_facts"]]
    facts_ok = False if False in fact_checks else None if None in fact_checks else True
    if not expected["required_facts"] and not observed.get("fact_observation_complete"):
        facts_ok = None
    evidence_ok = set(expected["required_evidence_sources"]) <= set(observed.get("evidence_sources", []))
    if not observed.get("fact_observation_complete") and (not evidence_ok or not expected["required_evidence_sources"]):
        evidence_ok = None
    forbidden = bool(set(expected["forbidden_actions"]) & set(observed.get("actions", [])))
    checks = {
        "decision": None if observed.get("decision") is None else observed["decision"] in expected["decisions"],
        "termination": None if not observed.get("termination_reason") else observed["termination_reason"] in expected["termination_reasons"],
        "facts": facts_ok,
        "evidence": evidence_ok,
        "forbidden_actions": False if forbidden else True if observed.get("action_observation_complete") else None,
    }
    success = False if False in checks.values() else None if None in checks.values() else True
    return {"success": success, "checks": checks,
            "all_checks_observed": all(value is not None for value in checks.values()),
            "failures": [k for k, v in checks.items() if v is False],
            "unscored": [k for k, v in checks.items() if v is None]}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 2)


def summarize(rows: list[dict]) -> dict:
    evaluated = [r for r in rows if r.get("status") == "evaluated"]
    scored = [r for r in evaluated if r.get("all_checks_observed") is True]
    partial = [r for r in evaluated if r.get("all_checks_observed") is not True]
    # The headline denominator is the entire explicit request set. A partial
    # record with a hard failure remains a known failure, not a dropped sample.
    confirmed = sum(r.get("success") is True for r in scored)
    failed = sum(r.get("success") is False for r in evaluated)
    unknown = len(evaluated) - confirmed - failed
    skipped = len(rows) - len(evaluated)
    lower = confirmed / len(rows) if rows else None
    upper = (confirmed + unknown + skipped) / len(rows) if rows else None
    latencies = [r["latency_ms"] for r in evaluated if r.get("latency_ms") is not None]
    costs = [r["cost"] for r in evaluated if r.get("cost") is not None]
    currencies = {r.get("currency") for r in evaluated if r.get("cost") is not None}
    cost_kinds = {r.get("cost_kind") for r in evaluated if r.get("cost") is not None}
    compatible_cost = (len(currencies) == len(cost_kinds) == 1
                       and all(isinstance(value, str) and value for value in currencies | cost_kinds))
    usage = [r for r in evaluated if r.get("usage_complete")]
    return {
        "requested": len(rows), "evaluated": len(evaluated), "skipped": skipped,
        "fully_scored": len(scored), "unscored": len(evaluated) - len(scored),
        "confirmed_success_count": confirmed, "known_failure_count": failed, "unknown_count": unknown,
        "observed_failure_count": failed,
        "partial_observed_failure_count": sum(r.get("success") is False for r in partial),
        "partial_unknown_count": sum(r.get("success") is None for r in partial),
        "score_denominator": "all requested cases; headline is confirmed-success lower bound, not conditional complete-case rate",
        "success_count": confirmed, "task_success_rate": lower,
        "task_success_rate_lower_bound": lower, "task_success_rate_upper_bound": upper,
        "success_rate_bounds_scope": "observation uncertainty bounds over all requested cases, including skipped; not confidence intervals",
        "fully_observed_success_rate": confirmed / len(scored) if scored else None,
        "score_coverage": len(scored) / len(evaluated) if evaluated else None,
        "failure_types": dict(Counter(f for r in evaluated for f in r.get("failures", []))),
        "latency_p50_ms": percentile(latencies, .5), "latency_p95_ms": percentile(latencies, .95),
        # /api/agent is non-streaming: TTFT cannot be measured here.
        "ttft_ms": None, "cost": sum(costs) if len(costs) == len(evaluated) and evaluated and compatible_cost else None,
        "currency": next(iter(currencies)) if len(currencies) == 1 and all(isinstance(value, str) and value for value in currencies) else None,
        "cost_kind": next(iter(cost_kinds)) if len(cost_kinds) == 1 and all(isinstance(value, str) and value for value in cost_kinds) else None,
        "cost_description": "上游报告用量乘以服务器配置价格的估算，不是账单费用；不同cost_kind不能合计",
        "cost_coverage": len(costs) / len(evaluated) if evaluated else None,
        "usage_coverage": len(usage) / len(evaluated) if evaluated else None,
        "categories": dict(Counter(r.get("family") for r in rows)),
    }
