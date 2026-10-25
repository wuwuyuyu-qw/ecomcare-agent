"""Replay recorded, sanitized tool data against current pure program guards.

No model call, database, authentication service or HTTP request is performed.
Replay is a guard diagnostic, never a baseline/candidate Agent quality result.
"""
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from app.graph import task_context as tasks
from app.logistics_agent.decision import decide_logistics

_SECRET = re.compile(r"(?i)(?:bearer\s+\S+|(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{8,}|(?:password|api_key|authorization)\s*[:=])")
_TOOLS = {"query_order", "query_logistics", "query_faq"}
_STATUSES = {"ok", "forbidden", "not_found", "insufficient", "transient_failure", "permanent_failure", "untrusted"}


def load_snapshots(path: Path) -> tuple[dict[str, dict], str]:
    if path.name.lower() == ".env" or path.stat().st_size > 20_000_000:
        raise ValueError("snapshot file must be JSON under 20 MB")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if value.get("schema_version") != 1 or value.get("sanitized") is not True:
        raise ValueError("snapshot header must declare schema_version=1 and sanitized=true")
    if value.get("provenance") not in {"recorded_synthetic", "recorded_redacted"}:
        raise ValueError("snapshot provenance must be recorded_synthetic or recorded_redacted")
    if _SECRET.search(raw.decode("utf-8")):
        raise ValueError("snapshot contains a credential-like string; redact before replay")
    out = {}
    for case in value.get("cases", []):
        if not isinstance(case, dict) or set(case) != {"case_id", "turns"} or case["case_id"] in out:
            raise ValueError("snapshot case format/ID is invalid")
        if not isinstance(case["turns"], list):
            raise ValueError("snapshot turns must be a list")
        out[case["case_id"]] = case
    return out, hashlib.sha256(raw).hexdigest()


def _claim_type(message: str) -> str:
    if any(t in message for t in ("没收到", "未收到", "没拿到", "未拿到", "不是我签收", "没有收到", "收件处没有")):
        return "signed_not_received"
    if any(t in message for t in ("没更新", "没动", "不动", "停滞", "卡住")):
        return "stalled"
    return "status"


def replay_turn(context: dict | None, case_id: str, index: int, message: str, snapshot: dict) -> tuple[dict, dict]:
    allowed = {"message", "user_id", "conversation_id", "observed_at", "claim_type", "tool_runs", "http_status", "termination_reason"}
    if not isinstance(snapshot, dict) or not set(snapshot) <= allowed or snapshot.get("message") != message:
        raise ValueError("snapshot message must exactly match the frozen case turn")
    if not isinstance(snapshot.get("user_id"), str) or not isinstance(snapshot.get("conversation_id"), int):
        raise ValueError("snapshot requires recorded synthetic scope")
    if not isinstance(snapshot.get("observed_at"), str) or not snapshot["observed_at"]:
        raise ValueError("snapshot requires recorded observation time")
    try:
        observed_time = datetime.fromisoformat(snapshot["observed_at"].replace("Z", "+00:00"))
        if observed_time.tzinfo is None:
            raise ValueError("missing timezone")
    except ValueError:
        raise ValueError("snapshot requires a valid timezone-aware observation time") from None
    runs = snapshot.get("tool_runs", [])
    if not isinstance(runs, list):
        raise ValueError("snapshot tool_runs must be a list")
    for run in runs:
        if (not isinstance(run, dict) or not set(run) <= {"name", "args", "data", "business_status", "format", "error_code"}
                or run.get("name") not in _TOOLS or run.get("business_status") not in _STATUSES
                or run.get("format", "raw") not in {"raw", "public"}
                or not isinstance(run.get("args"), dict)):
            raise ValueError("recorded tool run is not a supported safe snapshot")
    ctx = tasks.begin_turn(context, snapshot["user_id"], snapshot["conversation_id"], message,
                           {"task_id": f"replay-{case_id}-{index}", "request_id": f"replay-{case_id}-{index}", "now": snapshot["observed_at"]})
    facts, sources, actions = {}, set(), []
    decision, termination = None, ""
    if snapshot.get("http_status") in (401, 403, 404):
        decision, termination = "deny", "permission_denied"
    elif ctx.get("cancelled"):
        decision, termination = "cancelled", "cancelled"
    elif not ctx.get("active_order_id"):
        decision, termination = "clarify", "clarification_required"
    order, logistics = None, None
    # Import only local return contracts; they do not read settings or .env.
    from app.tools.contracts import ContractError, public_data, validate_input, validate_result
    for run in runs if decision is None else []:
        name, args, data, status = run["name"], run["args"], run.get("data"), run["business_status"]
        try:
            validate_input(name, args)
        except ContractError:
            decision, termination = "human_review", "tool_evidence_unavailable"
            break
        guard = tasks.guard_tool_call(ctx, name, args)
        if guard:
            decision, termination = "human_review", "tool_evidence_unavailable"
            break
        ctx = tasks.consume_tool(ctx, name, args)
        if data is not None:
            try:
                result = validate_result(name, data, args, formatted=run.get("format") == "public")
                if result.business_status != status:
                    raise ValueError("recorded business_status disagrees with local contract")
                data = public_data(name, result.data) if status == "ok" and run.get("format") != "public" else result.data
            except (ContractError, TypeError, ValueError):
                status, data = "untrusted", None
        elif status == "ok":
            status = "untrusted"
        ctx = tasks.record_tool_result(ctx, name, args, data, status)
        if name == "query_order" and status in {"forbidden", "not_found"}:
            decision, termination = "deny", "permission_denied"
            break
        if status not in {"ok", "insufficient"}:
            decision, termination = "human_review", "tool_evidence_unavailable"
            break
        if status == "ok":
            sources.add(name)
            if name == "query_order":
                order = data
            elif name == "query_logistics":
                logistics = data
    if decision is None:
        if order is None or logistics is None:
            decision, termination = "human_review", "tool_evidence_unavailable"
        else:
            claim = snapshot.get("claim_type") or _claim_type(message)
            if claim not in {"status", "stalled", "signed_not_received", "other"}:
                raise ValueError("recorded claim_type is invalid")
            result = decide_logistics(claim_type=claim, user_text=message, order=order, logistics=logistics)
            decision = result.verdict
            termination = {"no_ticket": "completed", "offer_ticket": "draft_ready", "human_review": "human_review"}[decision]
    ctx = tasks.finish(ctx, termination)
    facts.update({k: v.get("value") for k, v in ctx.get("verified_facts", {}).items()})
    if decision == "offer_ticket":
        actions.append("create_ticket")
    return {"decision": decision, "termination_reason": termination, "facts": facts,
            "evidence_sources": sorted(sources), "actions": actions,
            "measurement": "pure_guard_replay", "claim_source": "recorded" if snapshot.get("claim_type") else "conservative_rule",
            "http_status": snapshot.get("http_status", 200), "tool_calls": ctx.get("budgets", {}).get("tool_calls", 0),
            "fact_observation_complete": True, "action_observation_complete": True}, ctx
