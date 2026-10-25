"""Bounded, credential-screened projection of actual HTTP observations.

This is a local audit record, not a signature or proof that the server is honest.
Only explicit fields needed by the task rubric are retained. Whole projections
are withheld on invalid size/value/safety constraints; no truncated object is
presented as a complete observation. HTTP error bodies are never recorded.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import re

OBSERVATION_PROTOCOL_VERSION = "http-observation-v1"
PROJECTION_BYTE_LIMIT = 262_144
TEXT_CHAR_LIMIT = 20_000
LIST_LIMIT = 128
FACT_LIMIT = 40
_CREDENTIAL_LIKE = re.compile(
    r"(?i)(?:bearer\s+\S+|(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{8,}|"
    r"[\"']?(?:password|passwd|api[_ -]?key|access[_ -]?token|authorization|secret)[\"']?\s*[:=]|"
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})"
)
_DISALLOWED_KEYS = {
    "password", "passwd", "api_key", "access_token", "authorization", "secret",
    "reasoning", "reasoning_content", "chain_of_thought", "thinking", "messages", "headers",
}
_TOP_FIELDS = (
    "conversation_id", "task_context", "logistics_decision", "suggested_actions",
    "tool_calls", "tool_results", "execution_summary",
)
_SCOPE_FIELDS = ("user_id", "conversation_id", "task_id", "active_order_id")
_CONTEXT_FIELDS = (
    "schema_version", "request_id", "updated_at", "termination_reason", "decision", "decision_source",
    *_SCOPE_FIELDS,
)
_FACT_FIELDS = ("value", "source", "scope", "observed_at", "evidence_id")
_ACTION_FIELDS = (
    "type", "kind", "tool", "business_status", "error_code", "scope", "request_id", "observed_at",
    "order_id", "status",
)
_EVENT_FIELDS = ("tool", "name", "ok", "error_code", "duration_ms", "retry_count")
_EXECUTION_FIELDS = (
    "task_id", "request_id", "elapsed_ms", "model_calls", "tool_calls", "termination_reason",
    "measured_input_tokens", "measured_output_tokens", "usage_coverage", "price_coverage",
    "unfinished_model_calls", "cost", "measured_partial_cost", "currency", "cost_complete",
    "cost_kind", "scope", "excluded",
)
_USAGE_FIELDS = (
    "stage", "model", "kind", "input_tokens", "output_tokens", "cached_input_tokens", "failed", "cost",
)


def canonical_sha256(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _pick(value, fields: tuple[str, ...]):
    # Preserve malformed scalar/container shapes so the scorer can mark unknown;
    # turning them into an empty valid object would fabricate completeness.
    if not isinstance(value, dict):
        return value
    return {key: value[key] for key in fields if key in value}


def _items(value, project):
    if not isinstance(value, list):
        return value
    if len(value) > LIST_LIMIT:
        raise ValueError("list_exceeds_recording_limit")
    return [project(item) for item in value]


def _result(value):
    out = _pick(value, ("tool_call_id", "name", "ok"))
    if not isinstance(value, dict):
        return out
    content = value.get("content")
    marker = {"present": "content" in value, "is_string": isinstance(content, str)}
    if isinstance(content, str):
        if len(content) > TEXT_CHAR_LIMIT:
            raise ValueError("tool_content_exceeds_recording_limit")
        # The original body is never retained, even when it is a valid JSON
        # object. Hash it and retain only explicit actual write receipt fields.
        marker["sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
        try:
            def distinct_object(pairs):
                obj = {}
                for key, item in pairs:
                    if key in obj:
                        raise ValueError("duplicate JSON key")
                    obj[key] = item
                return obj

            def reject_constant(_value):
                raise ValueError("nonfinite JSON constant")

            parsed = json.loads(content, object_pairs_hook=distinct_object, parse_constant=reject_constant)
        except (TypeError, ValueError):
            parsed = None
            marker["parse_status"] = "invalid_json"
        else:
            marker["parse_status"] = "json_object" if isinstance(parsed, dict) else "json_nonobject"
        if isinstance(parsed, dict):
            out["receipt_fields"] = _pick(parsed, ("ticket_no", "refund_no", "error", "code", "status"))
    out["content_recording"] = marker
    return out


def _project(payload: dict) -> dict:
    out = {}
    if "conversation_id" in payload:
        out["conversation_id"] = payload["conversation_id"]
    if "task_context" in payload:
        context = payload["task_context"]
        projected = _pick(context, _CONTEXT_FIELDS)
        if isinstance(context, dict):
            if "verified_facts" in context:
                facts = context["verified_facts"]
                if isinstance(facts, dict) and len(facts) > FACT_LIMIT:
                    raise ValueError("facts_exceed_recording_limit")
                projected["verified_facts"] = {key: _pick(fact, _FACT_FIELDS)
                                                for key, fact in facts.items()} if isinstance(facts, dict) else facts
            if "evidence" in context:
                projected["evidence"] = _items(context["evidence"], lambda item: _pick(item, ("id", "field", *_FACT_FIELDS)))
            for key in ("actions", "pending_actions"):
                if key in context:
                    projected[key] = _items(context[key], lambda item: _pick(item, _ACTION_FIELDS))
        out["task_context"] = projected
    if "logistics_decision" in payload:
        detail = payload["logistics_decision"]
        projected = _pick(detail, ("decision", "issue_type", "order_id"))
        if isinstance(detail, dict):
            if "evidence" in detail:
                projected["evidence"] = _items(detail["evidence"], lambda item: _pick(
                    item, ("label", "value", "source", "source_tool", "chunk_id", "scope", "observed_at", "evidence_id")))
            if "tool_trace" in detail:
                projected["tool_trace"] = _items(detail["tool_trace"], lambda item: _pick(item, _EVENT_FIELDS))
        out["logistics_decision"] = projected
    for key in ("suggested_actions", "tool_calls", "tool_results"):
        if key in payload:
            project = (_result if key == "tool_results" else
                       (lambda item: _pick(item, ("id", "name"))) if key == "tool_calls" else
                       (lambda item: _pick(item, _ACTION_FIELDS)))
            out[key] = _items(payload[key], project)
    if "execution_summary" in payload:
        execution = payload["execution_summary"]
        projected = _pick(execution, _EXECUTION_FIELDS)
        if isinstance(execution, dict):
            if "tool_events" in execution:
                projected["tool_events"] = _items(execution["tool_events"], lambda item: _pick(item, _EVENT_FIELDS))
            if "usage" in execution:
                projected["usage"] = _items(execution["usage"], lambda item: _pick(item, _USAGE_FIELDS))
        out["execution_summary"] = projected
    return out


def _safe_json(value, *, depth: int = 0) -> None:
    if depth > 10:
        raise ValueError("projection_exceeds_depth_limit")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("projection_contains_nonfinite_number")
        return
    if isinstance(value, str):
        if len(value) > TEXT_CHAR_LIMIT:
            raise ValueError("projection_text_exceeds_recording_limit")
        if _CREDENTIAL_LIKE.search(value):
            raise ValueError("credential_like_projection_withheld")
        return
    if isinstance(value, list):
        if len(value) > LIST_LIMIT:
            raise ValueError("list_exceeds_recording_limit")
        for item in value:
            _safe_json(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > LIST_LIMIT:
            raise ValueError("object_exceeds_recording_limit")
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 128:
                raise ValueError("projection_key_invalid")
            if key.lower() in _DISALLOWED_KEYS:
                raise ValueError("sensitive_projection_field_withheld")
            _safe_json(key, depth=depth + 1)
            _safe_json(item, depth=depth + 1)
        return
    raise ValueError("projection_contains_non_json_value")


def utc_datetime(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def record_response(payload: dict | None, *, http_status: int | None, authenticated_user_id: str,
                    request_conversation_id: int | None, request_sent_at: str,
                    response_received_at: str | None, unavailable_reason: str | None = None) -> dict:
    """Record a complete allowlist projection of this exact response, or null.

    Presence is recorded independently of validity. ``complete`` refers to the
    recording, never to task success or the semantic completeness of evidence.
    The login ID is server ``/auth/me`` output; no login token is retained.
    """
    safe_identity = authenticated_user_id
    try:
        _safe_json(safe_identity)
    except (ValueError, TypeError, OverflowError):
        safe_identity = None
    metadata = {
        "observation_protocol_version": OBSERVATION_PROTOCOL_VERSION,
        "source": "actual_http_response_allowlist", "complete": False,
        "byte_limit": PROJECTION_BYTE_LIMIT, "text_char_limit": TEXT_CHAR_LIMIT,
        "http_status": http_status, "authenticated_user_id": safe_identity,
        "request_conversation_id": request_conversation_id,
        "request_sent_at": request_sent_at, "response_received_at": response_received_at,
        "field_presence": {key: isinstance(payload, dict) and key in payload for key in _TOP_FIELDS},
        "projection": None, "projection_sha256": None,
    }
    if unavailable_reason:
        return {**metadata, "reason": unavailable_reason}
    sent, received = utc_datetime(request_sent_at), utc_datetime(response_received_at)
    if not isinstance(payload, dict):
        return {**metadata, "reason": "response_object_unavailable"}
    if sent is None or received is None or received < sent:
        return {**metadata, "reason": "recording_timestamps_invalid"}
    if safe_identity is None or not isinstance(authenticated_user_id, str) or not authenticated_user_id.strip() or len(authenticated_user_id) > 128:
        return {**metadata, "reason": "authenticated_identity_invalid"}
    try:
        projection = _project(payload)
        _safe_json(projection)
        _safe_json(authenticated_user_id)
        raw = json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(raw.encode("utf-8")) > PROJECTION_BYTE_LIMIT:
            raise ValueError("projection_exceeds_byte_limit")
    except (TypeError, ValueError, OverflowError) as exc:
        reason = str(exc) if isinstance(exc, ValueError) and re.fullmatch(r"[a-z_]+", str(exc)) else "projection_invalid"
        return {**metadata, "reason": reason}
    return {**metadata, "complete": True, "reason": "complete", "projection": projection,
            "projection_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()}


def valid_recording(recording: dict | None) -> bool:
    """Integrity validation for comparison; a SHA is not server authentication."""
    if not isinstance(recording, dict) or recording.get("observation_protocol_version") != OBSERVATION_PROTOCOL_VERSION:
        return False
    if recording.get("source") != "actual_http_response_allowlist" or recording.get("complete") is not True:
        return False
    if not {"http_status", "authenticated_user_id", "request_conversation_id", "request_sent_at",
            "response_received_at", "field_presence", "projection", "projection_sha256",
            "byte_limit", "text_char_limit", "reason"} <= set(recording):
        return False
    projection, presence = recording.get("projection"), recording.get("field_presence")
    if (recording.get("reason") != "complete" or not isinstance(projection, dict) or not set(projection) <= set(_TOP_FIELDS)
            or not isinstance(presence, dict) or set(presence) != set(_TOP_FIELDS)
            or any(not isinstance(value, bool) for value in presence.values())
            or any(presence[key] != (key in projection) for key in _TOP_FIELDS)):
        return False
    identity, cid, status = (recording.get("authenticated_user_id"), recording.get("request_conversation_id"),
                             recording.get("http_status"))
    if (not isinstance(identity, str) or not identity.strip() or len(identity) > 128
            or (cid is not None and (isinstance(cid, bool) or not isinstance(cid, int) or cid < 1))
            or isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599
            or recording.get("byte_limit") != PROJECTION_BYTE_LIMIT or recording.get("text_char_limit") != TEXT_CHAR_LIMIT):
        return False
    sent, received = utc_datetime(recording.get("request_sent_at")), utc_datetime(recording.get("response_received_at"))
    if sent is None or received is None or received < sent:
        return False
    try:
        _safe_json(identity)
        _safe_json(projection)
        raw = json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return len(raw.encode("utf-8")) <= PROJECTION_BYTE_LIMIT and canonical_sha256(projection) == recording.get("projection_sha256")
    except (TypeError, ValueError, OverflowError):
        return False
