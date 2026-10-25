"""Actual loopback HTTP lifecycle observations, coordinated by an owned driver.

Use a separate short-deadline service for ``deadline``. Run ``before`` against
the normal service, inspect its private checkpoint in the parent, stop/restart
that same owned service with the same checkpoint, then run ``after``. This probe
never starts/stops services or opens their DB/checkpoint/.env/holdout files.
Credentials stay in process memory. Client closure is not proof of cancellation.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.task_recording import canonical_sha256, record_response, utc_datetime, valid_recording
from app.evaluation.task_report import assert_new_output, write_json_new
from app.evaluation.task_scoring import observe_response
from scripts.probe_agent_boundaries import SECRET_PATTERN, strict_json, validate_url

VERSION = "http-lifecycle-v1"
BODY_LIMIT = 262_144
QUERY = "查询订单3003的物流状态，请核对工具记录。"
SELECT_QUERY = "我要申请退货退款，请先查我的订单，订单号忘了，请让我选择自己的订单。"
WRITE_TOOLS = {"create_ticket", "submit_refund"}


def now():
    return datetime.now(timezone.utc).isoformat()


def _positive(value):
    return type(value) is int and value > 0


def _safe(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if len(raw.encode("utf-8")) > BODY_LIMIT or SECRET_PATTERN.search(raw):
        raise ValueError("unsafe_or_oversized_lifecycle_observation")
    return value


def _fresh_order(context, *, cid, identity, received, selected="3003"):
    if not isinstance(context, dict):
        return None
    scope = {key: context.get(key) for key in ("user_id", "conversation_id", "task_id", "active_order_id")}
    if (scope["user_id"] != identity or type(scope["conversation_id"]) is not int
            or scope["conversation_id"] != cid or scope["active_order_id"] != selected
            or not isinstance(scope["task_id"], str) or not scope["task_id"].strip()):
        return False
    clock = utc_datetime(received)
    facts = context.get("verified_facts")
    if clock is None or not isinstance(facts, dict):
        return None
    for name, expected in (("order_owned", True), ("order_id", selected)):
        fact = facts.get(name)
        if not isinstance(fact, dict):
            return None
        stamp = utc_datetime(fact.get("observed_at"))
        if (fact.get("scope") != scope or fact.get("source") != "query_order"
                or not isinstance(fact.get("evidence_id"), str) or not fact["evidence_id"]
                or stamp is None or not -5 <= (clock - stamp).total_seconds() <= 900
                or type(fact.get("value")) is not type(expected) or fact["value"] != expected):
            return False
    return True


def _resume_checks(frames, *, cid, identity, received, complete):
    done = [frame for frame in frames if frame.get("event") == "done"]
    executions = [frame for frame in frames if frame.get("event") == "execution_summary"]
    error = any(frame.get("event") == "error" for frame in frames)
    interrupted = any(frame.get("event") == "interrupt" for frame in frames)
    same_cid = len(done) == 1 and done[0].get("conversation_id") == cid
    context = done[0].get("task_context") if len(done) == 1 else None
    no_writes = None
    if len(executions) == 1:
        execution = executions[0]
        events = execution.get("tool_events")
        count = execution.get("tool_calls")
        if (isinstance(events, list) and type(count) is int and count == len(events)
                and all(isinstance(item, dict) and isinstance(item.get("tool"), str) for item in events)):
            no_writes = not any(item["tool"] in WRITE_TOOLS for item in events)
    if any(frame.get("event") == "tool" and frame.get("name") in WRITE_TOOLS for frame in frames):
        no_writes = False
    if isinstance(context, dict):
        for action in (context.get("actions") or []) + (context.get("pending_actions") or []):
            if isinstance(action, dict) and (action.get("kind") == "committed_write" or action.get("type") == "committed_write"):
                no_writes = False
    return {"sse_complete": complete, "no_sse_error": not error if complete else None,
            "resumed_same_conversation": same_cid if complete else None,
            "selection_interrupt_ended": not interrupted and same_cid if complete else None,
            "fresh_scoped_order_facts": _fresh_order(context, cid=cid, identity=identity, received=received),
            "no_write_tool_calls": no_writes}


def _stream_projection(frame):
    if not isinstance(frame, dict):
        raise ValueError("sse_frame_object_required")
    event = frame.get("event")
    if event == "done":
        return _safe({key: frame[key] for key in ("event", "conversation_id", "task_id", "request_id", "termination_reason", "task_context") if key in frame})
    if event == "execution_summary":
        return _safe({key: frame[key] for key in ("event", "task_id", "request_id", "termination_reason", "model_calls", "tool_calls", "tool_events") if key in frame})
    if event == "tool":
        return {key: frame[key] for key in ("event", "name", "ok") if key in frame}
    if event == "interrupt":
        return {key: frame[key] for key in ("event", "kind", "conversation_id") if key in frame}
    if event == "termination":
        return {key: frame[key] for key in ("event", "reason", "conversation_id") if key in frame}
    if event == "actions":
        items = frame.get("items")
        return {"event": event, "items": [{"type": item.get("type")} for item in items if isinstance(item, dict)] if isinstance(items, list) else None}
    if event == "error" or "message" in frame:
        return {"event": "error"}
    return {"event": "unrecorded_content"}


async def _json_request(client, method, path, body=None):
    async with client.stream(method, path, json=body) as response:
        status = response.status_code
        if status != 200:
            return status, None  # Error bodies are never read or retained.
        chunks, size = [], 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > BODY_LIMIT:
                raise ValueError("http_body_exceeds_lifecycle_limit")
            chunks.append(chunk)
        value = strict_json(b"".join(chunks))
        return status, value if isinstance(value, dict) else None


async def _close_at_headers(client, cid):
    started = time.perf_counter()
    async with client.stream("POST", "/api/chat", json={"message": QUERY, "conversation_id": cid}) as response:
        status = response.status_code
        content_type = response.headers.get("content-type", "").split(";", 1)[0]
        await asyncio.sleep(0.75)
        await response.aclose()
    return {"http_status": status, "content_type": content_type,
            "headers_received_elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "client_stream_closed": True, "body_events_consumed": 0, "wait_before_client_close_ms": 750,
            "server_execution_started": None, "server_cancelled": None,
            "server_completed_before_close": None,
            "scope": "actual response headers and client aclose only; parent checkpoint observation required"}


async def _resume_stream(client, cid):
    frames, complete, size = [], False, 0
    async with client.stream("POST", "/api/actions/resume", json={"conversation_id": cid, "order_id": "3003"}) as response:
        status = response.status_code
        if status != 200:
            return {"http_status": status, "complete": False, "frames": frames, "received_at": now()}
        event = None
        async for line in response.aiter_lines():
            size += len(line.encode("utf-8")) + 1
            if size > BODY_LIMIT or len(frames) > 128:
                raise ValueError("sse_exceeds_lifecycle_limit")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
                if data == "[DONE]":
                    complete = True
                    break
                frame = strict_json(data)
                if event == "error":
                    frame = {"event": "error"}
                frames.append(_stream_projection(frame))
                event = None
    return {"http_status": status, "complete": complete, "frames": frames, "received_at": now(),
            "observation_protocol": "sse-lifecycle-v1", "raw_prose_recorded": False}


async def _create_owned(client, identity):
    sent = now()
    status, payload = await _json_request(client, "POST", "/api/agent", {"message": "取消这个请求"})
    if status != 200 or not isinstance(payload, dict) or not _positive(payload.get("conversation_id")):
        raise ValueError("new_owned_conversation_creation_failed")
    cid = payload["conversation_id"]
    recording = record_response(payload, http_status=status, authenticated_user_id=identity,
                                request_conversation_id=None, request_sent_at=sent, response_received_at=now())
    if (not valid_recording(recording) or payload.get("task_context", {}).get("user_id") != identity
            or payload.get("execution_summary", {}).get("model_calls") != 0):
        raise ValueError("new_owned_conversation_preflight_failed")
    return cid, recording


async def _history(client, cid):
    status, payload = await _json_request(client, "GET", f"/api/conversations/{cid}/messages")
    items = payload.get("items") if isinstance(payload, dict) else None
    return {"http_status": status, "message_count": len(items) if isinstance(items, list) else None,
            "scope": "owned public history only; does not prove private checkpoint contents"}


def _state_data(path):
    if path.suffix.lower() != ".json" or path.name.lower() == "holdout.jsonl" or path.stat().st_size > BODY_LIMIT:
        raise ValueError("bounded_lifecycle_state_json_required")
    state = _safe(strict_json(path.read_bytes()))
    required = {"version", "identity", "base_url", "profile", "disconnect_cid", "interrupt_cid",
                "selection_interrupt_observed", "before_report_sha256", "created_at"}
    if (not isinstance(state, dict) or set(state) != required or state.get("version") != VERSION
            or state.get("identity") != "customer_demo" or not _positive(state.get("disconnect_cid"))
            or not _positive(state.get("interrupt_cid")) or state["disconnect_cid"] == state["interrupt_cid"]
            or state.get("selection_interrupt_observed") is not True):
        raise ValueError("lifecycle_state_invalid_or_no_selection_interrupt")
    validate_url(state["base_url"])
    return state


async def probe(args):
    base = validate_url(args.base_url)
    report = {"version": VERSION, "mode": "actual_http_lifecycle_probe", "phase": args.phase,
              "base_url": base, "code_version": args.code_version, "started_at": now(),
              "declared_turn_timeout_seconds": args.declared_turn_timeout_seconds,
              "server_timeout_setting_verified_by_probe": None, "cases": [], "checks": {},
              "parent_checkpoint_observation_required": True,
              "limitations": ["HTTP 504 and public history cannot prove checkpoint termination",
                  "client closure cannot prove server start, cancellation or prior completion",
                  "service stop/restart and checkpoint continuity require independent parent observations",
                  "synthetic lifecycle diagnostics, not dataset accuracy, baseline improvement or provider bill"],
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    state = _state_data(args.state) if args.phase == "after" else None
    password = os.environ.get("MEWHELP_PROBE_OWNER_PASSWORD")
    if not password:
        raise ValueError("named_owner_password_environment_required")
    saved_state = None
    try:
        async with httpx.AsyncClient(base_url=base, timeout=args.http_timeout_seconds,
                                     trust_env=False, follow_redirects=False) as client:
            status, login = await _json_request(client, "POST", "/api/auth/login", {"username": "customer_demo", "password": password})
            token = login.get("access_token") if isinstance(login, dict) else None
            if status != 200 or not isinstance(token, str) or not token:
                raise ValueError("owner_login_failed")
            client.headers["Authorization"] = "Bearer " + token
            status, me = await _json_request(client, "GET", "/api/auth/me")
            if status != 200 or not isinstance(me, dict) or me.get("user_id") != "customer_demo" or me.get("role") != "customer":
                raise ValueError("owner_identity_preflight_failed")
            identity = me["user_id"]
            profile = _safe({key: me.get(key) for key in ("fixture_profile", "fixture_version", "fixture_sha256", "execution_policy")})
            report.update(identity=identity, profile=profile)
            report["checks"]["authenticated_owner"] = True
            if args.phase == "deadline":
                cid = args.conversation_id
                if cid is None:
                    cid, created = await _create_owned(client, identity)
                    report["cases"].append({"id": "new_deadline_conversation", "response_recording": created})
                else:
                    history = await _history(client, cid)
                    if history["http_status"] != 200:
                        raise ValueError("deadline_conversation_owner_preflight_failed")
                report["deadline_cid"] = cid
                sent, started = now(), time.perf_counter()
                status, _ = await _json_request(client, "POST", "/api/agent", {"message": QUERY, "conversation_id": cid})
                report["cases"].append({"id": "actual_deadline_request", "conversation_id": cid,
                                        "sent_at": sent, "received_at": now(), "http_status": status,
                                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)})
                report["checks"]["actual_http_504"] = status == 504
                history = await _history(client, cid)
                report["cases"].append({"id": "owned_history_after_deadline", **history})
                report["checks"]["owned_history_readable"] = history["http_status"] == 200
            elif args.phase == "before":
                cid, created = await _create_owned(client, identity)
                report["disconnect_cid"] = cid
                report["cases"].append({"id": "new_disconnect_conversation", "response_recording": created})
                closed = await _close_at_headers(client, cid)
                report["cases"].append({"id": "close_after_headers", "conversation_id": cid, **closed})
                report["checks"]["stream_headers_200"] = closed["http_status"] == 200
                report["checks"]["stream_content_type"] = closed["content_type"] == "text/event-stream"
                report["checks"]["client_stream_closed"] = closed["client_stream_closed"]
                await asyncio.sleep(3)  # Parent can inspect cancellation without a later turn overwriting this cid.
                selection_cid, created = await _create_owned(client, identity)
                report["interrupt_cid"] = selection_cid
                report["cases"].append({"id": "new_selection_conversation", "response_recording": created})
                sent = now()
                status, payload = await _json_request(client, "POST", "/api/agent", {"message": SELECT_QUERY, "conversation_id": selection_cid})
                recording = record_response(payload, http_status=status, authenticated_user_id=identity,
                                            request_conversation_id=selection_cid, request_sent_at=sent, response_received_at=now())
                interrupt = payload.get("interrupt") if isinstance(payload, dict) else None
                selected = status == 200 and isinstance(interrupt, dict) and interrupt.get("type") == "select_order"
                report["cases"].append({"id": "selection_interrupt_before_restart", "response_recording": recording,
                                        "interrupt_type": interrupt.get("type") if isinstance(interrupt, dict) else None})
                report["checks"]["selection_interrupt_observed"] = selected
                report["checks"]["distinct_new_conversations"] = cid != selection_cid
                saved_state = {"version": VERSION, "identity": identity, "base_url": base, "profile": profile,
                               "disconnect_cid": cid, "interrupt_cid": selection_cid,
                               "selection_interrupt_observed": selected, "created_at": now(), "before_report_sha256": None}
            else:
                report["checks"]["same_origin_identity_and_fixture"] = state["base_url"] == base and state["identity"] == identity and state["profile"] == profile
                if not report["checks"]["same_origin_identity_and_fixture"]:
                    raise ValueError("after_identity_origin_or_fixture_changed")
                report["before_report_sha256"] = state["before_report_sha256"]
                report["restart_observed_by_probe"] = None
                cid = state["disconnect_cid"]
                sent = now()
                status, payload = await _json_request(client, "POST", "/api/agent", {"message": QUERY, "conversation_id": cid})
                recording = record_response(payload, http_status=status, authenticated_user_id=identity,
                                            request_conversation_id=cid, request_sent_at=sent, response_received_at=now())
                observed = observe_response({}, http_status=status, recording=recording)
                report["cases"].append({"id": "same_cid_request_after_disconnect", "conversation_id": cid, "response_recording": recording})
                report["checks"]["same_cid_request_succeeded"] = status == 200 and payload.get("conversation_id") == cid if isinstance(payload, dict) else False
                report["checks"]["recovery_fresh_scoped_order_facts"] = _fresh_order(payload.get("task_context") if isinstance(payload, dict) else None, cid=cid, identity=identity, received=now())
                report["checks"]["recovery_no_write_effect_observed"] = not any(action == "committed_write" or action.startswith("execute:") for action in observed["actions"]) if observed["action_observation_complete"] else None
                selection_cid = state["interrupt_cid"]
                report["cases"].append({"id": "owned_selection_history_after_restart", **await _history(client, selection_cid)})
                stream = await _resume_stream(client, selection_cid)
                report["cases"].append({"id": "resume_selection_after_restart", "conversation_id": selection_cid, **stream})
                report["checks"].update({"resume_" + key: value for key, value in _resume_checks(stream["frames"], cid=selection_cid, identity=identity,
                                                                                                received=stream["received_at"], complete=stream["complete"]).items()})
    except Exception as exc:
        # Never retain exception messages, request headers, credentials or vendor bodies.
        report["error_class"] = type(exc).__name__
        report["completed_planned_http_operations"] = False
    else:
        report["completed_planned_http_operations"] = True
    report["completed_at"] = now()
    report["summary"] = {"observed_checks": sum(value is not None for value in report["checks"].values()),
                         "failed_checks": [key for key, value in report["checks"].items() if value is False],
                         "unknown_checks": [key for key, value in report["checks"].items() if value is None],
                         "scope": "declared public HTTP checks only; lifecycle completion requires parent checkpoint/process evidence"}
    _safe(report)
    if saved_state:
        saved_state["before_report_sha256"] = canonical_sha256(report)
        write_json_new(args.state, _safe(saved_state))
    write_json_new(args.output, report)
    return int(not report["completed_planned_http_operations"] or bool(report["summary"]["failed_checks"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("deadline", "before", "after"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conversation-id", type=int)
    parser.add_argument("--declared-turn-timeout-seconds", type=float, required=True)
    parser.add_argument("--http-timeout-seconds", type=float, default=180)
    args = parser.parse_args()
    if (not 0 < args.declared_turn_timeout_seconds <= 600 or not 0 < args.http_timeout_seconds <= 600
            or (args.phase == "deadline" and args.declared_turn_timeout_seconds > 5)
            or (args.conversation_id is not None and not _positive(args.conversation_id))):
        parser.error("valid bounded timeout and conversation settings required")
    if args.phase in {"before", "after"} and args.state is None:
        parser.error("before and after require --state")
    assert_new_output(args.output)
    if args.phase == "before":
        assert_new_output(args.state)
        if args.state.resolve() == args.output.resolve():
            parser.error("state and report must use different new paths")
    return asyncio.run(probe(args))


if __name__ == "__main__":
    raise SystemExit(main())
