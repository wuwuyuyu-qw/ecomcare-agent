"""Explicit online HTTP boundary probe for the local task-v1 demo.

Run this script only against an operator-prepared isolated service. It neither
starts services nor provisions accounts, reads .env, or opens any evaluation
split. Login bodies and authorization headers are never report observations.
This probe is not a dataset score, a baseline comparison, or a complete bill.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.task_recording import record_response, valid_recording

MESSAGE = "取消这个请求"
QUERY = "查询订单3001的物流状态"
ACCOUNTS = {"owner": ("customer_demo", "customer"),
            "other": ("customer_other", "customer"), "admin": ("admin_probe", "admin")}
PASSWORD_ENVS = {"owner": "MEWHELP_PROBE_OWNER_PASSWORD",
                 "other": "MEWHELP_PROBE_OTHER_PASSWORD", "admin": "MEWHELP_PROBE_ADMIN_PASSWORD"}
SEED_ENVS = ("MEWHELP_PROBE_STALE_OWNER_CID", "MEWHELP_PROBE_STALE_SCOPE_CID")
BODY_LIMIT = 4_000_000
TEXT_LIMIT = 20_000
SECRET_PATTERN = re.compile(
    r"(?i)(?:bearer\s+\S+|sk-[A-Za-z0-9_-]{8,}|"
    r"[\"']?(?:password|passwd|api[_ -]?key|access[_ -]?token|authorization|secret)[\"']?\s*[:=]|"
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})")
SUCCESS_PATTERN = re.compile(r"(?:已提交|已创建|已受理|确认成功|操作成功|创建成功|退款成功|申请成功)")

# Every requested case and check is declared before the first HTTP request.
SPECS = [
    ("anonymous_agent", 401, ()), ("invalid_session_me", 401, ()),
    ("customer_jobs", 403, ()), ("customer_kb_overview", 403, ()),
    ("admin_jobs", 200, ()),
    ("owner_new_session", 200, ("projection_complete", "fresh_conversation", "context_identity",
                               "context_conversation", "decision_cancelled", "termination_cancelled", "zero_model_calls")),
    ("other_new_session_forged_user", 200, ("projection_complete", "fresh_conversation", "context_identity",
                                           "context_conversation", "decision_cancelled", "termination_cancelled", "zero_model_calls")),
    ("owner_reads_other_history", 404, ()), ("other_reads_own_history", 200, ()),
    ("owner_reads_own_history", 200, ()), ("admin_reads_owner_history", 404, ()),
    ("other_reads_owner_history", 404, ()), ("other_continues_owner_agent", 404, ()),
    ("other_continues_owner_chat", 404, ()), ("other_feedback_owner", 404, ()),
    ("other_resumes_owner", 404, ()), ("other_creates_owner_ticket", 404, ()),
    ("other_creates_owner_refund", 404, ()),
    ("owner_legacy_ticket_disabled", 410, ()), ("owner_legacy_refund_disabled", 410, ()),
    ("owner_confirmed_resume_true", 409, ()), ("owner_confirmed_resume_false", 409, ()),
    ("owner_ended_order_resume", 200, ("sse_complete", "sse_error_event", "no_delta", "no_tool",
                                      "no_done", "no_confirmation_success")),
    ("other_unknown_history", 404, ("same_status_as_other_owner_history",)),
    ("owner_unknown_history", 404, ()),
    ("other_forged_order_lookup", 200, ("projection_complete", "fresh_conversation", "context_identity",
                                      "context_conversation", "decision_deny", "termination_permission_denied",
                                      "no_verified_facts", "no_evidence", "suggested_orders_observed",
                                      "no_owner_logistics_orders")),
    ("owner_stale_checkpoint_identity", 404, ()), ("owner_stale_checkpoint_scope", 404, ()),
    ("owner_logout", 204, ()), ("owner_revoked_session_me", 401, ()),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strict_json(raw: bytes | str):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate_json_key")
            out[key] = value
        return out

    def constant(_value):
        raise ValueError("nonfinite_json_value")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def positive_integer(value) -> bool:
    return type(value) is int and value > 0


def validate_url(value: str) -> str:
    parsed = urlsplit(value)
    loopback = parsed.hostname == "localhost"
    if parsed.hostname and not loopback:
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            pass
    if (parsed.scheme not in {"http", "https"} or not loopback or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
        raise ValueError("base_url_requires_loopback_origin")
    # Also validate malformed or out-of-range ports before any network access.
    parsed.port
    return value.rstrip("/")


def load_config(path: Path) -> tuple[dict, str, str]:
    if path.name != "run-config.json" or not path.is_file() or path.stat().st_size > 1_000_000:
        raise ValueError("existing_run_config_json_required")
    raw = path.read_bytes()
    value = strict_json(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("config_object_required")
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if (SECRET_PATTERN.search(encoded) or any(key in encoded.lower() for key in
            ('"password"', '"passwd"', '"api_key"', '"access_token"', '"authorization"', '"secret"'))):
        raise ValueError("credential_like_config_rejected")
    fixture = ROOT / "data" / "agent_fixtures" / "task-v1.json"
    expected = {"fixture_profile": "task-v1", "fixture_version": "task-v1.0",
                "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()}
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("config_task_v1_profile_version_hash_mismatch")
    execution = value.get("execution")
    if not isinstance(execution, dict) or type(execution.get("skip_coref_without_history")) is not bool:
        raise ValueError("config_execution_policy_required")
    return value, hashlib.sha256(encoded.encode("utf-8")).hexdigest(), hashlib.sha256(raw).hexdigest()


class Probe:
    def __init__(self, base_url: str, code_version: str):
        self.passwords = {role: os.environ.get(name, "") for role, name in PASSWORD_ENVS.items()}
        self.tokens: dict[str, str] = {}
        self.identities: dict[str, str] = {}
        self.seed_ids: list[int] = []
        self.cids: dict[str, int] = {}
        self.config: dict = {}
        self.report = {"mode": "online_boundary_probe", "report_version": 1,
                       "started_at": utc_now(), "completed_at": None, "base_url": base_url,
                       "code_version": code_version, "config_sha256": None, "config_file_sha256": None,
                       "predeclared": {"requested_count": len(SPECS), "requested_case_ids": [s[0] for s in SPECS]},
                       "preflight": {"status": "unknown", "checks": []}, "cases": [],
                       "fresh_conversation_ids": [], "summary": {},
                       "scope": "Synthetic HTTP permissions only; not a dataset score, human review, baseline improvement, or complete provider bill."}
        self.rows = {}
        for case_id, expected, names in SPECS:
            row = {"id": case_id, "status": "unknown", "expected_http_status": expected,
                   "request": None, "http_status": None, "request_sent_at": None, "response_received_at": None,
                   "code_version": code_version, "config_sha256": None,
                   "checks": [{"id": name, "status": "unknown", "reason": "not_observed",
                               "checked_at": None} for name in ("http_status", *names)]}
            self.rows[case_id] = row
            self.report["cases"].append(row)
        self.client = httpx.Client(base_url=base_url, trust_env=False, follow_redirects=False, timeout=120)

    def safe_text(self, value) -> str | None:
        if not isinstance(value, str) or len(value) > TEXT_LIMIT or SECRET_PATTERN.search(value):
            return None
        secrets = [*self.passwords.values(), *self.tokens.values()]
        if any(secret and (secret in value if len(secret) >= 4 else secret == value) for secret in secrets):
            return None
        return value

    def safe_value(self, value, depth=0):
        if depth > 8:
            return {"withheld": "depth_limit"}
        if value is None or type(value) in {bool, int, float}:
            return value
        if isinstance(value, str):
            text = self.safe_text(value)
            return text if text is not None else {"withheld": "unsafe_text"}
        if isinstance(value, list) and len(value) <= 128:
            return [self.safe_value(item, depth + 1) for item in value]
        if isinstance(value, dict) and len(value) <= 128:
            return {key: self.safe_value(item, depth + 1) for key, item in value.items()
                    if isinstance(key, str) and self.safe_text(key) is not None}
        return {"withheld": "invalid_shape"}

    def request(self, method, endpoint, *, role=None, body=None, token=None, capture=None):
        sent, started = utc_now(), time.monotonic()
        result = {"status": None, "payload": None, "sse": None, "sent": sent,
                  "received": None, "elapsed_ms": None, "unavailable_reason": None}
        selected_token = self.tokens.get(role) if token is None else token
        headers = {"Authorization": "Bearer " + selected_token} if selected_token else {}
        try:
            with self.client.stream(method, endpoint, json=body, headers=headers) as response:
                result["status"] = response.status_code
                # HTTP errors are recorded by status only. Their bodies are not read.
                if 200 <= response.status_code < 300 and capture is not None:
                    raw = bytearray()
                    for chunk in response.iter_bytes():
                        raw.extend(chunk)
                        if len(raw) > BODY_LIMIT:
                            result["unavailable_reason"] = "response_exceeds_recording_limit"
                            break
                    if result["unavailable_reason"] is None:
                        if capture == "json":
                            try:
                                payload = strict_json(bytes(raw))
                                if not isinstance(payload, dict):
                                    raise ValueError("response_not_object")
                                result["payload"] = payload
                            except (ValueError, UnicodeError, RecursionError):
                                result["unavailable_reason"] = "response_json_object_unavailable"
                        elif capture == "sse":
                            try:
                                result["sse"] = self.parse_sse(bytes(raw).decode("utf-8"))
                            except (ValueError, UnicodeError, RecursionError):
                                result["unavailable_reason"] = "sse_recording_unavailable"
        except httpx.HTTPError as exc:
            result["unavailable_reason"] = "transport_" + type(exc).__name__
        result.update(received=utc_now(), elapsed_ms=round((time.monotonic() - started) * 1000, 2))
        return result

    def parse_sse(self, raw: str) -> dict:
        frames = []
        flags = {"has_error_event": False, "has_delta": False, "has_tool": False,
                 "has_done": False, "has_confirmation_success_content": False}
        complete = True

        def has_success(value):
            if isinstance(value, str):
                return bool(SUCCESS_PATTERN.search(value))
            if isinstance(value, list):
                return any(has_success(item) for item in value)
            if isinstance(value, dict):
                return any((key in {"ticket_no", "refund_no"} and bool(item)) or has_success(item)
                           for key, item in value.items())
            return False

        for block in re.split(r"\r?\n\r?\n", raw):
            if not block.strip():
                continue
            if len(frames) >= 128:
                raise ValueError("sse_frame_limit")
            event, data = None, []
            for line in block.splitlines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
            if not data and event is None:
                continue
            text = "\n".join(data)
            frame = {"explicit_event": self.safe_text(event), "json_event": None,
                     "payload_fields": [], "text": {}, "done_marker": text == "[DONE]"}
            flags["has_error_event"] |= event == "error"
            flags["has_delta"] |= event == "delta"
            flags["has_tool"] |= event == "tool"
            flags["has_done"] |= event == "done"
            if event is not None and frame["explicit_event"] is None:
                complete = False
            if text == "[DONE]":
                flags["has_done"] = True
            elif text:
                try:
                    payload = strict_json(text)
                except ValueError:
                    payload = None
                if not isinstance(payload, dict):
                    frame["parse_status"] = "json_object_unavailable"
                    complete = False
                else:
                    frame["parse_status"] = "json_object"
                    frame["json_event"] = self.safe_text(payload.get("event"))
                    frame["payload_fields"] = [key for key in payload if self.safe_text(key) is not None]
                    flags["has_error_event"] |= payload.get("event") == "error"
                    flags["has_delta"] |= "delta" in payload or payload.get("event") == "delta" or event == "delta"
                    flags["has_tool"] |= payload.get("event") == "tool" or event == "tool"
                    flags["has_done"] |= payload.get("event") == "done" or event == "done"
                    flags["has_confirmation_success_content"] |= has_success(payload)
                    for key in ("message", "delta", "answer", "content"):
                        if key in payload:
                            actual = self.safe_text(payload[key])
                            frame["text"][key] = actual
                            if actual is None:
                                complete = False
            frames.append(frame)
        return {"source": "actual_http_200_sse_allowlist", "complete": complete,
                "frame_count": len(frames), "frames": frames, **flags}

    def check(self, row, name, expected, observed, *, available=True, reason=None):
        item = next(item for item in row["checks"] if item["id"] == name)
        item.update(status=("passed" if observed == expected else "failed") if available else "unknown",
                    expected=self.safe_value(expected), observed=self.safe_value(observed) if available else None,
                    checked_at=utc_now(), code_version=self.report["code_version"],
                    config_sha256=self.report["config_sha256"], http_status=row.get("http_status"))
        item.pop("reason", None)
        if reason or not available:
            item["reason"] = reason or "observation_unavailable"

    def preflight_check(self, name, passed, observed=None, expected=True):
        self.report["preflight"]["checks"].append(
            {"id": name, "status": "passed" if passed else "failed", "expected": self.safe_value(expected),
             "observed": self.safe_value(observed), "checked_at": utc_now(),
             "code_version": self.report["code_version"], "config_sha256": self.report["config_sha256"]})

    def preflight(self, config_path):
        try:
            self.config, canonical_sha, file_sha = load_config(config_path)
            self.report.update(config_sha256=canonical_sha, config_file_sha256=file_sha)
            for row in self.rows.values():
                row["config_sha256"] = canonical_sha
            self.preflight_check("config_task_v1_and_execution_policy", True, True)
        except (OSError, ValueError, UnicodeError, RecursionError):
            self.preflight_check("config_task_v1_and_execution_policy", False, False)
        self.preflight_check("all_named_passwords_present", all(self.passwords.values()),
                             {role: bool(value) for role, value in self.passwords.items()},
                             {role: True for role in ACCOUNTS})
        for name in SEED_ENVS:
            raw = os.environ.get(name, "")
            self.seed_ids.append(int(raw) if re.fullmatch(r"[1-9][0-9]{0,17}", raw) else 0)
        seeds_valid = all(positive_integer(cid) for cid in self.seed_ids) and len(set(self.seed_ids)) == 2
        self.preflight_check("both_distinct_seed_conversation_ids_present", seeds_valid, seeds_valid)
        if seeds_valid:
            self.report["fresh_conversation_ids"] = list(self.seed_ids)
        if any(item["status"] != "passed" for item in self.report["preflight"]["checks"]):
            self.report["preflight"]["status"] = "failed"
            return False
        expected_fixture = {key: self.config[key] for key in ("fixture_profile", "fixture_version", "fixture_sha256")}
        policy = {"skip_coref_without_history": self.config["execution"]["skip_coref_without_history"]}
        self.report["verified_environment"] = {**expected_fixture, "expected_execution_policy": policy}
        for role, (username, expected_role) in ACCOUNTS.items():
            login_body = {"username": username, "password": self.passwords[role]}
            if role == "other":
                login_body["role"] = "admin"
            login = self.request("POST", "/api/auth/login", body=login_body, capture="json")
            payload = login["payload"] or {}
            token = payload.get("access_token")
            login_ok = (login["status"] == 200 and isinstance(token, str) and bool(token)
                        and payload.get("token_type") == "bearer")
            self.preflight_check(role + "_login", login_ok,
                                 {"http_status": login["status"], "session_received": login_ok},
                                 {"http_status": 200, "session_received": True})
            # Only times and status are retained; neither login body nor response is a report observation.
            self.report["preflight"]["checks"][-1].update(
                request_sent_at=login["sent"], response_received_at=login["received"])
            if not login_ok:
                continue
            self.tokens[role] = token
            me = self.request("GET", "/api/auth/me", role=role, capture="json")
            actual = me["payload"] or {}
            expected = {"user_id": username, "role": expected_role, **expected_fixture,
                        "execution_policy": policy}
            observed = {key: actual.get(key) for key in expected}
            me_ok = me["status"] == 200 and observed == expected
            actual_policy = actual.get("execution_policy")
            observed["execution_policy"] = {"skip_coref_without_history":
                actual_policy.get("skip_coref_without_history") if isinstance(actual_policy, dict) else None}
            self.preflight_check(role + "_me_identity_fixture_policy", me_ok,
                                 {"http_status": me["status"], **observed}, {"http_status": 200, **expected})
            self.report["preflight"]["checks"][-1].update(
                request_sent_at=me["sent"], response_received_at=me["received"])
            if me_ok:
                self.identities[role] = actual["user_id"]
        passed = len(self.identities) == len(ACCOUNTS)
        self.report["preflight"]["status"] = "passed" if passed else "failed"
        return passed

    def case(self, case_id, role, method, endpoint, body=None, *, token=None, capture=None):
        row = self.rows[case_id]
        row["request"] = {"actor": role or "anonymous", "method": method, "path": endpoint,
                          **({"body": body} if body is not None else {})}
        if endpoint == "/api/agent":
            capture = "json"
        obs = self.request(method, endpoint, role=role, body=body, token=token, capture=capture)
        row.update(http_status=obs["status"], request_sent_at=obs["sent"],
                   response_received_at=obs["received"], elapsed_ms=obs["elapsed_ms"])
        if obs["unavailable_reason"]:
            row["unavailable_reason"] = obs["unavailable_reason"]
        self.check(row, "http_status", row["expected_http_status"], obs["status"], available=obs["status"] is not None)
        if endpoint == "/api/agent" and obs["status"] == 200 and role in self.identities:
            recording = record_response(obs["payload"], http_status=obs["status"],
                                        authenticated_user_id=self.identities[role],
                                        request_conversation_id=(body or {}).get("conversation_id"),
                                        request_sent_at=obs["sent"], response_received_at=obs["received"],
                                        unavailable_reason=obs["unavailable_reason"])
            # The existing projection supplies its own structural credential screen.
            # Also withhold it if any actual password/session string appears.
            if recording.get("complete") and self.safe_text(json.dumps(recording["projection"], ensure_ascii=False)) is None:
                recording = record_response(None, http_status=obs["status"],
                    authenticated_user_id=self.identities[role], request_conversation_id=(body or {}).get("conversation_id"),
                    request_sent_at=obs["sent"], response_received_at=obs["received"],
                    unavailable_reason="credential_like_projection_withheld")
            row["response_recording"] = recording
        if obs["sse"] is not None:
            row["sse_recording"] = obs["sse"]
        return obs

    def new_session(self, case_id, role, message, *, forged=False):
        body = {"message": message, **({"user_id": "customer_demo"} if forged else {})}
        obs = self.case(case_id, role, "POST", "/api/agent", body)
        row = self.rows[case_id]
        payload = obs["payload"] or {}
        recording = row.get("response_recording")
        ready = obs["status"] == 200 and isinstance(obs["payload"], dict)
        self.check(row, "projection_complete", True, valid_recording(recording), available=obs["status"] == 200)
        cid = payload.get("conversation_id")
        fresh = positive_integer(cid) and cid not in self.report["fresh_conversation_ids"]
        self.check(row, "fresh_conversation", True, fresh, available=ready)
        if fresh:
            self.report["fresh_conversation_ids"].append(cid)
            self.cids[case_id] = cid
        context = payload.get("task_context")
        context = context if isinstance(context, dict) else {}
        self.check(row, "context_identity", ACCOUNTS[role][0], context.get("user_id"), available=ready)
        self.check(row, "context_conversation", cid, context.get("conversation_id"), available=ready and positive_integer(cid))
        if message == MESSAGE:
            self.check(row, "decision_cancelled", "cancelled", context.get("decision"), available=ready)
            self.check(row, "termination_cancelled", "cancelled", context.get("termination_reason"), available=ready)
            execution = payload.get("execution_summary")
            count = execution.get("model_calls") if isinstance(execution, dict) else None
            self.check(row, "zero_model_calls", True, type(count) is int and count == 0, available=ready)
        else:
            self.check(row, "decision_deny", "deny", context.get("decision"), available=ready)
            self.check(row, "termination_permission_denied", "permission_denied", context.get("termination_reason"), available=ready)
            self.check(row, "no_verified_facts", {}, context.get("verified_facts"), available=ready)
            self.check(row, "no_evidence", [], context.get("evidence"), available=ready)
            actions = payload.get("suggested_actions")
            selectors = [action for action in actions if isinstance(action, dict) and action.get("type") == "select_order"] if isinstance(actions, list) else []
            orders_observed = bool(selectors) and all(isinstance(action.get("orders"), list) for action in selectors)
            orders = [item for action in selectors for item in action.get("orders", [])] if orders_observed else []
            orders_observed = orders_observed and bool(orders) and all(isinstance(item, dict) and isinstance(item.get("order_id"), str) for item in orders)
            ids = [item["order_id"] for item in orders] if orders_observed else []
            self.check(row, "suggested_orders_observed", True, orders_observed, available=ready)
            self.check(row, "no_owner_logistics_orders", [], sorted(set(ids) & {"3001", "3002", "3003"}),
                       available=ready and orders_observed)
            # task_recording intentionally omits order menu bodies. Retain only
            # these observed synthetic order IDs to support this extra check.
            row["suggested_order_ids"] = self.safe_value(ids) if orders_observed else None
            row["classification"] = "business_permission_denial_in_http_200"

    def run_cases(self):
        self.case("anonymous_agent", None, "POST", "/api/agent", {"message": MESSAGE})
        self.case("invalid_session_me", None, "GET", "/api/auth/me", token="invalid-synthetic-probe-session")
        self.case("customer_jobs", "owner", "GET", "/api/jobs")
        self.case("customer_kb_overview", "owner", "GET", "/api/kb/overview")
        self.case("admin_jobs", "admin", "GET", "/api/jobs")
        self.new_session("owner_new_session", "owner", MESSAGE)
        self.new_session("other_new_session_forged_user", "other", MESSAGE, forged=True)
        owner = self.cids.get("owner_new_session")
        other = self.cids.get("other_new_session_forged_user")
        if other:
            self.case("owner_reads_other_history", "owner", "GET", f"/api/conversations/{other}/messages")
            self.case("other_reads_own_history", "other", "GET", f"/api/conversations/{other}/messages")
        if owner:
            history = f"/api/conversations/{owner}/messages"
            for case_id, role in (("owner_reads_own_history", "owner"), ("admin_reads_owner_history", "admin"),
                                  ("other_reads_owner_history", "other")):
                self.case(case_id, role, "GET", history)
            self.case("other_continues_owner_agent", "other", "POST", "/api/agent", {"conversation_id": owner, "message": MESSAGE})
            self.case("other_continues_owner_chat", "other", "POST", "/api/chat", {"conversation_id": owner, "message": MESSAGE}, capture="sse")
            self.case("other_feedback_owner", "other", "POST", "/api/feedback", {"conversation_id": owner, "rating": "up", "question": MESSAGE})
            self.case("other_resumes_owner", "other", "POST", "/api/actions/resume", {"conversation_id": owner, "order_id": "3001"}, capture="sse")
            ticket = {"conversation_id": owner, "description": "权限probe", "ticket_type": "咨询"}
            refund = {"conversation_id": owner, "order_id": "3001", "reason": "其他"}
            for case_id, role, endpoint, body in (
                ("other_creates_owner_ticket", "other", "/api/actions/create-ticket", ticket),
                ("other_creates_owner_refund", "other", "/api/actions/create-refund", refund),
                ("owner_legacy_ticket_disabled", "owner", "/api/actions/create-ticket", ticket),
                ("owner_legacy_refund_disabled", "owner", "/api/actions/create-refund", refund)):
                self.case(case_id, role, "POST", endpoint, body)
            for case_id, confirmed in (("owner_confirmed_resume_true", True), ("owner_confirmed_resume_false", False)):
                self.case(case_id, "owner", "POST", "/api/actions/resume", {"conversation_id": owner, "confirmed": confirmed}, capture="sse")
            obs = self.case("owner_ended_order_resume", "owner", "POST", "/api/actions/resume",
                            {"conversation_id": owner, "order_id": "3001"}, capture="sse")
            row, sse = self.rows["owner_ended_order_resume"], obs["sse"] or {}
            available = obs["status"] == 200 and obs["sse"] is not None
            self.check(row, "sse_complete", True, sse.get("complete") is True and sse.get("frame_count", 0) > 0, available=available)
            for name, field, expected in (("sse_error_event", "has_error_event", True), ("no_delta", "has_delta", False),
                                          ("no_tool", "has_tool", False), ("no_done", "has_done", False),
                                          ("no_confirmation_success", "has_confirmation_success_content", False)):
                self.check(row, name, expected, sse.get(field), available=available)
            row["classification"] = "resume_rejection_via_http_200_sse_error"
        # A fixed high synthetic ID is still checked over HTTP, never presumed absent.
        unknown = 2_147_483_647
        obs = self.case("other_unknown_history", "other", "GET", f"/api/conversations/{unknown}/messages")
        known_status = self.rows["other_reads_owner_history"]["http_status"]
        self.check(self.rows["other_unknown_history"], "same_status_as_other_owner_history", known_status,
                   obs["status"], available=known_status is not None and obs["status"] is not None)
        self.case("owner_unknown_history", "owner", "GET", f"/api/conversations/{unknown}/messages")
        self.new_session("other_forged_order_lookup", "other", QUERY, forged=True)
        for case_id, cid in zip(("owner_stale_checkpoint_identity", "owner_stale_checkpoint_scope"), self.seed_ids):
            self.case(case_id, "owner", "POST", "/api/agent", {"conversation_id": cid, "message": MESSAGE})
        self.case("owner_logout", "owner", "POST", "/api/auth/logout")
        self.case("owner_revoked_session_me", "owner", "GET", "/api/auth/me")

    def finish(self):
        preflight_failed = self.report["preflight"]["status"] != "passed"
        for row in self.rows.values():
            for check in row["checks"]:
                if check["status"] == "unknown" and check.get("reason") == "not_observed":
                    check.update(reason="preflight_failed" if preflight_failed else "required_observation_or_conversation_unavailable",
                                 code_version=self.report["code_version"], config_sha256=self.report["config_sha256"],
                                 http_status=row["http_status"])
            states = [item["status"] for item in row["checks"]]
            row["status"] = "failed" if "failed" in states else "unknown" if "unknown" in states else "passed"
        totals = {state: sum(row["status"] == state for row in self.rows.values()) for state in ("passed", "failed", "unknown")}
        agent_rows = [row for row in self.rows.values() if (row.get("request") or {}).get("path") == "/api/agent" and row["http_status"] == 200] if not preflight_failed else []
        calls, covered, usage_coverage, price_coverage = 0, 0, [], []
        for row in agent_rows:
            recording = row.get("response_recording")
            projection = recording.get("projection") if valid_recording(recording) else {}
            execution = (projection or {}).get("execution_summary")
            if isinstance(execution, dict) and type(execution.get("model_calls")) is int and execution["model_calls"] >= 0:
                calls += execution["model_calls"]
                covered += 1
                usage_coverage.append(self.safe_value(execution.get("usage_coverage")))
                price_coverage.append(self.safe_value(execution.get("price_coverage")))
        self.report["summary"] = {"requested_count": len(SPECS), **totals,
            "observed_http_requests_in_cases": sum(row["request"] is not None for row in self.rows.values()),
            "foreground_chat_calls_from_agent_responses": calls if covered == len(agent_rows) and agent_rows else None,
            "foreground_chat_calls_observed_partial": calls,
            "agent_response_ledger_coverage": {"observed_http_200_agent_responses": len(agent_rows),
                "responses_with_valid_foreground_model_call_ledger": covered,
                "coverage": covered / len(agent_rows) if agent_rows else None,
                "reported_usage_coverage_by_observed_response": usage_coverage,
                "reported_price_coverage_by_observed_response": price_coverage,
                "scope": "Only actual recorded HTTP 200 /api/agent response ledgers; background work and provider internal retries excluded."}}
        self.report["completed_at"] = utc_now()
        return self.report


def write_reports(path: Path, report: dict):
    markdown = path.with_suffix(".md")
    summary = report["summary"]
    lines = ["# Online boundary probe", "", "Mode: `online_boundary_probe`; report version: 1.", "",
             f"Code version: `{report['code_version']}`", f"Config SHA-256: `{report['config_sha256']}`", "",
             f"Preflight: **{report['preflight']['status']}**", "",
             f"Requested: {summary['requested_count']}; passed: {summary['passed']}; failed: {summary['failed']}; unknown: {summary['unknown']}.", "",
             "Synthetic HTTP observations only. This is not a 120-case dataset score, human evaluation, baseline improvement, or complete provider bill.", "",
             "| Case | Result | HTTP | Checks passed / requested |", "| --- | --- | --- | --- |"]
    for row in report["cases"]:
        passed = sum(check["status"] == "passed" for check in row["checks"])
        lines.append(f"| {row['id']} | {row['status']} | {row['http_status']} | {passed}/{len(row['checks'])} |")
    ledger = summary["agent_response_ledger_coverage"]
    lines.extend(["", f"Foreground chat calls from observed Agent response ledgers: {summary['foreground_chat_calls_from_agent_responses']}.",
                  f"Ledger coverage: {ledger['responses_with_valid_foreground_model_call_ledger']}/{ledger['observed_http_200_agent_responses']} observed HTTP 200 Agent responses.",
                  "", "HTTP 200 business permission denial and HTTP 200 SSE resume errors are recorded with their actual transport status.",
                  "The JSON contains every declared check, timestamps, safe SSE observations, and actual allowlist response projections with SHA-256.", ""])
    # Exclusive creation for both outputs; an existing sibling Markdown also
    # rejects the request before HTTP begins (main) and at final write (here).
    json_file = path.open("x", encoding="utf-8")
    try:
        md_file = markdown.open("x", encoding="utf-8")
    except OSError:
        json_file.close()
        path.unlink()
        raise
    with json_file, md_file:
        json.dump(report, json_file, ensure_ascii=False, indent=2, allow_nan=False)
        json_file.write("\n")
        md_file.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--config", type=Path, required=True, help="Existing run-config.json; no credential inputs")
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--output", type=Path, required=True, help="NEW JSON report; also creates a NEW same-name Markdown report")
    args = parser.parse_args()
    try:
        base_url = validate_url(args.base_url)
        output = args.output.expanduser().resolve()
        if output.suffix.lower() != ".json" or not output.parent.is_dir() or output.exists() or output.with_suffix(".md").exists():
            raise ValueError("new_json_and_markdown_outputs_required")
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@:+-]{0,159}", args.code_version)
                or SECRET_PATTERN.search(args.code_version)):
            raise ValueError("nonsecret_code_version_identifier_required")
    except (OSError, ValueError):
        parser.error("Require a loopback origin, a nonsecret code version, and unused JSON/Markdown paths in an existing directory.")
    probe = Probe(base_url, args.code_version)
    try:
        if probe.preflight(args.config.expanduser().resolve()):
            probe.run_cases()
    except (OSError, ValueError, TypeError, KeyError, RecursionError, OverflowError):
        # Do not print exception text: it can include upstream or credential data.
        probe.report["run_unavailable_reason"] = "unexpected_local_probe_error"
    finally:
        probe.client.close()
    report = probe.finish()
    try:
        write_reports(output, report)
    except (OSError, ValueError):
        print("Unable to create both new report files; existing files were not overwritten.", file=sys.stderr)
        return 2
    print(f"Boundary probe: requested={len(SPECS)} passed={report['summary']['passed']} failed={report['summary']['failed']} unknown={report['summary']['unknown']}")
    return 0 if report["preflight"]["status"] == "passed" and report["summary"]["failed"] == report["summary"]["unknown"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
