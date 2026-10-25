"""Evaluation-only fixed dev whitelist and global provider circuit breaker.

Never reads held-out labels. Existing task/final graders remain authoritative;
this module records availability, stop reasons and failure attribution separately.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from app.evaluation.stage1 import ProviderBudgetExceeded, budget_snapshot

ROOT = Path(__file__).resolve().parents[2]
VERSION = "stage1-dev12-v1"
CASE_IDS = ("task-01-01", "task-01-02", "task-02-01", "task-02-02", "task-03-01", "task-03-02",
            "task-05-01", "task-05-02", "task-12-01", "task-12-02", "task-04-01", "task-06-01")
LIMITS = (80, 20, 800000)


def validate_ids(ids):
    if not isinstance(ids, (list, tuple)) or tuple(ids) != CASE_IDS:
        raise ValueError("dev12 requires exactly the frozen 12 Case IDs in declared order")


def definition():
    path = ROOT / "data/agent_eval/dev12_v1.json"
    value = json.loads(path.read_text(encoding="utf8"))
    validate_ids(value.get("case_ids"))
    if (value.get("version"), value.get("split"), value.get("tasks"), value.get("turns"),
        value.get("chat_requests"), value.get("retrieval_requests"), value.get("reserved_output_tokens")) != (VERSION, "dev", 12, 13, *LIMITS):
        raise ValueError("dev12 definition or budgets changed")
    return value


def select_cases(cases, ids):
    validate_ids(ids)
    index = {c["id"]: c for c in cases}
    selected = []
    for case_id in CASE_IDS:
        case = index.get(case_id)
        if case is None or case["split"] != "dev" or "task-v1" not in case["online_profiles"]:
            raise ValueError("dev12 missing, wrong-split or unavailable case")
        selected.append(deepcopy(case))
    if sum(len(c["turns"]) for c in selected) != 13:
        raise ValueError("dev12 turn denominator changed")
    return selected


def stopped(path):
    state = budget_snapshot(path)
    attempts, limits = state["attempts"], state["limits"]
    if any(limits[k] > 0 and attempts.get(k, {}).get("requests", 0) >= limits[k] for k in ("chat", "retrieval")):
        return "provider_request_cap_reached"
    if sum(v["requested_output_tokens"] for v in attempts.values()) >= limits["requested_output_tokens"]:
        return "reserved_output_token_cap_reached"
    with sqlite3.connect(path) as conn:
        if conn.execute("SELECT COUNT(*) FROM batch_blocks").fetchone()[0]:
            return "provider_gate_blocked"
    return None


def initialize(path):
    from app.evaluation.stage1 import initialize_budget
    initialize_budget(path, chat=LIMITS[0], retrieval=LIMITS[1], output_tokens=LIMITS[2])
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE batch_blocks (reason TEXT, observed_at REAL)")
        conn.execute("CREATE TABLE provider_usage (attempt_id INTEGER, model TEXT, input_tokens INTEGER, output_tokens INTEGER, cache_read INTEGER, complete INTEGER)")
        from app.evaluation.usage_meter import initialize as initialize_usage
        initialize_usage(conn)


def reserve(path, kind, output_tokens, input_bytes):
    if (kind not in {"chat", "retrieval"} or type(output_tokens) is not int or type(input_bytes) is not int
            or output_tokens < 0 or not 0 <= input_bytes <= 1_000_000):
        raise ProviderBudgetExceeded("invalid_or_oversized_provider_request")
    blocked = None
    with sqlite3.connect(path, timeout=10) as conn:
        conn.execute("BEGIN IMMEDIATE")
        chat, retrieval, tokens = conn.execute("SELECT * FROM limits").fetchone()
        counts = dict(conn.execute("SELECT kind,COUNT(*) FROM attempts GROUP BY kind"))
        total = conn.execute("SELECT COALESCE(SUM(output_tokens),0) FROM attempts").fetchone()[0]
        if conn.execute("SELECT COUNT(*) FROM batch_blocks").fetchone()[0]:
            blocked = "batch_already_stopped"
        elif ((chat > 0 and counts.get("chat", 0) >= chat) or
              (retrieval > 0 and counts.get("retrieval", 0) >= retrieval) or
              {"chat": chat, "retrieval": retrieval}[kind] == 0):
            blocked = "provider_request_cap_reached"
        elif total + output_tokens > tokens:
            blocked = "reserved_output_token_cap_would_be_exceeded"
        if blocked is None:
            from app.evaluation.money_guard import check_locked
            blocked = check_locked(conn, kind, input_bytes)
        if blocked:
            conn.execute("INSERT INTO batch_blocks VALUES (?,?)", (blocked, time.time()))
        else:
            key = conn.execute("INSERT INTO attempts(kind,output_tokens,input_bytes) VALUES (?,?,?)",
                               (kind, output_tokens, input_bytes)).lastrowid
            from app.evaluation.money_guard import admit_locked
            admit_locked(conn, key, kind)
    if blocked:
        raise ProviderBudgetExceeded(blocked)
    return key


def install_http_budget(path, hosts):
    """Shared global cap includes every HTTP attempt/retry and background call.

    No endpoint/model fallback is permitted. Only metadata and vendor usage are
    persisted, never request bodies/credentials or model reasoning.
    """
    import httpx
    from app.evaluation import money_guard as money, usage_meter
    usage_meter.install_sdk_observer()
    sync_send, async_send = httpx.Client.send, httpx.AsyncClient.send

    def prepare(request):
        if request.url.host in {"127.0.0.1", "localhost", "::1"}:
            return None
        if (request.method == "GET" and request.url.scheme == "https" and request.url.host == "api.deepseek.com"
                and request.url.path == "/user/balance" and not request.url.query and money.enabled(path)):
            return None  # Read-only wallet checks are recorded separately, never as inference.
        if request.url.host not in hosts:
            with sqlite3.connect(path) as conn:
                conn.execute("INSERT INTO batch_blocks VALUES (?,?)", ("unconfigured_host", time.time()))
            raise ProviderBudgetExceeded("unconfigured_external_http_host_blocked")
        body = json.loads(request.content)
        model = body.get("model")
        suffix = request.url.path.rstrip("/").rsplit("/", 1)[-1]
        allowed = {"completions": ("api.deepseek.com", "chat", {"deepseek-v4-flash"}),
                   "embeddings": ("api.siliconflow.cn", "retrieval", {"BAAI/bge-m3"}),
                   "rerank": ("api.siliconflow.cn", "retrieval", {"BAAI/bge-reranker-v2-m3"})}
        rule = allowed.get(suffix)
        if not rule or request.url.host != rule[0] or model not in rule[2]:
            with sqlite3.connect(path) as conn:
                conn.execute("INSERT INTO batch_blocks VALUES (?,?)", ("unapproved_endpoint_or_model", time.time()))
            raise ProviderBudgetExceeded("unapproved_endpoint_or_model")
        kind = rule[1]
        output = body.get("max_tokens", body.get("max_completion_tokens")) if kind == "chat" else 0
        if type(output) is not int or output < (1 if kind == "chat" else 0) or output > 10000:
            raise ProviderBudgetExceeded("explicit_provider_output_bound_required")
        if money.enabled(path):
            from app.config import settings
            money.refresh(path, settings.chat_api_key, force=False)
        try:
            key = reserve(path, kind, output, len(request.content))
        except ProviderBudgetExceeded as exc:
            raw = request.headers.get("x-stainless-retry-count")
            retry = int(raw) if raw is not None and raw.isdecimal() else None
            with sqlite3.connect(path) as conn:
                conn.execute("INSERT INTO provider_denials VALUES (?,?,?)", (kind, retry, str(exc)))
            raise
        usage_meter.attempt(path, key, request)
        return key

    def finish(key, status, started):
        if key is not None:
            with sqlite3.connect(path) as conn:
                conn.execute("UPDATE attempts SET status=?,elapsed_ms=? WHERE id=?",
                             (status, round((time.perf_counter()-started)*1000, 3), key))

    def usage(key, payload):
        if key is None:
            return
        with sqlite3.connect(path) as conn:
            status = conn.execute("SELECT status FROM attempts WHERE id=?", (key,)).fetchone()[0]
            kind = conn.execute("SELECT kind FROM attempts WHERE id=?", (key,)).fetchone()[0]
        if kind == "retrieval" and isinstance(payload, dict) and isinstance(payload.get("usage"), dict):
            payload = deepcopy(payload)
            value = payload["usage"]
            value.setdefault("prompt_tokens", value.get("total_tokens"))
            value.setdefault("completion_tokens", 0)
        observation = usage_meter.Observation(path, key, status)
        observation.record(payload, "httpx_json")
        observation.finish(True)

    def sending(self, request, *args, **kwargs):
        key, start = prepare(request), time.perf_counter()
        try:
            response = sync_send(self, request, *args, **kwargs)
            finish(key, response.status_code, start)
            if not kwargs.get("stream"):
                try: usage(key, response.json())
                except (ValueError, httpx.ResponseNotRead): pass
            money.settle(path, key, status=response.status_code, complete=not kwargs.get("stream", False))
            return response
        except BaseException:
            finish(key, None, start)
            if key is not None:
                usage_meter.Observation(path, key, None).finish(False, "transport_error")
            money.settle(path, key, status=None, complete=False)
            raise

    async def asending(self, request, *args, **kwargs):
        key, start = prepare(request), time.perf_counter()
        try:
            response = await async_send(self, request, *args, **kwargs)
            finish(key, response.status_code, start)
            if key is not None and kwargs.get("stream"):
                usage_meter.attach(response, path, key)
            elif key is not None:
                try: usage(key, response.json())
                except (ValueError, httpx.ResponseNotRead): pass
                money.settle(path, key, status=response.status_code, complete=True)
            return response
        except BaseException:
            finish(key, None, start)
            if key is not None:
                usage_meter.Observation(path, key, None).finish(False, "transport_error")
            money.settle(path, key, status=None, complete=False)
            raise

    httpx.Client.send, httpx.AsyncClient.send = sending, asending


def not_run(case, reason):
    return {"id": case["id"], "family": case["family"], "split": "dev", "status": "NOT_RUN",
            "execution_status": "NOT_RUN", "batch_status": "ABORTED", "baseline_eligible": False,
            "task_success": None, "abort_reason": reason, "expected_turn_count": len(case["turns"]), "observed_turn_count": 0,
            "turns": [], "failure_type": "not_executed", "measurement": "requested denominator retained; no model execution"}


def finalize(row, case, stop_reason=None):
    turns = row.get("turns", [])
    components = [t.get("score", {}).get("success") for t in turns] + [t.get("final_score", {}).get("success") for t in turns]
    complete = len(turns) == len(case["turns"])
    row["task_success"] = False if False in components else None if not complete or not components or None in components else True
    statuses = [(t.get("response_recording") or {}).get("http_status") for t in turns]
    if not complete and stop_reason:
        kind = "budget_interrupted"
    elif any(s != 200 for s in statuses):
        kind = "external_or_service_error"
    elif row["task_success"] is True:
        kind = "correct_refusal" if all(set(t["expected"]["decisions"]) == {"deny"} for t in case["turns"]) else "satisfied"
    elif row["task_success"] is None:
        kind = "scoring_observation_insufficient"
    else:
        kind = "business_or_contract_failure"
    row.update(failure_type=kind, budget_stop_reason=stop_reason, expected_turn_count=len(case["turns"]), observed_turn_count=len(turns),
               execution_status="ABORTED" if not complete and stop_reason else "COMPLETED_WITH_PROTECTION" if stop_reason else "COMPLETED",
               baseline_eligible=complete and not stop_reason)
    if not complete and stop_reason:
        row["status"] = "ABORTED"
    return row


def denominator(rows):
    from collections import Counter
    def bucket(r):
        if r.get("status") in {"NOT_RUN", "ABORTED", "skipped"}:
            return {"NOT_RUN": "not_run", "ABORTED": "aborted", "skipped": "skipped"}[r["status"]]
        return "pass" if r.get("task_success") is True else "fail" if r.get("task_success") is False else "unknown"
    counts = Counter(bucket(r) for r in rows)
    return {"requested": 12, "saved": len(rows), **{k: counts[k] for k in ("pass", "fail", "unknown", "skipped", "not_run", "aborted")},
        "baseline_eligible": sum(r.get("baseline_eligible") is True for r in rows),
        "expected_turns": 13, "observed_turns": sum(len(r.get("turns", [])) for r in rows),
        "by_family": {family: dict(Counter(bucket(r) for r in rows if r["family"] == family)) for family in {r["family"] for r in rows}}}


def prepare(directory, ids):
    validate_ids(ids)
    spec = definition()
    import os
    os.environ["MEWHELP_FIXTURE_PROFILE"] = "task-v1"
    from app.tools import business, contracts, synthetic_fixtures
    from mcp_servers.logistics_server import query_logistics
    from app.evaluation.stage1 import adapted_cases
    from app.evaluation.task_cases import load_dataset
    from scripts import eval_stage1_baseline as base
    import asyncio
    cases, manifest = load_dataset(ROOT/"data/agent_eval", "dev")
    extended, adapters = adapted_cases(cases, business.owns_order)
    selected = select_cases(extended, ids)
    audit = []
    for case in selected:
        fixture = case["fixture_requirements"]
        checks = {"dev_only": case["split"] == "dev", "fixture_profile_available": "task-v1" in case["online_profiles"],
                  "expected_unchanged": case["turns"] == next(c for c in cases if c["id"] == case["id"])["turns"]}
        orders = []
        for target in [fixture, fixture.get("replacement_order", {})]:
            oid = target.get("order_id")
            if not oid: continue
            owned = business.owns_order("customer_demo", oid)
            expected_owned = fixture.get("ownership") is True
            checks["ownership_"+oid] = owned is expected_owned
            if owned:
                order = business.order_snapshot(oid)
                contracts.validate_result("query_order", order, {"order_id": oid})
                logistics = asyncio.run(query_logistics(order["tracking_no"]))
                contracts.validate_result("query_logistics", logistics, {"tracking_no": order["tracking_no"]})
                checks["world_"+oid] = all(logistics.get(k) == target.get(k) for k in ("tracking_no", "stalled_hours", "status_code"))
                orders.append({"order": order, "logistics": logistics})
            else:
                orders.append({"order_id": oid, "independent_owned": False, "protected_snapshot_read": False})
        audit.append({"id": case["id"], "family": case["family"], "fixture": fixture,
                      "turns": case["turns"], "checks": checks, "fixture_snapshots": orders,
                      "identity": "authenticated customer_demo from actual server; client assertions ignored",
                      "success": all(checks.values())})
    if not all(r["success"] for r in audit):
        raise ValueError("dev12 fixture audit failed")
    frozen = base.source_state(exclude_holdout=True)
    base.write(directory/"source-manifest.json", frozen)
    import subprocess
    git = lambda *args: subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    base.write(directory/"git-state.json", {"head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
        "status": git("status", "--short"), "dirty_source_sha256": frozen["sha256"]})
    base.write(directory/"case-audit.json", {"rows": audit, "all_passed": True, "requested": 12, "turns": 13,
        "validation_scope": manifest["validation_scope"], "adapters": adapters, "real_provider_requests": 0})
    base.write(directory/"frozen-cases.json", {"cases": selected, "manifest": manifest})
    from app.evaluation.stage1_v2 import VERSION as FINAL
    base.write(directory/"frozen-plan.json", {"created_at": datetime.now(timezone.utc).isoformat(), "batch_version": VERSION,
        "case_ids": list(CASE_IDS), "phase": "dev12", "contract": "v2", "final_contract": FINAL,
        "source_sha256": frozen["sha256"], "dataset_sha256": manifest["dataset_sha256"],
        "scorer": "task-scorer-v3", "limits": spec, "approval": "pending_user_budget_confirmation",
        "frozen_cases_sha256": base.digest(directory/"frozen-cases.json"),
        "monetary_policy_sha256": base.digest(directory/"monetary-policy.json") if (directory/"monetary-policy.json").exists() else None,
        "no_holdout_labels_loaded": True, "real_business_data": False})
    print(json.dumps({"event": "dev12_prepared_no_model_calls", "tasks": 12, "turns": 13, "approval": "pending"}))


def approved_cases(args, directory):
    validate_ids(getattr(args, "case_ids", None))
    if getattr(args, "budget_confirmed", False) is not True or args.contract != "v2" or getattr(args, "authorized_full", False):
        raise ValueError("dev12 needs explicit confirmed batch budget; full flags are forbidden")
    definition()
    from app.evaluation.money_guard import validate_policy
    validate_policy(directory)
    plan = json.loads((directory/"frozen-plan.json").read_text())
    payload = directory/"frozen-cases.json"
    if (plan.get("phase") != "dev12" or plan.get("batch_version") != VERSION
            or plan.get("case_ids") != list(CASE_IDS) or hashlib.sha256(payload.read_bytes()).hexdigest() != plan["frozen_cases_sha256"]
            or plan.get("limits") != definition()):
        raise ValueError("dev12 frozen selection changed")
    if hashlib.sha256((directory/"monetary-policy.json").read_bytes()).hexdigest() != plan.get("monetary_policy_sha256"):
        raise ValueError("monetary policy changed after freeze")
    data = json.loads(payload.read_text())
    return select_cases(data["cases"], args.case_ids), data["manifest"]


def verify_configuration(settings):
    from urllib.parse import urlsplit
    frozen_controls = {"max_agent_steps": 6, "max_task_tool_calls": 12, "max_identical_tool_calls": 2,
        "turn_timeout_seconds": 120.0, "mcp_tool_timeout": 10.0, "chat_thinking": "adaptive",
        "chat_reasoning_effort": "", "retrieval_collection": "knowledge"}
    pairs = ((settings.chat_base_url, "api.deepseek.com"), (settings.embed_base_url, "api.siliconflow.cn"), (settings.rerank_base_url, "api.siliconflow.cn"))
    if (any(urlsplit(url).hostname != host or urlsplit(url).scheme != "https" for url, host in pairs)
            or settings.chat_model != "deepseek-v4-flash" or settings.embed_model != "BAAI/bge-m3"
            or settings.rerank_model != "BAAI/bge-reranker-v2-m3" or settings.max_output_tokens != 10000
            or settings.intent_model not in {"", settings.chat_model} or settings.summary_model not in {"", settings.chat_model}
            or settings.intent_base_url not in {"", settings.chat_base_url} or settings.summary_base_url not in {"", settings.chat_base_url}
            or settings.policy_catalog_path or settings.compound_agent_enabled or settings.compound_joint_triage_enabled
            or settings.evidence_selection_enabled or settings.skip_coref_without_history):
        raise ValueError("dev12 config drift: re-review provider pricing/models/production factors before spending")
    if any(getattr(settings, name, None) != value for name, value in frozen_controls.items()):
        raise ValueError("dev12 production guard/runtime factors changed; no provider spending permitted")


def install_tool_observer(path):
    """Record direct investigator calls as well as model-issued tool calls.

    Results are existing engine-validated Mock data, never unvalidated remote
    payloads, exception prose or model reasoning. Original return/errors propagate.
    """
    import os
    from app.core.execution import current_execution
    from app.tools import engine, contracts
    from app.evaluation.task_replay import _SECRET
    original = engine.execute_tool_call

    async def observed(tool_call, conversation_id, specs, **kwargs):
        scope = current_execution()
        row = {"request_id": scope.request_id if scope else None, "task_id": scope.task_id if scope else None,
               "conversation_id": conversation_id, "authenticated_user_id": kwargs.get("user_id"),
               "tool_call": contracts.redact(tool_call, audit=False),
               "started_at": datetime.now(timezone.utc).isoformat(), "source": "actual_engine_boundary"}
        try:
            run = await original(tool_call, conversation_id, specs, **kwargs)
            row.update(ok=run.ok, status=run.status, error_code=run.error_code, business_status=run.business_status,
                       retry_count=run.retry_count, duration_ms=run.duration_ms,
                       validated_result=contracts.redact(run.data, audit=False))
            return run
        except BaseException as exc:
            row.update(ok=False, raised_error_class=type(exc).__name__, validated_result=None)
            raise
        finally:
            row["ended_at"] = datetime.now(timezone.utc).isoformat()
            encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)
            if _SECRET.search(encoded):
                encoded = json.dumps({"request_id": row["request_id"], "observation_withheld": True,
                                      "reason": "credential_like_content"})
            with Path(path).open("a", encoding="utf8") as stream:
                stream.write(encoded + "\n"); stream.flush(); os.fsync(stream.fileno())
    engine.execute_tool_call = observed
