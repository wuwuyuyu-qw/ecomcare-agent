"""Actual HTTP compound coverage diagnostic; not semantic task success.

Reuses authenticated transport, safe response/text recordings and strict fact
observation. Only an explicitly requested development split is read here.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.compound_cases import load_cases
from app.evaluation.task_recording import record_response, canonical_sha256, _safe_json
from app.evaluation.task_scoring import observe_response
from scripts.eval_task_agent import _post, _get, _record_text


def now():
    return datetime.now(timezone.utc).isoformat()


def coverage_checks(expected: dict, observed: dict, response: dict) -> dict:
    events = (response.get("execution_summary") or {}).get("tool_events", [])
    blocked = {"compound_tool_not_ready", "order_evidence_required", "tracking_scope_mismatch", "missing_order_id", "write_service_not_ready", "order_lookup_not_requested"}
    called = {e["tool"] for e in events if isinstance(e, dict) and isinstance(e.get("tool"), str)
              and e.get("error_code") not in blocked}
    checks = {"requested_tools_observed": set(expected["required_tools"]) <= called,
              "required_facts_observed": all(name in observed.get("facts", {}) for name in expected.get("required_facts", [])),
              "no_write_tool_event": not bool(called & {"create_ticket", "submit_refund"})}
    if expected.get("decision"):
        checks["program_control_decision"] = observed.get("decision") == expected["decision"]
    if expected.get("order_id"):
        checks["current_order_matches"] = observed.get("facts", {}).get("order_id") == expected["order_id"]
    if expected.get("no_tools"):
        checks["no_tool_execution"] = not events
    if expected.get("forbidden_tools"):
        checks["no_forbidden_downstream_tool"] = not (called & set(expected["forbidden_tools"]))
    return checks


def run(args):
    routing = getattr(args, "dataset", "compound") == "routing"
    cases, manifest = load_cases(ROOT / ("data/compound_routing_eval" if routing else "data/compound_eval"), "dev",
                                "compound-routing-v1.0" if routing else "compound-v1.1")
    if args.case_ids:
        wanted = set(args.case_ids)
        if not wanted <= {c["id"] for c in cases}:
            raise ValueError("unknown compound development case")
        cases = [c for c in cases if c["id"] in wanted]
    username, password = os.environ.get("MEWHELP_EVAL_USERNAME"), os.environ.get("MEWHELP_EVAL_PASSWORD")
    if not username or not password:
        raise ValueError("explicit isolated login required")
    status, login = _post(args.base_url, "/api/auth/login", {"username": username, "password": password}, 30)
    if status != 200:
        raise ValueError("isolated login failed")
    token = login["access_token"]
    me_status, me = _get(args.base_url, "/api/auth/me", 30, token)
    if me_status != 200 or me["user_id"] != "customer_demo" or me.get("fixture_profile") != "task-v1":
        raise ValueError("synthetic fixture identity differs")
    rows = []
    for case in cases:
        cid, turns = None, []
        case_start = time.monotonic()
        for turn in case["turns"]:
            body = {"message": turn["message"]}
            if cid is not None:
                body["conversation_id"] = cid
            sent, start = now(), time.monotonic()
            try:
                status, response = _post(args.base_url, "/api/agent", body, args.timeout, token)
            except (OSError, ValueError, TimeoutError):
                # Preserve incomplete requests instead of silently dropping them.
                status, response = None, {}
            received, elapsed = now(), time.monotonic() - start
            record = record_response(response, http_status=status, authenticated_user_id=me["user_id"],
                request_conversation_id=cid, request_sent_at=sent, response_received_at=received,
                unavailable_reason="http_error_body_withheld" if status != 200 else None)
            observed = observe_response(record.get("projection") or {}, http_status=status, recording=record)
            message, input_record = _record_text(body["message"], source="actual_http_input")
            answer, answer_record = _record_text(response.get("answer"), source="actual_http_response_answer")
            extra = {key: response.get(key) for key in ("compound_plan", "compound_completion", "answer_validation", "policy_assessment", "request_triage")}
            _safe_json(extra)
            checks = coverage_checks(turn, observed, response) if status == 200 and record["complete"] else {"complete_http": False}
            turns.append({"user_message": message, "input_recording": input_record,
                "final_answer": answer, "answer_recording": answer_record, "response_recording": record,
                "observed": observed, "diagnostic_checks": checks, "elapsed_seconds": elapsed,
                "compound_http_recording": {"source": "actual_http_extra_fields", "projection": extra, "sha256": canonical_sha256(extra)}})
            cid = response.get("conversation_id", cid)
            if status != 200:
                break
        passed = len(turns) == len(case["turns"]) and all(all(t["diagnostic_checks"].values()) for t in turns)
        rows.append({"id": case["id"], "status": "executed", "expected_turn_count": len(case["turns"]),
                     "turns": turns, "coverage_passed": passed, "elapsed_seconds": time.monotonic() - case_start})
        print(json.dumps({"event": "compound_case", "case_id": case["id"], "coverage_passed": passed}), flush=True)
    config_raw = args.config.read_bytes()
    return {"mode": "online", "measurement": "actual_compound_tool_fact_coverage_not_semantic_success",
        "observation_protocol_version": "actual-http-compound-coverage-v1", "split": "compound-routing-dev" if routing else "compound-dev",
        "dataset_sha256": manifest["dataset_sha256"], "run_config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "code_version": args.code_version, "implementation_label": "compound-" + args.strategy,
        "created_at": now(), "rows": rows, "requested_cases": len(cases),
        "coverage_passed_cases": sum(r["coverage_passed"] for r in rows),
        "limit": "coverage checks do not prove answer semantics, policy eligibility or background writes"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--strategy", choices=("fixed", "frontier", "joint-frontier"), required=True)
    parser.add_argument("--dataset", choices=("compound", "routing"), default="compound")
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--timeout", type=float, default=130)
    parser.add_argument("--output", "--out", dest="out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("choose a new compound output")
    report = run(args)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit("compound diagnostic failed: " + type(exc).__name__) from None
