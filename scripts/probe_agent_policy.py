"""Actual HTTP policy integration on an owned one-document synthetic index.

This focused test proves retrieval-to-condition-check wiring, not Recall@k on
the course corpus. Common task-v3 recording remains unchanged.
"""
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
from app.evaluation.task_recording import record_response, valid_recording, canonical_sha256
from app.evaluation.task_report import assert_new_output, write_json_new
from eval_task_agent import _post, _get
from probe_agent_boundaries import load_config, validate_url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-url", "config", "code-version", "output"):
        parser.add_argument("--" + name, required=True, type=Path if name in {"config", "output"} else str)
    args = parser.parse_args()
    base = validate_url(args.base_url)
    config, digest, _ = load_config(args.config)
    assert_new_output(args.output, markdown=True)
    setup = config.get("policy_diagnostic") or {}
    if (setup.get("indexed_document_ids") != [41003] or
            not str(setup.get("collection", "")).startswith("diagnostic_policy_")):
        raise ValueError("explicit isolated synthetic policy setup required")
    password = os.environ.get("MEWHELP_PROBE_OWNER_PASSWORD")
    if not password:
        raise ValueError("named password environment required")
    plan = [
        {"id": "approved_applicable", "message": "查询订单3001的物流，两天多没更新，按最新物流异常政策能怎么处理？", "order": "3001", "outcome": "applicable"},
        {"id": "unknown_order_scope", "message": "查询订单1001的物流，两天多没更新，按最新物流异常政策能怎么处理？", "order": "1001", "outcome": "unknown"},
    ]
    report = {"report_version": 1, "code_version": args.code_version, "config_sha256": digest,
              "mode": "actual_http_policy_integration", "plan": plan, "cases": [],
              "fresh_conversation_ids": [], "limitations": [
                  "two targeted pipeline cases on one owned synthetic indexed document; not corpus retrieval accuracy",
                  "metadata applicability and structured thresholds do not prove arbitrary generated claims",
                  "setup embedding and readiness calls excluded from foreground metrics; prices unknown"]}
    token = None
    try:
        status, login = _post(base, "/api/auth/login", {"username": "customer_demo", "password": password}, 20)
        token = login.get("access_token")
        if status != 200 or not token:
            raise ValueError("identity_preflight_failed")
        status, me = _get(base, "/api/auth/me", 20, token)
        if status != 200 or me.get("fixture_sha256") != config["fixture_sha256"]:
            raise ValueError("fixture_preflight_failed")
        for item in plan:
            started = time.perf_counter()
            sent = datetime.now(timezone.utc).isoformat()
            status, payload = _post(base, "/api/agent", {"message": item["message"]}, 120, token)
            recording = record_response(payload, http_status=status, authenticated_user_id="customer_demo",
                        request_conversation_id=None, request_sent_at=sent,
                        response_received_at=datetime.now(timezone.utc).isoformat())
            # Separate strict allowlist of actual policy fields. Missing state
            # is unknown; expected outcomes never fill in an observation.
            assessment = payload.get("policy_assessment") if isinstance(payload, dict) else None
            detail = payload.get("logistics_decision") or {}
            allowed = ("outcome", "supported_claims", "missing_facts", "scope", "catalog_digest", "catalog_revision", "verification_kind")
            projected = {k: assessment[k] for k in allowed if k in assessment} if isinstance(assessment, dict) else None
            policy_observation = {"protocol_version": "policy-observation-v1", "source": "actual_http_policy_allowlist",
                                  "projection": projected, "sha256": canonical_sha256(projected) if projected else None}
            cid = payload.get("conversation_id")
            if type(cid) is int and cid > 0:
                report["fresh_conversation_ids"].append(cid)
            citations = payload.get("citations") or []
            checks = {"http_ok": status == 200, "task_projection_complete": valid_recording(recording),
                "policy_projection_available": projected is not None,
                "outcome": assessment.get("outcome") == item["outcome"] if assessment else None,
                "scope_order": (assessment.get("scope") or {}).get("active_order_id") == item["order"] if assessment else None,
                "retrieved_exact_document": bool(citations) and all(c.get("id") == 41003 for c in citations),
                "body_bound": bool(citations) and all((c.get("policy_source") or {}).get("body_sha256") == setup["body_sha256"] for c in citations),
                "version_bound": bool(citations) and all((c.get("policy_source") or {}).get("version") == "v1" for c in citations),
                "compound_completion": detail.get("policy_subtask_complete") is (item["outcome"] == "applicable"),
                "no_write_receipt": not any(e.get("tool") in {"create_ticket", "create_refund"} and e.get("ok")
                                          for e in (payload.get("execution_summary") or {}).get("tool_events", []))}
            claims = assessment.get("supported_claims") if assessment else None
            checks["structured_claims"] = (bool(claims) and any(c.get("claim_key") == "shipping_stall_hours" and c.get("value") == 48 for c in claims)) if item["outcome"] == "applicable" else claims == []
            report["cases"].append({"id": item["id"], "latency_ms": round((time.perf_counter()-started)*1000, 2),
                "checks": checks, "response_recording": recording, "policy_observation": policy_observation,
                "citations": citations, "passed": all(v is True for v in checks.values())})
    except Exception as exc:
        report["error_class"] = type(exc).__name__
    finally:
        if token:
            try:
                _post(base, "/api/auth/logout", {}, 10, token)
            except Exception:
                pass
    passed = sum(row["passed"] for row in report["cases"])
    report["summary"] = {"requested": len(plan), "observed": len(report["cases"]), "passed": passed,
                         "failed_or_unknown": len(plan)-passed}
    write_json_new(args.output, report)
    with args.output.with_suffix(".md").open("x", encoding="utf-8") as stream:
        stream.write("# 政策链路实际 HTTP 验证\n\n" + json.dumps(report["summary"], ensure_ascii=False) + "\n\n")
        for row in report["cases"]:
            stream.write(f"- {row['id']}: {row['checks']}\n")
        stream.write("\n只验证两条预设链路与一份合成文档，不是检索总体准确率。\n")
    print(json.dumps(report["summary"]))
    return 0 if passed == len(plan) else 1


if __name__ == "__main__":
    raise SystemExit(main())
