"""Actual HTTP multi-turn task memory and visible-stream diagnostic.

Predeclared synthetic messages; observations are actual response projections.
No holdout or annotation files are read. Credentials remain in process memory.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.task_recording import record_response, valid_recording, canonical_sha256
from app.evaluation.task_report import assert_new_output, write_json_new
from eval_task_agent import _post, _get
from probe_agent_boundaries import validate_url, load_config, strict_json, SECRET_PATTERN, BODY_LIMIT


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_new_output(args.output, markdown=True)
    base = validate_url(args.base_url)
    config, digest, _ = load_config(args.config)
    password = os.environ.get("MEWHELP_PROBE_OWNER_PASSWORD")
    if not password:
        raise ValueError("named password environment required")
    # This plan is fixed before login or Agent execution, independent of outputs.
    sequence = [dict(id="initial_stall", message="查询订单3001的物流状态", order="3001", decision="offer_ticket")]
    for i in range(10):
        sequence.append(dict(id=f"long_followup_{i+1}",
            message="请继续核对这笔订单的物流状态。" + "我只需要你说明工具查询到的状态和更新时间，请保持解释清晰，不要执行任何处理动作。" * 3,
            order="3001", decision="offer_ticket"))
    sequence += [dict(id="correct_order", message="订单号说错了，改成3003，查询物流状态", order="3003", decision="no_ticket"),
                 dict(id="phone_not_order", message="我的手机号改成13800003301，我仍然只查询这笔订单的物流状态", order="3003", decision="no_ticket"),
                 dict(id="cancel_current", message="取消这个请求", order="", decision="cancelled"),
                 dict(id="after_cancel_missing_order", message="帮我查一下包裹的物流状态", order="", decision="clarify"),
                 dict(id="new_task", message="查询订单3002物流，显示已签收但我没收到包裹", order="3002", decision="offer_ticket"),
                 dict(id="switch_topic", message="换个话题，猫爬架有哪些商品规格？", order="", decision=None),
                 dict(id="after_topic_missing_order", message="帮我查询包裹物流状态", order="", decision="clarify")]
    report = {"report_version": 1, "mode": "online_memory_stream_probe", "code_version": args.code_version,
              "config_sha256": digest, "planned_at": now(), "plan": sequence, "cases": [], "stream_cases": [],
              "fresh_conversation_ids": [], "summary": {},
              "limitations": ["one serialized conversation under a forced small window; not memory accuracy on a dataset",
                 "window trimming/summary corroborated separately, restart recovery not implied by HTTP turn reuse",
                 "SSE visible-content wait is not provider TTFT; foreground cost remains unknown"]}
    token, cid = None, None
    try:
        status, login = _post(base, "/api/auth/login", {"username": "customer_demo", "password": password}, 20)
        token = login.get("access_token")
        if status != 200 or not isinstance(token, str) or not token:
            raise ValueError("identity_preflight_failed")
        status, me = _get(base, "/api/auth/me", 20, token)
        if status != 200 or me.get("user_id") != "customer_demo" or me.get("fixture_sha256") != config["fixture_sha256"]:
            raise ValueError("world_preflight_failed")
        for item in sequence:
            request = {"message": item["message"]}
            if cid is not None:
                request["conversation_id"] = cid
            sent, start = now(), time.perf_counter()
            status, payload = _post(base, "/api/agent", request, 120, token)
            recording = record_response(payload, http_status=status, authenticated_user_id="customer_demo",
                        request_conversation_id=cid, request_sent_at=sent, response_received_at=now())
            row = {"id": item["id"], "http_status": status, "latency_ms": round((time.perf_counter()-start)*1000, 2),
                   "response_recording": recording, "checks": {}, "passed": False}
            if status == 200 and valid_recording(recording):
                p = recording["projection"]
                if cid is None:
                    cid = p["conversation_id"]
                    report["fresh_conversation_ids"].append(cid)
                ctx = p.get("task_context") or {}
                checks = row["checks"]
                checks["scope"] = ctx.get("user_id") == "customer_demo" and ctx.get("conversation_id") == cid
                checks["active_order"] = ctx.get("active_order_id") == item["order"]
                checks["decision"] = ctx.get("decision") == item["decision"] if item["decision"] is not None else True
                facts = ctx.get("verified_facts") or {}
                checks["no_other_order_facts"] = all(f.get("scope", {}).get("active_order_id") == item["order"]
                                                   for f in facts.values() if isinstance(f, dict))
                if not item["order"]:
                    checks["invalidated_facts"] = not facts
                if item["id"] == "cancel_current":
                    checks["zero_cancel_model_calls"] = p.get("execution_summary", {}).get("model_calls") == 0
                    checks["no_cancel_actions"] = p.get("suggested_actions") == []
                row["passed"] = all(checks.values())
            report["cases"].append(row)

        # First-turn non-logistics coverage and timings. Keep raw HTTP SSE
        # timestamps separate from the older task recording protocol.
        stream_messages = ["你好", "猫粮有哪些商品规格？", "退货政策是什么？", "查询订单3003的物流状态"]
        with httpx.Client(base_url=base, trust_env=False, follow_redirects=False, timeout=120) as client:
            for message in stream_messages:
                start, frames, size, first_event, first_content = time.perf_counter(), [], 0, None, None
                row = {"message": message, "http_status": None, "complete": False, "passed": False}
                try:
                    with client.stream("POST", "/api/chat", json={"message": message},
                                       headers={"Authorization": "Bearer " + token}) as response:
                        row["http_status"] = response.status_code
                        for line in response.iter_lines():
                            size += len(line.encode("utf-8"))
                            if size > BODY_LIMIT:
                                raise ValueError("stream_recording_too_large")
                            if not line.startswith("data: "):
                                continue
                            raw = line[6:]
                            if raw == "[DONE]":
                                row["complete"] = True
                                continue
                            if SECRET_PATTERN.search(raw) or token in raw or password in raw:
                                raise ValueError("stream_recording_unsafe")
                            event = strict_json(raw)
                            if not isinstance(event, dict):
                                raise ValueError("invalid_stream_event")
                            elapsed = round((time.perf_counter()-start)*1000, 2)
                            if first_event is None:
                                first_event = elapsed
                            if event.get("delta") and first_content is None:
                                first_content = elapsed
                            if event.get("event") == "done" and type(event.get("conversation_id")) is int:
                                report["fresh_conversation_ids"].append(event["conversation_id"])
                            # All content is synthetic; record an allowlist of
                            # observed public event fields, never headers/login.
                            frames.append({k: event[k] for k in ("event", "delta", "items", "name", "ok", "conversation_id",
                                "termination_reason", "elapsed_ms", "first_stream_event_ms", "first_visible_content_ms",
                                "model_calls", "tool_calls", "usage_coverage", "cost", "cost_complete") if k in event})
                    row.update(first_event_ms=first_event, first_visible_content_ms=first_content,
                               elapsed_ms=round((time.perf_counter()-start)*1000, 2), frames=frames,
                               recording_sha256=canonical_sha256({"frames": frames}))
                    row["passed"] = (row["http_status"] == 200 and row["complete"] and first_content is not None
                                     and any(f.get("event") == "done" for f in frames))
                except Exception as exc:
                    row["unavailable_reason"] = type(exc).__name__
                report["stream_cases"].append(row)
    except Exception as exc:
        report["unavailable_reason"] = type(exc).__name__
    finally:
        if token:
            try:
                _post(base, "/api/auth/logout", {}, 10, token)
            except Exception:
                pass
    total = len(sequence) + 4
    passed = sum(r["passed"] for r in report["cases"]) + sum(r["passed"] for r in report["stream_cases"])
    report["summary"] = {"requested": total, "passed": passed, "failed_or_unobserved": total-passed}
    report["completed_at"] = now()
    write_json_new(args.output, report)
    with args.output.with_suffix(".md").open("x", encoding="utf-8") as stream:
        stream.write("# 多轮任务记忆与可见内容等待实验\n\n" + json.dumps(report["summary"], ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"]))
    return int(passed != total)


if __name__ == "__main__":
    raise SystemExit(main())
