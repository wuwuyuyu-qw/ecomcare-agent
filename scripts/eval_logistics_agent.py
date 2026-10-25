"""Replay synthetic logistics cases against a running MewHelp chat endpoint.

This reports observed behavior, not a model-graded quality claim. Use the same
dataset, model configuration and service versions for baseline/candidate runs.
The pre-extension endpoint has no explicit logistics_decision event; its action
is inferred conservatively and is marked as a heuristic in the report.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "tests" / "data" / "logistics_agent_cases.jsonl"


def load_cases(path: Path, split: str) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if split != "all":
        cases = [case for case in cases if case["split"] == split]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case id must be unique")
    return cases


def infer_legacy_decision(interrupt: dict | None, answer: str) -> str:
    kind = (interrupt or {}).get("kind") or (interrupt or {}).get("type")
    if kind == "confirm_ticket":
        return "offer_ticket"
    if kind == "select_order":
        return "clarify"
    if any(phrase in answer for phrase in ("没有找到您的这笔订单", "不是您的订单", "无权查询")):
        return "deny"
    if any(phrase in answer for phrase in ("转人工", "人工客服", "无法核实")):
        return "human_review"
    return "no_ticket"


async def run_case(client: httpx.AsyncClient, base_url: str, case: dict, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {"user_id": case["user_id"], "message": case["message"]}
    start = time.perf_counter()
    answer_parts: list[str] = []
    tool_names: list[str] = []
    decision_event: dict | None = None
    interrupt: dict | None = None
    error: str | None = None
    status_code: int | None = None
    try:
        async with client.stream("POST", f"{base_url.rstrip('/')}/api/chat", json=payload, headers=headers) as response:
            status_code = response.status_code
            if status_code != 200:
                error = f"HTTP {status_code}"
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        frame = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if frame.get("event") == "logistics_decision":
                        decision_event = frame
                    elif frame.get("event") == "interrupt":
                        interrupt = frame
                    elif frame.get("event") == "tool":
                        tool_names.append(str(frame.get("name", "")))
                    elif isinstance(frame.get("delta"), str):
                        answer_parts.append(frame["delta"])
                    elif frame.get("message") and error is None:
                        error = str(frame["message"])
    except (httpx.HTTPError, TimeoutError) as exc:
        error = type(exc).__name__
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    answer = "".join(answer_parts)
    if decision_event:
        predicted = decision_event.get("decision", "unknown")
        measurement = "explicit_event"
    elif error:
        predicted = "error"
        measurement = "transport_error"
    else:
        predicted = infer_legacy_decision(interrupt, answer)
        measurement = "legacy_heuristic"
    evidence = decision_event.get("evidence", []) if decision_event else []
    evidence_sources = sorted({e.get("source", "") for e in evidence if isinstance(e, dict)})
    trace = decision_event.get("tool_trace", []) if decision_event else []
    return {
        "id": case["id"], "split": case["split"], "group": case["group"],
        "expected_decision": case["expected_decision"], "predicted_decision": predicted,
        "expected_issue_type": case["expected_issue_type"],
        "predicted_issue_type": decision_event.get("issue_type") if decision_event else None,
        "expected_order_id": case.get("expected_order_id"),
        "predicted_order_id": decision_event.get("order_id") if decision_event else None,
        "required_evidence": case["required_evidence"], "evidence_sources": evidence_sources,
        "tool_names": tool_names, "tool_trace": trace, "measurement": measurement,
        "latency_ms": elapsed_ms, "status_code": status_code, "error": error,
        "answer": answer[:1000], "interrupt_kind": (interrupt or {}).get("kind"),
        "attack": case["attack"],
    }


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    by_group: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    for group, items in sorted(grouped.items()):
        by_group[group] = {
            "count": len(items),
            "decision_correct": sum(r["predicted_decision"] == r["expected_decision"] for r in items),
        }
    latencies = sorted(r["latency_ms"] for r in rows)
    evidence_cases = [r for r in rows if r["required_evidence"]]
    order_cases = [r for r in rows if r.get("expected_order_id")]
    traced_calls = [step for r in rows for step in r.get("tool_trace", [])]
    failed_calls = [step for step in traced_calls if step.get("status") != "成功"]
    return {
        "count": total,
        "decision_accuracy": round(sum(r["predicted_decision"] == r["expected_decision"] for r in rows) / total, 4) if total else None,
        "issue_accuracy_on_labeled": round(sum(r["predicted_issue_type"] == r["expected_issue_type"] for r in rows if r["expected_issue_type"] is not None) / max(1, sum(r["expected_issue_type"] is not None for r in rows)), 4),
        "evidence_coverage": round(sum(set(r["required_evidence"]).issubset(r["evidence_sources"]) for r in evidence_cases) / len(evidence_cases), 4) if evidence_cases else None,
        "order_id_accuracy": round(sum(r["predicted_order_id"] == r["expected_order_id"] for r in order_cases) / len(order_cases), 4) if order_cases else None,
        "unauthorized_offer_count": sum(r["group"] == "nonowned" and r["predicted_decision"] == "offer_ticket" for r in rows),
        "normal_offer_count": sum(r["group"] == "normal" and r["predicted_decision"] == "offer_ticket" for r in rows),
        "tool_call_count": len(traced_calls),
        "tool_failure_count": len(failed_calls),
        "transport_error_count": sum(r["error"] is not None for r in rows),
        "measurement_modes": dict(Counter(r["measurement"] for r in rows)),
        "latency_p95_ms": latencies[max(0, math.ceil(0.95 * len(latencies)) - 1)] if latencies else None,
        "latency_mean_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "by_group": by_group,
        "token_cost": None,  # SSE currently does not expose usage; do not fabricate a number.
    }


async def run(args: argparse.Namespace) -> dict:
    cases = load_cases(args.dataset, args.split)
    timeout = httpx.Timeout(args.timeout)
    rows = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for index, case in enumerate(cases, 1):
            row = await run_case(client, args.base_url, case, args.token)
            if args.require_explicit and row["measurement"] == "legacy_heuristic":
                row["error"] = "missing_logistics_decision_event"
                row["predicted_decision"] = "error"
                row["measurement"] = "missing_explicit_event"
            rows.append(row)
            print(f"{index:02d}/{len(cases)} {case['id']}: {row['predicted_decision']} ({row['latency_ms']} ms)")
    report = {"label": args.label, "split": args.split, "dataset": str(args.dataset),
              "summary": summarize(rows), "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("all", "dev", "test"), default="test")
    parser.add_argument("--label", required=True, help="baseline or candidate commit/run label")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token", default=os.getenv("MEWHELP_EVAL_TOKEN", ""))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--require-explicit", action="store_true",
                        help="Candidate run: treat missing logistics_decision events as errors")
    args = parser.parse_args()
    report = asyncio.run(run(args))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
