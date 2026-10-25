"""Compare two recorded logistics Agent evaluations on identical case IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "decision_accuracy", "issue_accuracy_on_labeled", "evidence_coverage", "order_id_accuracy",
    "unauthorized_offer_count", "normal_offer_count", "transport_error_count",
    "tool_call_count", "tool_failure_count", "latency_p95_ms", "latency_mean_ms", "token_cost",
)


def compare(baseline: dict, candidate: dict) -> dict:
    old = {row["id"]: row for row in baseline["cases"]}
    new = {row["id"]: row for row in candidate["cases"]}
    if old.keys() != new.keys():
        raise ValueError("baseline and candidate must use the same case IDs")
    if any(old[key]["expected_decision"] != new[key]["expected_decision"] for key in old):
        raise ValueError("gold labels differ between runs")
    before, after = baseline["summary"], candidate["summary"]
    metrics = {}
    for name in METRICS:
        a, b = before.get(name), after.get(name)
        metrics[name] = {"baseline": a, "candidate": b,
                         "delta": round(b - a, 4) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None}
    changed = [
        {"id": key, "expected": old[key]["expected_decision"],
         "baseline": old[key]["predicted_decision"], "candidate": new[key]["predicted_decision"]}
        for key in sorted(old)
        if old[key]["predicted_decision"] != new[key]["predicted_decision"]
    ]
    return {"baseline": baseline["label"], "candidate": candidate["label"],
            "count": len(old), "metrics": metrics, "changed_cases": changed,
            "measurement_warning": (
                "Baseline contains legacy heuristic labels; review raw answers before claiming a causal improvement."
                if before.get("measurement_modes", {}).get("legacy_heuristic", 0) else None)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    old = json.loads(args.baseline.read_text(encoding="utf-8"))
    new = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = compare(old, new)
    result = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
