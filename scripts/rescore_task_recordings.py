"""Recheck saved HTTP projections offline; never rerun or repair an Agent answer.

The source report and selected frozen rubric are hash-bound. This is scorer
reproducibility, not a new execution, task sample, latency or quality result.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.task_cases import load_dataset
from app.evaluation.task_recording import valid_recording
from app.evaluation.task_review import read_json
from app.evaluation.task_scoring import observe_response, score_observation, SCORER_VERSION
from app.evaluation.task_report import write_json_new, assert_new_output


def rescore(report, cases):
    by_id = {c["id"]: c for c in cases}
    if report.get("mode") != "online":
        raise ValueError("only actual saved HTTP task reports are supported")
    rows = []
    seen = set()
    for row in report.get("rows", []):
        cid = row.get("id")
        if cid not in by_id or cid in seen:
            raise ValueError("unknown or duplicate selected case")
        seen.add(cid)
        case = by_id[cid]
        turns = row.get("turns", [])
        if not isinstance(turns, list) or len(turns) > len(case["turns"]):
            raise ValueError("invalid recorded turn sequence")
        results = []
        for i, turn in enumerate(turns):
            if turn.get("user_message") != case["turns"][i]["message"]:
                raise ValueError("recorded input differs from frozen rubric")
            recording = turn.get("response_recording")
            if not valid_recording(recording):
                # No unsafe/partial projection is promoted to a complete record.
                results.append({"turn_index": i, "status": "not_rescored",
                                "reason": "complete_valid_http_recording_unavailable"})
                continue
            status = recording["http_status"]
            observed = observe_response(recording["projection"], http_status=status, recording=recording)
            if status >= 500 or status in {400, 409, 422}:
                observed.update(decision="transport_or_service_error", termination_reason="request_failed",
                                decision_source="actual_http_error_status")
            score = score_observation(case["turns"][i]["expected"], observed)
            results.append({"turn_index": i, "status": "rescored", "score": score,
                            "matches_saved_score": score == turn.get("score")})
        rows.append({"id": cid, "original_status": row.get("status"),
                     "expected_turn_count": len(case["turns"]), "observed_turn_count": len(turns), "turns": results})
    if not rows:
        raise ValueError("report has no requested cases")
    rescored = [t for r in rows for t in r["turns"] if t["status"] == "rescored"]
    return rows, {"requested_cases": len(rows), "rescored_turns": len(rescored),
                  "unchanged_scores": sum(t["matches_saved_score"] for t in rescored),
                  "changed_scores": sum(not t["matches_saved_score"] for t in rescored),
                  "unavailable_turns": sum(t["status"] != "rescored" for r in rows for t in r["turns"]),
                  "unobserved_tail_turns": sum(r["expected_turn_count"] - r["observed_turn_count"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "holdout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_new_output(args.output)
    report, digest = read_json(args.report)
    if report.get("split") != args.split:
        raise ValueError("explicit split differs from report")
    cases, manifest = load_dataset(ROOT / "data/agent_eval", args.split)
    if report.get("dataset_sha256") != manifest["dataset_sha256"]:
        raise ValueError("frozen rubric identity differs from saved report")
    rows, summary = rescore(report, cases)
    value = {"mode": "offline_recording_rescore", "created_at": datetime.now(timezone.utc).isoformat(),
             "source_report_sha256": digest, "dataset_sha256": manifest["dataset_sha256"],
             "split": args.split, "scorer_version": SCORER_VERSION, "rows": rows, "summary": summary,
             "scope": "saved HTTP projections only; no model/network/database/environment loading",
             "limitations": ["not a new Agent run, performance sample or independent quality review",
                             "source report unchanged; missing projections/tails remain unavailable"]}
    write_json_new(args.output, value)
    print(json.dumps(summary))
    return int(summary["changed_scores"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
