"""Reuse the existing OpenEvals judge on immutable component answer records.

No regeneration, gate retries or label changes. Bind source hashes before calls,
retain all failed recordings as unreviewable, and never show arm/gold labels to
the judge. This retrospective component packet is not an HTTP task evaluation.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_rag_development import encode, write_new
from scripts.judge_common_answers import judge_packet
from app.evaluation.common_judge import protocol_digest, _success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--frozen-plan", type=Path)
    args = parser.parse_args()
    args.source, args.env_file, args.output_dir = args.source.resolve(), args.env_file.resolve(), args.output_dir.resolve()
    args.judge_model = None
    if not 0 < args.timeout <= 90:
        raise ValueError("bounded review timeout required")
    names = ("answers.jsonl", "ratings.jsonl", "retrieval.jsonl", "run-config.json", "report.json", "frozen-plan.json", "source-snapshot.zip", "source-stability.json")
    source_hashes = {n: hashlib.sha256((args.source / n).read_bytes()).hexdigest() for n in names}
    source_rows = [json.loads(line) for line in (args.source / "ratings.jsonl").read_text(encoding="utf-8").splitlines()]
    answers = [json.loads(line) for line in (args.source / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    key = lambda r: (r["id"], r["arm"], r["repetition"])
    if len(source_rows) != len(answers) or len({key(r) for r in source_rows}) != len(source_rows) or len(source_rows) > 120:
        raise ValueError("complete unique bounded answer cohort required")
    by_key = {key(r): r for r in answers}
    entries, bindings = [], {}
    for index, r in enumerate(source_rows):
        answer = by_key[key(r)]
        if answer["query"] != r["entry"]["user_message"] or answer["answer"] != r["entry"]["answer"]:
            raise ValueError("recorded input/answer differs")
        review_id = f"review-{index+1:04d}"
        entries.append({**r["entry"], "review_id": review_id,
            "recording_status": "reviewable" if answer["status"] == "observed" else "not_reviewable"})
        bindings[review_id] = {"id": r["id"], "arm": r["arm"], "repetition": r["repetition"]}
    packet = {"protocol": "isolated-rag-visible-packet-v1", "entries": entries,
              "limits": "actual component retrieval/answers; no HTTP or real order tool observations"}
    code_names = ("scripts/review_rag_answers.py", "scripts/judge_common_answers.py", "app/evaluation/common_judge.py",
                  "app/evaluation/task_review.py", "pyproject.toml", "uv.lock")
    code_hashes = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in code_names}
    plan = {"source_hashes": source_hashes, "review_source_hashes": code_hashes, "protocol_sha256": protocol_digest(),
        "packet_sha256": hashlib.sha256(encode(packet)).hexdigest(), "entries": len(entries), "timeout": args.timeout,
        "scope": "retrospective same-family AI review, no human ratings, no regeneration"}
    if args.frozen_plan:
        if plan != json.loads(args.frozen_plan.read_text(encoding="utf-8")):
            raise ValueError("review binding changed")
    elif not args.plan_only:
        raise ValueError("freeze review before provider calls")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_new(args.output_dir / "frozen-plan.json", plan)
    write_new(args.output_dir / "packet.json", packet)
    write_new(args.output_dir / "bindings.json", bindings)
    with zipfile.ZipFile(args.output_dir / "review-source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for n in code_names:
            archive.write(ROOT / n, n)
    if args.plan_only:
        print(json.dumps({"event": "review_planned_no_env_or_network", "entries": len(entries)}))
        return
    ratings = asyncio.run(judge_packet(packet, plan["packet_sha256"], args))
    write_new(args.output_dir / "ratings.json", ratings)
    if any(hashlib.sha256((args.source / n).read_bytes()).hexdigest() != h for n, h in source_hashes.items()):
        raise ValueError("answer source changed during review")
    if any(hashlib.sha256((ROOT / n).read_bytes()).hexdigest() != h for n, h in code_hashes.items()):
        raise ValueError("judge source changed during review")
    summary = {}
    for arm in ("legacy", "selected"):
        rows = [r for r in ratings["entries"] if bindings[r["review_id"]]["arm"] == arm]
        decisions = [_success([r["rating"]]) for r in rows]
        summary[arm] = {"requested": len(rows), "pass": sum(x is True for x in decisions),
            "fail": sum(x is False for x in decisions), "unknown": sum(x is None for x in decisions)}
    write_new(args.output_dir / "comparison.json", {"summary": summary, "source_hashes_unchanged": True,
        "review_source_unchanged": True, "limits": "same-family AI rubric; unknowns retained; no independent human/held-out proof"})
    print(json.dumps({"event": "reviewed", "summary": summary}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit("RAG review failed: " + type(exc).__name__) from None
