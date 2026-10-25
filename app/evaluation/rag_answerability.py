"""Pure answerability definitions, confusion counts and development threshold scan."""
import hashlib
import json
from pathlib import Path

VERSION = "rag-answerability-v1"


def load_bundle(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != VERSION:
        raise ValueError("unknown answerability version")
    parsed = {}
    for name in ("corpus.json", "dev.jsonl"):
        raw = (directory / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError("answerability dataset changed")
        parsed[name] = (json.loads(raw) if name.endswith("json") else
                        [json.loads(line) for line in raw.decode().splitlines() if line.strip()])
        if len(parsed[name]) != manifest["files"][name]["count"]:
            raise ValueError("dataset count changed")
    docs, cases = parsed["corpus.json"], parsed["dev.jsonl"]
    ids = {d["id"] for d in docs}
    if len(ids) != len(docs) or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("duplicate identities")
    for c in cases:
        if (type(c["answerable"]) is not bool or not c["query"].strip()
                or bool(c["support_ids"]) != c["answerable"] or not set(c["support_ids"]) <= ids):
            raise ValueError("inconsistent answerability annotation")
    return docs, cases, manifest


def confusion(rows):
    counts = {k: 0 for k in ("true_accept", "false_accept", "true_reject", "false_reject", "unknown")}
    for r in rows:
        if r["status"] != "observed":
            counts["unknown"] += 1
            continue
        expected, accepted = r["answerable"], r["accepted"]
        counts[("true_" if expected == accepted else "false_") + ("accept" if accepted else "reject")] += 1
    positive = sum(r["answerable"] for r in rows)
    negative = len(rows) - positive
    return {**counts, "requested": len(rows), "positive": positive, "negative": negative,
            "answerable_accept_rate": counts["true_accept"] / positive if positive else None,
            "unanswerable_accept_rate": counts["false_accept"] / negative if negative else None,
            "scope": "development sufficiency decision, not generated answer quality"}


def threshold_scan(rows):
    """Do not change runtime settings from these results. No held-out estimate."""
    scores = [r for r in rows if r["retrieval_status"] == "observed"]
    out = []
    for threshold in [i / 20 for i in range(21)]:
        evaluated = [{"status": "observed", "answerable": r["answerable"],
                      "accepted": r["top_score"] >= threshold} for r in scores]
        out.append({"threshold": threshold, **confusion(evaluated)})
    return {"feature": "top rerank score only; no semantic check", "observed": len(scores),
            "omitted_retrieval_failures": len(rows) - len(scores), "scan": out,
            "runtime_threshold_changed": False,
            "limit": "small balanced synthetic development corpus; no production or held-out calibration claim"}
