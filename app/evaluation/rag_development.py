"""Pure development retrieval metrics and oracle-scoped policy diagnostics.

Labels never enter retrieval. Facts and target group are explicit diagnostic
oracles, so these checks cannot be described as end-to-end Agent accuracy.
"""
import hashlib
import json
import math
from pathlib import Path

from app.core.policy_evidence import catalog_from_dict, evaluate_citations

VERSION = "rag-development-v1"
STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")


def load_bundle(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != VERSION:
        raise ValueError("unknown retrieval development version")
    values = {}
    for name in ("corpus.json", "catalog.json", "dev.jsonl"):
        raw = (directory / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError("retrieval development file changed")
        values[name] = ([json.loads(line) for line in raw.decode().splitlines() if line.strip()]
                        if name.endswith("jsonl") else json.loads(raw))
    docs, cases = values["corpus.json"], values["dev.jsonl"]
    if (len(docs) != manifest["files"]["corpus.json"]["count"]
            or len(cases) != manifest["files"]["dev.jsonl"]["count"]
            or len({d["id"] for d in docs}) != len(docs) or len({c["id"] for c in cases}) != len(cases)):
        raise ValueError("retrieval identities/counts differ")
    catalog = catalog_from_dict(values["catalog.json"], digest=manifest["files"]["catalog.json"]["sha256"])
    bindings = {e.document_id: e.body_sha256 for e in catalog.documents}
    ids = {str(d["id"]) for d in docs}
    if set(bindings) != ids or any(bindings[str(d["id"])] != hashlib.sha256(d["answer"].encode()).hexdigest() for d in docs):
        raise ValueError("catalog and corpus body bindings differ")
    for case in cases:
        if not case["query"].strip() or not set(case["relevance"]) <= ids:
            raise ValueError("invalid query relevance")
        if not {str(i) for i in case["expected_applicable_ids"]} <= ids:
            raise ValueError("unknown applicable document")
        if any(type(v) is not int or not 1 <= v <= 3 for v in case["relevance"].values()):
            raise ValueError("invalid relevance grade")
    return docs, cases, catalog, manifest


def ranking_metrics(hits: list[dict], relevance: dict, k: int = 3) -> dict:
    if type(k) is not int or k < 1:
        raise ValueError("positive retrieval rank required")
    ids = [str(h["id"]) for h in hits[:k]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate retrieved document")
    if not relevance:
        return {"recall_at_k": None, "mrr_at_k": None, "ndcg_at_k": None,
                "out_of_scope_nonempty": bool(ids), "k": k}
    grades = [relevance.get(i, 0) for i in ids]
    dcg = sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(ideal))
    first = next((i + 1 for i, grade in enumerate(grades) if grade), None)
    return {"recall_at_k": len(set(ids) & set(relevance)) / len(relevance),
            "mrr_at_k": 1 / first if first else 0.0, "ndcg_at_k": dcg / idcg,
            "out_of_scope_nonempty": None, "k": k}


def policy_probe(case: dict, hits: list[dict], catalog) -> dict:
    group = case["policy_group"]
    if group is None:
        return {"scope": "oracle_topic_scope_diagnostic", "outcome": "not_evaluated",
                "outcome_matches_expected": None, "applicable_target_recall": None,
                "target_group_observed": None, "supported_claims": []}
    target_ids = {e.document_id for e in catalog.documents if e.policy_group == group}
    target = [h for h in hits if str(h["id"]) in target_ids]
    args = {"order_facts": case["synthetic_verified_facts"], "now": case["reference_time"], "require_latest": True}
    raw = evaluate_citations(target, catalog, **args)
    eligible = [h for h in target if evaluate_citations([h], catalog, **args)["outcome"] == "applicable"]
    if eligible:
        selected = evaluate_citations(eligible, catalog, **args)
    else:
        selected = {**raw, "supported_claims": []}
    expected_ids = {str(i) for i in case["expected_applicable_ids"]}
    available = {str(h["id"]) for h in eligible}
    return {"scope": "oracle_topic_scope_diagnostic", "outcome": selected["outcome"],
            "outcome_matches_expected": bool(target) and selected["outcome"] == case["policy_probe_expected"],
            "raw_target_citation_outcome": raw["outcome"],
            "target_group_observed": bool(target), "eligible_ids": sorted(available),
            "applicable_target_recall": len(available & expected_ids) / len(expected_ids) if expected_ids else None,
            "unsupported_release_on_no_answer_case": bool(selected["supported_claims"]) if not expected_ids else None,
            "supported_claims": selected["supported_claims"], "assessment": selected,
            "limit": "supplied topic/facts are diagnostic oracles; no generated answer evaluated"}
