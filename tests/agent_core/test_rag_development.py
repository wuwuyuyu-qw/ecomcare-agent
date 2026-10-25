from pathlib import Path
import pytest

from app.evaluation.rag_development import load_bundle, ranking_metrics, policy_probe


def bundle():
    return load_bundle(Path(__file__).resolve().parents[2] / "data/rag_development")


def test_corpus_catalog_and_cases_have_exact_identity_body_bindings():
    docs, cases, catalog, manifest = bundle()
    assert len(docs) == len(catalog.documents) == 16 and len(cases) == 12
    assert manifest["limits"] and manifest["dataset_sha256"]


def test_ranking_metrics_account_for_order_partial_recall_and_no_relevance():
    rank = ranking_metrics([{"id": 2}, {"id": 1}], {"1": 3, "2": 1})
    assert rank["recall_at_k"] == rank["mrr_at_k"] == 1
    assert 0 < rank["ndcg_at_k"] < 1
    assert ranking_metrics([{"id": 2}], {"1": 3})["recall_at_k"] == 0
    assert ranking_metrics([{"id": 2}], {})["recall_at_k"] is None
    assert ranking_metrics([{"id": 2}], {})["out_of_scope_nonempty"] is True
    with pytest.raises(ValueError):
        ranking_metrics([{"id": 1}, {"id": 1}], {"1": 3})


def test_expired_future_draft_and_wrong_region_cannot_release_current_policy():
    docs, cases, catalog, _ = bundle()
    hits = [d for d in docs if d["id"] in {42001, 42003, 42004, 42005}]
    assessment = policy_probe(cases[0], hits, catalog)
    assert not assessment["supported_claims"] and assessment["applicable_target_recall"] == 0
    hits.append(next(d for d in docs if d["id"] == 42002))
    assessment = policy_probe(cases[0], hits, catalog)
    assert assessment["outcome"] == "applicable" and assessment["applicable_target_recall"] == 1
    assert assessment["raw_target_citation_outcome"] != "applicable"
    assert {c["value"] for c in assessment["supported_claims"]} == {48}


@pytest.mark.parametrize("case_index,ids,outcome", [
    (8, {42012}, "unknown"), (9, {42002, 42003}, "unknown"),
    (10, {42013}, "conflict"), (10, {42013, 42014}, "conflict"),
])
def test_missing_facts_and_catalog_conflicts_remain_unresolved(case_index, ids, outcome):
    docs, cases, catalog, _ = bundle()
    result = policy_probe(cases[case_index], [d for d in docs if d["id"] in ids], catalog)
    assert result["outcome"] == outcome and result["outcome_matches_expected"] is True
    assert not result["supported_claims"] and result["unsupported_release_on_no_answer_case"] is False


def test_missing_target_retrieval_cannot_pass_policy_diagnostic_by_abstaining():
    docs, cases, catalog, _ = bundle()
    result = policy_probe(cases[8], [docs[0]], catalog)
    assert not result["target_group_observed"] and not result["outcome_matches_expected"]


def test_failed_rows_remain_in_summary_denominator():
    from scripts.eval_rag_development import metrics_summary
    docs, cases, catalog, _ = bundle()
    row = {"strategy": "vector", "status": "failed", "elapsed_seconds": 10,
           "ranking": ranking_metrics([], cases[0]["relevance"]), "policy": policy_probe(cases[0], [], catalog)}
    summary = metrics_summary([row])["vector"]
    assert summary["requested"] == summary["errors"] == 1 and summary["observed"] == 0
    assert summary["recall_at_k"] == summary["applicable_target_recall"] == 0
