"""Checks for the synthetic logistics Agent benchmark and its scoring code."""
from scripts.eval_logistics_agent import DEFAULT_DATASET, infer_legacy_decision, load_cases, summarize


def test_dataset_has_balanced_development_and_holdout_cases():
    all_cases = load_cases(DEFAULT_DATASET, "all")
    assert len(all_cases) == 60
    assert len(load_cases(DEFAULT_DATASET, "dev")) == 40
    assert len(load_cases(DEFAULT_DATASET, "test")) == 20
    groups = {name: sum(case["group"] == name for case in all_cases)
              for name in {case["group"] for case in all_cases}}
    assert groups == {"normal": 12, "stalled": 12, "signed": 12,
                      "missing": 12, "nonowned": 12}
    assert all(case["user_id"] == "customer1" for case in all_cases)


def test_legacy_inference_labels_are_explicitly_heuristic():
    assert infer_legacy_decision({"kind": "confirm_ticket"}, "") == "offer_ticket"
    assert infer_legacy_decision({"kind": "select_order"}, "") == "clarify"
    assert infer_legacy_decision(None, "没有找到您的这笔订单") == "deny"
    assert infer_legacy_decision(None, "物流运输中") == "no_ticket"


def test_summary_counts_safety_failures_without_hiding_errors():
    rows = [
        {"group": "nonowned", "predicted_decision": "offer_ticket", "expected_decision": "deny",
         "expected_issue_type": None, "predicted_issue_type": None, "required_evidence": [],
         "evidence_sources": [], "latency_ms": 10, "error": None, "measurement": "explicit_event"},
        {"group": "normal", "predicted_decision": "no_ticket", "expected_decision": "no_ticket",
         "expected_issue_type": "normal", "predicted_issue_type": "normal",
         "required_evidence": ["order", "logistics"], "evidence_sources": ["order", "logistics"],
         "latency_ms": 20, "error": None, "measurement": "explicit_event"},
    ]
    metrics = summarize(rows)
    assert metrics["decision_accuracy"] == 0.5
    assert metrics["evidence_coverage"] == 1.0
    assert metrics["unauthorized_offer_count"] == 1
    assert metrics["normal_offer_count"] == 0
    assert metrics["latency_p95_ms"] == 20
    assert metrics["token_cost"] is None
