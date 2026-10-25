import importlib.util
from pathlib import Path

import pytest

from app.evaluation.task_recording import record_response
from app.evaluation.task_scoring import observe_response, score_observation

spec = importlib.util.spec_from_file_location("synthetic_rescore", Path(__file__).resolve().parents[2] / "scripts/rescore_task_recordings.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _inputs():
    expected = {"decisions": ["clarify"], "termination_reasons": ["clarification_required"],
                "required_facts": [], "required_evidence_sources": [], "forbidden_actions": []}
    case = {"id": "synthetic", "turns": [{"message": "查询物流", "expected": expected}]}
    record = record_response({"conversation_id": 1, "task_context": {"schema_version": 1,
        "user_id": "synthetic", "conversation_id": 1, "task_id": "t", "decision": "clarify",
        "termination_reason": "clarification_required", "verified_facts": {}, "evidence": [],
        "actions": [], "pending_actions": []}, "suggested_actions": [], "tool_calls": [], "tool_results": [],
        "execution_summary": {"termination_reason": "clarification_required", "tool_events": []}},
        http_status=200, authenticated_user_id="synthetic", request_conversation_id=None,
        request_sent_at="2026-10-05T00:00:00Z", response_received_at="2026-10-05T00:00:01Z")
    score = score_observation(expected, observe_response(record["projection"], recording=record))
    report = {"mode": "online", "rows": [{"id": "synthetic", "status": "evaluated", "turns": [
        {"user_message": "查询物流", "response_recording": record, "score": score}]}]}
    return case, report


def test_rescore_saved_projection_without_using_old_score_as_evidence():
    case, report = _inputs()
    rows, summary = module.rescore(report, [case])
    assert summary["unchanged_scores"] == 1
    report["rows"][0]["turns"][0]["score"] = {}
    rows, summary = module.rescore(report, [case])
    assert summary["changed_scores"] == 1 and rows[0]["turns"][0]["score"]


def test_tampered_projection_is_unavailable_and_not_repaired():
    case, report = _inputs()
    report["rows"][0]["turns"][0]["response_recording"]["projection"]["conversation_id"] = 99
    _, summary = module.rescore(report, [case])
    assert summary["rescored_turns"] == 0 and summary["unavailable_turns"] == 1


def test_changed_frozen_input_is_rejected():
    case, report = _inputs()
    report["rows"][0]["turns"][0]["user_message"] = "另一轮"
    with pytest.raises(ValueError):
        module.rescore(report, [case])
