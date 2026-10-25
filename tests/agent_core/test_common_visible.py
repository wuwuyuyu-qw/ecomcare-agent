from copy import deepcopy
import json

import pytest

from app.evaluation.common_visible import normalize_report, export_common_reports, validate_common_report
from app.evaluation.common_judge import validate_rating, compare_ratings, protocol_digest, PROTOCOL, rate_with_format_retry
from app.evaluation.course_baseline import SafeRecorder
from app.evaluation.task_recording import record_response, canonical_sha256
from app.evaluation.task_review import export_packet, read_json, _report, CRITERIA

AT = "2026-10-05T00:00:01Z"
ORDER = {"order_id": "3003", "status": "已发货", "amount": 100, "product": "合成商品",
         "created_at": "2026-10-01 12:00", "tracking_no": "SYNTH3003"}


def _sources():
    common = {"split": "dev", "dataset_sha256": "a" * 64, "run_config_sha256": "b" * 64,
              "code_version": "c" * 40}
    scope = {"user_id": "synthetic", "conversation_id": 1, "task_id": "task", "active_order_id": "3003"}
    values = {"order_status" if k == "status" else k: v for k, v in ORDER.items()}
    values["order_owned"] = True
    facts = {k: {"value": v, "source": "query_order", "scope": scope, "observed_at": AT,
                  "evidence_id": "source-" + k} for k, v in values.items()}
    record = record_response({"conversation_id": 1, "task_context": {**scope, "verified_facts": facts,
        "evidence": [], "actions": [], "pending_actions": [], "decision": "no_ticket"},
        "suggested_actions": [], "tool_calls": [], "tool_results": [], "execution_summary": {"tool_events": []}},
        http_status=200, authenticated_user_id="synthetic", request_conversation_id=None,
        request_sent_at="2026-10-05T00:00:00Z", response_received_at="2026-10-05T00:00:02Z")
    visible = {"user_message": "查订单3003", "final_answer": "订单3003金额100元。"}
    candidate = {**common, "mode": "online", "implementation_label": "candidate", "rows": [{
        "id": "synthetic-case", "status": "evaluated", "expected_turn_count": 1, "turns": [{**visible,
        "input_recording": {"complete": True}, "answer_recording": {"complete": True},
        "response_recording": record, "observed": {"facts": {"invented": 999}}, "score": {"success": True}}]}]}
    recorder = SafeRecorder()
    tool = {"tool": "query_order", "arguments": {"order_id": "3003"}, "ok": True,
            "business_status": "ok", "data": ORDER, "observed_at": AT, "user_id": "synthetic", "conversation_id": 1}
    p = {"answer": visible["final_answer"], "tool_results": [tool], "user_id": "synthetic", "conversation_id": 1,
         "request_sent_at": "2026-10-05T00:00:00Z", "response_received_at": "2026-10-05T00:00:02Z"}
    baseline = {**common, "mode": "course_baseline_local", "implementation_label": "baseline",
        "observation_protocol_version": "course-baseline-transcript-v1", "comparable_with_http_v3": False,
        "rows": [{"id": "synthetic-case", "status": "evaluated", "expected_turn_count": 1, "turns": [{**visible,
            "input_recording": recorder.record(visible["user_message"], "actual_local_baseline_input"),
            "answer_recording": recorder.record(visible["final_answer"], "actual_local_baseline_answer"),
            "baseline_recording": recorder.record(p, "actual_local_model_and_shared_executor")}]}]}
    return baseline, candidate


def _write_sources(tmp_path):
    paths = [tmp_path / "baseline.json", tmp_path / "candidate.json"]
    for p, report in zip(paths, _sources()):
        p.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return paths


def _packet(tmp_path):
    sources = export_common_reports(_write_sources(tmp_path), tmp_path / "common")
    packet, private = tmp_path / "packet.json", tmp_path / "private.json"
    export_packet(sources, packet, private)
    p, digest = read_json(packet)
    b, _ = read_json(private)
    return p, b, digest


def _rating():
    return {"criteria": {k: {"outcome": "not_applicable" if k == "policy_conditions" else "pass",
            "rationale": "合成测试评分，不是实际人工或模型成绩。", "evidence_refs": ["e0001"] if k == "factual_support" else []}
            for k in CRITERIA}}


def test_common_evidence_comes_from_actual_records_not_machine_score():
    reports = _sources()
    adapted = [normalize_report(v, {"report_path": "synthetic.json", "report_sha256": "d" * 64}) for v in reports]
    facts = [v["rows"][0]["turns"][0]["observed"]["facts"] for v in adapted]
    assert facts[0]["evidence_records"][0]["values"] == facts[1]["evidence_records"][0]["values"]
    assert "invented" not in json.dumps(facts)
    assert "decision" not in json.dumps(facts)
    assert facts[0]["backend_side_effects"] == "not_independently_observed"


def test_old_evidence_keeps_actual_history_and_expiry():
    baseline, _ = _sources()
    turn = deepcopy(baseline["rows"][0]["turns"][0])
    p = turn["baseline_recording"]["projection"]
    p.update(tool_results=[], request_sent_at="2026-10-05T00:20:00Z", response_received_at="2026-10-05T00:20:01Z")
    turn["baseline_recording"]["sha256"] = canonical_sha256(p)
    baseline["rows"][0].update(expected_turn_count=2, turns=[baseline["rows"][0]["turns"][0], turn])
    result = normalize_report(baseline, {})["rows"][0]["turns"][1]
    facts = result["observed"]["facts"]
    assert len(facts["conversation_history"]) == 2
    assert facts["evidence_records"][0]["within_observation_ttl"] is False


@pytest.mark.parametrize("change", ["identity", "object", "contract", "time"])
def test_baseline_invalid_result_is_not_admitted(change):
    baseline, _ = _sources()
    r = baseline["rows"][0]["turns"][0]["baseline_recording"]
    t = r["projection"]["tool_results"][0]
    if change == "identity": t["user_id"] = "another"
    elif change == "object": t["data"]["order_id"] = "3001"
    elif change == "contract": t["data"]["amount"] = "100"
    else: t["observed_at"] = "2026-10-05T00:30:00Z"
    r["sha256"] = canonical_sha256(r["projection"])
    result = normalize_report(baseline, {})["rows"][0]["turns"][0]
    assert result["observed"]["facts"]["evidence_records"] == []
    assert result["observed"]["facts"]["observation_limits"]


def test_tampered_derived_report_and_changed_original_are_rejected(tmp_path):
    paths = _write_sources(tmp_path)
    common = export_common_reports(paths, tmp_path / "common")
    report, _ = read_json(common[0])
    validate_common_report(report)
    report["rows"][0]["turns"][0]["final_answer"] = "已退款。"
    with pytest.raises(ValueError): validate_common_report(report)
    original = json.loads(paths[0].read_text(encoding="utf-8"))
    original["extra"] = "changed"
    paths[0].write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError): _report(common[0])


def test_derived_origin_cycle_is_rejected_without_recursion(tmp_path):
    p = tmp_path / "cycle.json"
    p.write_text(json.dumps({"mode": "common_visible_review", "origin": {
        "report_path": str(p), "report_sha256": "a" * 64}}), encoding="utf-8")
    with pytest.raises(ValueError): _report(p)


@pytest.mark.parametrize("change", ["input", "config", "dataset", "cases"])
def test_unpaired_sources_fail_before_writing_outputs(tmp_path, change):
    paths = _write_sources(tmp_path)
    report = json.loads(paths[1].read_text(encoding="utf-8"))
    if change == "input": report["rows"][0]["turns"][0]["user_message"] = "另一订单"
    elif change == "cases": report["rows"][0]["id"] = "another-case"
    else: report[change + "_sha256" if change == "dataset" else "run_config_sha256"] = "f" * 64
    paths[1].write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError): export_common_reports(paths, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", ["missing_refs", "unknown_ref", "duplicate_ref", "expired_ref", "decision_na"])
def test_judge_cannot_pass_without_fresh_actual_evidence(tmp_path, change):
    packet, _, _ = _packet(tmp_path)
    entry, rating = packet["entries"][0], _rating()
    if change == "missing_refs": rating["criteria"]["factual_support"]["evidence_refs"] = []
    elif change == "unknown_ref": rating["criteria"]["factual_support"]["evidence_refs"] = ["made-up"]
    elif change == "duplicate_ref": rating["criteria"]["factual_support"]["evidence_refs"] *= 2
    elif change == "expired_ref": entry["observed"]["facts"]["evidence_records"][0]["within_observation_ttl"] = False
    else: rating["criteria"]["decision_quality"]["outcome"] = "not_applicable"
    with pytest.raises(ValueError): validate_rating(rating, entry)


def test_model_comparison_is_separate_from_task_and_human_scores(tmp_path):
    packet, private, digest = _packet(tmp_path)
    ratings = {"schema_version": PROTOCOL, "packet_sha256": digest, "protocol_sha256": protocol_digest(),
               "entries": [{"review_id": e["review_id"], "status": "rated", "rating": _rating()} for e in packet["entries"]]}
    result = compare_ratings(packet, private, digest, ratings)
    assert result["paired_complete_cases"] == 1 and result["human_ratings"] == 0
    assert "task_success_rate" not in result
    ratings["entries"][0]["rating"]["criteria"]["policy_conditions"]["outcome"] = "unknown"
    assert compare_ratings(packet, private, digest, ratings)["paired_complete_cases"] == 0
    ratings["entries"].pop()
    with pytest.raises(ValueError): compare_ratings(packet, private, digest, ratings)


@pytest.mark.asyncio
async def test_format_retry_recovers_structure_and_records_both_attempts(tmp_path):
    packet, _, _ = _packet(tmp_path)
    calls = []

    async def evaluator(**kwargs):
        calls.append(kwargs)
        return None if len(calls) == 1 else _rating()

    result = await rate_with_format_retry(evaluator, packet["entries"][0], 2)
    assert result["status"] == "rated" and len(result["attempts"]) == len(calls) == 2
    assert calls[0] == calls[1]


@pytest.mark.parametrize("outcome", ["fail", "unknown"])
@pytest.mark.asyncio
async def test_valid_negative_or_unknown_assessment_is_never_retried(tmp_path, outcome):
    packet, _, _ = _packet(tmp_path)
    calls = []

    async def evaluator(**kwargs):
        calls.append(kwargs)
        rating = _rating()
        rating["criteria"]["decision_quality"]["outcome"] = outcome
        return rating

    result = await rate_with_format_retry(evaluator, packet["entries"][0], 2)
    assert len(calls) == 1 and result["rating"]["criteria"]["decision_quality"]["outcome"] == outcome


@pytest.mark.asyncio
async def test_provider_error_stops_without_format_retry(tmp_path):
    packet, _, _ = _packet(tmp_path)
    calls = []

    async def evaluator(**kwargs):
        calls.append(kwargs)
        raise ConnectionError("synthetic provider failure")

    result = await rate_with_format_retry(evaluator, packet["entries"][0], 2)
    assert result["status"] == "judge_error" and len(calls) == 1


@pytest.mark.asyncio
async def test_format_retry_remains_bounded_and_does_not_invent_scores(tmp_path):
    packet, _, _ = _packet(tmp_path)

    async def evaluator(**kwargs):
        return None

    result = await rate_with_format_retry(evaluator, packet["entries"][0], 2)
    assert result["status"] == "judge_error" and result["rating"] is None and len(result["attempts"]) == 2
