import json

import pytest

from app.evaluation.course_baseline import SafeRecorder
from app.evaluation.task_review import export_packet, _screen


def _report():
    recorder = SafeRecorder()
    return {"mode": "course_baseline_local", "observation_protocol_version": "course-baseline-transcript-v1",
            "comparable_with_http_v3": False, "split": "dev", "dataset_sha256": "a" * 64,
            "run_config_sha256": "b" * 64, "rows": [{"id": "synthetic-case", "expected_turn_count": 1,
            "turns": [{"user_message": "查询合成订单", "final_answer": "请提供订单号。",
                       "input_recording": recorder.record("查询合成订单", "actual_local_baseline_input"),
                       "answer_recording": recorder.record("请提供订单号。", "actual_local_baseline_answer"),
                       "observed": {"facts": {}, "evidence_sources": [], "actions": []}}]}]}


def test_actual_baseline_can_be_reviewed_without_fabricating_http_state(tmp_path):
    source = tmp_path / "baseline.json"
    source.write_text(json.dumps(_report(), ensure_ascii=False), encoding="utf-8")
    packet, binding = tmp_path / "packet.json", tmp_path / "private.json"
    result = export_packet([source], packet, binding)
    assert result["reviewable_count"] == 1
    value = json.loads(packet.read_text(encoding="utf-8"))
    assert value["entries"][0]["observed"]["facts"] == {}
    assert "course_baseline_local" not in packet.read_text(encoding="utf-8")
    assert json.loads(source.read_text(encoding="utf-8"))["mode"] == "course_baseline_local"


@pytest.mark.parametrize("mutation", ["answer", "hash", "mode"])
def test_baseline_review_rejects_changed_recording_or_replay(tmp_path, mutation):
    report = _report()
    if mutation == "mode":
        report["mode"] = "replay"
    elif mutation == "answer":
        report["rows"][0]["turns"][0]["final_answer"] = "改变的答复"
    else:
        report["rows"][0]["turns"][0]["answer_recording"]["sha256"] = "c" * 64
    source = tmp_path / "baseline.json"
    source.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        export_packet([source], tmp_path / "packet.json", tmp_path / "private.json")


def test_review_protocol_label_is_not_a_key_but_real_key_is_blocked():
    _screen({"schema_version": "task-blind-review-v1"})
    with pytest.raises(ValueError):
        _screen({"answer": "key sk-synthetic123456789"})
