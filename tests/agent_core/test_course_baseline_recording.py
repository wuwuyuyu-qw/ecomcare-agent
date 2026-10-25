from pathlib import Path

from app.evaluation.course_baseline import SafeRecorder, load_course_prompt, digest


def test_baseline_recording_never_substitutes_hidden_or_unsafe_values():
    recorder = SafeRecorder(("synthetic-private-value",))
    clean = recorder.record({"answer": "合成公开答复"}, "actual_local_model")
    assert clean["complete"] is True and clean["sha256"] == digest(clean["projection"])
    for value in ({"answer": "synthetic-private-value"}, {"answer": "Bearer synthetictoken"}, {"value": float("nan")}):
        result = recorder.record(value, "actual_local_model")
        assert result["complete"] is False and result["projection"] is None and result["sha256"] is None


def test_course_prompt_is_read_as_git_data_with_explicit_adaptation():
    root = Path(__file__).resolve().parents[2]
    prompt, source = load_course_prompt(root)
    assert "只读" in prompt and "禁止创建工单" in prompt
    assert source["course_commit"].startswith("90e4b24")
    assert source["course_prompt_sha256"] != source["effective_prompt_sha256"]
