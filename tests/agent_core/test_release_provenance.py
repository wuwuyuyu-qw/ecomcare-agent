"""The public release must work without distributing the course Git history."""
import hashlib
import json
from pathlib import Path

import pytest

from app.evaluation.course_baseline import load_course_prompt


def _snapshot(root, prompt="synthetic course prompt", **changes):
    path = root / "provenance" / "course-agent-system.json"
    path.parent.mkdir(parents=True)
    value = {"course_commit": "90e4b2465b45ee99ebc9792e588108a2f3646275",
             "source_path": "app/core/prompts.py", "source_blob_sha256": "a" * 64,
             "agent_system": prompt,
             "course_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    value.update(changes)
    path.write_text(json.dumps(value), encoding="utf8")


def test_snapshot_loads_without_git(tmp_path):
    _snapshot(tmp_path)
    prompt, source = load_course_prompt(tmp_path)
    assert prompt.startswith("synthetic course prompt") and "只读" in prompt
    assert source["source_kind"] == "attributed_snapshot_without_original_git_history"


@pytest.mark.parametrize("change", [{"course_prompt_sha256": "0" * 64},
                                   {"course_commit": "f" * 40},
                                   {"source_path": "unexpected.py"}])
def test_invalid_snapshot_is_not_silently_replaced(tmp_path, change):
    _snapshot(tmp_path, **change)
    with pytest.raises(ValueError, match="provenance invalid"):
        load_course_prompt(tmp_path)
