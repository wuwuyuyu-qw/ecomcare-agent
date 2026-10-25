"""Small new compound dataset; read one named split, never both implicitly."""
import hashlib
import json
from pathlib import Path

VERSION = "compound-v1.1"


def load_cases(directory: Path, split: str = "dev", expected_version: str = VERSION):
    if split not in {"dev", "holdout"}:
        raise ValueError("explicit compound split required")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if expected_version not in {VERSION, "compound-routing-v1.0"} or manifest.get("version") != expected_version:
        raise ValueError("compound manifest version differs")
    raw = (directory / f"{split}.jsonl").read_bytes()
    metadata = manifest["files"][f"{split}.jsonl"]
    if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
        raise ValueError("compound dataset digest differs")
    cases = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(cases) != metadata["count"] or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("compound case count/identity differs")
    for case in cases:
        if not isinstance(case["turns"], list) or not 1 <= len(case["turns"]) <= 4:
            raise ValueError("bounded compound turns required")
        for turn in case["turns"]:
            if not isinstance(turn["message"], str) or not turn["message"].strip():
                raise ValueError("actual compound input required")
            if not set(turn["required_tools"]) <= {"query_order", "query_logistics", "query_faq", "query_warranty", "query_return_status"}:
                raise ValueError("unknown required compound tool")
    return cases, manifest
