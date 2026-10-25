"""Versioned task annotations and split integrity checks, using only stdlib."""
import hashlib
import json
from pathlib import Path
import re

VERSION = "task-v1.1"
DECISIONS = {"no_ticket", "offer_ticket", "clarify", "deny", "human_review", "cancelled", "answer_policy"}
TERMINATIONS = {"completed", "awaiting_confirmation", "clarification_required", "permission_denied",
                "human_review", "tool_evidence_unavailable", "cancelled", "tool_budget_exhausted",
                "deadline_exceeded", "repeated_tool_call", "order_evidence_required"}
TERMINATIONS |= {"tool_budget_exceeded", "repeated_call_limit", "step_budget_exceeded", "context_budget_exceeded",
                 "execution_failed", "forbidden", "superseded", "client_cancelled", "draft_ready"}


def load_dataset(directory: Path, split: str = "all") -> tuple[list[dict], dict]:
    if split not in {"all", "dev", "holdout"}:
        raise ValueError("unsupported dataset split")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != VERSION:
        raise ValueError("unsupported dataset manifest version")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"dev.jsonl", "holdout.jsonl"}:
        raise ValueError("frozen manifest must declare both dev and holdout files")
    for name, count in (("dev", 80), ("holdout", 40)):
        entry = files[name + ".jsonl"]
        if (not isinstance(entry, dict) or entry.get("count") != count
                or not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
            raise ValueError(f"invalid frozen manifest metadata: {name}")
    frozen_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if manifest.get("dataset_sha256") != frozen_hash:
        raise ValueError("dataset manifest digest mismatch")
    selected = ("dev", "holdout") if split == "all" else (split,)
    cases = []
    for name in selected:
        path = directory / f"{name}.jsonl"
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if manifest["files"][path.name]["sha256"] != digest:
            raise ValueError(f"frozen dataset hash mismatch: {path.name}")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        if len(rows) != manifest["files"][path.name]["count"]:
            raise ValueError(f"manifest count mismatch: {path.name}")
        if any(row.get("split") != name for row in rows):
            raise ValueError(f"case split differs from its frozen file: {path.name}")
        cases.extend(rows)
    expected_counts = {name: files[name + ".jsonl"]["count"] if name in selected else 0 for name in ("dev", "holdout")}
    inventory = validate_cases(cases, expected_splits=expected_counts)
    declared_profiles = manifest.get("online_by_profile") or {}
    for profile in ("demo", "task-v1"):
        for name in selected:
            if (declared_profiles.get(profile) or {}).get(name) != inventory["online_by_profile"][profile][name]:
                raise ValueError("manifest profile eligibility counts differ from parsed frozen cases")
    # Selecting dev must not expose holdout labels during diagnosis/tuning.
    # Its frozen digest contributes to dataset identity, but is not a statement
    # that the unparsed file's bytes or cross-split isolation were rechecked.
    scope = {"parsed_splits": list(selected), "file_hash_checked_splits": list(selected),
             "unparsed_splits": [name for name in ("dev", "holdout") if name not in selected],
             "cross_split_isolation_checked": split == "all",
             "unparsed_digest_source": "frozen_manifest_only; file bytes not read or rehashed" if split != "all" else None}
    return cases, {**manifest, "validation_scope": scope}


def validate_cases(cases: list[dict], *, expected_splits: dict[str, int] | None = None) -> dict:
    ids, group_splits, fixture_splits, entity_splits = set(), {}, {}, {}
    families = {}
    split_counts = {"dev": 0, "holdout": 0}
    for case in cases:
        for field in ("id", "annotation_version", "split", "scenario_group", "fixture_group", "family",
                      "description", "turns", "fixture_requirements", "online_supported", "online_profiles"):
            if field not in case:
                raise ValueError(f"{case.get('id', '?')}: missing {field}")
        if case["id"] in ids:
            raise ValueError(f"duplicate case id: {case['id']}")
        ids.add(case["id"])
        split = case["split"]
        if split not in split_counts or case["annotation_version"] != VERSION:
            raise ValueError(f"{case['id']}: unsupported split/version")
        split_counts[split] += 1
        families[case["family"]] = families.get(case["family"], 0) + 1
        for name, mapping in (("scenario_group", group_splits), ("fixture_group", fixture_splits)):
            group = case[name]
            if not isinstance(group, str) or not group:
                raise ValueError(f"{case['id']}: invalid {name}")
            if group in mapping and mapping[group] != split:
                raise ValueError(f"cross-split leakage: {name}={group}")
            mapping[group] = split
        if not isinstance(case["online_supported"], bool) or not isinstance(case["fixture_requirements"], dict):
            raise ValueError(f"{case['id']}: invalid fixture declaration")
        profiles = case["online_profiles"]
        if (not isinstance(profiles, list) or not set(profiles) <= {"demo", "task-v1"}
                or len(profiles) != len(set(profiles)) or case["online_supported"] != bool(profiles)):
            raise ValueError(f"{case['id']}: inconsistent online profile declaration")
        fixture = case["fixture_requirements"]
        entities = [fixture, fixture.get("replacement_order") or {}, fixture.get("policy") or {}]
        for entity in entities:
            for key in ("order_id", "tracking_no", "version"):
                value = entity.get(key)
                if isinstance(value, str) and value:
                    identity = key + ":" + value
                    if identity in entity_splits and entity_splits[identity] != split:
                        raise ValueError(f"cross-split entity leakage: {identity}")
                    entity_splits[identity] = split
        if not case["turns"] or not isinstance(case["turns"], list):
            raise ValueError(f"{case['id']}: empty turns")
        for turn in case["turns"]:
            if not isinstance(turn.get("message"), str) or not turn["message"].strip():
                raise ValueError(f"{case['id']}: empty message")
            expected = turn.get("expected") or {}
            required = {"decisions", "required_facts", "forbidden_actions", "termination_reasons", "required_evidence_sources"}
            if not required <= expected.keys():
                raise ValueError(f"{case['id']}: incomplete rubric")
            if not expected["decisions"] or not set(expected["decisions"]) <= DECISIONS:
                raise ValueError(f"{case['id']}: invalid decision label")
            if not expected["termination_reasons"] or not set(expected["termination_reasons"]) <= TERMINATIONS:
                raise ValueError(f"{case['id']}: invalid termination label")
            for field in ("required_facts", "forbidden_actions", "required_evidence_sources"):
                if not isinstance(expected[field], list):
                    raise ValueError(f"{case['id']}: {field} must be a list")
            for fact in expected["required_facts"]:
                if set(fact) != {"name", "equals"} or not isinstance(fact["name"], str):
                    raise ValueError(f"{case['id']}: malformed fact expectation")
    expected_splits = expected_splits if expected_splits is not None else {"dev": 80, "holdout": 40}
    if split_counts != expected_splits:
        raise ValueError(f"expected frozen split counts {expected_splits}, found {split_counts}")
    by_profile = {p: {split: sum(p in c["online_profiles"] and c["split"] == split for c in cases)
                      for split in ("dev", "holdout")} for p in ("demo", "task-v1")}
    return {"count": len(cases), "splits": split_counts, "families": families,
            "scenario_groups": len(group_splits), "fixture_groups": len(fixture_splits),
            "online_supported": sum(c["online_supported"] for c in cases), "online_by_profile": by_profile}
