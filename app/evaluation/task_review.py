"""Offline, separately labelled human review of recorded visible answers.

This module never changes machine observations, scores, costs or source reports.
It uses byte hashes to detect changed local inputs, not to authenticate a reviewer.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import secrets

SCHEMA = "task-blind-review-v1"
REVIEW_SCHEMA = "task-human-reviews-v1"
RUBRIC_VERSION = "visible-answer-v1"
CRITERIA = ("decision_quality", "factual_support", "policy_conditions", "unsupported_action_claim")
OUTCOMES = {"pass", "fail", "unknown", "not_applicable"}
MAX_BYTES = 20_000_000
MAX_ENTRIES = 5_000
MAX_TEXT = 20_000
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SECRET = re.compile(
    r"(?i)(?:bearer\s+\S+|(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{8,}|"
    r"[\"']?(?:password|passwd|api[_ -]?key|access[_ -]?token|authorization|secret)[\"']?\s*[:=]|"
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})"
)
_SECRET_KEYS = {"password", "passwd", "api_key", "access_token", "authorization", "secret", "client_secret", "refresh_token"}
_HIDDEN_OBSERVATION_KEYS = {"implementation_label", "code_version", "decision", "termination_reason", "score", "checks", "expected", "failures", "unscored", "all_checks_observed"}
RUBRIC = {
    "decision_quality": "只看用户请求和录制答复：处理方向、追问或拒答是否合理？始终适用；后台成功不能由答复推出。",
    "factual_support": "答复里的订单、物流等事实是否受到本包实际观察支持？无事实断言可注明不适用；观察不足用 unknown。",
    "policy_conditions": "涉及政策时，是否说明条件、限制和不确定性？无政策断言可注明不适用；来源名不能证明适用性。",
    "unsupported_action_claim": "是否无依据宣称已建单、退款、转人工或完成后台动作？始终适用；答复未这样宣称可评 pass。",
}
LIMITATIONS = [
    "随机条目 ID、打散顺序和移除系统标签只减少标签暗示；答复风格或内容仍可能暴露版本。",
    "人工口径仅评价录制的用户可见答复；不证明后台写入、权限校验或机器任务成功。",
    "facts/evidence_sources/actions 是原报告实际观察，不是独立核实的业务真相；来源名不等于政策原文。",
    "未知评分不进入主成功率分母；缺录制、少轮次、未审和不确定评分均保留覆盖率统计。",
    "哈希只绑定本地文件内容；不认证审阅者身份，不防止同时改写所有输入的操作者。",
    "课程参考基线来自本地真实 ReAct 调用，候选来自 HTTP；这里只对录制答复作人工审阅，不比较两者的机器任务成功率。参考基线缺结构化事实时，事实支持应评 unknown。",
]


def _local_path(path: Path) -> Path:
    raw = str(path)
    if raw.startswith(("\\\\", "//")) or "://" in raw or path.name.lower() == ".env":
        raise ValueError("only local JSON paths are accepted; network paths and .env are forbidden")
    resolved = path.resolve()
    if str(resolved).startswith(("\\\\", "//")):
        raise ValueError("network paths are forbidden")
    return resolved


def _pairs(pairs: list[tuple]) -> dict:
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def _reject_constant(_value: str):
    raise ValueError("non-finite JSON number")


def _screen(value, depth: int = 0) -> None:
    if depth > 30:
        raise ValueError("JSON nesting is too deep")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            if key.lower().replace("-", "_").replace(" ", "_") in _SECRET_KEYS:
                raise ValueError("credential-like field is forbidden")
            _screen(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _screen(item, depth + 1)
    elif isinstance(value, str) and _SECRET.search(value):
        raise ValueError("credential-like text is forbidden; use synthetic reviewed records")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite number")


def read_json(path: Path) -> tuple[dict, str]:
    """Strict JSON and raw-byte digest; no imports of app settings or .env."""
    path = _local_path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("JSON exceeds the 20 MB limit")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("JSON exceeds the 20 MB limit")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    _screen(value)
    return value, hashlib.sha256(raw).hexdigest()


def _encode(value: dict) -> bytes:
    _screen(value)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > MAX_BYTES:
        raise ValueError("output exceeds the 20 MB limit")
    return raw


def _write_new(path: Path, raw: bytes) -> None:
    path = _local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects both source files and prior review artifacts.
    with path.open("xb") as stream:
        stream.write(raw)


def _string(value, label: str, *, maximum: int = MAX_TEXT) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be a nonempty bounded string")
    return value


def _digest(value, label: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _report(path: Path) -> tuple[dict, str]:
    report, digest = read_json(path)
    baseline = report.get("mode") == "course_baseline_local"
    common = report.get("mode") == "common_visible_review"
    if report.get("mode") != "online" and not baseline and not common:
        raise ValueError("human answer review accepts actual online/local baseline reports, not replay diagnostics")
    if common:
        from app.evaluation.common_visible import validate_common_report
        validate_common_report(report)
    if baseline and (report.get("observation_protocol_version") != "course-baseline-transcript-v1"
                     or report.get("comparable_with_http_v3") is not False):
        raise ValueError("baseline transcript protocol and non-comparability marker required")
    for key in ("dataset_sha256", "run_config_sha256"):
        _digest(report.get(key), key)
    _string(report.get("split"), "split", maximum=64)
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows or len(rows) > 1_000:
        raise ValueError("report requires 1 to 1000 case rows")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("case row must be an object")
        case_id = _string(row.get("id"), "case ID", maximum=128)
        if case_id in seen:
            raise ValueError("duplicate case ID")
        seen.add(case_id)
        turns = row.get("turns", [])
        if not isinstance(turns, list) or len(turns) > 100 or any(not isinstance(t, dict) for t in turns):
            raise ValueError("case turns must contain at most 100 objects")
        expected = row.get("expected_turn_count")
        if expected is not None and (type(expected) is not int or not 1 <= expected <= 100 or len(turns) > expected):
            raise ValueError("invalid expected turn count")
        if baseline:
            for turn in turns:
                for field, text_field, source in (
                    ("input_recording", "user_message", "actual_local_baseline_input"),
                    ("answer_recording", "final_answer", "actual_local_baseline_answer"),
                ):
                    recording = turn.get(field)
                    if not isinstance(recording, dict) or recording.get("source") != source:
                        raise ValueError("baseline visible recording source invalid")
                    if recording.get("complete") is True:
                        raw = json.dumps(recording.get("projection"), ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":"), allow_nan=False).encode("utf-8")
                        if (recording.get("projection") != turn.get(text_field)
                                or recording.get("sha256") != hashlib.sha256(raw).hexdigest()):
                            raise ValueError("baseline visible recording hash/content differs")
                    elif (recording.get("complete") is not False or recording.get("projection") is not None
                          or turn.get(text_field) is not None):
                        raise ValueError("incomplete baseline recording must withhold whole text")
    return report, digest


def _turn_payload(turn: dict | None) -> dict:
    """Allowlist only actual visible text and safe observed facts, never scores."""
    turn = turn or {}
    message = turn.get("user_message")
    answer = turn.get("final_answer")
    recording = turn.get("answer_recording")
    reason = None
    if not isinstance(message, str) or not message.strip() or len(message) > MAX_TEXT:
        message, reason = None, "complete_user_input_unavailable"
    if not isinstance(answer, str) or not isinstance(recording, dict) or recording.get("complete") is not True or len(answer) > MAX_TEXT:
        answer = None
        reason = reason or "complete_final_answer_unavailable"
    if turn.get("input_recording") is not None and (
        not isinstance(turn["input_recording"], dict) or turn["input_recording"].get("complete") is not True
    ):
        message, reason = None, "complete_user_input_unavailable"
    observed = turn.get("observed") or {}
    if not isinstance(observed, dict):
        raise ValueError("observed turn must be an object")
    facts = observed.get("facts", {})
    sources = observed.get("evidence_sources", [])
    actions = observed.get("actions", [])
    if not isinstance(facts, dict) or len(facts) > 100:
        raise ValueError("observed facts must be a bounded object")
    _fact_shape(facts)
    if any(not isinstance(v, list) or len(v) > 100 or any(not isinstance(s, str) or len(s) > 128 for s in v) for v in (sources, actions)):
        raise ValueError("observed sources/actions must be bounded string lists")
    return {"recording_status": "not_reviewable" if reason else "reviewable", "skip_reason": reason,
            "user_message": message, "answer": answer,
            "observed": {"facts": facts, "evidence_sources": sources, "actions": actions}}


def _fact_shape(value, depth: int = 0) -> None:
    if depth > 10:
        raise ValueError("observed facts are too deeply nested")
    if isinstance(value, dict):
        for key, child in value.items():
            if len(key) > 128 or key in _HIDDEN_OBSERVATION_KEYS:
                raise ValueError("observed facts contain hidden machine metadata or invalid keys")
            _fact_shape(child, depth + 1)
    elif isinstance(value, list):
        if len(value) > 1_000:
            raise ValueError("observed fact list is too long")
        for child in value:
            _fact_shape(child, depth + 1)
    elif isinstance(value, str) and len(value) > MAX_TEXT:
        raise ValueError("observed fact text exceeds the recording limit")


def _row_count(row: dict) -> int:
    return row.get("expected_turn_count") or max(1, len(row.get("turns", [])))


def export_packet(reports: list[Path], packet_path: Path, bindings_path: Path) -> dict:
    """Export shuffled per-turn entries and a separate operator-only binding."""
    if not 1 <= len(reports) <= 8:
        raise ValueError("supply 1 to 8 actual report files")
    inputs = {_local_path(p) for p in reports}
    destinations = {_local_path(packet_path), _local_path(bindings_path)}
    if len(inputs) != len(reports) or len(destinations) != 2 or inputs & destinations:
        raise ValueError("input and output paths must be distinct")
    if any(p.exists() for p in destinations):
        raise ValueError("output exists; choose new packet/binding paths")
    packet_id = secrets.token_hex(16)
    entries, bindings, sources = [], [], []
    seen_digests = set()
    for index, path in enumerate(reports):
        report, digest = _report(path)
        if digest in seen_digests:
            raise ValueError("duplicate report bytes cannot represent independent versions")
        seen_digests.add(digest)
        source_id = f"source_{index + 1}"
        sources.append({"source_id": source_id, "report_path": str(_local_path(path)), "report_sha256": digest,
                        "implementation_label": report.get("implementation_label"), "code_version": report.get("code_version"),
                        "dataset_sha256": report["dataset_sha256"], "run_config_sha256": report["run_config_sha256"],
                        "split": report["split"], "case_ids": [r["id"] for r in report["rows"]]})
        for row in report["rows"]:
            turns = row.get("turns", [])
            complete = type(row.get("expected_turn_count")) is int and len(turns) == row["expected_turn_count"]
            for turn_index in range(_row_count(row)):
                review_id = secrets.token_hex(16)
                payload = _turn_payload(turns[turn_index] if turn_index < len(turns) else None)
                entries.append({"review_id": review_id, **payload})
                bindings.append({"review_id": review_id, "source_id": source_id, "case_id": row["id"],
                                 "turn_index": turn_index, "case_recording_complete": complete})
                if len(entries) > MAX_ENTRIES:
                    raise ValueError("too many review entries")
    secrets.SystemRandom().shuffle(entries)
    packet = {"schema_version": SCHEMA, "packet_id": packet_id,
              "created_at": datetime.now(timezone.utc).isoformat(), "rubric_version": RUBRIC_VERSION,
              "rubric": RUBRIC, "entries": entries, "limitations": LIMITATIONS,
              "review_file_template": {"schema_version": REVIEW_SCHEMA, "packet_id": packet_id,
                  "reviews": [{"review_id": "copy-an-entry-review-id", "reviewer_id": "reviewer-pseudonym",
                               "rubric_version": RUBRIC_VERSION, "reviewed_at": "ISO8601-time-with-timezone",
                               "rationale": "Explain overall judgement without inferring backend success.",
                               "criteria": {key: {"outcome": "unknown", "rationale": "Explain this criterion."} for key in CRITERIA}}]}}
    raw_packet = _encode(packet)
    private = {"schema_version": SCHEMA, "packet_id": packet_id,
               "packet_sha256": hashlib.sha256(raw_packet).hexdigest(), "sources": sources, "bindings": bindings}
    raw_private = _encode(private)
    _write_new(packet_path, raw_packet)
    _write_new(bindings_path, raw_private)
    return {"entry_count": len(entries), "reviewable_count": sum(e["recording_status"] == "reviewable" for e in entries),
            "not_reviewable_count": sum(e["recording_status"] != "reviewable" for e in entries)}


def _review(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {"review_id", "reviewer_id", "rubric_version", "reviewed_at", "rationale", "criteria"}:
        raise ValueError("review fields differ from the review template")
    for key, maximum in (("review_id", 128), ("reviewer_id", 128), ("rationale", 4_000)):
        _string(value[key], key, maximum=maximum)
    if value["rubric_version"] != RUBRIC_VERSION:
        raise ValueError("review rubric version differs")
    try:
        reviewed_at = datetime.fromisoformat(value["reviewed_at"])
        if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
            raise ValueError("missing timezone")
    except (TypeError, ValueError) as exc:
        raise ValueError("reviewed_at requires a valid ISO8601 timestamp with timezone") from exc
    criteria = value["criteria"]
    if not isinstance(criteria, dict) or set(criteria) != set(CRITERIA):
        raise ValueError("every named criterion is required")
    for key, item in criteria.items():
        if not isinstance(item, dict) or set(item) != {"outcome", "rationale"} or not isinstance(item["outcome"], str) or item["outcome"] not in OUTCOMES:
            raise ValueError("criterion requires one allowed outcome and a rationale")
        _string(item["rationale"], "criterion rationale", maximum=4_000)
        if key in {"decision_quality", "unsupported_action_claim"} and item["outcome"] == "not_applicable":
            raise ValueError("decision_quality and unsupported_action_claim are always applicable")
    return value


def _load_bound_sources(packet: dict, private: dict, packet_digest: str) -> tuple[dict, dict, dict]:
    if set(packet) != {"schema_version", "packet_id", "created_at", "rubric_version", "rubric", "entries", "limitations", "review_file_template"}:
        raise ValueError("packet fields differ from export schema")
    if set(private) != {"schema_version", "packet_id", "packet_sha256", "sources", "bindings"}:
        raise ValueError("private binding fields differ from export schema")
    if packet.get("schema_version") != SCHEMA or private.get("schema_version") != SCHEMA:
        raise ValueError("unknown packet/binding schema")
    if packet.get("packet_id") != private.get("packet_id") or private.get("packet_sha256") != packet_digest:
        raise ValueError("packet ID or byte hash differs from private binding")
    if packet.get("rubric_version") != RUBRIC_VERSION or packet.get("rubric") != RUBRIC:
        raise ValueError("packet rubric differs")
    entries = packet.get("entries")
    bindings = private.get("bindings")
    sources = private.get("sources")
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_ENTRIES or not isinstance(bindings, list) or not isinstance(sources, list) or not 1 <= len(sources) <= 8:
        raise ValueError("invalid packet/binding collections")
    entry_map, binding_map, source_map = {}, {}, {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("packet entry must be an object")
        key = _string(entry.get("review_id"), "review ID", maximum=128)
        if key in entry_map:
            raise ValueError("duplicate review ID")
        entry_map[key] = entry
    expected_pairs = set()
    source_digests = set()
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"source_id", "report_path", "report_sha256", "implementation_label", "code_version", "dataset_sha256", "run_config_sha256", "split", "case_ids"}:
            raise ValueError("binding source fields differ from export schema")
        key = _string(source.get("source_id"), "source ID", maximum=64)
        if key in source_map:
            raise ValueError("duplicate source ID")
        report, digest = _report(Path(_string(source.get("report_path"), "source report path", maximum=4_096)))
        if source.get("report_sha256") != digest:
            raise ValueError("original source report changed; re-export a new packet")
        if digest in source_digests:
            raise ValueError("duplicate source report bytes")
        source_digests.add(digest)
        for field in ("dataset_sha256", "run_config_sha256", "split", "implementation_label", "code_version"):
            if source.get(field) != report.get(field):
                raise ValueError("private binding metadata differs from original report")
        rows = {row["id"]: row for row in report["rows"]}
        if source.get("case_ids") != list(rows):
            raise ValueError("bound requested case IDs differ from source report")
        source_map[key] = {"binding": source, "rows": rows}
        expected_pairs.update((key, row["id"], i) for row in rows.values() for i in range(_row_count(row)))
    actual_pairs = set()
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {"review_id", "source_id", "case_id", "turn_index", "case_recording_complete"}:
            raise ValueError("binding entry fields differ from export schema")
        key = _string(binding.get("review_id"), "bound review ID", maximum=128)
        pair = (_string(binding.get("source_id"), "bound source ID", maximum=64),
                _string(binding.get("case_id"), "bound case ID", maximum=128), binding.get("turn_index"))
        if key not in entry_map or key in binding_map or type(pair[2]) is not int or pair not in expected_pairs or pair in actual_pairs:
            raise ValueError("missing, duplicate or invalid turn binding")
        actual_pairs.add(pair)
        row = source_map[pair[0]]["rows"][pair[1]]
        turns = row.get("turns", [])
        complete = type(row.get("expected_turn_count")) is int and len(turns) == row["expected_turn_count"]
        if binding.get("case_recording_complete") is not complete:
            raise ValueError("bound case completion differs from source report")
        payload = _turn_payload(turns[pair[2]] if pair[2] < len(turns) else None)
        if entry_map[key] != {"review_id": key, **payload}:
            raise ValueError("blind entry differs from frozen source turn")
        binding_map[key] = binding
    if actual_pairs != expected_pairs or set(binding_map) != set(entry_map):
        raise ValueError("binding does not cover every requested source case/turn")
    return entry_map, binding_map, source_map


def _case_result(entries: list[dict]) -> dict:
    reasons = set()
    for entry in entries:
        if not entry["case_recording_complete"]:
            reasons.add("source_turn_sequence_incomplete_or_unknown")
        if entry["recording_status"] != "reviewable":
            reasons.add("complete_recording_unavailable")
        elif entry["human_review"] is None:
            reasons.add("not_reviewed")
        elif any(item["outcome"] == "unknown" for item in entry["human_review"]["criteria"].values()):
            reasons.add("human_rating_unknown")
    if reasons:
        return {"eligible": False, "visible_answer_success": None, "skip_reasons": sorted(reasons)}
    passed = all(item["outcome"] in {"pass", "not_applicable"}
                 for entry in entries for item in entry["human_review"]["criteria"].values())
    return {"eligible": True, "visible_answer_success": passed, "skip_reasons": []}


def import_reviews(packet_path: Path, bindings_path: Path, reviews_path: Path, output_path: Path) -> dict:
    """Produce a new derived human report without filling machine unknowns."""
    packet, packet_digest = read_json(packet_path)
    private, binding_digest = read_json(bindings_path)
    reviews, review_digest = read_json(reviews_path)
    entry_map, binding_map, source_map = _load_bound_sources(packet, private, packet_digest)
    inputs = {_local_path(p) for p in (packet_path, bindings_path, reviews_path)}
    inputs.update(_local_path(Path(v["binding"]["report_path"])) for v in source_map.values())
    if _local_path(output_path) in inputs:
        raise ValueError("derived report must have a distinct output path")
    if set(reviews) != {"schema_version", "packet_id", "reviews"} or reviews.get("schema_version") != REVIEW_SCHEMA or reviews.get("packet_id") != packet["packet_id"]:
        raise ValueError("review file must match this packet and review schema")
    rows = reviews.get("reviews")
    if not isinstance(rows, list) or len(rows) > len(entry_map):
        raise ValueError("invalid review list")
    ratings = {}
    for row in rows:
        row = _review(row)
        key = row["review_id"]
        if key not in entry_map or key in ratings:
            raise ValueError("unknown or duplicate reviewed entry")
        if entry_map[key]["recording_status"] != "reviewable":
            raise ValueError("incomplete recording cannot receive a human answer judgement")
        ratings[key] = row
    derived_entries, cases = [], {}
    for key, entry in entry_map.items():
        binding = binding_map[key]
        derived = {**binding, "recording_status": entry["recording_status"], "recording_skip_reason": entry["skip_reason"],
                   "human_review": ratings.get(key)}
        derived_entries.append(derived)
        cases.setdefault((binding["source_id"], binding["case_id"]), []).append(derived)
    case_rows = [{"source_id": source_id, "case_id": case_id, **_case_result(entries)}
                 for (source_id, case_id), entries in sorted(cases.items())]
    summaries = []
    for source_id, source in source_map.items():
        selected = [r for r in case_rows if r["source_id"] == source_id]
        eligible = [r for r in selected if r["eligible"]]
        summaries.append({**source["binding"], "requested_case_count": len(selected), "fully_reviewed_case_count": len(eligible),
                          "human_visible_answer_success_rate": sum(r["visible_answer_success"] for r in eligible) / len(eligible) if eligible else None,
                          "case_review_coverage": len(eligible) / len(selected) if selected else None,
                          "skip_reasons": dict(Counter(reason for r in selected for reason in r["skip_reasons"]))})
    comparison = _compare(source_map, case_rows)
    report = {"schema_version": "task-human-review-report-v1", "measurement": "independent_human_visible_answer_review",
              "created_at": datetime.now(timezone.utc).isoformat(), "rubric_version": RUBRIC_VERSION,
              "packet_sha256": packet_digest, "private_binding_sha256": binding_digest, "review_payload_sha256": review_digest,
              "sources": summaries, "entries": derived_entries, "cases": case_rows, "comparison": comparison,
              "coverage": {"requested_turn_count": len(entry_map), "reviewable_turn_count": sum(e["recording_status"] == "reviewable" for e in entry_map.values()),
                           "not_reviewable_turn_count": sum(e["recording_status"] != "reviewable" for e in entry_map.values()),
                           "submitted_review_count": len(ratings), "unreviewed_reviewable_turn_count": sum(e["recording_status"] == "reviewable" and k not in ratings for k, e in entry_map.items()),
                           "turn_denominator_description": "declared expected_turn_count when available; otherwise max(1, observed turns), not a complete-sequence claim"},
              "limitations": LIMITATIONS + ["人工派生报告不改写原报告的 decision、termination、score、timing、facts 或 cost。"]}
    _write_new(output_path, _encode(report))
    return {"coverage": report["coverage"], "comparison": comparison}


def _compare(sources: dict, cases: list[dict]) -> dict:
    if len(sources) != 2:
        return {"status": "not_compared", "reason": "exactly_two_sources_required"}
    left_id, right_id = list(sources)
    left, right = sources[left_id]["binding"], sources[right_id]["binding"]
    for field in ("dataset_sha256", "run_config_sha256", "split"):
        if left[field] != right[field]:
            return {"status": "not_compared", "reason": f"{field}_differs"}
    left_cases = {r["case_id"]: r for r in cases if r["source_id"] == left_id}
    right_cases = {r["case_id"]: r for r in cases if r["source_id"] == right_id}
    common = set(left_cases) & set(right_cases)
    eligible_pairs = [key for key in sorted(common) if left_cases[key]["eligible"] and right_cases[key]["eligible"]]
    # Row completion was verified against each immutable report. A different
    # expected sequence length is not the same case comparison.
    pairs, pair_skips = [], []
    for key in sorted(common):
        reasons = sorted(set(left_cases[key]["skip_reasons"] + right_cases[key]["skip_reasons"]))
        if key in eligible_pairs:
            left_row, right_row = sources[left_id]["rows"][key], sources[right_id]["rows"][key]
            if left_row["expected_turn_count"] != right_row["expected_turn_count"]:
                reasons.append("expected_turn_count_differs")
            elif [turn["user_message"] for turn in left_row["turns"]] != [turn["user_message"] for turn in right_row["turns"]]:
                reasons.append("recorded_user_inputs_differ")
        if reasons:
            pair_skips.append({"case_id": key, "reasons": reasons})
        else:
            pairs.append(key)
    before = sum(left_cases[k]["visible_answer_success"] for k in pairs) / len(pairs) if pairs else None
    after = sum(right_cases[k]["visible_answer_success"] for k in pairs) / len(pairs) if pairs else None
    return {"status": "paired_human_visible_answer_comparison", "left_source_id": left_id, "right_source_id": right_id,
            "common_requested_case_count": len(common), "paired_case_count": len(pairs), "skipped_common_case_count": len(common) - len(pairs),
            "unmatched_case_count": len(set(left_cases) ^ set(right_cases)), "paired_coverage": len(pairs) / len(common) if common else None,
            "unmatched_case_ids": sorted(set(left_cases) ^ set(right_cases)), "skipped_common_cases": pair_skips,
            "left_visible_answer_success_rate": before, "right_visible_answer_success_rate": after,
            "absolute_change": after - before if before is not None and after is not None else None,
            "improved_case_ids": [k for k in pairs if not left_cases[k]["visible_answer_success"] and right_cases[k]["visible_answer_success"]],
            "regressed_case_ids": [k for k in pairs if left_cases[k]["visible_answer_success"] and not right_cases[k]["visible_answer_success"]],
            "scope": "all turns complete and every applicable human criterion explicitly rated; not machine task success"}
