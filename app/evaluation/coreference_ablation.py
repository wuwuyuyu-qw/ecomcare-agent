"""Read-only, single-factor ABBA analysis of actual local development runs.

Only execution.skip_coref_without_history may differ. This is an ablation of
the current candidate, not a comparison with the purchased course baseline.
No provider calls, environment loading, or holdout annotation reads occur.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import zipfile

from app.evaluation.task_recording import (
    OBSERVATION_PROTOCOL_VERSION, canonical_sha256, utc_datetime, valid_recording,
)
from app.evaluation.task_report import assert_new_output, write_json_new
from app.evaluation.task_scoring import (
    CHECK_NAMES, SCORER_VERSION, observe_response, percentile, score_observation, summarize,
)

FACTOR = "skip_coref_without_history"
FACTOR_PATH = "execution." + FACTOR
ANALYSIS_VERSION = "coreference-abba-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DEV_PATH = "data/agent_eval/dev.jsonl"
_MANIFEST_PATH = "data/agent_eval/manifest.json"
_JSON_LIMIT = 20_000_000


def _digest(value, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name}: expected lowercase SHA256")
    return value


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name}: expected integer >= {minimum}")
    return value


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}: expected finite nonnegative number")
    return float(value)


def _json(raw: str, name: str) -> dict:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"{name}: duplicate JSON key")
            out[key] = value
        return out

    def reject(_value):
        raise ValueError(f"{name}: nonfinite JSON value")

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected JSON object")
    return value


def _read(path: Path) -> dict:
    if path.suffix.lower() != ".json" or path.stat().st_size > _JSON_LIMIT:
        raise ValueError("analysis inputs must be JSON documents smaller than 20 MB")
    return _json(path.read_text(encoding="utf-8"), path.name)


def _same(actual, expected, name: str) -> None:
    # bool is an int subclass: canonical JSON distinguishes true from 1.
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError(f"{name}: inconsistent recording")


def _archive_definitions(directory: Path, source: dict) -> tuple[dict, dict]:
    """Read just frozen dev annotations and manifest; never extract the ZIP."""
    files = source.get("files")
    if not isinstance(files, dict) or not files or any(not isinstance(k, str) for k in files):
        raise ValueError("source manifest lacks file digest mapping")
    for name, digest in files.items():
        _digest(digest, "source file digest " + name)
    # Driver's source fingerprint uses its declared file mapping, including
    # default JSON spacing. Keep that format distinct from config hashing.
    calculated = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    _same(source.get("source_sha256"), calculated, "source fingerprint")
    if source.get("holdout_annotations_included") is not False:
        raise ValueError("source snapshot must explicitly exclude holdout annotations")
    with zipfile.ZipFile(directory / "source-snapshot.zip") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("source snapshot has duplicate members")
        if any("holdout" in name.lower() for name in names):
            raise ValueError("holdout annotations must not be included in this dev-only snapshot")
        raw = {}
        for name in (_DEV_PATH, _MANIFEST_PATH):
            info = archive.getinfo(name)
            if info.file_size > 4_000_000:
                raise ValueError("frozen dev metadata exceeds analysis limit")
            raw[name] = archive.read(name)
            _same(hashlib.sha256(raw[name]).hexdigest(), files.get(name), "frozen " + name)
    manifest = _json(raw[_MANIFEST_PATH].decode("utf-8"), "frozen dataset manifest")
    dev_digest = ((manifest.get("files") or {}).get("dev.jsonl") or {}).get("sha256")
    _same(dev_digest, hashlib.sha256(raw[_DEV_PATH]).hexdigest(), "manifest dev digest")
    definitions = {}
    for line in raw[_DEV_PATH].decode("utf-8").splitlines():
        if not line.strip():
            continue
        case = _json(line, "frozen dev case")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in definitions or case.get("split") != "dev":
            raise ValueError("frozen dev cases have invalid or duplicate IDs/splits")
        if not isinstance(case.get("turns"), list) or not case["turns"]:
            raise ValueError("frozen dev case has no turns")
        definitions[case_id] = case
    _same(((manifest.get("files") or {}).get("dev.jsonl") or {}).get("count"), len(definitions),
          "manifest dev count")
    return definitions, manifest


def _verify_turn(turn: dict, definition: dict, case_id: str, index: int) -> dict:
    label = f"{case_id} turn {index + 1}"
    if not isinstance(turn, dict):
        raise ValueError(label + ": invalid turn")
    inp, message = turn.get("input_recording"), turn.get("user_message")
    if (not isinstance(inp, dict) or inp.get("complete") is not True
            or inp.get("source") != "actual_http_input" or inp.get("reason") != "complete"
            or not isinstance(message, str) or not message.strip()
            or isinstance(inp.get("char_limit"), bool) or not isinstance(inp.get("char_limit"), int)
            or not 0 < inp["char_limit"] <= 20_000 or len(message) > inp["char_limit"]):
        raise ValueError(label + ": complete actual input required")
    _same(message, definition.get("message"), label + " actual input versus frozen input")
    recording = turn.get("response_recording")
    if not isinstance(recording, dict) or not isinstance(recording.get("complete"), bool):
        raise ValueError(label + ": response recording marker missing")
    _same(recording.get("observation_protocol_version"), OBSERVATION_PROTOCOL_VERSION, label + " protocol")
    if recording.get("source") != "actual_http_response_allowlist":
        raise ValueError(label + ": actual HTTP allowlist recording required")
    score = turn.get("score")
    if (not isinstance(score, dict) or not isinstance(score.get("all_checks_observed"), bool)
            or not (score.get("success") is None or isinstance(score.get("success"), bool))):
        raise ValueError(label + ": score/completeness marker missing")
    checks = score.get("checks")
    if not isinstance(checks, dict) or not set(checks) <= set(CHECK_NAMES):
        raise ValueError(label + ": invalid checks")
    if any(value is not None and not isinstance(value, bool) for value in checks.values()):
        raise ValueError(label + ": check must be true, false or null")
    complete = set(checks) == set(CHECK_NAMES) and all(isinstance(value, bool) for value in checks.values())
    _same(score["all_checks_observed"], complete, label + " check completeness")
    if recording["complete"]:
        if not valid_recording(recording):
            raise ValueError(label + ": invalid projection or integrity digest")
        status = recording["http_status"]
        observed = observe_response(recording["projection"], http_status=status, recording=recording)
        if status >= 500 or status in {400, 409, 422}:
            observed.update(decision="transport_or_service_error", termination_reason="request_failed",
                            decision_source="actual_http_error_status")
        execution = observed.pop("execution", {})
        _same(turn.get("observed"), observed, label + " recomputed observation")
        _same(score, score_observation(definition["expected"], observed), label + " recomputed rubric")
    else:
        if (recording.get("projection") is not None or recording.get("projection_sha256") is not None
                or not isinstance(recording.get("reason"), str) or recording["reason"] == "complete" or complete):
            raise ValueError(label + ": withheld projection cannot claim complete observations")
        execution = None
        # A transport failure is an observed execution failure. Other legal
        # withheld records have rubric checks only where actually observed.
        if checks:
            actual_success = False if False in checks.values() else None
            _same(score.get("success"), actual_success, label + " partial score")
        elif score.get("success") not in {False, None}:
            raise ValueError(label + ": unavailable response cannot pass")
    return {"complete": complete, "success": score.get("success"), "execution": execution,
            "recording_valid": valid_recording(recording),
            "input_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
            "sent": utc_datetime(recording.get("request_sent_at")),
            "received": utc_datetime(recording.get("response_received_at"))}


def _usage(turn_audits: list[dict], expected_turns: int) -> dict:
    executions = [t["execution"] for t in turn_audits if isinstance(t.get("execution"), dict)]
    calls, inputs, outputs, coref, unfinished, measured_entries, entries = 0, 0, 0, 0, 0, 0, 0
    usage_complete_turns = 0
    for execution in executions:
        call_count = _integer(execution.get("model_calls"), "foreground model_calls")
        input_count = _integer(execution.get("measured_input_tokens"), "measured_input_tokens")
        output_count = _integer(execution.get("measured_output_tokens"), "measured_output_tokens")
        unfinished_count = _integer(execution.get("unfinished_model_calls"), "unfinished_model_calls")
        calls += call_count
        inputs += input_count
        outputs += output_count
        unfinished += unfinished_count
        usage = execution.get("usage")
        if not isinstance(usage, list):
            raise ValueError("execution usage must be a list")
        actual_chat, actual_input, actual_output, actual_measured = 0, 0, 0, 0
        for entry in usage:
            if not isinstance(entry, dict):
                raise ValueError("execution usage entry must be an object")
            if entry.get("kind") not in {"chat", "sdk"} or not isinstance(entry.get("stage"), str):
                raise ValueError("usage entry kind/stage missing or invalid")
            actual_chat += entry["kind"] == "chat"
            entries += 1
            coref += entry.get("kind") == "chat" and entry.get("stage") == "resolve_reference"
            if entry.get("input_tokens") is not None and entry.get("output_tokens") is not None:
                actual_input += _integer(entry["input_tokens"], "entry input tokens")
                actual_output += _integer(entry["output_tokens"], "entry output tokens")
                actual_measured += 1
                measured_entries += 1
        _same(input_count, actual_input, "execution input token subtotal")
        _same(output_count, actual_output, "execution output token subtotal")
        if actual_chat > call_count:
            raise ValueError("recorded chat usage exceeds actual foreground chat calls")
        _same(unfinished_count, call_count - actual_chat, "execution unfinished chat calls")
        count = len(usage) + unfinished_count
        _same(execution.get("usage_coverage"), actual_measured / count if count else None,
              "execution token usage coverage")
        usage_complete_turns += execution.get("usage_coverage") == 1
    return {"foreground_chat_calls_observed": calls, "resolve_reference_recorded_chat_calls": coref,
            "unfinished_chat_calls": unfinished, "measured_input_tokens": inputs,
            "measured_output_tokens": outputs, "usage_entries": entries,
            "entries_with_token_usage": measured_entries, "expected_turns": expected_turns,
            "execution_observed_turns": len(executions), "token_usage_complete_turns": usage_complete_turns,
            "token_usage_turn_coverage": usage_complete_turns / expected_turns if expected_turns else None,
            "cost": None,
            "scope": "foreground server ledger only; recorded calls/tokens are observations, not provider billing",
            "excluded": ["separate readiness probe", "background summaries", "unreported provider internal retries"],
            "stage_count_scope": "recorded chat entries; unfinished calls cannot be assigned to a stage"}


def load_run(report_path: Path) -> dict:
    path = report_path.resolve(strict=True)
    report, config = _read(path), _read(path.parent / "run-config.json")
    manifest, source = _read(path.parent / "run-manifest.json"), _read(path.parent / "source-manifest.json")
    if report.get("report_version") != 3 or report.get("mode") != "online" or report.get("split") != "dev":
        raise ValueError("ablation requires report_version=3, mode=online and split=dev")
    _same(report.get("observation_protocol_version"), OBSERVATION_PROTOCOL_VERSION, "observation protocol")
    _same(report.get("scorer_version"), SCORER_VERSION, "scorer version")
    execution = config.get("execution")
    if not isinstance(execution, dict) or not isinstance(execution.get(FACTOR), bool):
        raise ValueError("config must explicitly declare boolean " + FACTOR_PATH)
    factor = execution[FACTOR]
    config_hash = canonical_sha256(config)
    _same(_digest(report.get("run_config_sha256"), "config hash"), config_hash, "actual config hash")
    _same(report.get("configuration"), config, "embedded configuration")
    common = deepcopy(config)
    del common["execution"][FACTOR]
    source_hash = _digest(source.get("source_sha256"), "source hash")
    code_version = "dirty-source-sha256:" + source_hash
    _same(report.get("code_version"), code_version, "report source version")
    _same(manifest.get("code_version"), code_version, "manifest source version")
    definitions, dataset_manifest = _archive_definitions(path.parent, source)
    _same(report.get("dataset_sha256"), dataset_manifest.get("dataset_sha256"), "dataset hash")
    _same(report.get("dataset_version"), dataset_manifest.get("dataset_version"), "dataset version")
    declared_fixture = (dataset_manifest.get("fixture_profiles") or {}).get(config.get("fixture_profile"))
    if not isinstance(declared_fixture, dict):
        raise ValueError("fixture profile is absent from frozen dataset manifest")
    _same(config.get("fixture_version"), declared_fixture.get("version"), "frozen fixture version")
    _same(config.get("fixture_sha256"), declared_fixture.get("sha256"), "frozen fixture digest")
    verification = report.get("fixture_verification")
    if not isinstance(verification, dict) or verification.get("identity_verified") is not True:
        raise ValueError("authenticated fixture identity verification is required")
    policy = verification.get("execution_policy")
    expected_policy = {FACTOR: factor}
    if not isinstance(policy, dict) or policy.get("verified") is not True:
        raise ValueError("actual server execution policy was not verified")
    _same(policy.get("server"), expected_policy, "server execution policy")
    _same(policy.get("expected"), expected_policy, "expected execution policy")
    for field in ("fixture_profile", "fixture_version", "fixture_sha256"):
        _same(verification.get(field), config.get(field), "authenticated " + field)
        _same((manifest.get("fixture") or {}).get(field), config.get(field), "manifest " + field)
    rows, selection = report.get("rows"), report.get("selection")
    if not isinstance(rows, list) or not rows or not isinstance(selection, dict):
        raise ValueError("report needs nonempty requested rows and selection")
    ids = [row.get("id") for row in rows if isinstance(row, dict)]
    if len(ids) != len(rows) or any(not isinstance(i, str) or i not in definitions for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("requested case IDs are invalid, duplicated or outside frozen dev")
    _same(selection.get("requested_case_ids"), ids, "requested IDs")
    _same(selection.get("requested_count"), len(ids), "requested count")
    if set(manifest.get("case_ids", [])) != set(ids):
        raise ValueError("run manifest cases differ from actual requested IDs")
    frozen_order = [case_id for case_id in definitions if case_id in set(ids)]
    _same(ids, frozen_order, "frozen dataset order")
    audits, inputs, all_turns = {}, {}, []
    for row in rows:
        case_id, definition = row["id"], definitions[row["id"]]
        expected_count = len(definition["turns"])
        if row.get("status") == "skipped":
            if row.get("success") is not None or row.get("turns"):
                raise ValueError(case_id + ": skipped case cannot have an outcome or turns")
            audits[case_id] = {"expected_turns": expected_count, "observed_turns": 0, "complete": False,
                               "success": None, "latency_ms": None, "turns": []}
            inputs[case_id] = []
            continue
        if row.get("status") != "evaluated" or not isinstance(row.get("turns"), list):
            raise ValueError(case_id + ": invalid case status/turns")
        turns = row["turns"]
        _same(row.get("expected_turn_count"), expected_count, case_id + " expected turns")
        _same(row.get("observed_turn_count"), len(turns), case_id + " observed turns")
        if not 1 <= len(turns) <= expected_count:
            raise ValueError(case_id + ": invalid turn count")
        verified = [_verify_turn(turn, definition["turns"][index], case_id, index)
                    for index, turn in enumerate(turns)]
        complete = len(turns) == expected_count and all(t["complete"] for t in verified)
        success = (False if len(turns) != expected_count or any(t["success"] is False for t in verified)
                   else None if any(t["success"] is None for t in verified) else True)
        _same(row.get("all_checks_observed"), complete, case_id + " case completeness")
        _same(row.get("success"), success, case_id + " case outcome")
        latency = _number(row.get("latency_ms"), case_id + " case wall-clock latency")
        audits[case_id] = {"expected_turns": expected_count, "observed_turns": len(turns),
                           "complete": complete, "success": success, "latency_ms": latency, "turns": verified}
        inputs[case_id] = [t["input_sha256"] for t in verified]
        all_turns.extend(verified)
    _same(report.get("summary"), summarize(rows), "recomputed all-requested summary")
    start, end = utc_datetime(manifest.get("started_at")), utc_datetime(report.get("run_at"))
    if start is None or end is None or end < start:
        raise ValueError("run start/end UTC timestamps are invalid")
    for turn in all_turns:
        if turn["sent"] is not None and not start <= turn["sent"] <= end:
            raise ValueError("actual request timestamp is outside run interval")
        if turn["received"] is not None and not start <= turn["received"] <= end:
            raise ValueError("actual response timestamp is outside run interval")
    requested_turns = sum(a["expected_turns"] for a in audits.values())
    summary = report["summary"]
    metrics = {key: summary[key] for key in (
        "requested", "evaluated", "skipped", "fully_scored", "confirmed_success_count", "known_failure_count",
        "unknown_count", "task_success_rate_lower_bound", "task_success_rate_upper_bound",
        "fully_observed_success_rate", "latency_p50_ms", "latency_p95_ms")}
    metrics.update(expected_turns=requested_turns, observed_turns=len(all_turns),
                   requested_score_coverage=summary["fully_scored"] / len(rows),
                   usage=_usage(all_turns, requested_turns))
    return {"path": str(path), "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "run_id": manifest.get("run_id"), "factor": factor, "common_config_sha256": canonical_sha256(common),
            "config_sha256": config_hash, "source_sha256": source_hash,
            "dataset_sha256": _digest(report.get("dataset_sha256"), "dataset hash"),
            "dataset_version": report["dataset_version"], "fixture_sha256": config.get("fixture_sha256"),
            "started": start, "ended": end, "ids": ids, "input_hashes": inputs,
            "metrics": metrics, "audits": audits,
            "provider_returned_model": (manifest.get("provider_preflight") or {}).get("returned_model")}


def analyze_reports(paths: list[Path]) -> dict:
    if len(paths) != 4 or len({p.resolve() for p in paths}) != 4:
        raise ValueError("supply four distinct actual report paths in chronological A B B A order")
    runs = [load_run(path) for path in paths]
    _same([run["factor"] for run in runs], [False, True, True, False], "ABBA factor sequence")
    for run in runs[1:]:
        for field in ("common_config_sha256", "source_sha256", "dataset_sha256", "dataset_version", "fixture_sha256", "ids"):
            _same(run[field], runs[0][field], "ABBA common " + field)
        # Compare full actual input vectors, not just the case IDs. Missing
        # tails cannot be silently accepted as a fair paired sequence.
        _same(run["input_hashes"], runs[0]["input_hashes"], "actual input sequences")
    for earlier, later in zip(runs, runs[1:]):
        if earlier["ended"] > later["started"]:
            raise ValueError("ABBA runs overlap or are not in chronological order")
    labels = ("A1", "B1", "B2", "A2")
    per_case, paired_ids, excluded = [], [], []
    pair1, pair2, merged = [], [], []
    for case_id in runs[0]["ids"]:
        audits = [run["audits"][case_id] for run in runs]
        complete = all(a["complete"] and a["observed_turns"] == a["expected_turns"]
                       and all(t["recording_valid"] for t in a["turns"]) for a in audits)
        row = {"id": case_id, "expected_turns": audits[0]["expected_turns"],
               "latency_ms": {label: a["latency_ms"] for label, a in zip(labels, audits)},
               "success": {label: a["success"] for label, a in zip(labels, audits)},
               "all_checks_observed": {label: a["complete"] for label, a in zip(labels, audits)},
               "paired": complete, "pair1_b_minus_a_ms": None, "pair2_b_minus_a_ms": None,
               "abba_b_mean_minus_a_mean_ms": None}
        if complete:
            a1, b1, b2, a2 = [a["latency_ms"] for a in audits]
            d1, d2, delta = b1 - a1, b2 - a2, (b1 + b2 - a1 - a2) / 2
            pair1.append(d1)
            pair2.append(d2)
            merged.append(delta)
            paired_ids.append(case_id)
            row.update(pair1_b_minus_a_ms=round(d1, 2), pair2_b_minus_a_ms=round(d2, 2),
                       abba_b_mean_minus_a_mean_ms=round(delta, 2))
        else:
            excluded.append({"id": case_id, "reason": "at least one run lacks complete sequence/checks/actual recording"})
        per_case.append(row)
    run_records = [{"label": label, "factor": run["factor"], "report_path": run["path"],
                    "report_sha256": run["report_sha256"], "run_id": run["run_id"],
                    "config_sha256": run["config_sha256"], "started_at": run["started"].isoformat(),
                    "ended_at": run["ended"].isoformat(), "provider_returned_model": run["provider_returned_model"],
                    "metrics": run["metrics"]} for label, run in zip(labels, runs)]
    groups = {}
    for name, indexes in (("A", (0, 3)), ("B", (1, 2))):
        metrics = [runs[i]["metrics"] for i in indexes]
        requested = sum(m["requested"] for m in metrics)
        confirmed = sum(m["confirmed_success_count"] for m in metrics)
        unknown = sum(m["unknown_count"] for m in metrics)
        skipped = sum(m["skipped"] for m in metrics)
        full = sum(m["fully_scored"] for m in metrics)
        latencies = [a["latency_ms"] for i in indexes for a in runs[i]["audits"].values() if a["latency_ms"] is not None]
        groups[name] = {"factor": name == "B", "requested": requested, "confirmed_success_count": confirmed,
                        "known_failure_count": sum(m["known_failure_count"] for m in metrics),
                        "unknown_count": unknown, "skipped": skipped, "fully_scored": full,
                        "requested_score_coverage": full / requested,
                        "success_rate_lower_bound": confirmed / requested,
                        "success_rate_upper_bound": (confirmed + unknown + skipped) / requested,
                        "latency_p50_ms": percentile(latencies, .5), "latency_p95_ms": percentile(latencies, .95),
                        "foreground_chat_calls_observed": sum(m["usage"]["foreground_chat_calls_observed"] for m in metrics),
                        "resolve_reference_recorded_chat_calls": sum(m["usage"]["resolve_reference_recorded_chat_calls"] for m in metrics),
                        "measured_input_tokens": sum(m["usage"]["measured_input_tokens"] for m in metrics),
                        "measured_output_tokens": sum(m["usage"]["measured_output_tokens"] for m in metrics),
                        "cost": None}
    return {"analysis_version": ANALYSIS_VERSION, "mode": "candidate_single_factor_abba",
            "analyzed_at": datetime.now(timezone.utc).isoformat(), "factor_path": FACTOR_PATH,
            "factor_sequence": [False, True, True, False], "run_order": list(labels),
            "common_config_sha256": runs[0]["common_config_sha256"], "source_sha256": runs[0]["source_sha256"],
            "dataset_sha256": runs[0]["dataset_sha256"], "dataset_version": runs[0]["dataset_version"],
            "observation_protocol_version": OBSERVATION_PROTOCOL_VERSION, "scorer_version": SCORER_VERSION,
            "requested_case_ids": runs[0]["ids"], "input_sequence_sha256": canonical_sha256(runs[0]["input_hashes"]),
            "runs": run_records, "groups": groups, "per_case": per_case,
            "paired_latency": {"paired_case_count": len(paired_ids), "requested_unique_case_count": len(per_case),
                               "paired_coverage": len(paired_ids) / len(per_case), "paired_ids": paired_ids,
                               "exclusions": excluded,
                               "pair1_median_b_minus_a_ms": round(median(pair1), 2) if pair1 else None,
                               "pair2_median_b_minus_a_ms": round(median(pair2), 2) if pair2 else None,
                               "abba_median_case_b_mean_minus_a_mean_ms": round(median(merged), 2) if merged else None,
                               "pooled_pair_delta_median_ms": round(median(pair1 + pair2), 2) if pair1 else None,
                               "negative_delta_means": "B faster; per-case wall clock including all declared turns"},
            "limitations": ["small explicitly selected development subset; not a holdout or course baseline comparison",
                            "two repeats per factor; no p-value, statistical significance or population improvement claim",
                            "ABBA reduces simple time drift but does not control provider caching, scheduling or model randomness",
                            "allowlist integrity hashes do not authenticate an honest server or truthful operator",
                            "structured rubric only; natural-language quality and policy assertion support remain unscored",
                            "existing Milvus index is not independently versioned",
                            "foreground tokens may be incomplete; readiness/background/unreported retries excluded",
                            "cost unknown in this experiment; no zero cost or billing savings claim"]}


def write_analysis(path: Path, result: dict) -> None:
    assert_new_output(path, markdown=True)
    lines = ["# 无历史指代消解：单因素 ABBA 实验", "",
             "当前候选代码的消融：A 保留首轮指代消解；B 在没有历史时跳过。四次顺序为 A1、B1、B2、A2。", "",
             f"代码 SHA256：`{result['source_sha256']}`。仅删除配置字段 `{FACTOR_PATH}` 后，其余配置摘要一致：`{result['common_config_sha256']}`。", "",
             "## 全部请求的任务结果", "",
             "| 组 | 请求案例次数 | 确认通过 | 已知失败 | 未知 | 跳过 | 完整评分覆盖率 | 成功率下界 | 成功率上界 | P50 ms | P95 ms |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label, group in result["groups"].items():
        values = [group[k] for k in ("requested", "confirmed_success_count", "known_failure_count", "unknown_count", "skipped",
                                    "requested_score_coverage", "success_rate_lower_bound", "success_rate_upper_bound", "latency_p50_ms", "latency_p95_ms")]
        lines.append("| " + label + " | " + " | ".join(str(v) for v in values) + " |")
    latency = result["paired_latency"]
    lines += ["", "成功率分母是各组两次运行的全部请求；上下界表示缺测范围，不是置信区间。", "", "## 配对延迟", "",
              f"完整配对：{latency['paired_case_count']}/{latency['requested_unique_case_count']} 个案例。负差值表示 B 更快。", "",
              f"- 第一对 B1 − A1 的案例差值中位数：{latency['pair1_median_b_minus_a_ms']} ms。",
              f"- 第二对 B2 − A2 的案例差值中位数：{latency['pair2_median_b_minus_a_ms']} ms。",
              f"- 每案例两次 B 平均 − 两次 A 平均，再取案例中位数：{latency['abba_median_case_b_mean_minus_a_mean_ms']} ms。",
              "", "| 案例 | 轮数 | A1 ms | B1 ms | B2 ms | A2 ms | ABBA 差值 ms | 完整配对 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in result["per_case"]:
        values = [row["id"], row["expected_turns"], *[row["latency_ms"][label] for label in result["run_order"]],
                  row["abba_b_mean_minus_a_mean_ms"], row["paired"]]
        lines.append("| " + " | ".join(str(v) for v in values) + " |")
    lines += ["", "## 已观察调用与用量", "",
              "| 组 | 前台聊天调用 | 记录到 resolve_reference 的聊天调用 | 已测输入 token | 已测输出 token | 费用 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for label, group in result["groups"].items():
        lines.append("| " + label + " | " + " | ".join(str(group[k]) for k in (
            "foreground_chat_calls_observed", "resolve_reference_recorded_chat_calls", "measured_input_tokens", "measured_output_tokens", "cost")) + " |")
    lines += ["", "已测 token 是前台记录量，缺测、后台摘要及供应商未上报重试不计入；费用 null 表示未知。", "",
              "## 原始运行", "", "| 顺序 | 因素值 | 开始 UTC | 报告 | 完整 token 用量轮数 / 预期轮数 |",
              "| --- | --- | --- | --- | --- |"]
    for run in result["runs"]:
        usage = run["metrics"]["usage"]
        lines.append(f"| {run['label']} | {run['factor']} | {run['started_at']} | {run['report_path']} | {usage['token_usage_complete_turns']} / {usage['expected_turns']} |")
    lines += ["", "## 局限", "", *["- " + item for item in result["limitations"]], ""]
    write_json_new(path, result)
    with path.with_suffix(".md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
