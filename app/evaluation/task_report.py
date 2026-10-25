"""Reports and strictly paired comparison; missing measurements are not zero."""
import json
from pathlib import Path
import re

from app.evaluation.task_recording import OBSERVATION_PROTOCOL_VERSION, valid_recording
from app.evaluation.task_scoring import CHECK_NAMES, SCORER_VERSION

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TEXT_LIMIT = 20_000


def _input_issue(turn: dict) -> str | None:
    recording = turn.get("input_recording")
    if not isinstance(recording, dict):
        return "input_recording_missing"
    if recording.get("complete") is not True:
        return "input_recording_incomplete"
    if recording.get("source") != "actual_http_input":
        return "input_recording_source_invalid"
    limit = recording.get("char_limit")
    message = turn.get("user_message")
    if (isinstance(limit, bool) or not isinstance(limit, int) or not 0 < limit <= _TEXT_LIMIT
            or not isinstance(message, str) or not message.strip() or len(message) > limit
            or recording.get("reason") != "complete"):
        return "input_recording_content_invalid"
    return None


def _answer_complete(turn: dict, *, case_id: str) -> bool:
    """Absent recording remains unknown; contradictory declarations are invalid."""
    recording, answer = turn.get("answer_recording"), turn.get("final_answer")
    if recording is None and answer is None:
        return False
    if not isinstance(recording, dict) or not isinstance(recording.get("complete"), bool):
        raise ValueError(f"case {case_id}: missing or invalid answer recording marker")
    limit = recording.get("char_limit")
    if (recording.get("source") != "actual_http_response_answer"
            or isinstance(limit, bool) or not isinstance(limit, int) or not 0 < limit <= _TEXT_LIMIT
            or not isinstance(recording.get("reason"), str) or not recording["reason"]):
        raise ValueError(f"case {case_id}: invalid answer recording metadata")
    if recording["complete"]:
        if not isinstance(answer, str) or len(answer) > limit or recording["reason"] != "complete":
            raise ValueError(f"case {case_id}: complete answer recording disagrees with content")
        return True
    if answer is not None or recording["reason"] == "complete":
        raise ValueError(f"case {case_id}: incomplete answer recording must withhold the entire answer")
    return False


def _answer_coverage(rows: dict[str, dict], flags: dict[str, list[bool]]) -> dict:
    selected = [row for row in rows.values() if row.get("status") == "evaluated"]
    expected = sum(row["expected_turn_count"] for row in selected)
    observed = sum(len(row["turns"]) for row in selected)
    complete = sum(sum(flags[row["id"]]) for row in selected)
    return {"complete_turns": complete, "observed_turns": observed, "expected_turns": expected,
            "coverage": complete / expected if expected else None,
            "scope": "declared turns of evaluated cases, including unobserved tails; skipped cases excluded"}


def _complete_coverage(row: dict) -> bool:
    """Reject old or inconsistent completeness claims before paired scoring."""
    if row.get("status") != "evaluated":
        return False
    if not isinstance(row.get("all_checks_observed"), bool):
        raise ValueError(f"case {row.get('id', '?')}: missing observation completeness; record or rescore this report explicitly")
    turns = row.get("turns")
    requested, observed = row.get("expected_turn_count"), row.get("observed_turn_count")
    if (not isinstance(turns, list) or not turns or isinstance(requested, bool) or not isinstance(requested, int)
            or requested < 1 or isinstance(observed, bool) or not isinstance(observed, int)
            or observed != len(turns) or observed > requested):
        raise ValueError(f"case {row.get('id', '?')}: invalid turn completeness metadata")
    complete = observed == requested
    turn_successes = []
    for turn in turns:
        if not isinstance(turn, dict) or not isinstance(turn.get("score"), dict):
            raise ValueError(f"case {row.get('id', '?')}: invalid turn score record")
        score = turn.get("score") or {}
        recording = turn.get("response_recording")
        if (not isinstance(recording, dict) or recording.get("observation_protocol_version") != OBSERVATION_PROTOCOL_VERSION
                or recording.get("source") != "actual_http_response_allowlist"
                or not isinstance(recording.get("complete"), bool)):
            raise ValueError(f"case {row.get('id', '?')}: actual HTTP observation recording is missing or invalid")
        if recording["complete"]:
            if not valid_recording(recording):
                raise ValueError(f"case {row.get('id', '?')}: observation projection or SHA256 is invalid")
        elif (recording.get("projection") is not None or recording.get("projection_sha256") is not None
              or not isinstance(recording.get("reason"), str) or recording["reason"] == "complete"):
            raise ValueError(f"case {row.get('id', '?')}: withheld observation must not retain a partial projection")
        checks = score.get("checks")
        marker = score.get("all_checks_observed")
        if not isinstance(marker, bool) or not isinstance(checks, dict):
            raise ValueError(f"case {row.get('id', '?')}: missing turn observation completeness")
        if not set(checks) <= set(CHECK_NAMES) or any(value is not None and not isinstance(value, bool) for value in checks.values()):
            raise ValueError(f"case {row.get('id', '?')}: invalid observed check values")
        actual_complete = set(checks) == set(CHECK_NAMES) and all(isinstance(value, bool) for value in checks.values())
        if marker != actual_complete:
            raise ValueError(f"case {row.get('id', '?')}: turn completeness claim disagrees with observed checks")
        if actual_complete and not recording["complete"]:
            raise ValueError(f"case {row.get('id', '?')}: complete checks lack a complete actual HTTP observation")
        if actual_complete:
            actual_success = all(checks.values())
            if score.get("success") is not actual_success:
                raise ValueError(f"case {row.get('id', '?')}: turn outcome disagrees with observed checks")
            turn_successes.append(actual_success)
        complete = complete and actual_complete
    if row["all_checks_observed"] != complete:
        raise ValueError(f"case {row.get('id', '?')}: case completeness claim disagrees with turns")
    if complete and row.get("success") is not all(turn_successes):
        raise ValueError(f"case {row.get('id', '?')}: case outcome disagrees with observed turns")
    return complete


def assert_new_output(path: Path, *, markdown: bool = False) -> None:
    """Fail before expensive calls rather than replacing an earlier experiment."""
    if path.suffix.lower() != ".json":
        raise ValueError("evaluation output must use a new .json path")
    destinations = [path, path.with_suffix(".md")] if markdown else [path]
    if any(destination.exists() for destination in destinations):
        raise ValueError("evaluation output already exists; choose new paths to preserve prior results")


def write_json_new(path: Path, value: dict) -> None:
    assert_new_output(path)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(raw)


def write_report(path: Path, report: dict) -> None:
    assert_new_output(path, markdown=True)
    summary = report["summary"]
    lines = ["# Task Agent 评测报告", "", f"模式：**{report['mode']}**。运行标签：{report['implementation_label']}。", "",
             "replay 仅检查当前程序守卫，不是模型能力评测；online 才实际调用服务。缺测字段保留 null。", "",
             "主指标 task_success_rate 是全部请求案例的已确认通过下界。已知失败、未知和跳过均保留；另报成功率上界与完整观察案例的条件成功率。上下界不是统计置信区间。", "",
             f"观察协议：`{report.get('observation_protocol_version')}`；评分版本：`{report.get('scorer_version')}`。", "",
             f"数据版本：{report['dataset_version']}；数据 SHA256：`{report['dataset_sha256']}`。", "",
             f"配置 SHA256：`{report['run_config_sha256']}`。模型/知识配置由操作者填写；online fixture 元数据经服务端核对。", "",
             "| 指标 | 实测值 |", "| --- | --- |"]
    for key, value in summary.items():
        lines.append(f"| {key} | {json.dumps(value, ensure_ascii=False)} |")
    lines += ["", "## 错例与未评分", "", "| case | 状态 | 失败项或跳过原因 |", "| --- | --- | --- |"]
    for row in report["rows"]:
        if row.get("success") is not True:
            detail = row.get("skip_reason") or ", ".join(row.get("failures", []) + row.get("unscored", [])) or row.get("error") or "unknown"
            lines.append(f"| {row['id']} | {row['status']} | {detail} |")
    lines += ["", "## 口径与局限", "",
              "- 任务成功要求所有轮次的决策、关键事实、证据来源、禁止动作及终止状态满足标注。",
              "- confirmed_success_count + known_failure_count + unknown_count + skipped = requested；局部已知失败计入 known_failure_count，不能从主指标分母中消失。",
              "- fully_observed_success_rate 与配对比较仅使用完整观察；它们必须连同覆盖率报告，不能代替全部请求结果。",
              "- online response_recording 保存真实 HTTP 白名单 projection、字段存在性、摘要、认证身份及请求/接收 UTC 时间；不保存登录令牌、历史消息、隐藏推理或原始工具正文。",
              "- 事实必须在合法账号/会话/任务/活动订单 scope 内，并在真实响应接收时间前 900 秒内；最多容忍 5 秒服务器未来时钟偏差。缺字段、过期或不一致为不可用观察。",
              "- 三种动作容器必须均为合法列表；已观察写调用的失败、超时、无回执或未知工具效果保持副作用未知。显式写入回执才支持 committed_write。",
              "- 证据来源出现不证明最终政策断言成立；本版本不评分政策版本适用性和自然语言断言支持率。",
              "- online 只运行登录后已核对profile的场景；其余场景需接入适配器或提供安全录制快照。",
              "- /api/agent 非流式，本脚本无法测量 TTFT；延迟包含 HTTP 等待和服务内重试。",
              "- 成本是上游报告用量乘以服务器配置价格的估算，不是账单费用；只有完整覆盖、相同币种和cost_kind才合计，未知为null。",
              "- 后台摘要与供应商内部未上报重试可能未计入估算。",
              "- online 逐轮 final_answer 来自真实 HTTP 响应；缺失、超限或命中凭据样式时为 null，answer_recording.complete=false。过滤不能替代操作者脱敏审查。",
              "- 合成数据集尚需人工复核；此报告不能据此承诺线上表现或面试结果。", ""]
    markdown = "\n".join(lines)
    write_json_new(path, report)
    with path.with_suffix(".md").open("x", encoding="utf-8") as stream:
        stream.write(markdown)


def compare_reports(baseline: dict, candidate: dict) -> dict:
    if baseline.get("report_version") != 3 or candidate.get("report_version") != 3:
        raise ValueError("comparison requires report_version=3 with actual HTTP observation recordings; old reports stay frozen and require a new explicitly recorded run")
    for report in (baseline, candidate):
        if report.get("observation_protocol_version") != OBSERVATION_PROTOCOL_VERSION or report.get("scorer_version") != SCORER_VERSION:
            raise ValueError("comparison requires the supported identical observation_protocol_version and scorer_version")
        if not isinstance(report.get("split"), str) or report["split"] not in {"dev", "holdout", "all"}:
            raise ValueError("cannot compare: split must be dev, holdout or all")
        for field in ("dataset_sha256", "run_config_sha256"):
            if not isinstance(report.get(field), str) or not _SHA256.fullmatch(report[field]):
                raise ValueError(f"cannot compare: {field} must be a nonempty lowercase SHA256 digest")
        if (not isinstance(report.get("rows"), list)
                or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] for row in report["rows"])):
            raise ValueError("cannot compare: rows require nonempty case IDs")
    for field in ("mode", "dataset_sha256", "run_config_sha256", "split", "observation_protocol_version", "scorer_version"):
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"cannot compare: {field} differs")
    if baseline.get("mode") != "online":
        raise ValueError("replay reports are guard diagnostics, not Agent improvement comparisons")
    left = {r["id"]: r for r in baseline["rows"]}
    right = {r["id"]: r for r in candidate["rows"]}
    if len(left) != len(baseline["rows"]) or len(right) != len(candidate["rows"]):
        raise ValueError("cannot compare reports with duplicate case IDs")
    if set(left) != set(right):
        raise ValueError("cannot compare reports with different requested case IDs")
    left_complete = {key: _complete_coverage(row) for key, row in left.items()}
    right_complete = {key: _complete_coverage(row) for key, row in right.items()}
    left_answers = {key: [_answer_complete(turn, case_id=key) for turn in row["turns"]]
                    if row.get("status") == "evaluated" else [] for key, row in left.items()}
    right_answers = {key: [_answer_complete(turn, case_id=key) for turn in row["turns"]]
                     if row.get("status") == "evaluated" else [] for key, row in right.items()}
    paired, exclusions = [], []
    for key in sorted(left):
        a, b = left[key], right[key]
        reasons, input_issues = [], []
        for side, row, coverage in (("baseline", a, left_complete[key]), ("candidate", b, right_complete[key])):
            if row.get("status") != "evaluated":
                reasons.append(side + "_not_evaluated")
                continue
            if not coverage:
                reasons.append(side + "_observation_incomplete")
            for index, turn in enumerate(row["turns"]):
                issue = _input_issue(turn)
                if issue:
                    input_issues.append({"side": side, "turn_index": index, "reason": issue})
            if any(issue["side"] == side for issue in input_issues):
                reasons.append(side + "_actual_input_unavailable")
        if a.get("status") == b.get("status") == "evaluated":
            if a["expected_turn_count"] != b["expected_turn_count"]:
                reasons.append("expected_turn_count_differs")
            elif not input_issues and [turn["user_message"] for turn in a["turns"]] != [turn["user_message"] for turn in b["turns"]]:
                reasons.append("actual_user_messages_differ")
        if reasons:
            exclusions.append({"id": key, "reasons": reasons, "input_issues": input_issues,
                               "baseline_unscored": a.get("unscored", []), "candidate_unscored": b.get("unscored", [])})
        else:
            paired.append((a, b))
    before = sum(a["success"] for a, b in paired) / len(paired) if paired else None
    after = sum(b["success"] for a, b in paired) / len(paired) if paired else None
    left_answer_stats, right_answer_stats = _answer_coverage(left, left_answers), _answer_coverage(right, right_answers)
    paired_turns = sum(a["expected_turn_count"] for a, b in paired)
    paired_left_answers = sum(sum(left_answers[a["id"]]) for a, b in paired)
    paired_right_answers = sum(sum(right_answers[b["id"]]) for a, b in paired)
    paired_both_answers = sum(sum(la and ra for la, ra in zip(left_answers[a["id"]], right_answers[b["id"]])) for a, b in paired)
    return {"mode": "paired_online_comparison", "comparison_status": "compared" if paired else "not_compared",
            "observation_protocol_version": OBSERVATION_PROTOCOL_VERSION, "scorer_version": SCORER_VERSION,
            "not_compared_reason": None if paired else "no complete machine observations with matching complete actual input recordings",
            "paired_count": len(paired), "requested_count": len(left),
            "paired_coverage": len(paired) / len(left) if left else None, "baseline_success_rate": before,
            "candidate_success_rate": after, "absolute_change": after - before if paired else None,
            "baseline_fully_observed": sum(left_complete.values()), "candidate_fully_observed": sum(right_complete.values()),
            "excluded_ids": [row["id"] for row in exclusions], "exclusions": exclusions,
            "baseline_partial_observed_failures": [key for key in sorted(left) if not left_complete[key] and left[key].get("status") == "evaluated" and left[key].get("success") is False],
            "candidate_partial_observed_failures": [key for key in sorted(right) if not right_complete[key] and right[key].get("status") == "evaluated" and right[key].get("success") is False],
            "baseline_answer_recording": left_answer_stats, "candidate_answer_recording": right_answer_stats,
            "baseline_answer_recording_coverage": left_answer_stats["coverage"], "candidate_answer_recording_coverage": right_answer_stats["coverage"],
            "paired_answer_recording": {"expected_turns": paired_turns, "baseline_complete_turns": paired_left_answers,
                                         "candidate_complete_turns": paired_right_answers, "both_complete_turns": paired_both_answers,
                                         "baseline_coverage": paired_left_answers / paired_turns if paired_turns else None,
                                         "candidate_coverage": paired_right_answers / paired_turns if paired_turns else None,
                                         "both_coverage": paired_both_answers / paired_turns if paired_turns else None},
            "fixed": [a["id"] for a, b in paired if not a["success"] and b["success"]],
            "regressed": [a["id"] for a, b in paired if a["success"] and not b["success"]],
            "conclusion_scope": "structured machine rubric only; final natural-language answer quality and assertion support are not compared",
            "limitations": ["model/knowledge configuration is operator supplied, not server verified", "synthetic fixtures",
                            "allowlist projection SHA256 detects content changes, not forged recordings or a dishonest server",
                            "only complete common machine observation coverage with identical actual inputs is paired; excluded partial failures remain diagnostics",
                            "answer recording coverage is reported separately and is not a natural-language quality metric"]}
