"""Explicit local course-inspired ReAct measurement; not a v3 HTTP baseline.

No provider, service or database call occurs on import or --help. Credentials
come from named environment variables. Default split is dev; holdout requires
an explicit frozen-experiment digest and is read only when that run is invoked.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.course_baseline import CourseBaseline, SafeRecorder, PROTOCOL, load_course_prompt
from app.evaluation.task_cases import load_dataset
from app.evaluation.task_report import assert_new_output, write_json_new
from app.evaluation.task_scoring import SCORER_VERSION, score_observation, summarize
from eval_task_agent import _config, _select_cases, _post, _get, _aggregate, _skip
from probe_agent_boundaries import validate_url


def verify_local_configuration(config: dict) -> dict:
    """Verify labels and budgets without serializing provider URLs/credentials."""
    from app.config import settings
    from app.tools.synthetic_fixtures import profile_info
    info = profile_info()
    expected = {k: config[k] for k in ("fixture_profile", "fixture_version", "fixture_sha256")}
    if info != expected:
        raise ValueError("local baseline fixture differs from declared run configuration")
    model = config["model_config"]
    comparisons = {
        "chat_model": settings.chat_model, "embedding_model": settings.embed_model,
        "rerank_model": settings.rerank_model, "thinking": settings.chat_thinking,
        "reasoning_effort": settings.chat_reasoning_effort, "max_output_tokens": settings.max_output_tokens,
    }
    for key, actual in comparisons.items():
        if key in model and model[key] != actual:
            raise ValueError(f"local baseline model setting differs: {key}")
    declared = config.get("execution")
    if not isinstance(declared, dict):
        raise ValueError("shared execution budgets must be explicitly declared")
    budgets = {"turn_timeout_seconds": settings.turn_timeout_seconds,
               "max_agent_steps": settings.max_agent_steps,
               "max_task_tool_calls": settings.max_task_tool_calls,
               "max_identical_tool_calls": settings.max_identical_tool_calls,
               "mcp_timeout_seconds": settings.mcp_tool_timeout}
    for key, actual in budgets.items():
        if key not in declared or isinstance(declared[key], bool) or declared[key] != actual:
            raise ValueError(f"local baseline execution budget differs: {key}")
    temperature = (model.get("stage_temperatures") or {}).get("main_agent")
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise ValueError("model_config.stage_temperatures.main_agent must be explicit")
    return {"fixture": info, "model_labels": comparisons, "execution_budgets": budgets,
            "main_agent_temperature": temperature,
            "knowledge_version": config["knowledge_version"],
            "knowledge_verification": "operator_supplied; actual database/vector contents not attested",
            "credential_configuration": "runtime settings; omitted from report"}


async def run(args, cases, manifest, config, prompt) -> tuple[list[dict], dict]:
    from app.config import settings
    local = verify_local_configuration(config)
    username, password = os.environ.get(args.username_env), os.environ.get(args.password_env)
    if not username or not password:
        raise ValueError("set the named username/password environment variables")
    declared_fixture = (manifest.get("fixture_profiles") or {}).get(config["fixture_profile"])
    if (not isinstance(declared_fixture, dict) or
            declared_fixture.get("version") != config["fixture_version"] or
            declared_fixture.get("sha256") != config["fixture_sha256"]):
        raise ValueError("configuration fixture differs from frozen dataset metadata")
    status, login = _post(args.base_url, "/api/auth/login", {"username": username, "password": password}, args.auth_timeout)
    token = login.get("access_token")
    if status != 200 or not isinstance(token, str) or not token:
        raise ValueError("baseline account authentication failed; credentials were not recorded")
    rows, fresh = [], []
    try:
        status, me = _get(args.base_url, "/api/auth/me", args.auth_timeout, token)
        expected_fixture = {k: config[k] for k in ("fixture_profile", "fixture_version", "fixture_sha256")}
        actual_fixture = {k: me.get(k) for k in expected_fixture}
        uid = me.get("user_id")
        if (status != 200 or me.get("role") != "customer" or actual_fixture != expected_fixture
                or not isinstance(uid, str) or not uid.strip() or len(uid) > 128
                or (config["fixture_profile"] == "task-v1" and uid != "customer_demo")):
            raise ValueError("actual baseline account/fixture differs; no case was run")
        recorder = SafeRecorder((password, token, settings.chat_api_key, settings.embed_api_key, settings.rerank_api_key))
        baseline = CourseBaseline(prompt, temperature=local["main_agent_temperature"], recorder=recorder)
        for case in cases:
            if config["fixture_profile"] not in case["online_profiles"]:
                rows.append(_skip(case, "no installed adapter in the declared and verified fixture profile"))
                continue
            session = await baseline.new_session(uid)
            fresh.append(session.conversation_id)
            turns = []
            for number, definition in enumerate(case["turns"], 1):
                turn = await baseline.run_turn(session, definition["message"])
                # expected is used only after actual execution, never as model,
                # tool or runtime input. No decision is inferred from answer text.
                turn.update(turn_index=number, score=score_observation(definition["expected"], turn["observed"]))
                turns.append(turn)
                if turn["runtime_termination_reason"] != "model_finished":
                    break
            row = _aggregate(case, turns, sum(t["latency_ms"] for t in turns))
            row.update(comparable_with_http_v3=False,
                       comparability_reason="local ReAct transcript; no explicit task decision or scoped TaskContext",
                       conversation_id=session.conversation_id)
            rows.append(row)
        return rows, {"authenticated_user_id": uid, "role": "customer", "server_fixture": actual_fixture,
                      "local": local, "fresh_conversation_ids": fresh,
                      "identity_source": "actual server login and /api/auth/me; token omitted",
                      "execution_source": "local model/shared executor, not /api/agent"}
    finally:
        try:
            _post(args.base_url, "/api/auth/logout", {}, args.auth_timeout, token)
        except (OSError, ValueError, TimeoutError):
            pass  # Logout may return HTTP 204 with no JSON; retain the measurements.
        finally:
            from app.db.base import engine
            await engine.dispose()


def write_baseline_report(path: Path, report: dict) -> None:
    assert_new_output(path, markdown=True)
    lines = ["# 课程启发的安全 ReAct 基线", "",
             "这是实际本地模型/安全工具执行，不是课程原版等价，也不是实际 HTTP v3 观察。",
             "缺少显式任务决策、业务终态与 TaskContext 的评分维度保留 unknown；不从答复文本猜测。", "",
             f"源码标识：`{report['code_version']}`。配置：`{report['run_config_sha256']}`。",
             f"原课程 commit：`{report['course_provenance']['course_commit']}`。", "",
             "| Case | 状态 | V3 缺测 | 本地总耗时 ms |", "| --- | --- | --- | --- |"]
    for row in report["rows"]:
        lines.append(f"| {row['id']} | {row['status']} | {', '.join(row.get('unscored', [])) or row.get('skip_reason', '')} | {row.get('latency_ms')} |")
    lines.extend(["", "没有可共同完整评分的配对时，整体任务提升保持未知。",
                  "安全改造包括同账号验证、订单归属、运单授权、严格工具契约、只读限制、有限重试和预算。",
                  "知识实际版本仍由操作者说明；录制 SHA 证明内容一致性，不能证明语义正确或服务器身份。",
                  "TTFT 未测；前台调用及用量不包含全部后台/供应商调用，价格或用量缺失时费用未知。", ""])
    write_json_new(path, report)
    with path.with_suffix(".md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/agent_eval")
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--course-ref", default="90e4b24")
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--implementation-label", default="course-inspired-safe-react")
    parser.add_argument("--frozen-experiment-sha256", help="required for explicit holdout execution; preregistered experiment plan digest")
    parser.add_argument("--username-env", default="MEWHELP_EVAL_USERNAME")
    parser.add_argument("--password-env", default="MEWHELP_EVAL_PASSWORD")
    parser.add_argument("--auth-timeout", type=float, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_new_output(args.output, markdown=True)
    args.base_url = validate_url(args.base_url)
    if not re.fullmatch(r"(?:dirty-source-sha256:[0-9a-f]{64}|[0-9a-f]{7,40})", args.code_version):
        raise ValueError("an actual frozen source/commit identifier is required")
    if not math.isfinite(args.auth_timeout) or not 0 < args.auth_timeout <= 120:
        raise ValueError("auth timeout must be positive and bounded")
    if args.split == "holdout" and not re.fullmatch(r"[0-9a-f]{64}", args.frozen_experiment_sha256 or ""):
        raise ValueError("holdout requires an explicit frozen experiment plan digest")
    config, config_hash = _config(args.config)
    prompt, provenance = load_course_prompt(ROOT, args.course_ref)
    cases, manifest = load_dataset(args.dataset, args.split)
    cases, selection = _select_cases(cases, args.case_ids, config["fixture_profile"])
    rows, verification = asyncio.run(run(args, cases, manifest, config, prompt))
    diagnostic = summarize(rows)
    summary = {"requested": diagnostic["requested"], "evaluated": diagnostic["evaluated"],
               "skipped": diagnostic["skipped"], "fully_scored": diagnostic["fully_scored"],
               "unscored": diagnostic["unscored"], "task_success_rate": None,
               "task_success_rate_reason": "not comparable: missing explicit task state and HTTP-v3 recording",
               "latency_p50_ms": diagnostic["latency_p50_ms"], "latency_p95_ms": diagnostic["latency_p95_ms"],
               "ttft_ms": None, "cost": diagnostic["cost"], "usage_coverage": diagnostic["usage_coverage"]}
    report = {"report_version": 1, "mode": "course_baseline_local", "implementation_label": args.implementation_label,
              "observation_protocol_version": PROTOCOL, "scorer_version": SCORER_VERSION,
              "code_version": args.code_version, "run_at": datetime.now(timezone.utc).isoformat(),
              "split": args.split, "dataset_version": manifest["dataset_version"],
              "dataset_sha256": manifest["dataset_sha256"], "dataset_validation_scope": manifest["validation_scope"],
              "run_config_sha256": config_hash, "configuration": config, "selection": selection,
              "course_provenance": provenance, "environment_verification": verification,
              "frozen_experiment_sha256": args.frozen_experiment_sha256,
              "summary": summary, "shared_scorer_diagnostic": diagnostic, "rows": rows,
              "comparable_with_http_v3": False,
              "shared_safety": ["server-authenticated same-account scope", "read-only allowlist and hard write denial",
                                "shared tool contracts/ownership/timeouts/retries", "authorized tracking from successful owned order query",
                                "shared execution budgets and foreground usage ledger"],
              "differences_from_original_course": ["bare ReAct subset; no intent/coreference/forced graph branches",
                                                   "current shared read-only safety adapter", "full serial message history; no automatic summary",
                                                   "independent local transcript protocol; no HTTP business state"],
              "limitations": ["not an unmodified original-course end-to-end baseline", "no inferred decision/facts/side-effect proof",
                              "no task success or overall improvement claim without common complete observations",
                              "operator supplied knowledge version; no server attestation", "no TTFT or complete provider bill"]}
    write_baseline_report(args.output, report)
    print(json.dumps({"mode": report["mode"], "output": str(args.output), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Do not print provider/authentication exception text or configuration.
        raise SystemExit("course baseline input/environment rejected: " + type(exc).__name__) from None
