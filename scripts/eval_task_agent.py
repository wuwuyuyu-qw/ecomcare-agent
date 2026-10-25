"""Run explicitly labelled online evaluation or safe pure-guard replay.

Credentials are read from named environment variables, never .env or files.
No API calls occur until the operator explicitly runs the `online` command.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.task_cases import load_dataset
from app.evaluation.task_report import assert_new_output, compare_reports, write_json_new, write_report
from app.evaluation.task_recording import OBSERVATION_PROTOCOL_VERSION, record_response
from app.evaluation.task_scoring import SCORER_VERSION, observe_response, score_observation, summarize

_RECORD_CHAR_LIMIT = 20_000
_CREDENTIAL_LIKE = re.compile(
    r"(?i)(?:bearer\s+\S+|sk-[A-Za-z0-9_-]{8,}|"
    r"[\"']?(?:password|passwd|api[_ -]?key|access[_ -]?token|authorization|secret)[\"']?\s*[:=]|"
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})"
)


def _record_text(value, *, source: str) -> tuple[str | None, dict]:
    """Record complete actual text or explicitly withhold it, never a prefix.

    This narrow credential screen is not a guarantee of complete redaction.
    Local experiments must use synthetic inputs and operator-reviewed outputs.
    """
    metadata = {"complete": False, "source": source, "char_limit": _RECORD_CHAR_LIMIT}
    if not isinstance(value, str):
        return None, {**metadata, "reason": "missing_or_invalid_text"}
    if len(value) > _RECORD_CHAR_LIMIT:
        return None, {**metadata, "reason": "text_exceeds_recording_limit"}
    if _CREDENTIAL_LIKE.search(value):
        return None, {**metadata, "reason": "credential_like_text_withheld"}
    return value, {**metadata, "complete": True, "reason": "complete"}


class FixtureExperimentError(RuntimeError):
    """An invalid experiment environment is not a failed Agent decision."""


def _read_json(path: Path) -> dict:
    if path.name.lower() == ".env" or path.stat().st_size > 20_000_000:
        raise ValueError("use a JSON file under 20 MB; .env is not a supported input")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON document must be an object")
    return value


def _config(path: Path) -> tuple[dict, str]:
    value = _read_json(path)
    required = {"model_config", "knowledge_version", "fixture_profile", "fixture_version", "fixture_sha256", "pricing"}
    if not required <= value.keys():
        raise ValueError("config requires model_config, knowledge_version, fixture profile/version/hash and pricing")
    if (not isinstance(value["model_config"], dict) or not value["model_config"]
            or not isinstance(value["pricing"], dict)
            or not all(isinstance(value[k], str) and value[k] for k in ("knowledge_version", "fixture_version"))):
        raise ValueError("config field types/versions are incomplete")
    temperatures = value["model_config"].get("stage_temperatures")
    if temperatures is not None:
        if (not isinstance(temperatures, dict) or not temperatures
                or any(not isinstance(stage, str) or not stage.strip() for stage in temperatures)):
            raise ValueError("model_config.stage_temperatures must be a nonempty object of stage names")
        for temperature in temperatures.values():
            if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                                             or not math.isfinite(temperature) or not 0 <= temperature <= 2):
                raise ValueError("stage temperatures must be finite numbers in [0,2] or null for unknown")
    if not isinstance(value["fixture_profile"], str) or value["fixture_profile"] not in {"demo", "task-v1"} or (value["fixture_sha256"] is not None and
            (not isinstance(value["fixture_sha256"], str) or len(value["fixture_sha256"]) != 64)):
        raise ValueError("config fixture profile/digest is invalid")
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if "replace-with" in text:
        raise ValueError("replace example configuration placeholders before running an experiment")
    from app.evaluation.task_replay import _SECRET
    if _SECRET.search(text) or any(key in text.lower() for key in ('"api_key"', '"password"', '"access_token"', '"authorization"')):
        raise ValueError("evaluation config must not contain credentials")
    return value, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _select_cases(cases: list[dict], case_ids: list[str] | None, profile: str) -> tuple[list[dict], dict]:
    """Filter a declared request set without changing the frozen dataset/rubric."""
    available = {case["id"] for case in cases}
    if case_ids is not None:
        if not case_ids or len(case_ids) != len(set(case_ids)):
            raise ValueError("--case-ids must contain a nonempty whitelist without duplicate IDs")
        if not set(case_ids) <= available:
            # Do not inspect other splits to distinguish unknown from holdout.
            raise ValueError("--case-ids contains an unknown ID or an ID outside the selected split")
        selected = [case for case in cases if case["id"] in set(case_ids)]
    else:
        selected = cases
    requested_ids = [case["id"] for case in selected]
    eligible_ids = [case["id"] for case in selected if profile in case["online_profiles"]]
    identity = json.dumps(requested_ids, ensure_ascii=False, separators=(",", ":"))
    selection = {"mode": "explicit_case_ids" if case_ids is not None else "all_cases_in_split",
                 "requested_case_ids": requested_ids, "requested_count": len(selected),
                 "available_case_count": len(cases), "unselected_count": len(cases) - len(selected),
                 "eligible_case_ids": eligible_ids, "eligible_count": len(eligible_ids), "fixture_profile": profile,
                 "eligibility_scope": "online fixture profile declarations; replay separately requires recorded snapshots",
                 "order": "frozen_dataset_order", "requested_ids_sha256": hashlib.sha256(identity.encode("utf-8")).hexdigest()}
    return selected, selection


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _post(base_url: str, endpoint: str, body: dict, timeout: float, token: str = "") -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = Request(base_url.rstrip("/") + endpoint, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    try:
        with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            if response.headers.get("Content-Length") and int(response.headers["Content-Length"]) > 4_000_000:
                raise ValueError("API response exceeds recording limit")
            raw = response.read(4_000_001)
            if len(raw) > 4_000_000:
                raise ValueError("API response exceeds recording limit")
            payload = json.loads(raw)
            return response.status, payload if isinstance(payload, dict) else {}
    except HTTPError as exc:
        # Error bodies may contain provider/internal details. Store only code.
        return exc.code, {}


def _get(base_url: str, endpoint: str, timeout: float, token: str) -> tuple[int, dict]:
    request = Request(base_url.rstrip("/") + endpoint, headers={"Authorization": "Bearer " + token}, method="GET")
    try:
        with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            raw = response.read(1_000_001)
            if len(raw) > 1_000_000:
                raise ValueError("fixture metadata response is too large")
            payload = json.loads(raw)
            return response.status, payload if isinstance(payload, dict) else {}
    except HTTPError as exc:
        return exc.code, {}


def _aggregate(case: dict, turns: list[dict], latency: float | None) -> dict:
    checks = [t["score"] for t in turns]
    success = False if any(c["success"] is False for c in checks) else None if any(c["success"] is None for c in checks) else True
    complete_sequence = len(turns) == len(case["turns"])
    failures = {f for c in checks for f in c["failures"]}
    unscored = {u for c in checks for u in c["unscored"]}
    if not complete_sequence:
        success = False
        failures.add("incomplete_turn_sequence")
        unscored.add("remaining_turns")
    costs = [t.get("cost") for t in turns]
    currency = {t.get("currency") for t in turns if t.get("cost") is not None}
    cost_kinds = {t.get("cost_kind") for t in turns if t.get("cost") is not None}
    cost_complete = (costs and all(c is not None for c in costs) and len(currency) == len(cost_kinds) == 1
                     and all(isinstance(value, str) and value for value in currency | cost_kinds))
    return {"id": case["id"], "family": case["family"], "split": case["split"], "status": "evaluated",
            "success": success, "all_checks_observed": bool(complete_sequence and checks and all(c.get("all_checks_observed") is True for c in checks)),
            "expected_turn_count": len(case["turns"]), "observed_turn_count": len(turns),
            "failures": sorted(failures), "unscored": sorted(unscored),
            "latency_ms": round(latency, 2) if latency is not None else None,
            "cost": sum(costs) if cost_complete else None,
            "currency": next(iter(currency)) if len(currency) == 1 and all(isinstance(value, str) and value for value in currency) else None,
            "cost_kind": next(iter(cost_kinds)) if len(cost_kinds) == 1 and all(isinstance(value, str) and value for value in cost_kinds) else None,
            "usage_complete": bool(turns and all(t.get("usage_complete") for t in turns)),
            "turns": turns}


def _skip(case: dict, reason: str) -> dict:
    return {"id": case["id"], "family": case["family"], "split": case["split"],
            "status": "skipped", "success": None, "skip_reason": reason,
            "cost": None, "usage_complete": False}


def _online(args, cases: list[dict], manifest: dict, config: dict) -> tuple[list[dict], dict]:
    url = urlsplit(args.base_url)
    if url.scheme not in {"http", "https"} or url.hostname not in {"localhost", "127.0.0.1", "::1"} or url.username or url.password:
        raise ValueError("this local-demo runner only accepts loopback URLs without embedded credentials")
    profile = config["fixture_profile"]
    declared = (manifest.get("fixture_profiles") or {}).get(profile)
    if not declared or config["fixture_version"] != declared["version"] or config["fixture_sha256"] != declared["sha256"]:
        raise ValueError("run config fixture profile/version/hash differs from the frozen dataset manifest")
    username, password = os.environ.get(args.username_env), os.environ.get(args.password_env)
    if not username or not password:
        raise ValueError("set the named username/password environment variables before online evaluation")
    code, login = _post(args.base_url, "/api/auth/login", {"username": username, "password": password}, args.timeout)
    token = login.get("access_token")
    if code != 200 or not isinstance(token, str) or not token:
        raise ValueError(f"local demo login failed (HTTP {code}); no credential value was recorded")
    rows = []
    try:
        code, me = _get(args.base_url, "/api/auth/me", args.timeout, token)
        metadata = {key: me.get(key) for key in ("fixture_profile", "fixture_version", "fixture_sha256")}
        expected = {"fixture_profile": profile, "fixture_version": declared["version"], "fixture_sha256": declared["sha256"]}
        if code != 200 or metadata != expected:
            raise ValueError("actual authenticated server fixture metadata differs from the requested profile; no case was scored")
        if profile == "task-v1" and me.get("user_id") != "customer_demo":
            raise ValueError("task-v1 synthetic fixtures require the customer_demo account; no case was scored")
        authenticated_user_id = me.get("user_id")
        if not isinstance(authenticated_user_id, str) or not authenticated_user_id.strip() or len(authenticated_user_id) > 128:
            raise ValueError("actual authenticated server identity is missing or invalid; no case was scored")
        execution_policy_verification = None
        declared_execution = config.get("execution") or {}
        if isinstance(declared_execution, dict) and "skip_coref_without_history" in declared_execution:
            skip_coref = declared_execution["skip_coref_without_history"]
            server_policy = me.get("execution_policy")
            if (not isinstance(skip_coref, bool) or not isinstance(server_policy, dict)
                    or not isinstance(server_policy.get("skip_coref_without_history"), bool)
                    or server_policy["skip_coref_without_history"] != skip_coref):
                raise FixtureExperimentError("actual server coreference policy differs from declared factor; no case was scored")
            execution_policy_verification = {"server": server_policy,
                "expected": {"skip_coref_without_history": skip_coref}, "verified": True}
        for case in cases:
            if profile not in case["online_profiles"]:
                rows.append(_skip(case, "case has no installed adapter in the verified fixture profile"))
                continue
            turns, cid = [], None
            started = time.perf_counter()
            for turn in case["turns"]:
                # Optional evaluation-only lifecycle hooks; ordinary runners and
                # all existing grading contracts retain their prior behavior.
                if getattr(args, "before_turn", None) and not args.before_turn():
                    break
                body = {"message": turn["message"]}
                if cid is not None:
                    body["conversation_id"] = cid
                request_sent_at = datetime.now(timezone.utc).isoformat()
                try:
                    status, response = _post(args.base_url, "/api/agent", body, args.timeout, token)
                    response_received_at = datetime.now(timezone.utc).isoformat()
                    if status == 401:
                        raise FixtureExperimentError("evaluation login expired or was revoked; run aborted without treating authentication loss as model failure")
                    fixture_events = []
                    for field, events_field in (("execution_summary", "tool_events"), ("logistics_decision", "tool_trace")):
                        container = response.get(field)
                        if isinstance(container, dict) and isinstance(container.get(events_field), list):
                            fixture_events.extend(container[events_field])
                    if any(event.get("error_code") in {"fixture_profile_mismatch", "fixture_configuration_error"}
                           for event in fixture_events if isinstance(event, dict)):
                        raise FixtureExperimentError("API/MCP fixture worlds differ; run aborted without reporting model failures")
                    response_recording = record_response(
                        response, http_status=status, authenticated_user_id=authenticated_user_id,
                        request_conversation_id=body.get("conversation_id"), request_sent_at=request_sent_at,
                        response_received_at=response_received_at,
                        unavailable_reason="http_error_body_withheld" if status != 200 else None,
                    )
                    observed = observe_response(response_recording.get("projection") or {}, http_status=status,
                                                recording=response_recording)
                    if status >= 500 or status in {400, 409, 422}:
                        observed.update(decision="transport_or_service_error", termination_reason="request_failed",
                                        decision_source="actual_http_error_status")
                    score = score_observation(turn["expected"], observed)
                    cid = response.get("conversation_id", cid)
                    execution = observed.pop("execution", {})
                    user_message, input_recording = _record_text(body["message"], source="actual_http_input")
                    final_answer, answer_recording = _record_text(response.get("answer"), source="actual_http_response_answer")
                    turns.append({"user_message": user_message, "input_recording": input_recording,
                                  "final_answer": final_answer, "answer_recording": answer_recording,
                                  "response_recording": response_recording,
                                  "observed": observed, "score": score,
                                  "cost": execution.get("cost") if execution.get("cost_complete") else None,
                                  "currency": execution.get("currency"),
                                  "cost_kind": execution.get("cost_kind"),
                                  "usage_complete": execution.get("usage_coverage") == 1,
                                  "usage": {k: execution.get(k) for k in ("model_calls", "tool_calls", "measured_input_tokens", "measured_output_tokens", "usage_coverage", "price_coverage", "scope", "excluded")}})
                    if getattr(args, "after_turn", None): args.after_turn(turns[-1])
                    if status != 200:
                        break
                except (URLError, TimeoutError, OSError, ValueError):
                    user_message, input_recording = _record_text(body["message"], source="actual_http_input")
                    response_recording = record_response(
                        None, http_status=None, authenticated_user_id=authenticated_user_id,
                        request_conversation_id=body.get("conversation_id"), request_sent_at=request_sent_at,
                        response_received_at=None, unavailable_reason="response_unavailable",
                    )
                    turns.append({"user_message": user_message, "input_recording": input_recording,
                                  "final_answer": None, "answer_recording": {"complete": False, "source": "actual_http_response_answer", "char_limit": _RECORD_CHAR_LIMIT, "reason": "response_unavailable"},
                                  "response_recording": response_recording,
                                  "observed": {"decision": "transport_error", "decision_source": "actual_transport_error",
                                               "scorer_version": SCORER_VERSION},
                                  "score": {"success": False, "all_checks_observed": False, "failures": ["transport"], "unscored": ["response_unavailable"], "checks": {}},
                                  "cost": None, "usage_complete": False})
                    if getattr(args, "after_turn", None): args.after_turn(turns[-1])
                    break
            rows.append(_aggregate(case, turns, (time.perf_counter() - started) * 1000))
    finally:
        try:
            _post(args.base_url, "/api/auth/logout", {}, min(args.timeout, 5), token)
        except (URLError, TimeoutError, OSError, ValueError):
            pass
    return rows, {**metadata, "identity_verified": profile != "task-v1" or me.get("user_id") == "customer_demo",
                  "execution_policy": execution_policy_verification,
                  "scope": "API metadata verified; MCP consistency is separately enforced by server tool contracts"}


def _replay(args, cases: list[dict]) -> tuple[list[dict], str]:
    from app.evaluation.task_replay import load_snapshots, replay_turn
    snapshots, digest = load_snapshots(args.snapshots)
    rows = []
    for case in cases:
        record = snapshots.get(case["id"])
        if not record:
            rows.append(_skip(case, "no operator-provided recorded safe snapshot"))
            continue
        if len(record["turns"]) != len(case["turns"]):
            raise ValueError(f"snapshot turn count differs: {case['id']}")
        turns, context = [], None
        for index, turn in enumerate(case["turns"]):
            observed, context = replay_turn(context, case["id"], index, turn["message"], record["turns"][index])
            turns.append({"observed": observed, "score": score_observation(turn["expected"], observed),
                          "cost": None, "usage_complete": False})
        # Offline CPU time is not an online Agent latency measurement.
        rows.append(_aggregate(case, turns, None))
    return rows, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("online", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", type=Path, default=ROOT / "data/agent_eval")
        p.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
        p.add_argument("--case-ids", nargs="+", help="explicit case whitelist within --split; default requests the whole selected split")
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--implementation-label", required=True)
        p.add_argument("--code-version", required=True, help="commit/tree identifier; do not label a dirty tree as clean")
        p.add_argument("--output", type=Path, required=True)
        if name == "online":
            p.add_argument("--base-url", default="http://127.0.0.1:8000")
            p.add_argument("--username-env", default="MEWHELP_EVAL_USERNAME")
            p.add_argument("--password-env", default="MEWHELP_EVAL_PASSWORD")
            p.add_argument("--timeout", type=float, default=180)
        else:
            p.add_argument("--snapshots", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_new_output(args.output, markdown=args.mode != "compare")
    if args.mode == "compare":
        result = compare_reports(_read_json(args.baseline), _read_json(args.candidate))
        write_json_new(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    cases, manifest = load_dataset(args.dataset, args.split)
    config, config_hash = _config(args.config)
    cases, selection = _select_cases(cases, args.case_ids, config["fixture_profile"])
    snapshot_hash = None
    fixture_verification = None
    if args.mode == "online":
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout must be positive")
        rows, fixture_verification = _online(args, cases, manifest, config)
    else:
        rows, snapshot_hash = _replay(args, cases)
    report = {"report_version": 3, "mode": args.mode, "implementation_label": args.implementation_label,
              "observation_protocol_version": OBSERVATION_PROTOCOL_VERSION if args.mode == "online" else "pure-guard-replay-v1",
              "scorer_version": SCORER_VERSION,
              "code_version": args.code_version, "run_at": datetime.now(timezone.utc).isoformat(),
              "split": args.split, "dataset_version": manifest["dataset_version"],
              "dataset_sha256": manifest["dataset_sha256"], "run_config_sha256": config_hash,
              "dataset_validation_scope": manifest["validation_scope"], "selection": selection,
              "configuration_source": "model_and_knowledge_operator_supplied; online_fixture_metadata_verified" if args.mode == "online" else "operator_supplied_no_server_check", "configuration": config,
              "fixture_verification": fixture_verification,
              "snapshot_sha256": snapshot_hash, "summary": summarize(rows), "rows": rows}
    write_report(args.output, report)
    print(json.dumps({"mode": args.mode, "output": str(args.output), "summary": report["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FixtureExperimentError, OSError) as exc:
        raise SystemExit(f"evaluation input rejected: {exc}") from None
