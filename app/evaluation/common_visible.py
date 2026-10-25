"""Shared visible-answer evidence adapters; never fabricate HTTP/task observations.

Reuses the existing hash-bound recording and human-review infrastructure. These
derived records are retrospective answer audits, not new Agent executions.
"""
from __future__ import annotations

from pathlib import Path

from app.evaluation.task_recording import canonical_sha256, utc_datetime, valid_recording
from app.evaluation.task_scoring import observe_response
from app.evaluation.task_review import read_json, _encode, _write_new, _report
from app.tools.contracts import validate_result, ContractError

PROTOCOL = "common-visible-evidence-v1"
MODE = "common_visible_review"
ORDER_FIELDS = ("order_id", "tracking_no", "order_status", "amount", "product", "created_at",
                "destination_region", "fulfillment_type", "delivered_at", "order_owned")
LOGISTICS_FIELDS = ("logistics_status", "logistics_current_city", "logistics_stalled_hours", "logistics_trace")
LIMITATIONS = [
    "Derived from actual recorded answers and tool evidence; no new execution or task-success score.",
    "Tool observations are local evidence, not independent carrier truth or policy approval.",
    "Prior-turn evidence retains its turn/time; user correction can make it irrelevant. Missing evidence remains unknown.",
    "No backend-write proof is reconstructed from answer text or an empty receipt list.",
    "Input/answer byte bindings detect local edits; they do not authenticate servers or reviewers.",
]


def _baseline_projection(turn: dict) -> dict | None:
    record = turn.get("baseline_recording")
    if not isinstance(record, dict) or record.get("complete") is not True:
        return None
    p = record.get("projection")
    if (record.get("source") != "actual_local_model_and_shared_executor" or not isinstance(p, dict)
            or record.get("sha256") != canonical_sha256(p)
            or p.get("answer") != turn.get("final_answer")):
        raise ValueError("baseline evidence recording hash/source/answer differs")
    return p


def _flatten(tool: str, data: dict) -> dict:
    if tool == "query_order":
        out = {k: v for k, v in data.items() if k in ORDER_FIELDS}
        if "status" in data:
            out["order_status"] = data["status"]
        # This means the shared executor admitted an owned-order result, not
        # that the judge independently checked the database authorization.
        return {**out, "order_owned": True}
    if tool == "query_logistics":
        return {"logistics_" + k: data[k] for k in ("status", "current_city", "stalled_hours", "trace") if k in data}
    if tool == "query_faq":
        return {"policy_chunk_" + str(c["id"]): {k: c[k] for k in
                ("id", "section_path", "question", "answer", "content_type", "policy_source") if k in c}
                for c in data.get("citations", [])[:4]}
    return {k: data[k] for k in ("warranty", "warranty_until", "return_status", "updated_at",
            "product_name", "price", "stock", "spec") if k in data}


def _evidence(turn: dict, mode: str, index: int) -> tuple[list[dict], list[dict], list[str], str | None]:
    facts, events, notes = [], [], []
    if mode == "course_baseline_local":
        p = _baseline_projection(turn)
        if p is None:
            return [], [], ["complete_tool_recording_unavailable"], None
        sent, received = utc_datetime(p.get("request_sent_at")), utc_datetime(p.get("response_received_at"))
        if sent is None or received is None or received < sent:
            raise ValueError("baseline evidence timestamps invalid")
        for result in p.get("tool_results", []):
            if not isinstance(result, dict):
                raise ValueError("baseline tool result must be an object")
            tool, args, at = result.get("tool"), result.get("arguments"), utc_datetime(result.get("observed_at"))
            events.append({"tool": tool, "ok": result.get("ok"), "error_code": result.get("error_code"), "source_turn": index})
            if result.get("ok") is not True or result.get("business_status") != "ok":
                continue
            if (result.get("user_id") != p.get("user_id") or result.get("conversation_id") != p.get("conversation_id")
                    or at is None or not sent <= at <= received or not isinstance(args, dict)):
                notes.append("tool_scope_or_time_invalid")
                continue
            if tool not in {"query_order", "query_logistics", "query_faq", "query_product", "query_warranty", "query_return_status"}:
                notes.append("unknown_tool_not_admitted")
                continue
            try:
                admitted = validate_result(tool, result.get("data"), args, formatted=True)
            except (ContractError, TypeError, ValueError):
                notes.append("tool_contract_invalid")
                continue
            if admitted.business_status != "ok":
                notes.append("tool_business_result_not_admitted")
                continue
            values = _flatten(tool, admitted.data)
            if values:
                facts.append({"source_tool": tool, "source_turn": index,
                              "observed_at": result["observed_at"], "values": values,
                              "subject": {k: admitted.data[k] for k in ("order_id", "tracking_no") if k in admitted.data}})
        return facts, events, notes, p["response_received_at"]
    record = turn.get("response_recording")
    if not valid_recording(record):
        return [], [], ["complete_http_recording_unavailable"], None
    p = record["projection"]
    observed = observe_response(p, http_status=record["http_status"], recording=record)
    raw = (p.get("task_context") or {}).get("verified_facts", {})
    groups = {}
    for name, value in observed["facts"].items():
        if name not in (*ORDER_FIELDS, *LOGISTICS_FIELDS, "warranty", "warranty_until", "return_status", "updated_at") and not name.startswith("policy_chunk_"):
            continue
        item = raw[name]
        key = (item["source"], item["observed_at"])
        groups.setdefault(key, {})[name] = value
    for (tool, at), values in groups.items():
        scope = next(item.get("scope", {}) for item in raw.values()
                     if item.get("source") == tool and item.get("observed_at") == at)
        subject = {"order_id": scope["active_order_id"]} if scope.get("active_order_id") else {}
        if tool in {"query_order", "query_logistics"} and isinstance(observed["facts"].get("tracking_no"), str):
            subject["tracking_no"] = observed["facts"]["tracking_no"]
        facts.append({"source_tool": tool, "source_turn": index, "observed_at": at, "values": values, "subject": subject})
    if observed.get("fact_observation_complete") is not True:
        notes.append("some_scoped_facts_missing_or_rejected")
    execution = p.get("execution_summary") or {}
    for event in execution.get("tool_events", []):
        events.append({k: event.get(k) for k in ("tool", "ok", "error_code")} | {"source_turn": index})
    return facts, events, notes, record["response_received_at"]


def normalize_report(report: dict, origin: dict, case_ids: list[str] | None = None) -> dict:
    """No expected labels or answer-to-fact extraction; evidence comes from recordings."""
    if report.get("mode") not in {"online", "course_baseline_local"}:
        raise ValueError("normalization requires actual online or baseline source")
    available = {r["id"] for r in report["rows"]}
    if case_ids is not None and (not case_ids or len(set(case_ids)) != len(case_ids) or not set(case_ids) <= available):
        raise ValueError("select distinct case IDs present in this source")
    rows = []
    for row in report["rows"]:
        if case_ids is not None and row["id"] not in case_ids:
            continue
        history, evidence, all_events, notes, turns = [], [], [], [], []
        history_complete = True
        for index, turn in enumerate(row.get("turns", []), 1):
            complete_input = isinstance(turn.get("input_recording"), dict) and turn["input_recording"].get("complete") is True
            complete_answer = isinstance(turn.get("answer_recording"), dict) and turn["answer_recording"].get("complete") is True
            message = turn.get("user_message") if complete_input else None
            answer = turn.get("final_answer") if complete_answer else None
            if not isinstance(message, str) or not isinstance(answer, str):
                history_complete = False
            facts, events, limitations, received = _evidence(turn, report["mode"], index)
            for fact in facts:
                # Reused verified facts are not counted as new tool observations.
                if not any((e["source_tool"], e["observed_at"], e["values"]) ==
                           (fact["source_tool"], fact["observed_at"], fact["values"]) for e in evidence):
                    evidence.append(fact)
            all_events.extend(events)
            notes.extend(limitations)
            if len(evidence) > 100 or len(history) > 100 or len(all_events) > 100:
                raise ValueError("common review context exceeds bounded observation limit")
            refs = []
            end = utc_datetime(received)
            for n, item in enumerate(evidence, 1):
                at = utc_datetime(item["observed_at"])
                fresh = end is not None and at is not None and -5 <= (end - at).total_seconds() <= 900
                refs.append({"id": f"e{n:04d}", **item, "within_observation_ttl": fresh})
            observed = {"facts": {"evidence_records": refs, "conversation_history": list(history),
                        "tool_events": list(all_events), "observation_limits": sorted(set(notes)),
                        "backend_side_effects": "not_independently_observed"},
                        "evidence_sources": sorted({r["source_tool"] for r in refs}), "actions": []}
            turns.append({"user_message": message, "final_answer": answer,
                          "input_recording": {"complete": complete_input and history_complete, "source": PROTOCOL},
                          "answer_recording": {"complete": complete_answer, "source": PROTOCOL}, "observed": observed})
            if isinstance(message, str):
                history.append({"role": "user", "content": message, "source_turn": index})
            if isinstance(answer, str):
                history.append({"role": "assistant", "content": answer, "source_turn": index})
        rows.append({"id": row["id"], "status": row.get("status"),
                     "expected_turn_count": row.get("expected_turn_count"), "turns": turns})
    return {"mode": MODE, "observation_protocol_version": PROTOCOL, "split": report["split"],
            "dataset_sha256": report["dataset_sha256"], "run_config_sha256": report["run_config_sha256"],
            "code_version": report.get("code_version"), "implementation_label": report.get("implementation_label"),
            "origin": origin, "selected_case_ids": case_ids, "rows": rows, "limitations": LIMITATIONS}


def validate_common_report(report: dict) -> None:
    origin = report.get("origin")
    if not isinstance(origin, dict) or set(origin) != {"report_path", "report_sha256"}:
        raise ValueError("common review requires original report binding")
    source_path = Path(origin["report_path"])
    source, digest = read_json(source_path)
    if source.get("mode") not in {"online", "course_baseline_local"} or digest != origin["report_sha256"]:
        raise ValueError("original report changed or derived records cannot chain")
    source, _ = _report(source_path)
    if report != normalize_report(source, origin, report.get("selected_case_ids")):
        raise ValueError("common evidence differs from original source recordings")


def export_common_reports(paths: list[Path], output: Path, case_ids: list[str] | None = None) -> list[Path]:
    """Prepare all inputs first; shared config/dataset/input checks precede writes."""
    if len(paths) != 2 or len({p.resolve() for p in paths}) != 2 or output.exists():
        raise ValueError("two different actual sources and a new directory are required")
    normalized = []
    for path in paths:
        source, digest = _report(path)
        normalized.append(normalize_report(source, {"report_path": str(path.resolve()), "report_sha256": digest}, case_ids))
    a, b = normalized
    if any(a[k] != b[k] for k in ("dataset_sha256", "run_config_sha256", "split")):
        raise ValueError("shared dataset/config/split required")
    ar, br = {r["id"]: r for r in a["rows"]}, {r["id"]: r for r in b["rows"]}
    if set(ar) != set(br):
        raise ValueError("requested case sets differ")
    for key in ar:
        at, bt = ar[key]["turns"], br[key]["turns"]
        for x, y in zip(at, bt):
            if x["user_message"] is not None and y["user_message"] is not None and x["user_message"] != y["user_message"]:
                raise ValueError("actual paired user inputs differ")
    output.mkdir(parents=True, exist_ok=False)
    destinations = [output / "source-1.json", output / "source-2.json"]
    for value, path in zip(normalized, destinations):
        _write_new(path, _encode(value))
    return destinations
