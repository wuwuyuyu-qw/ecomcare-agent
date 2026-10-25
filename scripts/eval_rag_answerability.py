"""Isolated retrieval -> gate -> answer -> OpenEvals review experiment.

Same actual hits feed both gates. No routing/HTTP/tool execution is measured;
the generation context and answer guard reuse production helpers. Development
labels enter scoring only. Plan-only does not open .env or contact services.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from uuid import uuid4
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.rag_answerability import load_bundle, confusion, threshold_scan
from scripts.eval_rag_development import encode, write_new


def source_state():
    paths = [p for folder in ("app", "scripts", "tests/agent_core") for p in (ROOT / folder).rglob("*.py")]
    paths.extend((ROOT / "data/rag_answerability").glob("*"))
    paths.extend(ROOT / n for n in ("pyproject.toml", "uv.lock"))
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths) if p.is_file()}
    return {"files": hashes, "source_sha256": hashlib.sha256(encode(hashes)).hexdigest()}


def append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def finalize_component_answer(state, text):
    from app.graph.answer_guard import guard_answer
    return guard_answer(state, text)


async def measure(args, docs, cases, snapshot):
    from dotenv import dotenv_values
    os.environ.update({k: v for k, v in dotenv_values(args.env_file, interpolate=False).items() if v is not None})
    os.environ.update(LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY="", LANGFUSE_BASE_URL="",
                      LANGSMITH_TRACING="false", SUBQUERY_SPLIT="false", POLICY_CATALOG_PATH="")
    from app.config import settings
    from app.core import embeddings, rerank, retrieval, selfcheck, evidence_selection
    from app.core.confidence import compute_evidence_confidence
    from app.core.execution import execution_scope, UsageCollector, utc_now
    from app.core.llm import get_chat_model
    from app.core.policy_runtime import build_citations
    from app.graph import nodes, task_context
    from app.kb import milvus_client as mv
    from app.evaluation.common_judge import create_evaluator, rate_with_format_retry, _success, protocol_digest
    from scripts.judge_common_answers import create_judge_client
    from langchain_core.messages import HumanMessage
    from langsmith import tracing_context

    name = "raganswer_" + uuid4().hex[:24]
    config = {"chat_model": settings.chat_model, "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model, "collection": name,
        "top_k": 10, "recall_top_k": settings.recall_top_k, "subquery_split": False,
        "legacy_confidence_threshold": settings.evidence_confidence_threshold,
        "legacy_gate_temperature": 0.3, "selected_gate_temperature": 0,
        "generator_temperature": 0.3, "generator_thinking": False, "judge_temperature": 0,
        "judge_protocol_sha256": protocol_digest(), "repetitions": args.repetitions,
        "factor": "weighted score + binary check versus selected evidence semantic check; same hits",
        "limits": "guide corpus common to both; no route/tool/HTTP measurements; dev only",
        "cost": None, "prices": "unknown", "labels_given_to_generator_or_gate": False}
    write_new(args.output_dir / "run-config.json", config)
    client = await mv.acall(mv.get_client)
    if await mv.acall(client.has_collection, name):
        raise ValueError("new collection required")
    rows, observations, ratings = [], [], []
    try:
        with execution_scope(args.timeout) as ledger:
            texts = [d["category"] + "\n" + d["question"] + "\n" + d["answer"] for d in docs]
            vectors = await asyncio.wait_for(embeddings.embed_texts(texts), args.timeout)
            await mv.acall(mv.ensure_collection, client, name)
            await mv.acall(mv.upsert_vectors, client, [{**d, "dense": v, "text": t}
                for d, v, t in zip(docs, vectors, texts, strict=True)], name)
            await mv.acall(mv.flush, client, name)
            found = await mv.acall(client.query, name, filter="id >= 0", limit=len(docs), output_fields=["question", "answer"])
            expected = {str(d["id"]): (d["question"], d["answer"]) for d in docs}
            if {str(d["id"]): (d["question"], d["answer"]) for d in found} != expected:
                raise ValueError("index identities/bodies differ")
            write_new(args.output_dir / "index-observation.json", {"collection": name, "owned_created": True,
                "documents": len(found), "exact_match": True, "setup_execution": ledger.summary()})
        for case_index, case in enumerate(cases):
            started = time.monotonic()
            with execution_scope(args.timeout) as ledger:
                try:
                    with tracing_context(enabled=False):
                        hits = await asyncio.wait_for(retrieval.search_knowledge(case["query"], strategy="hybrid_rerank",
                            top_k=10, collection=name, split=False), args.timeout)
                    if any(str(h["id"]) not in expected or (h["question"], h["answer"]) != expected[str(h["id"])] for h in hits):
                        raise ValueError("retrieval source binding differs")
                    retrieval_status, error = "observed", None
                except Exception as exc:
                    hits, retrieval_status, error = [], "failed", type(exc).__name__
                observation = {"id": case["id"], "query": case["query"], "answerable": case["answerable"],
                    "hits": hits, "retrieval_status": retrieval_status, "error_class": error,
                    "top_score": float(hits[0]["rerank_score"]) if hits else 0,
                    "elapsed_seconds": time.monotonic() - started, "execution": ledger.summary()}
            observations.append(observation)
            append(args.output_dir / "retrieval.jsonl", observation)
            for repetition in range(args.repetitions):
                # ABBA when two repetitions; alternate first arm across cases.
                arms = ("legacy", "selected") if (case_index + repetition) % 2 == 0 else ("selected", "legacy")
                for arm in arms:
                    started = time.monotonic()
                    with execution_scope(args.timeout) as ledger:
                        callback_config = {"callbacks": [UsageCollector(ledger)]}
                        status, error, answer, citations = "observed", None, "", []
                        try:
                            async with asyncio.timeout(args.timeout):
                                if retrieval_status != "observed":
                                    raise RuntimeError("retrieval_failed")
                                if arm == "legacy":
                                    confidence = compute_evidence_confidence(hits)
                                    if confidence.score < settings.evidence_confidence_threshold:
                                        gate = {"useful": False, "status": "score_rejected", "reason": "legacy confidence threshold", "selected": []}
                                    else:
                                        # Existing binary chain with the same observable callbacks.
                                        raw = await selfcheck._chain().ainvoke({"query": case["query"],
                                            "evidence": "\n".join(f"[{i}] {h['question']} {h['answer']}" for i, h in enumerate(hits, 1))}, callback_config)
                                        gate = {"useful": raw.useful, "status": "full" if raw.useful else "none", "reason": raw.reason,
                                                "selected": hits if raw.useful else []}
                                else:
                                    gate = await evidence_selection.select_evidence(case["query"], hits, callback_config)
                                    if gate["status"] == "unavailable":
                                        raise RuntimeError("selection_unavailable")
                                accepted = bool(gate["useful"])
                                if accepted:
                                    citations = build_citations(retrieval.arrange_head_tail(gate["selected"]))
                                    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
                                    context = task_context.begin_turn(None, "synthetic_review", 0, case["query"], {"now": utc_now()})
                                    state = {"messages": [HumanMessage(case["query"])], "user_id": "synthetic_review", "conversation_id": 0,
                                        "route": "knowledge", "evidence_strong": True, "evidence": evidence, "citations": citations,
                                        "task_context": context, "request_triage": {"protocol": "joint-request-triage-v1", "status": "parsed",
                                            "intent": "商品咨询", "confidence": 1.0, "mode": "knowledge", "checks": [], "route": "knowledge"}}
                                    ai = await get_chat_model(streaming=True, thinking=False).ainvoke(nodes._agent_messages(state), callback_config)
                                    verdict = finalize_component_answer(state, ai.content)
                                    answer = verdict["answer"]
                                    gate["answer_guard_reason"] = verdict["reason"]
                                else:
                                    answer = evidence_selection.abstention_text(gate["status"]) if arm == "selected" else nodes.FALLBACK_REPLY
                        except Exception as exc:
                            status, error, accepted = "failed", type(exc).__name__, False
                            gate = {"status": "unavailable", "reason": error, "selected": []}
                        row = {"id": case["id"], "arm": arm, "repetition": repetition + 1, "query": case["query"],
                            "answerable": case["answerable"], "accepted": accepted, "status": status, "error_class": error,
                            "gate": {k: v for k, v in gate.items() if k != "selected"}, "selected_ids": [str(h["id"]) for h in gate["selected"]],
                            "answer": answer, "citations": citations, "elapsed_seconds": time.monotonic() - started,
                            "execution": ledger.summary()}
                    rows.append(row)
                    append(args.output_dir / "answers.jsonl", row)
                    entry = {"user_message": case["query"], "answer": answer, "observed": {"facts": {
                        "evidence_records": [{"id": f"doc:{h['id']}", "source_tool": "isolated_knowledge_retrieval",
                            "observed_at": utc_now(), "within_observation_ttl": True, "subject": {"document_id": str(h["id"])},
                            "values": {"question": h["question"], "answer": h["answer"]}} for h in hits],
                        "conversation_history": [], "tool_events": [], "backend_side_effects": "not_independently_observed",
                        "observation_limits": ["component experiment; no actual order tools or HTTP", "all documents are self-authored synthetic data"]}}}
                    if status == "observed":
                        with execution_scope(args.timeout) as review_ledger:
                            judge = create_judge_client({"CHAT_BASE_URL": settings.chat_base_url, "CHAT_API_KEY": settings.chat_api_key},
                                model=settings.chat_model, timeout=args.timeout, callbacks=[UsageCollector(review_ledger)])
                            try:
                                with tracing_context(enabled=False):
                                    rating = await rate_with_format_retry(create_evaluator(judge), entry, args.timeout)
                            finally:
                                await judge.root_async_client.close()
                                judge.root_client.close()
                            review_usage = review_ledger.summary()
                    else:
                        rating, review_usage = {"status": "not_reviewable", "rating": None}, None
                    result = {"id": case["id"], "arm": arm, "repetition": repetition + 1, "entry": entry,
                        **rating, "visible_quality_success": _success([rating["rating"]]), "execution": review_usage}
                    ratings.append(result)
                    append(args.output_dir / "ratings.jsonl", result)
                    print(json.dumps({"id": case["id"], "arm": arm, "repetition": repetition + 1,
                        "status": status, "accepted": accepted, "review_status": rating["status"]}), flush=True)
        after = source_state()
        write_new(args.output_dir / "source-stability.json", {"unchanged": snapshot == after})
        if snapshot != after:
            raise ValueError("source changed during experiment")
        summary = {}
        for arm in ("legacy", "selected"):
            selected = [r for r in rows if r["arm"] == arm]
            reviewed = [r for r in ratings if r["arm"] == arm]
            summary[arm] = {**confusion(selected), "elapsed_p50_seconds": statistics.median(r["elapsed_seconds"] for r in selected),
                "visible_ai_pass": sum(r["visible_quality_success"] is True for r in reviewed),
                "visible_ai_fail": sum(r["visible_quality_success"] is False for r in reviewed),
                "visible_ai_unknown": sum(r["visible_quality_success"] is None for r in reviewed),
                "model_calls": sum(r["execution"]["model_calls"] for r in selected)}
        return {"measurement": "isolated_rag_gate_and_generated_answer_development", "summary": summary,
            "threshold_diagnostic": threshold_scan(observations), "collection": name, "preserved": True,
            "source_sha256": snapshot["source_sha256"], "human_ratings": 0, "cost": None,
            "limits": ["same-family AI judge, not human or independent quality proof", "positive/negative labels balanced and visible",
                       "corpus includes tutorials in both arms; guide addition effect not isolated", "no end-to-end routing, tool success or ticket measurement"]}
    finally:
        await embeddings.close_client()
        await rerank.close_client()
        await mv.acall(client.close)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--frozen-plan", type=Path)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--repetitions", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if not 0 < args.timeout <= 180:
        raise ValueError("bounded timeout required")
    args.env_file, args.output_dir = args.env_file.resolve(), args.output_dir.resolve()
    docs, cases, manifest = load_bundle(ROOT / "data/rag_answerability")
    snapshot = source_state()
    plan = {**snapshot, "dataset_sha256": manifest["dataset_sha256"], "repetitions": args.repetitions,
        "timeout": args.timeout, "queries": len(cases), "documents": len(docs), "arms": ["legacy", "selected"],
        "order": "alternate AB/BA by case/repetition; two repetitions yield ABBA", "split": "development_only",
        "labels_enter_scoring_only": True, "quality": "OpenEvals common-visible rubric; same-family judge",
        "runtime_threshold_will_change": False}
    if args.frozen_plan:
        if json.loads(args.frozen_plan.read_text(encoding="utf-8")) != plan:
            raise ValueError("frozen source/dataset/plan differs")
    elif not args.plan_only:
        raise ValueError("freeze before provider calls")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_new(args.output_dir / "frozen-plan.json", plan)
    with zipfile.ZipFile(args.output_dir / "source-snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in snapshot["files"]:
            archive.write(ROOT / relative, relative)
    if args.plan_only:
        print(json.dumps({"event": "planned_no_env_or_network", "cases": len(cases)}))
        return
    with tempfile.TemporaryDirectory(prefix="mewhelp-answer-empty-") as temporary:
        old = Path.cwd()
        try:
            os.chdir(temporary)
            with __import__("langsmith").tracing_context(enabled=False):
                value = asyncio.run(measure(args, docs, cases, snapshot))
        finally:
            os.chdir(old)
    write_new(args.output_dir / "report.json", value)
    print(json.dumps({"event": "measured", "summary": value["summary"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit("Answerability experiment failed: " + type(exc).__name__) from None
