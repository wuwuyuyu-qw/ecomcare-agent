"""Explicit isolated RAG development comparison; no Agent or ticket writes.

Plan-only never opens the supplied .env or connects to services. Online mode
reuses the existing retrieval SDK path and keeps its new collection for replay.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import importlib.metadata
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
from app.evaluation.rag_development import STRATEGIES, load_bundle, ranking_metrics, policy_probe


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_new(path, value):
    with path.open("xb") as handle:
        handle.write(encode(value))


def source_state():
    names = ["app/evaluation/rag_development.py", "scripts/eval_rag_development.py", "app/config.py",
             "app/core/retrieval.py", "app/core/embeddings.py", "app/core/rerank.py",
             "app/core/execution.py", "app/core/vector_contract.py", "app/core/policy_evidence.py",
             "app/kb/milvus_client.py", "pyproject.toml", "uv.lock"]
    names += ["data/rag_development/" + n for n in ("manifest.json", "dev.jsonl", "corpus.json", "catalog.json")]
    hashes = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in sorted(names)}
    return {"files": hashes, "source_sha256": hashlib.sha256(encode(hashes)).hexdigest()}


def metrics_summary(rows):
    out = {}
    def mean(values):
        values = [v for v in values if v is not None]
        return statistics.mean(values) if values else None
    for strategy in STRATEGIES:
        selected = [r for r in rows if r["strategy"] == strategy]
        successful = [r for r in selected if r["status"] == "observed"]
        times = sorted(r["elapsed_seconds"] for r in selected)
        out[strategy] = {"requested": len(selected), "observed": len(successful),
            "errors": len(selected) - len(successful),
            **{k: mean(r["ranking"][k] for r in selected) for k in ("recall_at_k", "mrr_at_k", "ndcg_at_k")},
            "applicable_target_recall": mean(r["policy"]["applicable_target_recall"] for r in selected),
            "policy_oracle_outcome_matches": sum(r["policy"]["outcome_matches_expected"] is True for r in selected),
            "policy_oracle_scorable_cases": sum(r["policy"]["outcome_matches_expected"] is not None for r in selected),
            "unsafe_oracle_releases": sum(r["policy"].get("unsupported_release_on_no_answer_case") is True for r in selected),
            "out_of_scope_nonempty": sum(r["ranking"]["out_of_scope_nonempty"] is True for r in selected),
            "elapsed_p50_seconds": statistics.median(times) if times else None,
            "elapsed_mean_seconds": mean(times), "cost": None,
            "scope": "all requested calls; rank failures count zero, no semantic answer scoring"}
    return out


async def measure(args, docs, cases, catalog, snapshot):
    from dotenv import dotenv_values
    # Imports occur inside an empty cwd. Application Settings cannot silently
    # load an unrelated .env; only the named file supplies provider config.
    explicit = dotenv_values(args.env_file, interpolate=False)
    os.environ.update({k: v for k, v in explicit.items() if v is not None})
    os.environ.update(LANGFUSE_PUBLIC_KEY="", LANGFUSE_SECRET_KEY="", LANGFUSE_BASE_URL="",
                      LANGSMITH_TRACING="false", SUBQUERY_SPLIT="false")
    from app.config import settings
    from app.core import embeddings, rerank, retrieval
    from app.core.execution import execution_scope
    from app.kb import milvus_client as mv
    from langsmith import tracing_context

    name = "ragdev_" + uuid4().hex[:24]
    config = {"embed_model": settings.embed_model, "rerank_model": settings.rerank_model,
        "pymilvus_version": importlib.metadata.version("pymilvus"), "top_k": 3,
        "recall_top_k": settings.recall_top_k, "subquery_split": False,
        "collection": name, "index": "dense cosine + Milvus chinese BM25 + SDK RRFRanker(k=60)",
        "prices": "unknown; no monetary savings inferred", "query_embedding_reuse": False,
        "retry_scope": "existing embedding SDK and rerank retry adapters; unreported retries may omit tokens"}
    write_new(args.output_dir / "run-config.json", config)
    client = await mv.acall(mv.get_client)
    if await mv.acall(client.has_collection, name):
        raise ValueError("new isolated collection required")
    rows = []
    try:
        with execution_scope(args.timeout) as ledger:
            texts = [d["category"] + "\n" + d["question"] + "\n" + d["answer"] for d in docs]
            vectors = await asyncio.wait_for(embeddings.embed_texts(texts), args.timeout)
            await mv.acall(mv.ensure_collection, client, name)
            indexed = [{**d, "dense": v, "text": text} for d, v, text in zip(docs, vectors, texts, strict=True)]
            await mv.acall(mv.upsert_vectors, client, indexed, name)
            await mv.acall(mv.flush, client, name)
            observed = await mv.acall(client.query, name, filter="id >= 0", limit=len(docs), output_fields=["answer", "question"])
            actual = {str(d["id"]): (d["question"], d["answer"]) for d in observed}
            expected = {str(d["id"]): (d["question"], d["answer"]) for d in docs}
            if actual != expected:
                raise ValueError("indexed document identity/body differs")
            write_new(args.output_dir / "index-observation.json", {"collection": name, "owned_created": True,
                "observed_documents": len(actual), "exact_question_body_match": True,
                "setup_execution": ledger.summary(), "scope": "setup excluded from query latency/call metrics"})
        for index, case in enumerate(cases):
            offset = index % len(STRATEGIES)
            order = STRATEGIES[offset:] + STRATEGIES[:offset]
            for strategy in order:
                start = time.monotonic()
                with execution_scope(args.timeout) as ledger:
                    try:
                        with tracing_context(enabled=False):
                            hits = await asyncio.wait_for(retrieval.search_knowledge(case["query"], strategy=strategy,
                                top_k=3, collection=name, split=False), args.timeout)
                        if not {str(h["id"]) for h in hits} <= set(expected):
                            raise ValueError("retrieved document not in isolated corpus")
                        rank, policy, status, error = ranking_metrics(hits, case["relevance"]), policy_probe(case, hits, catalog), "observed", None
                    except Exception as exc:
                        hits, status, error = [], "failed", type(exc).__name__
                        rank, policy = ranking_metrics([], case["relevance"]), policy_probe(case, [], catalog)
                        policy["outcome_matches_expected"] = False if case["policy_group"] else None
                    row = {"id": case["id"], "query": case["query"], "strategy": strategy,
                           "status": status, "error_class": error, "elapsed_seconds": time.monotonic() - start,
                           "hits": hits, "ranking": rank, "policy": policy, "execution": ledger.summary()}
                rows.append(row)
                with (args.output_dir / "observations.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                print(json.dumps({"case": case["id"], "strategy": strategy, "status": status}), flush=True)
        after = source_state()
        write_new(args.output_dir / "source-stability.json", {"unchanged": snapshot == after})
        if snapshot != after:
            raise ValueError("source changed during retrieval experiment")
        return {"measurement": "synthetic_dev_retrieval_and_oracle_policy_diagnostics", "rows": rows,
            "summary": metrics_summary(rows), "source_sha256": snapshot["source_sha256"],
            "collection": name, "requested_queries": len(cases) * len(STRATEGIES),
            "cost": None, "collection_preserved": True,
            "limits": ["one small development pass; not an independent held-out result",
                       "rotated strategy order limits order bias, not cache/network confounding",
                       "topic scope and world facts are oracle inputs; not Agent extraction",
                       "no generated answers, human ratings, chat calls or ticket writes",
                       "all requested failures retained; unknown pricing and partial provider usage"]}
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
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if not 0 < args.timeout <= 300:
        raise ValueError("bounded query timeout required")
    args.env_file, args.output_dir = args.env_file.resolve(), args.output_dir.resolve()
    docs, cases, catalog, manifest = load_bundle(ROOT / "data/rag_development")
    snapshot = source_state()
    if args.frozen_plan:
        frozen = json.loads(args.frozen_plan.read_text(encoding="utf-8"))
        if frozen["source_sha256"] != snapshot["source_sha256"] or frozen["dataset_sha256"] != manifest["dataset_sha256"]:
            raise ValueError("frozen retrieval source/dataset differs")
    elif not args.plan_only:
        raise ValueError("freeze plan before online retrieval")
    args.output_dir.mkdir(exist_ok=False, parents=True)
    plan = {**snapshot, "dataset_sha256": manifest["dataset_sha256"], "documents": len(docs), "cases": len(cases),
            "strategies": STRATEGIES, "top_k": 3, "query_count": len(cases) * 4,
            "order": "rotate four strategies by case index; one pass, no code/label edits while running",
            "split": "development only", "policy_scope": "oracle topic and world; no generated answers",
            "failure_rule": "retain failures; ranking zero, policy outcome comparison false"}
    write_new(args.output_dir / "frozen-plan.json", plan)
    with zipfile.ZipFile(args.output_dir / "source-snapshot.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for relative in snapshot["files"]:
            archive.write(ROOT / relative, relative)
    if args.plan_only:
        print(json.dumps({"event": "planned_no_env_or_network", "documents": len(docs), "queries": len(cases) * 4}))
        return
    # Preserve the absolute authorized paths before moving away from .env.
    with tempfile.TemporaryDirectory(prefix="mewhelp-rag-empty-") as temporary:
        old = Path.cwd()
        try:
            os.chdir(temporary)
            value = asyncio.run(measure(args, docs, cases, catalog, snapshot))
        finally:
            os.chdir(old)
    write_new(args.output_dir / "report.json", value)
    print(json.dumps({"event": "measured", "requested": value["requested_queries"], "summary": value["summary"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit("RAG development experiment failed: " + type(exc).__name__) from None
