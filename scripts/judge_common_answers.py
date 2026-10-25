"""Explicitly call OpenEvals on a common-evidence blind packet; local reports only.

Uses an optional locked eval dependency and existing provider configuration.
No model calls on import, --help, or comparison-only mode. Does not import
application Settings, load a second .env, or upload records to LangSmith.
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.evaluation.common_judge import PROTOCOL, protocol_digest, create_evaluator, judge_inputs, rate_with_format_retry, compare_ratings
from app.evaluation.task_review import read_json, _encode, _write_new


def create_judge_client(config, *, model, timeout, callbacks):
    """Actual BaseChatModel for OpenEvals, using the proven FC adapter.

    Passing a RunnableBinding instead of a BaseChatModel makes OpenEvals reject
    the client before any provider request. Keep callbacks on the client itself.
    Caller closes both owned HTTP clients.
    """
    from langchain_openai import ChatOpenAI
    class FunctionCallingJudge(ChatOpenAI):
        def with_structured_output(self, schema=None, **kwargs):
            return super().with_structured_output(schema, **{**kwargs, "method": "function_calling"})
    return FunctionCallingJudge(model=model, base_url=config["CHAT_BASE_URL"], api_key=config["CHAT_API_KEY"],
        temperature=0, max_tokens=2500, timeout=timeout, max_retries=0, streaming=True, stream_usage=True,
        extra_body={"thinking": {"type": "disabled"}}, callbacks=callbacks)


async def judge_packet(packet: dict, digest: str, args) -> dict:
    from dotenv import dotenv_values
    from langchain_core.callbacks import UsageMetadataCallbackHandler
    from langsmith import tracing_context

    config = dotenv_values(args.env_file, interpolate=False)
    model = args.judge_model or config.get("CHAT_MODEL")
    if not all(isinstance(v, str) and v for v in (model, config.get("CHAT_BASE_URL"), config.get("CHAT_API_KEY"))):
        raise ValueError("explicit judge model/provider configuration required")
    usage = UsageMetadataCallbackHandler()
    # Reuse the course's documented streaming function-call transport choice:
    # this provider intermittently omits tool_calls on non-streaming requests.
    # Only the transport changes; no answer-text-to-rating fallback is allowed.
    client = create_judge_client(config, model=model, timeout=args.timeout, callbacks=[usage])
    evaluator = create_evaluator(client)
    results = []
    try:
        for index, entry in enumerate(packet["entries"], 1):
            result = {"review_id": entry["review_id"], "status": "not_reviewable", "rating": None}
            if entry["recording_status"] == "reviewable":
                with tracing_context(enabled=False):
                    result.update(await rate_with_format_retry(evaluator, entry, args.timeout))
            results.append(result)
            print(json.dumps({"event": "judge_entry", "index": index, "total": len(packet["entries"]),
                              "status": result["status"]}), flush=True)
    finally:
        await client.root_async_client.close()
        client.root_client.close()
    return {"schema_version": PROTOCOL, "mode": "model_answer_ratings", "packet_sha256": digest,
        "protocol_sha256": protocol_digest(), "rated_at": datetime.now(timezone.utc).isoformat(),
        "judge": {"model": model, "temperature": 0, "openevals_version": importlib.metadata.version("openevals"),
                  "independent_human": False, "provider_retries": 0, "max_format_attempts": 2,
                  "structured_transport": "streaming_function_calling"},
        "usage": usage.usage_metadata, "cost": None, "cost_reason": "no verified judge pricing; failed calls may omit usage",
        "entries": results, "scope": "retrospective visible-answer model assessment; not human or runtime task success"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    judge = commands.add_parser("judge")
    judge.add_argument("--packet", type=Path, required=True)
    judge.add_argument("--env-file", type=Path, required=True)
    judge.add_argument("--judge-model")
    judge.add_argument("--timeout", type=float, default=45)
    judge.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--packet", type=Path, required=True)
    compare.add_argument("--bindings", type=Path, required=True)
    compare.add_argument("--ratings", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output file")
    packet, digest = read_json(args.packet)
    if args.mode == "judge":
        if not math.isfinite(args.timeout) or not 0 < args.timeout <= 120:
            raise ValueError("positive bounded timeout required")
        if len(packet.get("entries", [])) > 120 or not packet.get("entries"):
            raise ValueError("select a pilot packet of 1 to 120 entries")
        for entry in packet["entries"]:
            if entry.get("recording_status") == "reviewable":
                judge_inputs(entry)  # Validate every entry before provider calls.
        value = asyncio.run(judge_packet(packet, digest, args))
    else:
        private, _ = read_json(args.bindings)
        ratings, _ = read_json(args.ratings)
        value = compare_ratings(packet, private, digest, ratings)
    _write_new(args.output, _encode(value))
    print(json.dumps({"mode": value["mode"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit("common judge input/environment rejected: " + type(exc).__name__) from None
