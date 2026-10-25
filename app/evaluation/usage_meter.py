"""Evaluation-only per-HTTP-response usage observer, before LangChain aggregation.

OpenAI CompletionUsage fields are request totals (cache is a subset of input),
not chunk deltas. Retain the last monotonic snapshot; identical reports are
idempotent. Never sum repeated snapshots or carry state across responses.
Only usage metadata is persisted, never prompts, generated text or credentials.
"""
from __future__ import annotations

import sqlite3

EXTENSION = "mewhelp_eval_usage_v2"
VERSION = "eval-usage-v2"


def initialize(conn):
    conn.execute("CREATE TABLE usage_events (attempt_id INTEGER, source TEXT, model TEXT, input_tokens INTEGER, output_tokens INTEGER, cache_read INTEGER, total_tokens INTEGER, error TEXT)")
    conn.execute("CREATE TABLE provider_attempt_meta (attempt_id INTEGER PRIMARY KEY, retry_index INTEGER, outcome TEXT, usage_error TEXT, usage_source TEXT)")
    conn.execute("CREATE TABLE provider_denials (kind TEXT, retry_index INTEGER, reason TEXT)")


def attempt(path, key, request):
    if key is None:
        return
    raw = request.headers.get("x-stainless-retry-count")
    retry = int(raw) if raw is not None and raw.isdecimal() else None
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO provider_attempt_meta(attempt_id,retry_index) VALUES (?,?)", (key, retry))


class Observation:
    def __init__(self, path, key, status):
        self.path, self.key, self.status = path, key, status
        self.sdk_owned = self.finished = self.raw_eof = self.decoding = self.sdk_done = False
        self.previous = None
        self.source = None
        self.error = None

    def record(self, payload, source):
        if self.finished or not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
            return
        # The typed SDK object is authoritative. Decoded SSE is only for clients
        # that do not use the SDK; do not record both layers for the same stream.
        if self.sdk_owned and source != "openai_sdk":
            return
        value = payload["usage"]
        incoming, outgoing, total = (value.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
        details = value.get("prompt_tokens_details") or {}
        cache = value.get("prompt_cache_hit_tokens")
        nested_cache = details.get("cached_tokens") if isinstance(details, dict) else None
        if cache is None:
            cache = nested_cache
        model = str(payload.get("model", ""))[:128]
        error = None
        if (type(incoming) is not int or not 0 <= incoming < 2**63 or type(outgoing) is not int or not 0 <= outgoing < 2**63
                or (cache is not None and (type(cache) is not int or not 0 <= cache <= incoming))
                or (total is not None and (type(total) is not int or total != incoming + outgoing))):
            error = "invalid_request_total_usage"
        elif nested_cache is not None and nested_cache != cache:
            error = "conflicting_cache_fields"
        current = (model, incoming, outgoing, cache)
        if not error and self.previous is not None:
            old_model, old_in, old_out, old_cache = self.previous
            if model != old_model or incoming != old_in or cache != old_cache or outgoing < old_out:
                # Prompt/cache counts cannot grow/change after a request starts.
                # Do not interpret a reset or a delta-only stream as a huge bill.
                error = "ambiguous_or_nonmonotonic_usage"
        self.error = self.error or error
        self.source = source
        if not error:
            self.previous = current
        # Invalid types are represented by nulls; no exception hides stream data.
        integer = lambda v: v if type(v) is int and -(2**63) <= v < 2**63 else None
        with sqlite3.connect(self.path) as conn:
            conn.execute("INSERT INTO usage_events VALUES (?,?,?,?,?,?,?,?)", (self.key, source, model,
                integer(incoming), integer(outgoing), integer(cache), integer(total), error))
            # Canonical, one snapshot per physical request; events remain above.
            conn.execute("DELETE FROM provider_usage WHERE attempt_id=?", (self.key,))
            conn.execute("INSERT INTO provider_usage VALUES (?,?,?,?,?,?)", (self.key, model,
                integer(incoming), integer(outgoing), integer(cache), int(not self.error)))
            conn.execute("UPDATE provider_attempt_meta SET usage_error=?,usage_source=? WHERE attempt_id=?",
                         (self.error, source, self.key))

    def finish(self, complete, outcome=None):
        if self.finished:
            return
        self.finished = True
        with sqlite3.connect(self.path) as conn:
            conn.execute("UPDATE provider_attempt_meta SET outcome=? WHERE attempt_id=?",
                         (outcome or ("complete" if complete else "interrupted"), self.key))
        from app.evaluation import money_guard
        money_guard.settle(self.path, self.key, status=self.status, complete=complete and not self.error)


def attach(response, path, key):
    """Decoded SSE fallback for plain HTTPX; SDK ownership supersedes it."""
    import httpx
    from openai._streaming import SSEDecoder
    state = Observation(path, key, response.status_code)
    response.extensions[EXTENSION] = state
    original = response.stream

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            async for chunk in original:
                yield chunk
            state.raw_eof = True

        async def aclose(self):
            try:
                await original.aclose()
            finally:
                if not state.sdk_owned and not (state.raw_eof and state.decoding):
                    state.finish(False, "closed_before_stream_completion")

    response.stream = Stream()
    decoded = response.aiter_bytes

    async def observed_bytes(*args, **kwargs):
        state.decoding = True
        decoder = SSEDecoder()
        done = False
        pending = b""
        try:
            async for chunk in decoded(*args, **kwargs):
                if not state.sdk_owned:
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        sse = decoder.decode(line.rstrip(b"\r").decode("utf8"))
                        if sse is None:
                            continue
                        if sse.data.startswith("[DONE]"):
                            done = True
                        else:
                            try:
                                state.record(sse.json(), "httpx_decoded_sse")
                            except (ValueError, UnicodeError):
                                state.error = "invalid_sse_usage_observation"
                    if len(pending) > 1_000_000:
                        state.error = "oversized_sse_observation"
                        pending = b""
                yield chunk
        finally:
            state.decoding = False
            if not state.sdk_owned:
                state.finish(done and state.raw_eof)

    response.aiter_bytes = observed_bytes
    return state


def install_sdk_observer():
    """Hook typed SDK chunks without changing yields, prompts or model options.

    The response carries the observer, so hooks can be installed once and do not
    retain a process-global ledger/model or mix concurrent requests. The private
    SDK boundary is tested against the installed version by offline integration.
    """
    from openai import AsyncStream
    if getattr(AsyncStream, "_mewhelp_eval_usage_v2", False):
        return
    original_init, original_stream, original_close = AsyncStream.__init__, AsyncStream.__stream__, AsyncStream.close
    original_events = AsyncStream._iter_events

    def init(self, *args, **kwargs):
        response = kwargs.get("response")
        state = response.extensions.get(EXTENSION) if response is not None else None
        if state:
            state.sdk_owned = True
        original_init(self, *args, **kwargs)

    async def stream(self):
        state = self.response.extensions.get(EXTENSION)
        complete = False
        try:
            async for chunk in original_stream(self):
                if state and getattr(chunk, "usage", None) is not None:
                    state.record({"model": getattr(chunk, "model", ""), "usage": chunk.usage.model_dump()}, "openai_sdk")
                yield chunk
            complete = state.sdk_done if state else True
        finally:
            if state:
                state.finish(complete)

    async def events(self):
        async for event in original_events(self):
            state = self.response.extensions.get(EXTENSION)
            if state and event.data.startswith("[DONE]"):
                state.sdk_done = True
            yield event

    async def close(self):
        try:
            await original_close(self)
        finally:
            state = self.response.extensions.get(EXTENSION)
            if state:
                state.finish(False, "sdk_closed_before_completion")

    AsyncStream.__init__, AsyncStream.__stream__, AsyncStream.close = init, stream, close
    AsyncStream._iter_events = events
    AsyncStream._mewhelp_eval_usage_v2 = True


def snapshot(path):
    """Physical attempts and known partial tokens, never logical model callbacks."""
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT a.kind,m.retry_index,m.outcome,m.usage_error,m.usage_source,u.input_tokens,u.output_tokens,u.cache_read "
                            "FROM attempts a LEFT JOIN provider_attempt_meta m ON m.attempt_id=a.id "
                            "LEFT JOIN provider_usage u ON u.attempt_id=a.id ORDER BY a.id").fetchall()
        denied = conn.execute("SELECT kind,retry_index FROM provider_denials").fetchall()
    result = {}
    for kind in ("chat", "retrieval"):
        selected = [r for r in rows if r[0] == kind]
        observed = [r for r in selected if not r[3] and r[5] is not None and r[6] is not None]
        valid = [r for r in selected if r[2] == "complete" and not r[3] and r[5] is not None and (kind != "chat" or r[6] is not None)]
        result[kind] = {"http_requests": len(selected), "retry_http_requests": sum((r[1] or 0) > 0 for r in selected),
            "retry_metadata_unknown_requests": sum(r[1] is None for r in selected),
            "known_partial_input_tokens": sum(r[5] for r in valid), "known_partial_output_tokens": sum(r[6] or 0 for r in valid),
            "known_partial_cache_hit_tokens": sum(r[7] or 0 for r in valid),
            "usage_unknown_requests": len(selected)-len(valid), "cache_unknown_requests": sum(r[7] is None for r in valid),
            "sources": sorted({r[4] for r in selected if r[4]})}
        result[kind].update(locally_blocked_sends=sum(r[0] == kind for r in denied),
            locally_blocked_retry_sends=sum(r[0] == kind and (r[1] or 0) > 0 for r in denied),
            observed_latest_input_tokens=sum(r[5] for r in observed),
            observed_latest_output_tokens=sum(r[6] for r in observed))
        result[kind]["input_tokens"] = result[kind]["known_partial_input_tokens"] if len(valid) == len(selected) else None
        result[kind]["output_tokens"] = result[kind]["known_partial_output_tokens"] if len(valid) == len(selected) else None
        result[kind]["cache_hit_tokens"] = result[kind]["known_partial_cache_hit_tokens"] if len(valid) == len(selected) and not result[kind]["cache_unknown_requests"] else None
    return result
