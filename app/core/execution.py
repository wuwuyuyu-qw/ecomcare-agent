"""Request-scoped deadline and a usage ledger without prompts or credentials.

Monotonic deadlines never enter checkpoints. Resume gets a fresh HTTP deadline;
the task's logical tool/step budget remains in TaskContext.
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from anyio import CancelScope

from langchain_core.callbacks import AsyncCallbackHandler


@dataclass
class ExecutionScope:
    task_id: str
    request_id: str
    deadline: float
    started: float = field(default_factory=time.monotonic)
    model_calls: int = 0
    tool_calls: int = 0
    usage: list[dict] = field(default_factory=list)
    tool_events: list[dict] = field(default_factory=list)
    first_stream_event_ms: int | None = None
    first_visible_content_ms: int | None = None
    termination_reason: str = ""
    initial_context: dict = field(default_factory=dict, repr=False)

    def summary(self, prices: dict | None = None, currency: str = "CNY") -> dict:
        entries, inputs, outputs, measured, priced, cost = [], 0, 0, 0, 0, 0.0
        for entry in self.usage:
            known = entry.get("input_tokens") is not None and entry.get("output_tokens") is not None
            if known:
                measured += 1
                inputs += entry["input_tokens"]
                outputs += entry["output_tokens"]
            charge = _price(entry, (prices or {}).get(entry.get("model", "")))
            if charge is not None:
                priced += 1
                cost += charge
            entries.append({**entry, "cost": charge})
        missing_calls = max(0, self.model_calls - sum(e.get("kind") == "chat" for e in entries))
        count = len(entries)
        coverage_count = count + missing_calls
        complete = count > 0 and not missing_calls and measured == count and priced == count
        return {"task_id": self.task_id, "request_id": self.request_id,
                "elapsed_ms": int((time.monotonic() - self.started) * 1000),
                "first_stream_event_ms": self.first_stream_event_ms,
                "first_visible_content_ms": self.first_visible_content_ms,
                "stream_timing_scope": "server_generator_before_send; guarded_content_not_provider_first_token",
                "model_calls": self.model_calls, "tool_calls": self.tool_calls,
                "termination_reason": self.termination_reason, "usage": entries,
                "measured_input_tokens": inputs, "measured_output_tokens": outputs,
                "usage_coverage": measured / coverage_count if coverage_count else None,
                "price_coverage": priced / coverage_count if coverage_count else None,
                "unfinished_model_calls": missing_calls,
                "cost": round(cost, 8) if complete else None,
                "measured_partial_cost": round(cost, 8) if priced else None,
                "currency": currency, "cost_complete": complete,
                "cost_kind": "estimate_from_reported_usage_and_configured_prices",
                "scope": "foreground_provider_calls_observed_by_this_request",
                "excluded": ["background_summary", "unreported_provider_internal_retries"],
                "tool_events": list(self.tool_events)}


_CURRENT: ContextVar[ExecutionScope | None] = ContextVar("mewhelp_execution", default=None)


def current_execution() -> ExecutionScope | None:
    return _CURRENT.get()


async def finish_cancellation(operation):
    """Allow bounded local cleanup inside ASGI's cancelled task group.

    Starlette cancels the enclosing AnyIO scope on disconnect. Without a
    shield every awaited checkpoint/SQL cleanup is cancelled again immediately.
    The caller supplies its own small timeout; no provider work belongs here.
    """
    with CancelScope(shield=True):
        return await operation


def remaining_seconds() -> float | None:
    scope = current_execution()
    return max(0.0, scope.deadline - time.monotonic()) if scope else None


@contextmanager
def execution_scope(timeout: float, task_id: str | None = None):
    scope = ExecutionScope(task_id=task_id or uuid4().hex, request_id=uuid4().hex,
                           deadline=time.monotonic() + timeout)
    token = _CURRENT.set(scope)
    try:
        yield scope
    finally:
        _CURRENT.reset(token)


@contextmanager
def detached_execution():
    """A background task must not inherit the customer's foreground ledger."""
    token = _CURRENT.set(None)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_tool_event(name: str, *, ok: bool, error_code: str | None = None,
                      duration_ms: int = 0, retry_count: int = 0) -> None:
    scope = current_execution()
    if scope:
        scope.tool_calls += 1
        scope.tool_events.append({"tool": name[:128], "ok": bool(ok), "error_code": error_code,
                                  "duration_ms": duration_ms, "retry_count": retry_count})


def record_stream_event(kind: str) -> None:
    scope = current_execution()
    if scope is None:
        return
    elapsed = int((time.monotonic() - scope.started) * 1000)
    if scope.first_stream_event_ms is None:
        scope.first_stream_event_ms = elapsed
    if kind == "delta" and scope.first_visible_content_ms is None:
        scope.first_visible_content_ms = elapsed


def _token(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def record_usage(stage: str, model: str, usage: dict | None, *, failed: bool = False,
                 kind: str = "sdk") -> None:
    scope = current_execution()
    if scope is None:
        return
    data = usage or {}
    details = data.get("input_token_details") or data.get("prompt_tokens_details") or {}
    scope.usage.append({"stage": stage, "model": model, "kind": kind,
                        "input_tokens": _token(data.get("input_tokens", data.get("prompt_tokens"))),
                        "output_tokens": _token(data.get("output_tokens", data.get("completion_tokens"))),
                        "cached_input_tokens": _token(details.get("cache_read", details.get("cached_tokens"))),
                        "failed": failed})


def _price(entry: dict, rates: dict | None) -> float | None:
    if not isinstance(rates, dict) or entry.get("failed"):
        return None
    inp, out = entry.get("input_tokens"), entry.get("output_tokens")
    if inp is None or out is None:
        return None
    raw_cached = entry.get("cached_input_tokens")
    if raw_cached is None and "cached_input_per_million" in rates:
        return None  # Discounted token count is unknown, not a measured cache miss.
    cached = raw_cached or 0
    if cached > inp:
        return None
    keys = ["input_per_million", "output_per_million"]
    if cached:
        keys.append("cached_input_per_million")
    if any(not isinstance(rates.get(k), (int, float)) or isinstance(rates.get(k), bool)
           or not math.isfinite(rates[k]) or rates[k] < 0 for k in keys):
        return None
    return ((inp - cached) * rates["input_per_million"] + out * rates["output_per_million"]
            + cached * rates.get("cached_input_per_million", 0)) / 1_000_000


class UsageCollector(AsyncCallbackHandler):
    """Actual chat usage across graph nodes; structured retries count separately."""
    def __init__(self, scope: ExecutionScope):
        self.scope = scope
        self.runs: dict[str, dict] = {}

    async def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        info = metadata or {}
        self.scope.model_calls += 1
        self.runs[str(run_id)] = {"stage": info.get("langgraph_node") or "chat",
                                  "model": info.get("ls_model_name") or "unknown"}

    def _record(self, info, usage, failed=False):
        token = _CURRENT.set(self.scope)
        try:
            record_usage(info["stage"], info["model"], usage, failed=failed, kind="chat")
        finally:
            _CURRENT.reset(token)

    async def on_llm_end(self, response, *, run_id, **kwargs):
        info = self.runs.pop(str(run_id), {"stage": "chat", "model": "unknown"})
        usage = None
        for generations in response.generations:
            for generation in generations:
                msg = getattr(generation, "message", None)
                if msg is not None:
                    usage = getattr(msg, "usage_metadata", None) or usage
                    info["model"] = (getattr(msg, "response_metadata", None) or {}).get("model_name") or info["model"]
        self._record(info, usage or (response.llm_output or {}).get("token_usage"))

    async def on_llm_error(self, error, *, run_id, **kwargs):
        info = self.runs.pop(str(run_id), {"stage": "chat", "model": "unknown"})
        self._record(info, None, failed=True)
