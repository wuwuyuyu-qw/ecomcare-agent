"""Pure lifecycle observation guards; no graph startup or real HTTP/DB/model."""
from copy import deepcopy
import json

import httpx
import pytest

from scripts.probe_agent_lifecycle import (
    _close_at_headers, _fresh_order, _json_request, _resume_checks,
    _resume_stream, _stream_projection,
)

NOW = "2026-10-05T00:00:01Z"
SCOPE = {"user_id": "customer_demo", "conversation_id": 23, "task_id": "synthetic-task", "active_order_id": "3003"}


def context():
    return {**SCOPE, "actions": [], "pending_actions": [], "verified_facts": {
        name: {"value": value, "source": "query_order", "scope": deepcopy(SCOPE),
               "observed_at": NOW, "evidence_id": "synthetic-" + name}
        for name, value in (("order_owned", True), ("order_id", "3003"))}}


def frames():
    return [{"event": "execution_summary", "tool_calls": 1,
             "tool_events": [{"tool": "query_order", "ok": True}]},
            {"event": "done", "conversation_id": 23, "task_context": context()}]


def check(value, complete=True):
    return _resume_checks(value, cid=23, identity="customer_demo", received=NOW, complete=complete)


def test_resume_requires_actual_complete_stream_and_fresh_scoped_facts():
    assert all(value is True for value in check(frames()).values())
    partial = check(frames(), complete=False)
    assert partial["sse_complete"] is False and partial["selection_interrupt_ended"] is None


@pytest.mark.parametrize("mutate", [
    lambda ctx: ctx["verified_facts"]["order_owned"].update(value=1),
    lambda ctx: ctx["verified_facts"]["order_id"].update(value="3001"),
    lambda ctx: ctx["verified_facts"]["order_id"]["scope"].update(user_id="customer_other"),
    lambda ctx: ctx["verified_facts"]["order_id"].update(source="user_message"),
    lambda ctx: ctx["verified_facts"]["order_id"].update(observed_at="2026-10-04T23:00:00Z"),
    lambda ctx: ctx["verified_facts"]["order_id"].update(observed_at=None),
])
def test_resume_rejects_wrong_identity_claims_coercions_and_stale_facts(mutate):
    value = context()
    mutate(value)
    assert _fresh_order(value, cid=23, identity="customer_demo", received=NOW) is False


def test_missing_sse_context_or_execution_is_unknown():
    value = frames()
    value[-1].pop("task_context")
    value.pop(0)
    observed = check(value)
    assert observed["fresh_scoped_order_facts"] is None and observed["no_write_tool_calls"] is None


def test_stream_done_cannot_conceal_resume_error_or_write_invocation():
    value = frames() + [{"event": "error"}, {"event": "tool", "name": "create_ticket", "ok": False}]
    observed = check(value)
    assert observed["no_sse_error"] is False and observed["no_write_tool_calls"] is False


def test_sse_projection_withholds_generated_text_and_order_menu():
    assert _stream_projection({"delta": "synthetic generated prose"}) == {"event": "unrecorded_content"}
    assert _stream_projection({"event": "interrupt", "kind": "select_order", "orders": [{"order_id": "3003"}]}) == {"event": "interrupt", "kind": "select_order"}


async def test_client_close_at_headers_does_not_claim_server_cancellation():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"data: [DONE]\n\n"))
    async with httpx.AsyncClient(base_url="http://127.0.0.1:9", transport=transport) as client:
        result = await _close_at_headers(client, 23)
    assert result["client_stream_closed"] is True and result["body_events_consumed"] == 0
    assert result["server_cancelled"] is None and result["server_execution_started"] is None
    assert result["server_completed_before_close"] is None


async def test_resume_parser_requires_done_sentinel_and_records_actual_context():
    raw = "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames()) + "data: [DONE]\n\n"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=raw))
    async with httpx.AsyncClient(base_url="http://127.0.0.1:9", transport=transport) as client:
        result = await _resume_stream(client, 23)
    assert result["complete"] is True and result["frames"][-1]["task_context"] == context()


async def test_non_200_error_body_is_never_used_as_lifecycle_evidence():
    transport = httpx.MockTransport(lambda request: httpx.Response(504, text="synthetic private vendor detail"))
    async with httpx.AsyncClient(base_url="http://127.0.0.1:9", transport=transport) as client:
        status, value = await _json_request(client, "POST", "/api/agent", {"conversation_id": 23})
    assert status == 504 and value is None
