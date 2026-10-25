"""Actual installed SDK, synthetic streams/MockTransport, no external requests."""
import asyncio
from decimal import Decimal
import gzip
import json
import sqlite3

import httpx
from openai import AsyncOpenAI, APIConnectionError
import pytest

from app.evaluation import dev_batch as batch, money_guard as money, usage_meter as meter
from app.evaluation.stage1 import ProviderBudgetExceeded


def usage(incoming=874, outgoing=50, cache=640, **extra):
    return {"prompt_tokens": incoming, "completion_tokens": outgoing, "total_tokens": incoming+outgoing,
            "prompt_cache_hit_tokens": cache, **extra}


def event(value=None, text=None):
    return {"id": "offline", "object": "chat.completion.chunk", "created": 0, "model": "deepseek-flash",
            "choices": [] if text is None else [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            "usage": value}


def wire(values, done=True):
    return b"".join(b"data: "+json.dumps(v).encode()+b"\n\n" for v in values)+(b"data: [DONE]\n\n" if done else b"")


class Body(httpx.AsyncByteStream):
    def __init__(self, data, failure=None):
        self.data, self.failure = data, failure

    async def __aiter__(self):
        for offset in range(0, len(self.data), 17):
            yield self.data[offset:offset+17]
        if self.failure:
            raise self.failure


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path/"budget.sqlite"
    batch.initialize(path)
    monkeypatch.setattr(money, "query_balance", lambda key: Decimal("15.31"))
    money.activate(path, "offline-placeholder")
    monkeypatch.setattr(httpx.Client, "send", httpx.Client.send)
    monkeypatch.setattr(httpx.AsyncClient, "send", httpx.AsyncClient.send)
    batch.install_http_budget(path, {"api.deepseek.com", "api.siliconflow.cn"})
    return path


def client_for(handler, retries=0):
    return AsyncOpenAI(api_key="offline-placeholder", base_url="https://api.deepseek.com/v1", max_retries=retries,
                       http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def consume(client):
    stream = await client.chat.completions.create(model="deepseek-v4-flash", max_tokens=10000, stream=True,
        stream_options={"include_usage": True}, messages=[{"role": "user", "content": "offline synthetic"}])
    async with stream:
        return [chunk async for chunk in stream]


@pytest.mark.parametrize("compressed", [False, True])
async def test_two_consecutive_sdk_streams_tail_usage_and_cache_are_per_request(ledger, compressed):
    numbers = [usage(382,31,128), usage(874,50,640)]
    def handler(request):
        data = wire([event(text="private-prose-never-record"),event(numbers.pop(0))])
        return httpx.Response(200, headers={"content-type":"text/event-stream", **({"content-encoding":"gzip"} if compressed else {})},
                              stream=Body(gzip.compress(data) if compressed else data))
    async with client_for(handler) as client:
        await consume(client)
        assert money.snapshot(ledger)["unknown_cost_requests"] == 0
        await consume(client)
    state = money.snapshot(ledger)
    assert state["total_cost_upper_cny"] == "0.00165472"
    assert state["provider_usage"]["chat"] == {
        "http_requests":2,"retry_http_requests":0,"retry_metadata_unknown_requests":0,
        "known_partial_input_tokens":1256,"known_partial_output_tokens":81,"known_partial_cache_hit_tokens":768,
        "usage_unknown_requests":0,"cache_unknown_requests":0,"sources":["openai_sdk"],
        "input_tokens":1256,"output_tokens":81,"cache_hit_tokens":768,
        "locally_blocked_sends":0,"locally_blocked_retry_sends":0,
        "observed_latest_input_tokens":1256,"observed_latest_output_tokens":81}
    assert not state["blocks"]
    assert b"private-prose-never-record" not in ledger.read_bytes()


@pytest.mark.parametrize("values,expected", [([usage(),usage()],50), ([usage(outgoing=20),usage(outgoing=50)],50)])
async def test_duplicate_and_increasing_cumulative_output_are_not_summed(ledger, values, expected):
    async with client_for(lambda r:httpx.Response(200,stream=Body(wire([event(v) for v in values])))) as client:
        await consume(client)
    state = money.snapshot(ledger)
    assert state["provider_usage"]["chat"]["output_tokens"] == expected
    assert state["provider_usage"]["chat"]["input_tokens"] == 874
    with sqlite3.connect(ledger) as conn:
        assert conn.execute("SELECT COUNT(*) FROM usage_events").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM provider_usage").fetchone()[0] == 1


async def test_repeated_large_valid_snapshot_does_not_falsely_trip_fee_or_token_alarm(ledger):
    repeated = [event(usage(outgoing=4000)) for _ in range(3)]
    async with client_for(lambda r:httpx.Response(200,stream=Body(wire(repeated)))) as client:
        await consume(client)
    state=money.snapshot(ledger)
    assert state["provider_usage"]["chat"]["output_tokens"] == 4000
    assert state["total_cost_upper_cny"] == "0.0324936"
    assert not state["blocks"]


@pytest.mark.parametrize("values", [
    [usage(),usage(outgoing=10)], [usage(),usage(incoming=900)],
    [usage(prompt_tokens_details={"cached_tokens":500})],
    [usage(total_tokens=9999)], [usage(cache=900)],
    [usage(incoming=10**100)],
])
async def test_ambiguous_totals_are_unknown_not_zero_or_a_false_large_bill(ledger, values):
    async with client_for(lambda r:httpx.Response(200,stream=Body(wire([event(v) for v in values])))) as client:
        await consume(client)
    state = money.snapshot(ledger)
    assert state["total_cost_upper_cny"] is None and state["unknown_cost_cny"] is None
    assert state["unknown_cost_requests"] == 1
    assert "abnormally_large_observed_token_usage" not in state["blocks"]
    with pytest.raises(ProviderBudgetExceeded): batch.reserve(ledger, "chat", 10000, 1)


@pytest.mark.parametrize("value,cache,upper", [
    ({"prompt_tokens":1000,"completion_tokens":100,"prompt_tokens_details":{"cached_tokens":500},
      "completion_tokens_details":{"reasoning_tokens":70}},500,"0.00182"),
    ({"prompt_tokens":1000,"completion_tokens":100},None,"0.0028"),
])
async def test_cache_subset_and_reasoning_already_in_output(ledger, value, cache, upper):
    async with client_for(lambda r:httpx.Response(200,stream=Body(wire([event(value)])))) as client:
        await consume(client)
    state=money.snapshot(ledger)
    assert state["total_cost_upper_cny"] == upper
    assert state["provider_usage"]["chat"]["cache_hit_tokens"] == cache
    assert state["provider_usage"]["chat"]["output_tokens"] == 100


@pytest.mark.parametrize("failure", [httpx.ReadError("offline"),httpx.ReadTimeout("offline"),asyncio.CancelledError()])
async def test_interruption_timeout_cancellation_even_after_usage_are_unknown(ledger, failure):
    async with client_for(lambda r:httpx.Response(200,stream=Body(wire([event(usage())],done=False),failure))) as client:
        with pytest.raises(type(failure)):
            await consume(client)
    state=money.snapshot(ledger)
    assert state["unknown_cost_requests"] == 1 and state["total_cost_upper_cny"] is None
    assert state["provider_usage"]["chat"]["input_tokens"] is None
    assert state["provider_usage"]["chat"]["observed_latest_input_tokens"] == 874
    with pytest.raises(ProviderBudgetExceeded): batch.reserve(ledger,"retrieval",0,1)


@pytest.mark.parametrize("values,done", [([event(text="no usage")],True),([event(usage())],False)])
async def test_missing_usage_or_terminal_marker_is_missing_observation(ledger, values, done):
    async with client_for(lambda r:httpx.Response(200,stream=Body(wire(values,done)))) as client:
        await consume(client)
    state=money.snapshot(ledger)
    assert state["unknown_cost_requests"] == 1 and state["total_cost_upper_cny"] is None


async def test_close_without_consumption_is_unknown_and_blocks(ledger):
    async with client_for(lambda r:httpx.Response(200,stream=Body(wire([event(usage())])))) as client:
        stream=await client.chat.completions.create(model="deepseek-v4-flash",max_tokens=10000,stream=True,messages=[])
        await stream.close()
    assert money.snapshot(ledger)["unknown_cost_requests"] == 1
    with pytest.raises(ProviderBudgetExceeded): batch.reserve(ledger,"chat",10000,1)


async def test_unknown_first_send_prevents_sdk_retry_from_reaching_provider(ledger, monkeypatch):
    sent=[]
    def handler(request):
        sent.append(True)
        raise httpx.ConnectError("offline")
    async with client_for(handler,retries=1) as client:
        monkeypatch.setattr(client,"_calculate_retry_timeout",lambda *args:0)
        with pytest.raises(APIConnectionError): await consume(client)
    assert len(sent)==1
    state=money.snapshot(ledger)
    assert state["provider_usage"]["chat"]["http_requests"] == 1
    assert state["provider_usage"]["chat"]["retry_http_requests"] == 0
    assert state["provider_usage"]["chat"]["locally_blocked_retry_sends"] == 1
    assert state["unknown_cost_requests"] == 1


async def test_actual_sdk_retry_counts_as_another_http_request_without_money_policy(tmp_path, monkeypatch):
    path=tmp_path/"no-money.sqlite"; batch.initialize(path)
    monkeypatch.setattr(httpx.Client,"send",httpx.Client.send)
    monkeypatch.setattr(httpx.AsyncClient,"send",httpx.AsyncClient.send)
    batch.install_http_budget(path,{"api.deepseek.com"})
    sent=[]
    def handler(request):
        sent.append(request.headers["x-stainless-retry-count"])
        if len(sent)==1: raise httpx.ConnectError("offline")
        return httpx.Response(200,stream=Body(wire([event(usage())])))
    async with client_for(handler,retries=1) as client:
        monkeypatch.setattr(client,"_calculate_retry_timeout",lambda *args:0)
        await consume(client)
    result=meter.snapshot(path)["chat"]
    assert sent == ["0","1"]
    assert result["http_requests"] == 2 and result["retry_http_requests"] == 1
    assert result["usage_unknown_requests"] == 1 and result["input_tokens"] is None
    assert result["known_partial_input_tokens"] == 874


async def test_sdk_and_production_totals_agree_across_reused_model_calls(ledger):
    from app.core.llm import _CumulativeUsageChatOpenAI
    values=[usage(400,30,100),usage(900,70,200)]
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,stream=Body(wire([event(text="ok"),event(values.pop(0))]))))) as transport:
        model=_CumulativeUsageChatOpenAI(model="deepseek-v4-flash",api_key="offline-placeholder",base_url="https://api.deepseek.com/v1",
            max_tokens=10000,streaming=True,stream_usage=True,http_async_client=transport)
        reported=[]
        for _ in range(2):
            chunks=[c async for c in model.astream("offline")]
            reported.append(sum((c.usage_metadata or {}).get("input_tokens",0) for c in chunks))
    assert reported == [400,900]
    assert meter.snapshot(ledger)["chat"]["input_tokens"] == 1300
