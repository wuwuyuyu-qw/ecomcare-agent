"""No-network simulations for approved CNY warnings and unresolved paid usage."""
from decimal import Decimal
import json
import sqlite3
import time

import httpx
import pytest

from app.evaluation import dev_batch as batch, money_guard as money
from app.evaluation.stage1 import ProviderBudgetExceeded


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    path = tmp_path/"budget.sqlite"
    batch.initialize(path)
    monkeypatch.setattr(money, "query_balance", lambda key: Decimal("15.31"))
    money.activate(path, "offline-placeholder")
    return path


def reported(path, key, incoming=1000, outgoing=100, cache=500, model="deepseek-flash"):
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO provider_usage VALUES (?,?,?,?,?,?)", (key, model, incoming, outgoing, cache, 1))


def test_real_usage_is_charged_conservatively_and_next_request_can_run(ledger):
    key = batch.reserve(ledger, "chat", 10000, 1000)
    reported(ledger, key)
    money.settle(ledger, key, status=200, complete=True)
    state = money.snapshot(ledger)
    assert Decimal(state["measured_peak_cost_upper_cny"]) == Decimal("0.00182")
    assert state["unknown_cost_requests"] == 0
    batch.reserve(ledger, "retrieval", 0, 1000)


@pytest.mark.parametrize("status,complete,usage", [(None,False,False),(200,True,False),(200,False,True),(500,True,True)])
def test_missing_failed_or_incomplete_paid_usage_blocks_all_next_requests(ledger, status, complete, usage):
    key = batch.reserve(ledger, "chat", 10000, 1000)
    if usage: reported(ledger, key)
    money.settle(ledger, key, status=status, complete=complete)
    for kind in ("chat", "retrieval"):
        with pytest.raises(ProviderBudgetExceeded): batch.reserve(ledger, kind, 1, 1)
    assert money.snapshot(ledger)["unknown_cost_requests"] == 1


def test_pending_paid_request_cannot_be_assumed_zero(ledger):
    batch.reserve(ledger, "chat", 10000, 1000)
    with pytest.raises(ProviderBudgetExceeded): batch.reserve(ledger, "chat", 10000, 1000)
    assert "paid_request_cost_unresolved" in money.snapshot(ledger)["blocks"]


@pytest.mark.parametrize("balance", [Decimal("2.99"), Decimal("13.31")])
def test_low_balance_or_account_drop_stops_before_new_sends(ledger, monkeypatch, balance):
    monkeypatch.setattr(money, "query_balance", lambda key: balance)
    money.refresh(ledger, "offline-placeholder")
    with pytest.raises(ProviderBudgetExceeded): batch.reserve(ledger, "chat", 10000, 1000)
    assert money.snapshot(ledger)["blocks"]


def test_2_cny_warning_and_inflight_headroom(ledger):
    with sqlite3.connect(ledger) as conn:
        conn.execute("INSERT INTO money_requests(attempt_id,state,cost_upper) VALUES (999,'settled','1.8')")
    with pytest.raises(ProviderBudgetExceeded): batch.reserve(ledger, "chat", 10000, 1000)
    assert "insufficient_fee_or_balance_headroom" in money.snapshot(ledger)["blocks"]


def test_exact_2_cny_warning_also_blocks_free_retrieval(ledger):
    with sqlite3.connect(ledger) as conn:
        conn.execute("INSERT INTO money_requests(attempt_id,state,cost_upper) VALUES (999,'settled','2')")
    with pytest.raises(ProviderBudgetExceeded): batch.reserve(ledger, "retrieval", 0, 1)
    assert "fee_warning_2_cny_reached" in money.snapshot(ledger)["blocks"]


def test_unavailable_wallet_is_not_zero_balance_or_permission_to_continue(ledger, monkeypatch):
    def unavailable(key): raise TimeoutError("private error details must not be recorded")
    monkeypatch.setattr(money, "query_balance", unavailable)
    money.refresh(ledger, "offline-placeholder")
    snapshot = money.snapshot(ledger)
    assert snapshot["wallet_checks"][-1]["balance_cny"] is None
    assert snapshot["wallet_checks"][-1]["error_class"] == "TimeoutError"
    assert "private error details" not in json.dumps(snapshot)
    with pytest.raises(ProviderBudgetExceeded): batch.reserve(ledger, "retrieval", 0, 1)


@pytest.mark.parametrize("incoming,outgoing", [(20001,100),(1000,5001)])
def test_abnormal_observed_usage_stops_without_changing_output_configuration(ledger, incoming, outgoing):
    key = batch.reserve(ledger, "chat", 10000, 1000)
    reported(ledger, key, incoming=incoming, outgoing=outgoing, cache=0)
    money.settle(ledger, key, status=200, complete=True)
    assert "abnormally_large_observed_token_usage" in money.snapshot(ledger)["blocks"]


def test_missing_cache_is_billed_as_uncached_upper_bound(ledger):
    key = batch.reserve(ledger, "chat", 10000, 1000)
    reported(ledger, key, cache=None)
    money.settle(ledger, key, status=200, complete=True)
    assert Decimal(money.snapshot(ledger)["measured_peak_cost_upper_cny"]) == Decimal("0.0028")


def test_wallet_checked_again_after_five_chat_attempts(ledger, monkeypatch):
    for _ in range(5):
        key = batch.reserve(ledger, "chat", 10000, 1000)
        reported(ledger, key); money.settle(ledger, key, status=200, complete=True)
    calls=[]
    monkeypatch.setattr(money, "query_balance", lambda key: calls.append(True) or Decimal("15.30"))
    money.refresh(ledger, "offline-placeholder", force=False)
    assert len(calls)==1
    with sqlite3.connect(ledger) as conn:
        conn.execute("UPDATE wallet_checks SET checked_at=?", (time.time()-46,))
        assert money.check_locked(conn, "chat", 1000)=="balance_check_stale"


def test_transport_missing_usage_prevents_retry_and_any_other_provider(ledger, monkeypatch):
    monkeypatch.setattr(httpx.Client, "send", httpx.Client.send)
    monkeypatch.setattr(httpx.AsyncClient, "send", httpx.AsyncClient.send)
    batch.install_http_budget(ledger, {"api.deepseek.com","api.siliconflow.cn"})
    sent=[]
    def transport(request):
        sent.append(request.url.path)
        return httpx.Response(200, json={"choices":[]})
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        client.post("https://api.deepseek.com/chat/completions", json={"model":"deepseek-v4-flash","max_tokens":10000})
        with pytest.raises(ProviderBudgetExceeded):
            client.post("https://api.siliconflow.cn/v1/embeddings", json={"model":"BAAI/bge-m3"})
    assert sent==["/chat/completions"]


async def test_stream_usage_settles_only_after_complete_stream(ledger, monkeypatch):
    monkeypatch.setattr(httpx.Client, "send", httpx.Client.send)
    monkeypatch.setattr(httpx.AsyncClient, "send", httpx.AsyncClient.send)
    batch.install_http_budget(ledger, {"api.deepseek.com"})
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"model":"deepseek-flash","usage":{"prompt_tokens":1000,"completion_tokens":100}}\n\ndata: [DONE]\n\n'
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,stream=Stream()))) as client:
        async with client.stream("POST","https://api.deepseek.com/chat/completions",json={"model":"deepseek-v4-flash","max_tokens":10000}) as response:
            await response.aread()
    assert money.snapshot(ledger)["unknown_cost_requests"]==0
    assert not money.snapshot(ledger)["blocks"]
