"""Evaluation-only fail-closed CNY warning guard; never changes Agent settings.

Peak published Flash prices conservatively bound observed usage. Wallet deltas
are account-wide signals, not an attributable invoice. Pending/failed paid
requests cannot silently become zero cost. Official balance GETs are separate
from inference budgets and never log credentials or the full account response.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sqlite3
import time

from app.evaluation.stage1 import ProviderBudgetExceeded

POLICY = {"version": "dev12-money-v1", "warning_cny": "2", "minimum_balance_cny": "3",
    "per_request_headroom_cny": "0.25", "balance_max_age_seconds": 45, "balance_chat_interval": 5,
    "max_input_bytes_per_request": 65536, "max_input_tokens_per_request": 20000,
    "max_output_tokens_observed_per_request": 5000, "max_input_tokens_batch": 150000,
    "max_output_tokens_batch": 50000, "peak_cny_per_million": {"input": "2", "cache": "0.04", "output": "8"},
    "pricing_source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/",
    "free_retrieval_models": ["BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"],
    "retrieval_pricing_source": "https://www.siliconflow.cn/pricing"}


def enabled(path):
    with sqlite3.connect(path) as conn:
        return bool(conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='money_requests'").fetchone())


def validate_policy(directory):
    value = json.loads((Path(directory)/"monetary-policy.json").read_text())
    if value != POLICY:
        raise ValueError("monetary controls differ from approved dev12 scope")


def activate(path, key):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE money_requests (attempt_id INTEGER PRIMARY KEY, state TEXT, input_tokens INTEGER, output_tokens INTEGER, cache_read INTEGER, cost_upper TEXT, reason TEXT)")
        conn.execute("CREATE TABLE wallet_checks (id INTEGER PRIMARY KEY, checked_at REAL, label TEXT, http_status INTEGER, available INTEGER, balance_cny TEXT, chat_count INTEGER, cost_upper_at_check TEXT, error_class TEXT, duration_ms REAL)")
    refresh(path, key, label="before_run")


def block_locked(conn, reason):
    conn.execute("INSERT INTO batch_blocks VALUES (?,?)", (reason, time.time()))


def _cost(conn):
    return sum((Decimal(v[0]) for v in conn.execute("SELECT cost_upper FROM money_requests WHERE state='settled'")), Decimal(0))


def _last_wallet(conn):
    return conn.execute("SELECT checked_at,available,balance_cny,chat_count,cost_upper_at_check,error_class FROM wallet_checks ORDER BY id DESC LIMIT 1").fetchone()


def check_locked(conn, kind, input_bytes):
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='money_requests'").fetchone():
        return None
    if conn.execute("SELECT 1 FROM money_requests WHERE state!='settled' LIMIT 1").fetchone():
        return "paid_request_cost_unresolved"
    # Free pricing does not make unreported Token Usage observable. Also gate
    # in-flight retrieval so another send cannot race ahead of its settlement.
    if conn.execute("SELECT 1 FROM attempts a LEFT JOIN provider_attempt_meta m ON m.attempt_id=a.id "
                    "LEFT JOIN provider_usage u ON u.attempt_id=a.id WHERE a.kind='retrieval' AND "
                    "(a.status IS NULL OR a.status!=200 OR m.outcome IS NULL OR m.outcome!='complete' "
                    "OR u.complete IS NULL OR u.complete!=1) LIMIT 1").fetchone():
        return "retrieval_usage_unresolved"
    wallet = _last_wallet(conn)
    if not wallet or wallet[1] != 1 or wallet[2] is None or wallet[5]:
        return "reliable_balance_unavailable"
    if time.time()-wallet[0] > POLICY["balance_max_age_seconds"]:
        return "balance_check_stale"
    first = conn.execute("SELECT balance_cny FROM wallet_checks WHERE available=1 AND error_class IS NULL ORDER BY id LIMIT 1").fetchone()
    current = Decimal(wallet[2]); start = Decimal(first[0]); cost = _cost(conn)
    observed_drop = max(Decimal(0), start-current)
    effective_spend = max(cost, observed_drop)
    headroom = Decimal(POLICY["per_request_headroom_cny"]) if kind == "chat" else Decimal(0)
    estimated_balance = current-max(Decimal(0), cost-Decimal(wallet[4]))
    if current < Decimal("3") or estimated_balance < Decimal("3"):
        return "balance_below_3_cny"
    if effective_spend >= Decimal("2"):
        return "fee_warning_2_cny_reached"
    if effective_spend+headroom > Decimal("2") or estimated_balance-headroom < Decimal("3"):
        return "insufficient_fee_or_balance_headroom"
    if input_bytes > POLICY["max_input_bytes_per_request"]:
        return "abnormally_large_provider_input"
    return None


def admit_locked(conn, attempt_id, kind):
    if kind == "chat" and conn.execute("SELECT 1 FROM sqlite_master WHERE name='money_requests'").fetchone():
        conn.execute("INSERT INTO money_requests(attempt_id,state) VALUES (?,'pending')", (attempt_id,))


def settle(path, attempt_id, *, status, complete):
    if attempt_id is None or not enabled(path):
        return
    with sqlite3.connect(path, timeout=10) as conn:
        conn.execute("BEGIN IMMEDIATE")
        kind = conn.execute("SELECT kind FROM attempts WHERE id=?", (attempt_id,)).fetchone()
        if kind and kind[0] == "retrieval":
            value = conn.execute("SELECT input_tokens,output_tokens,complete FROM provider_usage "
                                 "WHERE attempt_id=? ORDER BY rowid DESC LIMIT 1", (attempt_id,)).fetchone()
            if (not complete or status != 200 or not value or value[2] != 1
                    or type(value[0]) is not int or value[0] < 0
                    or type(value[1]) is not int or value[1] < 0):
                block_locked(conn, "retrieval_usage_unavailable_or_unreliable")
            return  # Official free price remains separate from unknown Usage.
        row = conn.execute("SELECT state FROM money_requests WHERE attempt_id=?", (attempt_id,)).fetchone()
        if not row or row[0] != "pending": return
        value = conn.execute("SELECT model,input_tokens,output_tokens,cache_read FROM provider_usage WHERE attempt_id=? ORDER BY rowid DESC LIMIT 1", (attempt_id,)).fetchone()
        reason = None
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='provider_attempt_meta'").fetchone():
            metadata = conn.execute("SELECT usage_error FROM provider_attempt_meta WHERE attempt_id=?", (attempt_id,)).fetchone()
            if metadata and metadata[0]:
                reason = "paid_request_usage_observation_ambiguous:" + metadata[0]
        if reason:
            pass
        elif not complete or status != 200 or not value:
            reason = "paid_request_usage_unavailable"
        elif (value[0] not in {"deepseek-flash", "deepseek-v4-flash"} or type(value[1]) is not int or value[1] <= 0
              or type(value[2]) is not int or value[2] < 0 or (value[3] is not None and not 0 <= value[3] <= value[1])):
            reason = "paid_request_usage_unreliable"
        if reason:
            conn.execute("UPDATE money_requests SET state='unknown',reason=? WHERE attempt_id=?", (reason, attempt_id))
            block_locked(conn, reason); return
        model, incoming, outgoing, cache = value
        # Missing cache metadata uses the uncached peak rate, never zero cost.
        cost = (Decimal(incoming-(cache or 0))*2+Decimal(cache or 0)*Decimal("0.04")+Decimal(outgoing)*8)/Decimal(1000000)
        conn.execute("UPDATE money_requests SET state='settled',input_tokens=?,output_tokens=?,cache_read=?,cost_upper=? WHERE attempt_id=?",
                     (incoming, outgoing, cache, str(cost), attempt_id))
        totals = conn.execute("SELECT SUM(input_tokens),SUM(output_tokens) FROM money_requests WHERE state='settled'").fetchone()
        if (incoming > POLICY["max_input_tokens_per_request"] or outgoing > POLICY["max_output_tokens_observed_per_request"]
                or totals[0] > POLICY["max_input_tokens_batch"] or totals[1] > POLICY["max_output_tokens_batch"]):
            block_locked(conn, "abnormally_large_observed_token_usage")
        if _cost(conn) >= Decimal("2"):
            block_locked(conn, "fee_warning_2_cny_reached")


def query_balance(key):
    import httpx
    if not key: raise ValueError("missing_balance_credential")
    with httpx.Client(timeout=8, follow_redirects=False, trust_env=True) as client:
        response = client.get("https://api.deepseek.com/user/balance", headers={"Authorization": "Bearer "+key, "Accept": "application/json"})
    if response.status_code != 200: raise ValueError("official_balance_http_error")
    value = response.json()
    if value.get("is_available") is not True: raise ValueError("official_balance_not_available")
    balances = [v for v in value.get("balance_infos", []) if isinstance(v, dict) and v.get("currency") == "CNY"]
    if len(balances) != 1: raise ValueError("official_cny_balance_missing")
    amount = Decimal(balances[0]["total_balance"])
    if not amount.is_finite() or amount < 0: raise ValueError("invalid_official_balance")
    return amount


def refresh(path, key, *, label="periodic", force=True):
    if not enabled(path): return
    with sqlite3.connect(path) as conn:
        wallet = _last_wallet(conn)
        count = conn.execute("SELECT COUNT(*) FROM attempts WHERE kind='chat'").fetchone()[0]
        if not force and wallet and time.time()-wallet[0] <= POLICY["balance_max_age_seconds"] and count-wallet[3] < POLICY["balance_chat_interval"]:
            return
    amount, error = None, None
    started = time.perf_counter()
    try: amount = query_balance(key)
    except Exception as exc: error = type(exc).__name__
    with sqlite3.connect(path, timeout=10) as conn:
        conn.execute("BEGIN IMMEDIATE")
        count = conn.execute("SELECT COUNT(*) FROM attempts WHERE kind='chat'").fetchone()[0]
        conn.execute("INSERT INTO wallet_checks(checked_at,label,http_status,available,balance_cny,chat_count,cost_upper_at_check,error_class,duration_ms) VALUES (?,?,?,?,?,?,?,?,?)",
                     (time.time(), label, 200 if amount is not None else None, int(amount is not None), str(amount) if amount is not None else None, count, str(_cost(conn)), error, round((time.perf_counter()-started)*1000,3)))
        if amount is None: block_locked(conn, "official_balance_unavailable")
        elif amount < Decimal("3"): block_locked(conn, "balance_below_3_cny")
        else:
            first = conn.execute("SELECT balance_cny FROM wallet_checks WHERE available=1 ORDER BY id LIMIT 1").fetchone()
            if Decimal(first[0])-amount >= Decimal("2"): block_locked(conn, "account_balance_drop_2_cny")


def snapshot(path):
    if not enabled(path): return {"enabled": False}
    with sqlite3.connect(path) as conn:
        wallets = [dict(zip(("id", "checked_at", "label", "http_status", "available", "balance_cny", "chat_count", "cost_upper_at_check", "error_class", "duration_ms"), r)) for r in conn.execute("SELECT * FROM wallet_checks ORDER BY id")]
        requests = [dict(zip(("attempt_id", "state", "input_tokens", "output_tokens", "cache_read", "cost_upper", "reason"), r)) for r in conn.execute("SELECT * FROM money_requests ORDER BY attempt_id")]
        blocks = [r[0] for r in conn.execute("SELECT reason FROM batch_blocks")]
        upper = str(_cost(conn))
    successful = [w for w in wallets if w["balance_cny"] is not None]
    unknown = sum(r["state"] != "settled" for r in requests)
    from app.evaluation.usage_meter import VERSION as METER_VERSION, snapshot as usage_snapshot
    usage = usage_snapshot(path)
    return {"enabled": True, "policy": POLICY, "wallet_checks": wallets, "paid_requests": requests,
        "measured_peak_cost_upper_cny": upper, "known_partial_cost_upper_cny": upper,
        "total_cost_upper_cny": None if unknown else upper, "unknown_cost_cny": None if unknown else "0",
        "unknown_cost_requests": unknown, "provider_usage": usage,
        "unknown_usage_requests": sum(v["usage_unknown_requests"] for v in usage.values()),
        "free_retrieval_estimated_cost_cny": "0",
        "free_retrieval_price_basis": POLICY["retrieval_pricing_source"],
        "usage_stop_contract": "all-provider-usage-stop-v1; no free-model exception",
        "usage_meter_version": METER_VERSION,
        "cost_basis": "published peak-rate estimate; exact attributable billed fee unavailable",
        "observable_balance_delta_cny": str(Decimal(successful[0]["balance_cny"])-Decimal(successful[-1]["balance_cny"])) if successful else None,
        "blocks": blocks, "limitation": "balance is rounded, delayed and account-wide; peak usage estimate is conservative, not an attributable invoice; an in-flight request cannot be retroactively unbilled"}
