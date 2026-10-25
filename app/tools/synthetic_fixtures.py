"""Opt-in synthetic business worlds, independent of annotations or application config.

Default ``demo`` never opens the extended catalog. Only a locally registered
profile is accepted; invalid data/config fails closed instead of random fallback.
The selected profile and validated catalog are loaded once per process. Cached
data is recursively immutable, and every public snapshot is a fresh mutable copy.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping


_ROOT = Path(__file__).resolve().parents[2]
_REGISTERED_FILES = MappingProxyType({"task-v1": _ROOT / "data/agent_fixtures/task-v1.json"})
_PROFILE_VERSIONS = MappingProxyType({"demo": "demo", "task-v1": "task-v1.0"})
_OWNERS = frozenset({"customer_demo"})
_RESERVED_DEMO_IDS = frozenset({"1001", "2002", "3001", "3002", "3003", "9999"})
_STATUSES = frozenset({"PICKED_UP", "IN_TRANSIT", "DELIVERING", "DELIVERED"})
_MAX_BYTES = 512_000
_ORDER_FIELDS = frozenset({"order_id", "status", "amount", "created_at", "product", "tracking_no"})
_LOGISTICS_FIELDS = frozenset({"tracking_no", "status_code", "current_city", "trace", "stalled_hours", "carrier_code"})


class FixtureConfigurationError(ValueError):
    """Fixed safe error: a bad fixture must never downgrade to another world."""

    def __init__(self):
        super().__init__("合成数据环境无效；请使用登记的 demo 或 task-v1 配置。")


@dataclass(frozen=True)
class _Catalog:
    profile: str
    version: str
    digest: str | None
    orders: Mapping
    logistics: Mapping
    owners: Mapping


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


def _copy(value):
    if isinstance(value, Mapping):
        return {key: _copy(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_copy(child) for child in value]
    return value


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise FixtureConfigurationError()
        out[key] = value
    return out


def _invalid_constant(_value):
    raise FixtureConfigurationError()


def _text(value, *, max_length=512):
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise FixtureConfigurationError()


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise FixtureConfigurationError()


def _nonnegative_number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise FixtureConfigurationError()


def _validate_record(record):
    if not isinstance(record, dict) or set(record) != {"owner", "order", "logistics"}:
        raise FixtureConfigurationError()
    owner, order, logistics = record["owner"], record["order"], record["logistics"]
    if not isinstance(owner, str) or owner not in _OWNERS:
        raise FixtureConfigurationError()
    if (not isinstance(order, dict) or set(order) != _ORDER_FIELDS
            or not isinstance(logistics, dict) or set(logistics) != _LOGISTICS_FIELDS):
        raise FixtureConfigurationError()
    oid = order["order_id"]
    if (not isinstance(oid, str) or not re.fullmatch(r"[0-9]{4,16}", oid)
            or oid in _RESERVED_DEMO_IDS):
        raise FixtureConfigurationError()
    _identifier(order["tracking_no"])
    if logistics["tracking_no"] != order["tracking_no"]:
        raise FixtureConfigurationError()
    code = logistics["status_code"]
    if not isinstance(code, str) or code not in _STATUSES:
        raise FixtureConfigurationError()
    status = "已签收" if code == "DELIVERED" else "已发货"
    if order["status"] != status:
        raise FixtureConfigurationError()
    _nonnegative_number(order["amount"])
    _nonnegative_number(logistics["stalled_hours"])
    _text(order["product"])
    created = order["created_at"]
    if not isinstance(created, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", created):
        raise FixtureConfigurationError()
    datetime.fromisoformat(created)
    _text(logistics["current_city"])
    _text(logistics["carrier_code"])
    trace = logistics["trace"]
    if not isinstance(trace, list) or not 1 <= len(trace) <= 100:
        raise FixtureConfigurationError()
    for event in trace:
        _text(event)


@lru_cache(maxsize=1)
def current_profile() -> str:
    profile = os.environ.get("MEWHELP_FIXTURE_PROFILE", "demo")
    if profile not in _PROFILE_VERSIONS:
        raise FixtureConfigurationError()
    return profile


@lru_cache(maxsize=1)
def _catalog() -> _Catalog:
    profile = current_profile()
    if profile == "demo":
        # No stat/read of the optional file on the ordinary demonstration path.
        return _Catalog(profile, "demo", None, _freeze({}), _freeze({}), _freeze({}))
    try:
        path = _REGISTERED_FILES[profile]
        with path.open("rb") as source:
            raw = source.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            raise FixtureConfigurationError()
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_invalid_constant)
        if (not isinstance(data, dict) or set(data) != {"profile", "version", "orders"}
                or data["profile"] != profile or type(data["version"]) is not int or data["version"] != 1
                or not isinstance(data["orders"], list) or not 1 <= len(data["orders"]) <= 500):
            raise FixtureConfigurationError()
        orders, logistics, owners = {}, {}, {}
        for record in data["orders"]:
            _validate_record(record)
            order, track = record["order"], record["logistics"]
            oid, tracking_no = order["order_id"], order["tracking_no"]
            if oid in orders or tracking_no in logistics:
                raise FixtureConfigurationError()
            orders[oid], logistics[tracking_no], owners[oid] = order, track, record["owner"]
        return _Catalog(profile, _PROFILE_VERSIONS[profile], hashlib.sha256(raw).hexdigest(),
                        _freeze(orders), _freeze(logistics), _freeze(owners))
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
        raise FixtureConfigurationError() from None


def profile_info() -> dict:
    catalog = _catalog()
    return {"fixture_profile": catalog.profile, "fixture_version": catalog.version,
            "fixture_sha256": catalog.digest}


def catalog_digest() -> str | None:
    return _catalog().digest


def reserved_order_ids() -> frozenset[str]:
    """All extended IDs must be excluded from ordinary random ownership."""
    return frozenset(_catalog().orders)


def order_ids_for(user_id: str) -> tuple[str, ...]:
    catalog = _catalog()
    return tuple(oid for oid, owner in catalog.owners.items() if user_id == owner)


def owns_order(user_id: str, order_id: str) -> bool:
    return bool(user_id) and _catalog().owners.get(order_id) == user_id


def order_snapshot(order_id: str) -> dict | None:
    value = _catalog().orders.get(order_id)
    return _copy(value) if value is not None else None


def logistics_snapshot(tracking_no: str) -> dict | None:
    value = _catalog().logistics.get(tracking_no)
    return _copy(value) if value is not None else None
