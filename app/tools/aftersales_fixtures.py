"""Coherent synthetic aftersales snapshots for new diagnostic orders only.

The reference clock is fixed for replay, not a production warranty rule. Course
orders 1001/2002 and arbitrary legacy mocks retain their exact original behavior.
"""
from datetime import datetime, timedelta
import random

from app.tools import business, synthetic_fixtures

VERSION = "coherent-aftersales-v1"
REFERENCE_TIME = datetime(2026, 10, 5, 9)


def _order(order_id: str) -> dict | None:
    if order_id not in business.LOGISTICS_ORDER_IDS and order_id not in synthetic_fixtures.reserved_order_ids():
        return None
    order = business.order_snapshot(order_id)
    created = datetime.fromisoformat(order["created_at"])
    if created > REFERENCE_TIME:
        raise ValueError("synthetic order is after reference clock")
    return order


def warranty_snapshot(order_id: str) -> dict | None:
    order = _order(order_id)
    if order is None:
        return None
    created = datetime.fromisoformat(order["created_at"])
    duration = random.Random(f"coherent-warranty:{order_id}").choice([7, 365])
    until = created + timedelta(days=duration)
    return {"order_id": order_id, "warranty_code": "IN_WARRANTY" if until.date() >= REFERENCE_TIME.date() else "EXPIRED",
            "warranty_until": until.strftime("%Y-%m-%d"), "policy_ref": "SYNTHETIC-AS-V1"}


def return_snapshot(order_id: str) -> dict | None:
    order = _order(order_id)
    if order is None:
        return None
    created = datetime.fromisoformat(order["created_at"])
    rng = random.Random(f"return:{order_id}")
    code = rng.choice(["AUDITING", "RETURNING", "REFUNDED", "NONE"])
    # No later-than-clock or earlier-than-order record can be manufactured.
    elapsed_minutes = int((REFERENCE_TIME - created).total_seconds() // 60)
    updated = created + timedelta(minutes=rng.randint(0, elapsed_minutes))
    return {"order_id": order_id, "return_code": code, "updated_at": updated.strftime("%Y-%m-%d %H:%M")}
