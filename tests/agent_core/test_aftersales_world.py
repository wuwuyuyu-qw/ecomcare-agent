import random
from datetime import datetime

import pytest

from app.tools import aftersales_fixtures as fixtures, business
from app.tools.contracts import validate_result
from mcp_servers import aftersales_server as server


@pytest.mark.parametrize("oid", ["3001", "3002", "3003"])
def test_shared_order_dates_and_statuses_are_coherent_and_repeatable(oid):
    created = datetime.fromisoformat(business.order_snapshot(oid)["created_at"])
    warranty, returned = fixtures.warranty_snapshot(oid), fixtures.return_snapshot(oid)
    assert returned == fixtures.return_snapshot(oid) and warranty == fixtures.warranty_snapshot(oid)
    assert created <= datetime.fromisoformat(returned["updated_at"]) <= fixtures.REFERENCE_TIME
    until = datetime.fromisoformat(warranty["warranty_until"])
    assert until.date() >= created.date()
    assert (warranty["warranty_code"] == "IN_WARRANTY") == (until.date() >= fixtures.REFERENCE_TIME.date())
    validate_result("query_warranty", warranty, {"order_id": oid})
    validate_result("query_return_status", returned, {"order_id": oid})
    returned["updated_at"] = "mutated"
    assert fixtures.return_snapshot(oid)["updated_at"] != "mutated"


@pytest.mark.asyncio
@pytest.mark.parametrize("oid", ["1001", "2002", "unknown-demo"])
async def test_course_mocks_remain_byte_for_byte_equivalent(oid):
    assert fixtures.warranty_snapshot(oid) is None and fixtures.return_snapshot(oid) is None
    rng = random.Random(f"warranty:{oid}")
    expected = {"order_id": oid, "warranty_code": rng.choice(["IN_WARRANTY", "EXPIRED"]),
                "warranty_until": f"2026-{rng.randint(8, 12):02d}-{rng.randint(1, 28):02d}", "policy_ref": "AS-POLICY-07"}
    assert await server.query_warranty(oid) == expected
    rng = random.Random(f"return:{oid}")
    expected = {"order_id": oid, "return_code": rng.choice(["AUDITING", "RETURNING", "REFUNDED", "NONE"]),
                "updated_at": f"2026-07-{rng.randint(1, 16):02d} 10:00"}
    assert await server.query_return_status(oid) == expected
