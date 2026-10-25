"""These adapters exercise guards, not model-dependent task success."""
import pytest
from app.evaluation.readonly_recovery import load_fixture, probe_case, readonly_world
from app.tools import business


@pytest.mark.parametrize("case_id", ["task-09-07", "task-09-08", "task-09-10", "task-10-07", "task-10-09", "task-10-10"])
async def test_recovery_adapter(case_id, tmp_path):
    result = await probe_case(case_id, checkpoint_path=tmp_path / "private.sqlite")
    assert result["success"], result["checks"]
    assert result["model_calls"] == result["external_requests"] == 0


def test_private_catalog_checks_real_ownership_and_restores_dependency():
    data, digest = load_fixture()
    original = business.synthetic_fixtures._catalog
    with readonly_world(data, digest):
        for case in data["cases"]:
            oid = case["order_id"]
            assert business.owns_order(data["owner"], oid)
            assert not business.owns_order("customer_other", oid)
            assert not business.owns_order("", oid)
            copy = business.order_snapshot(oid)
            copy["tracking_no"] = "tampered"
            assert business.order_snapshot(oid)["tracking_no"] == case["tracking_no"]
    assert business.synthetic_fixtures._catalog is original
