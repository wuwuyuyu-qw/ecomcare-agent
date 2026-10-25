import asyncio

import pytest
from anyio import CancelScope
from app.core.execution import finish_cancellation


@pytest.mark.asyncio
async def test_asgi_cancel_scope_allows_bounded_local_finalization():
    saved = []

    async def save():
        async with asyncio.timeout(0.5):
            await asyncio.sleep(0)
            saved.append("client_cancelled")
            return "saved"

    with CancelScope() as cancelled:
        cancelled.cancel()
        assert await finish_cancellation(save()) == "saved"
    assert saved == ["client_cancelled"]


@pytest.mark.asyncio
async def test_shield_does_not_disable_cleanup_timeout():
    async def stalled_save():
        async with asyncio.timeout(0.01):
            await asyncio.sleep(1)
    with pytest.raises(TimeoutError):
        await finish_cancellation(stalled_save())
