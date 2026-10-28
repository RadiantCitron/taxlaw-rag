# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""graceful shutdown (서버 기동 불필요)."""
from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _reset_shutdown_state():
    import session_manager as sm

    sm._shutting_down.clear()
    with sm._sessions_lock:
        sm._sessions.clear()
    yield
    sm._shutting_down.clear()
    with sm._sessions_lock:
        sm._sessions.clear()


def test_begin_shutdown_cancels_open_sessions():
    import session_manager as sm

    sess = sm.ClientSession("127.0.0.1")
    sess.processing = True
    with sm._sessions_lock:
        sm._sessions[sess.session_id] = sess

    sm.begin_shutdown()

    assert sm.is_shutting_down()
    assert sess.cancel_event.is_set()
    assert not sess.presence_open


def test_drain_in_flight_returns_when_processing_ends():
    import session_manager as sm

    sess = sm.ClientSession("127.0.0.1")
    sess.processing = True
    with sm._sessions_lock:
        sm._sessions[sess.session_id] = sess

    async def _run():
        async def _finish():
            await asyncio.sleep(0.25)
            sess.processing = False

        asyncio.create_task(_finish())
        return await sm.drain_in_flight(2.0)

    remaining = asyncio.run(_run())
    assert remaining == 0


def test_raise_if_cancelled_during_shutdown():
    import session_manager as sm

    sm.begin_shutdown()
    with pytest.raises(sm.SessionCancelled):
        sm.raise_if_cancelled()


def test_unload_all_models_clears_slots():
    pytest.importorskip("torch")
    from model_registry import _ModelSlot, _slots, unload_all_models

    _slots[("rerank", "test-model")] = _ModelSlot("rerank", "test-model", object())
    unload_all_models()
    assert _slots == {}
