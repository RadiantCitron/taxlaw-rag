# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""접속 presence 해제·/ask cancel 시 슬롯 반납 회귀."""
from __future__ import annotations

import asyncio
import time

import pytest


class FakeRequest:
    def __init__(self, disconnected: bool = False):
        self._disconnected = disconnected

    async def is_disconnected(self) -> bool:
        return self._disconnected


@pytest.fixture(autouse=True)
def _reset_sessions_and_gate():
    import session_manager as sm
    from local_llm_gate import init_gate

    sm._shutting_down.clear()
    with sm._sessions_lock:
        sm._sessions.clear()
    sm.init_sessions(max_connected=5)
    init_gate(max_concurrent_user=1, max_waiting=2)
    yield
    sm._shutting_down.clear()
    with sm._sessions_lock:
        sm._sessions.clear()


def test_presence_close_frees_connected_count():
    import session_manager as sm

    s1 = sm.create_session("127.0.0.1")
    s2 = sm.create_session("127.0.0.1")
    assert sm.status_payload()["connected"] == 2

    sm.mark_presence_closed(s1.session_id)
    assert sm.status_payload()["connected"] == 1
    assert sm.get_session(s1.session_id) is None

    sm.mark_presence_closed(s2.session_id)
    assert sm.status_payload()["connected"] == 0


@pytest.mark.asyncio
async def test_presence_close_while_processing_waits_for_end_ask():
    import session_manager as sm

    sess = sm.create_session("127.0.0.1")
    assert sess.try_begin_ask()
    assert sess.processing

    sm.mark_presence_closed(sess.session_id)
    # processing 중이면 카운트에 남는다
    assert sm.status_payload()["connected"] == 1
    assert sm.get_session(sess.session_id) is not None
    assert not sess.presence_open
    assert sess.cancel_event.is_set()

    sess.end_ask("cancelled")
    assert sm.status_payload()["connected"] == 0
    assert sm.get_session(sess.session_id) is None


@pytest.mark.asyncio
async def test_heartbeat_loop_exits_quickly_on_disconnect():
    """poll 짧게 두면 disconnect 시 15초 sleep 없이 종료."""
    from api_server import presence_heartbeat_loop
    import session_manager as sm

    class DisconnectAfterPoll:
        def __init__(self):
            self._n = 0

        async def is_disconnected(self) -> bool:
            self._n += 1
            return self._n >= 2

    sess = sm.create_session("127.0.0.1")
    t0 = time.monotonic()
    frames = []
    async for frame in presence_heartbeat_loop(
        DisconnectAfterPoll(), sess, poll_sec=0.05, ping_sec=10.0
    ):
        frames.append(frame)
    elapsed = time.monotonic() - t0
    assert frames == []
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_cancelled_path_releases_ask_slot_before_slow_work():
    """cancel 시 ask slot 을 긴 파이프라인 대기 전에 반납하는 정책 검증.

    acquire_ask_slot 은 내부에서 최대 0.5s 폴링하므로, 슬롯이 비어 있어도
    즉시(0.2s 미만)는 아닐 수 있다. 느린 작업(2s)보다 훨씬 빨리 재획득되면 성공.
    """
    from local_llm_gate import acquire_ask_slot, release_ask_slot

    queue: asyncio.Queue = asyncio.Queue()
    assert await acquire_ask_slot(FakeRequest(), queue) is True

    # /ask finally 의 cancel 경로: 슬롯을 먼저 풀고, 느린 작업을 뒤에서 기다림
    release_ask_slot()

    slow_done = asyncio.Event()

    async def slow_pipeline():
        await asyncio.sleep(2.0)
        slow_done.set()

    task = asyncio.create_task(slow_pipeline())
    t0 = time.monotonic()
    assert await acquire_ask_slot(FakeRequest(), queue) is True
    acquired_after = time.monotonic() - t0
    assert acquired_after < 1.0
    assert not slow_done.is_set()

    await task
    release_ask_slot()
