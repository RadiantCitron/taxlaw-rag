# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""test_api start_presence SSE 스트림 정리."""
from __future__ import annotations

import threading
import time

import pytest


def test_start_presence_stop_closes_stream_quickly(monkeypatch):
    import test_api

    closed = threading.Event()

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def iter_lines(self, decode_unicode=True):
            yield 'data: {"event": "connected", "session_id": "sess-test-1"}'
            while not closed.is_set():
                time.sleep(0.01)

        def close(self):
            closed.set()

    class FakeSession:
        def __init__(self):
            self._resp = None

        def get(self, *_args, **_kwargs):
            self._resp = FakeResponse()
            return self._resp

        def close(self):
            if self._resp is not None:
                self._resp.close()

    monkeypatch.setattr(test_api.requests, "Session", lambda: FakeSession())

    session_id, stop, http, thread = test_api.start_presence("http://127.0.0.1:8000/session/connect")
    assert session_id == "sess-test-1"

    stop.set()
    closer = getattr(http, "_close_presence_stream", None)
    if callable(closer):
        closer()
    try:
        http.close()
    except Exception:
        pass
    thread.join(timeout=2)
    assert closed.is_set()
    assert not thread.is_alive()


def test_deepeval_modules_gives_install_hint(monkeypatch):
    import evaluator

    real_import = __import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "deepeval" or name.startswith("deepeval."):
            raise ImportError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", blocked_import)

    with pytest.raises(ImportError, match="DeepEval"):
        evaluator._deepeval_modules()
