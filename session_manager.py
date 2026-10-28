# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
# --- START OF FILE session_manager.py ---
"""연결 단위 세션, 접속/처리 카운트, 세션별 로그 라우팅."""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

logger = logging.getLogger("my_rag")

LOG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
SESSION_LOG_DIR = os.path.join(LOG_ROOT, "sessions")
SERVER_LOG_PATH = os.path.join(LOG_ROOT, "server.jsonl")

_current_session: contextvars.ContextVar["ClientSession | None"] = contextvars.ContextVar(
    "rag_session", default=None
)
_sessions: dict[str, "ClientSession"] = {}
_sessions_lock = threading.Lock()
_server_log_lock = threading.Lock()
_router_installed = False
_max_connected = 10
_shutting_down = threading.Event()
_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()
MAX_SESSION_TURNS = 200


class SessionCancelled(Exception):
    """클라가 끊었거나 Stop 해서, 우리 쪽 수신을 중단한다."""


class SessionCapacityError(Exception):
    """접속 정원 초과."""

    def __init__(self, current: int, maximum: int):
        self.current = current
        self.maximum = maximum
        super().__init__(f"접속 정원 초과 ({current}/{maximum})")


def init_sessions(max_connected: int = 10) -> None:
    global _max_connected
    _max_connected = max(1, int(max_connected))


def max_connected() -> int:
    return _max_connected


def get_current_session() -> "ClientSession | None":
    return _current_session.get()


def bind_session(session: "ClientSession") -> contextvars.Token:
    return _current_session.set(session)


def reset_session(token: contextvars.Token) -> None:
    _current_session.reset(token)


def raise_if_cancelled() -> None:
    session = get_current_session()
    if session is not None and session.cancel_event.is_set():
        raise SessionCancelled()
    if is_shutting_down():
        raise SessionCancelled()


def is_shutting_down() -> bool:
    return _shutting_down.is_set()


def count_processing_sessions() -> int:
    with _sessions_lock:
        return sum(1 for s in _sessions.values() if s.processing)


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            workers = max(1, int(os.getenv("SESSION_EXECUTOR_WORKERS", "8")))
            _executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="rag-worker",
            )
        return _executor


def _shutdown_executor() -> None:
    global _executor
    with _executor_lock:
        ex = _executor
        _executor = None
    if ex is None:
        return
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        ex.shutdown(wait=False)


def begin_shutdown(reason: str = "server_shutdown") -> None:
    """SIGINT/SIGTERM 시 신규 작업 차단 + 열린 세션 취소."""
    if _shutting_down.is_set():
        return
    _shutting_down.set()
    _shutdown_executor()
    with _sessions_lock:
        session_ids = list(_sessions.keys())
    for sid in session_ids:
        mark_presence_closed(sid, reason=reason)
    logger.info("[SHUTDOWN] 세션 %s건 종료 신호 (%s)", len(session_ids), reason)


async def drain_in_flight(timeout_sec: float) -> int:
    """processing 중인 /ask 요청이 끝날 때까지 기다린다. 남은 건수를 반환."""
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while time.monotonic() < deadline:
        remaining = count_processing_sessions()
        if remaining == 0:
            return 0
        await asyncio.sleep(0.2)
    return count_processing_sessions()


async def run_in_session_executor(func, /, *args, **kwargs):
    """워커 스레드에도 현재 세션 ContextVar를 복사한다."""
    ctx = contextvars.copy_context()
    loop = asyncio.get_running_loop()

    def _call():
        return ctx.run(lambda: func(*args, **kwargs))

    return await loop.run_in_executor(_get_executor(), _call)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_log_dirs() -> None:
    os.makedirs(SESSION_LOG_DIR, exist_ok=True)


def _append_jsonl(path: str, payload: dict, lock: threading.Lock) -> None:
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()


def _close_stream_quietly(stream: Any) -> None:
    if stream is None:
        return
    for closer in ("close", "aclose"):
        fn = getattr(stream, closer, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                logger.warning("[SESSION] 스트림 close 실패", exc_info=True)
            return
    resp = getattr(stream, "response", None)
    if resp is not None:
        fn = getattr(resp, "close", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                logger.warning("[SESSION] 스트림 response.close 실패", exc_info=True)


class ClientSession:
    def __init__(self, client_host: str | None):
        self.session_id = str(uuid4())
        self.client_host = client_host or ""
        self.connected_at = _utc_now()
        self.last_seen_ts = time.time()
        self.presence_open = True
        self.processing = False
        self.request_id: str | None = None
        self.cancel_event = threading.Event()
        self.stream_min_level = logging.INFO
        self.out_queue: asyncio.Queue | None = None
        self.ask_loop: asyncio.AbstractEventLoop | None = None
        self._file_lock = threading.Lock()
        self._stream_lock = threading.Lock()
        self._active_stream = None
        self._log_path = os.path.join(SESSION_LOG_DIR, f"{self.session_id}.jsonl")
        self.current_run: Any = None
        self.turns: dict[str, Any] = {}
        self.stream_logs: bool = True
        self.user_id: str | None = None
        self.dropped_frames: int = 0
        self.dropped_reasons: dict[str, int] = {}
        _ensure_log_dirs()
        self.write_event(
            "connected",
            client=self.client_host,
        )

    def _rotate_log_file(self, reason: str) -> None:
        """append 실패 시 이 세션만 새 파일로 갈아탄다. RAG는 중단하지 않는다."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        prev = self._log_path
        self._log_path = os.path.join(
            SESSION_LOG_DIR, f"{self.session_id}_{stamp}.jsonl"
        )
        try:
            _ensure_log_dirs()
            _append_jsonl(
                self._log_path,
                {
                    "ts": _utc_now(),
                    "session_id": self.session_id,
                    "event": "log_file_rotated",
                    "prev_file": os.path.basename(prev),
                    "reason": (reason or "")[:300],
                },
                self._file_lock,
            )
        except Exception:
            logger.warning("[SESSION] 로그 파일 교체 실패 session=%s", self.session_id, exc_info=True)

    def write_event(self, event: str, **fields) -> None:
        payload = {
            "ts": _utc_now(),
            "session_id": self.session_id,
            "event": event,
        }
        if self.request_id:
            payload["request_id"] = self.request_id
        payload.update(fields)
        try:
            _append_jsonl(self._log_path, payload, self._file_lock)
        except Exception as exc:
            self._rotate_log_file(str(exc))
            try:
                _append_jsonl(self._log_path, payload, self._file_lock)
            except Exception:
                logger.warning("[SESSION] 세션 로그 기록 실패 session=%s", self.session_id, exc_info=True)

    def write_log_line(self, level: str, message: str) -> None:
        text = (message or "").rstrip("\n")
        if not text.strip():
            return
        self.write_event("log", level=level, message=text)

    def record_client_text(self, message: str, event: str = "log") -> None:
        """logger를 타지 않고 클라에 가는 RAG 문구를 세션 파일에도 남긴다."""
        text = (message or "").replace("<br>", "\n").rstrip("\n")
        if not text.strip():
            return
        if event == "log":
            self.write_log_line("INFO", text)
        else:
            self.write_event(event, message=text)

    def _count_drop(self, reason: str) -> None:
        self.dropped_frames += 1
        key = reason or "unknown"
        self.dropped_reasons[key] = self.dropped_reasons.get(key, 0) + 1

    def emit_to_client(self, message: Any, kind: str = "log") -> None:
        loop = self.ask_loop
        queue = self.out_queue
        if loop is None or queue is None:
            self._count_drop("no_queue")
            return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (kind, message))
        except Exception as e:
            self._count_drop(f"exception:{type(e).__name__}")

    def touch(self) -> None:
        self.last_seen_ts = time.time()

    def try_begin_ask(self) -> bool:
        if self.processing:
            return False
        self.processing = True
        self.touch()
        self.request_id = str(uuid4())
        self.cancel_event.clear()
        self.out_queue = asyncio.Queue()
        self.ask_loop = asyncio.get_running_loop()
        self.dropped_frames = 0
        self.dropped_reasons = {}
        return True

    def remember_turn(self, run: Any) -> None:
        """질문 1회를 세션 이력에 넣는다. 오래된 턴은 앞에서 버린다."""
        run_id = getattr(run, "run_id", None)
        if not run_id:
            return
        self.turns[run_id] = run
        extra = len(self.turns) - MAX_SESSION_TURNS
        if extra <= 0:
            return
        for key in list(self.turns.keys())[:extra]:
            self.turns.pop(key, None)

    def end_ask(self, reason: str) -> None:
        if self.request_id:
            self.write_event("ask_end", reason=reason)
        self.processing = False
        self.touch()
        self.request_id = None
        self.current_run = None
        # turns 는 세션이 살아있는 동안 삽입 순서로 유지한다. 상한은 remember_turn.
        if self.dropped_frames > 0:
            logger.warning(
                "[SESSION] dropped_frames=%s reasons=%s session=%s",
                self.dropped_frames,
                self.dropped_reasons,
                self.session_id,
            )
        self.out_queue = None
        self.ask_loop = None
        with self._stream_lock:
            self._active_stream = None
        maybe_destroy(self.session_id)

    def request_cancel(self) -> None:
        self.cancel_event.set()
        with self._stream_lock:
            _close_stream_quietly(self._active_stream)

    def attach_stream(self, stream: Any) -> None:
        with self._stream_lock:
            self._active_stream = stream
            if self.cancel_event.is_set():
                _close_stream_quietly(stream)
                self._active_stream = None

    def detach_stream(self, stream: Any) -> None:
        with self._stream_lock:
            if self._active_stream is stream:
                self._active_stream = None
        _close_stream_quietly(stream)


def create_session(client_host: str | None) -> ClientSession:
    _ensure_log_dirs()
    with _sessions_lock:
        n = len(_sessions)
        if n >= _max_connected:
            raise SessionCapacityError(n, _max_connected)
        session = ClientSession(client_host)
        _sessions[session.session_id] = session
    logger.info(f"[SESSION] 접속 {session.session_id} client={session.client_host}")
    return session


def get_session(session_id: str | None) -> ClientSession | None:
    if not session_id:
        return None
    with _sessions_lock:
        return _sessions.get(session_id.strip())


def mark_presence_closed(session_id: str, *, reason: str = "disconnected") -> None:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        return
    session.presence_open = False
    session.request_cancel()
    session.write_event(reason)
    logger.info(f"[SESSION] 접속 종료 {session_id} reason={reason}")
    maybe_destroy(session_id)


def sweep_idle_sessions(max_idle_sec: float) -> int:
    """하트비트/질문 없이 유휴인 접속만 닫는다. 처리 중인 세션은 건드리지 않는다."""
    if max_idle_sec <= 0:
        return 0
    now = time.time()
    stale_ids: list[str] = []
    with _sessions_lock:
        for sid, session in _sessions.items():
            if session.processing or not session.presence_open:
                continue
            last_seen = getattr(session, "last_seen_ts", now)
            if (now - last_seen) >= max_idle_sec:
                stale_ids.append(sid)
    closed = 0
    for sid in stale_ids:
        with _sessions_lock:
            session = _sessions.get(sid)
            if session is None or session.processing or not session.presence_open:
                continue
            if (time.time() - session.last_seen_ts) < max_idle_sec:
                continue
        mark_presence_closed(sid, reason="idle_timeout")
        closed += 1
    return closed


def maybe_destroy(session_id: str) -> None:
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is None:
            return
        if session.presence_open or session.processing:
            return
        _sessions.pop(session_id, None)
    try:
        from session_auth import release as release_user_binding
        release_user_binding(session_id)
    except Exception:
        logger.exception("[SESSION] 사용자 바인딩 해제 실패 session=%s", session_id)


def status_payload() -> dict:
    with _sessions_lock:
        items = list(_sessions.values())
    return {
        "connected": len(items),
        "max_connected": _max_connected,
        "processing": sum(1 for s in items if s.processing),
        "sessions": [
            {
                "session_id": s.session_id,
                "processing": s.processing,
                "client": s.client_host,
                "connected_at": s.connected_at,
            }
            for s in items
        ],
    }


class SessionLogRouter(logging.Handler):
    """전역 로거 하나. 레코드는 ContextVar 세션 파일/큐로만 보낸다."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            logger.warning("[SESSION] 로그 포맷 실패", exc_info=True)
            return
        if not (msg or "").strip():
            return
        session = get_current_session()
        if session is None:
            _ensure_log_dirs()
            _append_jsonl(
                SERVER_LOG_PATH,
                {
                    "ts": _utc_now(),
                    "session_id": None,
                    "event": "log",
                    "level": record.levelname,
                    "message": msg.rstrip("\n"),
                },
                _server_log_lock,
            )
            return
        prompt_trace = bool(getattr(record, "prompt_trace", False))
        session.write_log_line(record.levelname, msg)
        if session.stream_logs and (
            prompt_trace or record.levelno >= session.stream_min_level
        ):
            kind = "prompt" if prompt_trace else "log"
            session.emit_to_client(msg + "\n", kind=kind)


class _SessionIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        session = get_current_session()
        record.session_id = session.session_id[:8] if session else "server"
        return True


def install_log_router() -> None:
    global _router_installed
    target = logging.getLogger("my_rag")
    if _router_installed or any(isinstance(h, SessionLogRouter) for h in target.handlers):
        _router_installed = True
        return
    _ensure_log_dirs()
    target.setLevel(logging.DEBUG)
    target.propagate = False
    router = SessionLogRouter()
    router.setLevel(logging.DEBUG)
    router.setFormatter(logging.Formatter("%(message)s"))
    target.addHandler(router)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.addFilter(_SessionIdFilter())
    console.setFormatter(logging.Formatter("[%(session_id)s] %(message)s"))
    target.addHandler(console)
    _router_installed = True
