# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
# --- START OF FILE session_auth.py ---
"""수업용 아이디 입구컷. RAG 파이프라인과 CS 프로세스를 모른다.

기동 시 --login-mode 가 켜져 있으면 /ask 에서 X-User-Id 를 검사한다.
명단은 session_users.json 시드 + 메모리. /admin 은 CS_ALLOW_IP 만 받는다.
비밀번호 필드는 없다. 식별만 하고 인증하지 않는다.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from fastapi import APIRouter, Depends, Request

from error_envelope import http_error
from pydantic import BaseModel, Field
from session_manager import get_session, status_payload

logger = logging.getLogger("my_rag")

USERS_FILENAME = "session_users.json"
USERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), USERS_FILENAME)

_lock = threading.Lock()
_login_mode = False
_users: dict[str, dict[str, Any]] = {}
_bound: dict[str, str] = {}
_bound_by_session: dict[str, str] = {}


def _cs_allow_ips() -> set[str]:
    raw = os.environ.get("CS_ALLOW_IP", "127.0.0.1")
    ips = {part.strip() for part in raw.split(",") if part.strip()}
    if not ips:
        ips = {"127.0.0.1"}
    if "127.0.0.1" in ips:
        ips.add("::1")
        ips.add("localhost")
        ips.add("::ffff:127.0.0.1")
    return ips


def _ttl_expired(rec: dict[str, Any], now: float | None = None) -> bool:
    ttl = int(rec.get("ttl_sec") or 0)
    started = rec.get("first_bind_ts")
    if ttl <= 0 or started is None:
        return False
    return ((now or time.time()) - float(started)) >= ttl


def _unbind_session(session_id: str) -> None:
    uid = _bound_by_session.pop(session_id, None)
    if uid and _bound.get(uid) == session_id:
        _bound.pop(uid, None)


def _user_public(rec: dict[str, Any], user_id: str) -> dict[str, Any]:
    sid = _bound.get(user_id)
    live = bool(sid and get_session(sid) is not None)
    return {
        "id": user_id,
        "ttl_sec": int(rec.get("ttl_sec") or 0),
        "bound": live,
        "session_id": sid if live else None,
        "expired": _ttl_expired(rec),
    }


def init(*, login_mode: bool) -> None:
    global _login_mode
    _login_mode = bool(login_mode)
    load_users_file()
    n = len(_users)
    msg = f"[AUTH] login_mode={str(_login_mode).lower()} users={n} file={USERS_PATH}"
    logger.info(msg)
    print(msg, flush=True)


def load_users_file(path: str | None = None) -> None:
    target = path or USERS_PATH
    with _lock:
        _users.clear()
        _bound.clear()
        _bound_by_session.clear()
        if not os.path.isfile(target):
            return
        with open(target, encoding="utf-8") as fh:
            data = json.load(fh)
        rows = data.get("users") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError(f"명단 JSON 형식 오류: {target}")
        for row in rows:
            if not isinstance(row, dict):
                continue
            uid = str(row.get("id") or "").strip()
            if not uid:
                continue
            _users[uid] = {
                "ttl_sec": max(0, int(row.get("ttl_sec") or 0)),
                "first_bind_ts": None,
            }


def release(session_id: str) -> None:
    """세션이 사라진 뒤 아이디를 다시 쓸 수 있게 한다. 명단·TTL 시계는 유지."""
    if not session_id:
        return
    with _lock:
        _unbind_session(session_id)


def gate(request: Request, session) -> None:
    """로그인 모드에서 /ask 입구컷. 공개 모드면 아무 것도 하지 않는다."""
    if not _login_mode:
        return
    user_id = (request.headers.get("X-User-Id") or "").strip()
    bind_session_user(session, user_id)


def bind_session_user(session, user_id: str) -> None:
    if not _login_mode:
        return
    already = getattr(session, "user_id", None)
    with _lock:
        if already:
            rec = _users.get(already)
            if rec is None:
                raise http_error("AUTH_FORBIDDEN", "이 세션의 아이디가 명단에서 제거되었습니다.", status=403)
            if _ttl_expired(rec):
                raise http_error("AUTH_FORBIDDEN", "아이디 유효 시간이 만료되었습니다.", status=403)
            if user_id and user_id != already:
                raise http_error("AUTH_FORBIDDEN", "이 세션은 다른 아이디로 이미 로그인되어 있습니다.", status=403)
            return
        if not user_id:
            raise http_error(
                "AUTH_REQUIRED",
                "로그인 모드입니다. X-User-Id 헤더가 필요합니다.",
                status=401,
            )
        rec = _users.get(user_id)
        if rec is None:
            raise http_error("AUTH_FORBIDDEN", "명단에 없는 아이디입니다.", status=403)
        if _ttl_expired(rec):
            raise http_error("AUTH_FORBIDDEN", "아이디 유효 시간이 만료되었습니다.", status=403)
        other = _bound.get(user_id)
        if other and other != session.session_id:
            other_session = get_session(other)
            if other_session is not None:
                logger.warning(
                    "[AUTH] 중복 로그인 입구컷 id=%s live_session=%s new_session=%s",
                    user_id,
                    other,
                    session.session_id,
                )
                raise http_error(
                    "AUTH_FORBIDDEN",
                    "이미 다른 클라이언트에서 사용 중인 아이디입니다.",
                    status=403,
                )
            _unbind_session(other)
        session.user_id = user_id
        _bound[user_id] = session.session_id
        _bound_by_session[session.session_id] = user_id
        if rec.get("first_bind_ts") is None:
            rec["first_bind_ts"] = time.time()
        logger.info("[AUTH] bind id=%s session=%s", user_id, session.session_id)


def login_mode_enabled() -> bool:
    return _login_mode


def require_cs_ip(request: Request) -> None:
    host = request.client.host if request.client else ""
    allowed = _cs_allow_ips()
    if host not in allowed:
        logger.warning("[AUTH] 관리 API 입구컷 ip=%s allowed=%s", host, sorted(allowed))
        raise http_error("ADMIN_FORBIDDEN_IP", "관리 요청이 허용되지 않은 IP입니다.")


class UserIn(BaseModel):
    id: str = Field(min_length=1)
    ttl_sec: int = Field(default=3600, ge=0)


router = APIRouter(prefix="/admin", dependencies=[Depends(require_cs_ip)])


def _admin_status_unlocked() -> dict[str, Any]:
    live_ids: list[str] = []
    stale: list[str] = []
    for uid, sid in _bound.items():
        if get_session(sid) is None:
            stale.append(sid)
        else:
            live_ids.append(uid)
    for sid in stale:
        _unbind_session(sid)
    sess = status_payload()
    return {
        "ok": True,
        "login_mode": _login_mode,
        "cs_allow_ip": sorted(_cs_allow_ips()),
        "connected": {
            "current": sess["connected"],
            "max": sess["max_connected"],
        },
        "processing": {"current": sess["processing"]},
        "user_ids": sorted(live_ids),
        "users": [_user_public(rec, uid) for uid, rec in sorted(_users.items())],
    }


@router.get("/status")
def admin_status():
    with _lock:
        return _admin_status_unlocked()


@router.get("/users")
def admin_users():
    with _lock:
        return {"users": [_user_public(rec, uid) for uid, rec in sorted(_users.items())]}


@router.post("/users")
def admin_add_user(body: UserIn):
    uid = body.id.strip()
    if not uid:
        raise http_error("ADMIN_INVALID_BODY", "아이디가 비어 있습니다.")
    with _lock:
        if uid in _users:
            raise http_error("ADMIN_DUPLICATE_ID", "이미 있는 아이디입니다.")
        _users[uid] = {
            "ttl_sec": int(body.ttl_sec),
            "first_bind_ts": None,
        }
        logger.info("[AUTH] user add id=%s ttl_sec=%s", uid, body.ttl_sec)
        return {"ok": True, "user": _user_public(_users[uid], uid)}


@router.delete("/users/{user_id}")
def admin_del_user(user_id: str):
    uid = (user_id or "").strip()
    with _lock:
        rec = _users.get(uid)
        if rec is None:
            raise http_error("ADMIN_NO_SUCH_ID", "없는 아이디입니다.")
        sid = _bound.get(uid)
        if sid and get_session(sid) is not None:
            raise http_error(
                "ADMIN_SESSION_BUSY",
                "접속 중인 아이디는 삭제할 수 없습니다.",
            )
        if sid:
            _unbind_session(sid)
        _users.pop(uid, None)
        logger.info("[AUTH] user del id=%s", uid)
        return {"ok": True, "id": uid}
