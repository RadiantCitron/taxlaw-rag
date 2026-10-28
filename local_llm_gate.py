# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
# --- START OF FILE local_llm_gate.py ---
import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from fastapi import Request
from session_manager import get_current_session
from error_envelope import build_envelope, http_error
from sse import SSE_KIND_ERROR, SSE_KIND_SYSTEM

logger = logging.getLogger("my_rag")

# --- 게이트 동시성 제어 상태 ---
_local_sem = None
_ask_sem = None
_local_waiting = 0
_ask_waiting = 0
_max_waiting = 5
_max_concurrent_user = 4

# --- 쿼터 & 쿨다운 정책 설정 ---
KST = timezone(timedelta(hours=9))
_quota_lock = threading.Lock()
_quota_file = os.getenv("RATE_LIMIT_FILE", "logs/quotas.json")
_quotas_cache: dict[str, dict[str, dict]] = {}  # { "YYYY-MM-DD": { client_key: {"count": int, "last_ask_ts": float} } }


def _today_kst() -> str:
    """한국 표준시(KST) 기준 날짜 문자열 YYYY-MM-DD"""
    return datetime.now(KST).strftime("%Y-%m-%d")


def _get_daily_limit() -> int:
    try:
        return max(1, int(os.getenv("RATE_LIMIT_DAILY", "5")))
    except Exception:
        return 5


def _get_cooldown_sec() -> int:
    try:
        return max(0, int(os.getenv("RATE_LIMIT_COOLDOWN_SEC", "900")))  # 기본 15분 (900초)
    except Exception:
        return 900


def _is_rate_limit_enabled() -> bool:
    val = os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower()
    return val not in ("false", "0", "no", "off")


def _load_quotas_from_disk() -> None:
    global _quotas_cache
    if not os.path.exists(_quota_file):
        _quotas_cache = {}
        return
    try:
        with open(_quota_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                _quotas_cache = data
    except Exception as e:
        logger.warning(f"[GATE] 쿼터 원장 로드 실패 (초기화 진행): {e}")
        _quotas_cache = {}


def _save_quotas_to_disk() -> None:
    try:
        os.makedirs(os.path.dirname(_quota_file) or ".", exist_ok=True)
        tmp_file = f"{_quota_file}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(_quotas_cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, _quota_file)
    except Exception as e:
        logger.error(f"[GATE] 쿼터 원장 저장 실패: {e}")


def extract_client_key(request: Request) -> tuple[str, str, str]:
    """클라우드플레어 IP, 프록시 X-Forwarded-For, 직접 호스트 및 X-Device-Id 추출"""
    cf_ip = (request.headers.get("cf-connecting-ip") or "").strip()
    xff = (request.headers.get("x-forwarded-for") or "").strip()
    direct_ip = (request.client.host if request.client else "127.0.0.1").strip()
    
    if cf_ip:
        ip = cf_ip
    elif xff:
        ip = xff.split(",")[0].strip()
    else:
        ip = direct_ip
        
    device_id = (request.headers.get("x-device-id") or "").strip()
    # 시크릿 창이나 localStorage 삭제로 우회하는 것을 막기 위해 IP를 1차 기준 키로 통제
    primary_key = ip
    return primary_key, ip, device_id


def get_quota_status(request: Request) -> dict:
    """현재 클라이언트의 쿼터 사용량 및 쿨다운 상태를 조회한다."""
    if not _is_rate_limit_enabled():
        return {
            "ok": True,
            "enabled": False,
            "daily_limit": 9999,
            "used_today": 0,
            "remaining_today": 9999,
            "cooldown_sec": 0,
            "cooldown_remaining_sec": 0,
            "allowed": True,
            "reason": None,
            "message": "레이트 리미트 비활성화 상태",
        }

def _evaluate_quota_record(
    used_count: int,
    last_ask_ts: float,
    daily_limit: int,
    cooldown_sec: int,
    now: float,
) -> tuple[bool, str | None, str, int, int]:
    """쿼터 레코드로부터 허용 여부, 거부 사유, 메시지, 남은 쿨다운초, 오늘 남은 횟수를 순수 계산한다."""
    elapsed = now - last_ask_ts
    cooldown_rem = max(0, int(cooldown_sec - elapsed)) if (last_ask_ts > 0 and cooldown_sec > 0) else 0
    remaining_today = max(0, daily_limit - used_count)

    allowed = True
    reason = None
    message = "질문 가능"

    # [중요 버그 수정]: 일일 5회 소진 여부를 쿨다운보다 '먼저' 검사해야 함!
    # 5회를 다 썼는데 "15분 쿨다운 중"이라는 오해의 소지가 있는 메시지가 나오는 것을 방지
    if remaining_today <= 0:
        allowed = False
        reason = "DAILY_QUOTA_EXCEEDED"
        message = f"오늘 무료 질문 한도({daily_limit}회)를 모두 소진했습니다. 내일 00:00에 초기화됩니다."
    elif cooldown_rem > 0:
        allowed = False
        reason = "COOLDOWN_ACTIVE"
        min_rem = (cooldown_rem + 59) // 60
        message = f"15분 쿨다운 중입니다. 약 {min_rem}분 ({cooldown_rem}초) 후 다시 질문하실 수 있습니다."

    return allowed, reason, message, cooldown_rem, remaining_today


def get_quota_status(request: Request) -> dict:
    """
    현재 클라이언트(Cloudflare IP 기반)의 일일 질문 잔여 횟수 및 쿨다운 상태를 반환한다.
    UI 렌더링용 API (/api/quota-status) 및 선제 검증에서 호출된다.
    """
    if not _is_rate_limit_enabled():
        return {
            "ok": True,
            "enabled": False,
            "allowed": True,
            "message": "레이트 리밋이 비활성화되어 있습니다.",
        }

    key, ip, dev = extract_client_key(request)
    today = _today_kst()
    daily_limit = _get_daily_limit()
    cooldown_sec = _get_cooldown_sec()
    now = time.time()

    with _quota_lock:
        today_records = _quotas_cache.setdefault(today, {})
        client_rec = today_records.get(key, {"count": 0, "last_ask_ts": 0.0})
        used_count = int(client_rec.get("count", 0))
        last_ask_ts = float(client_rec.get("last_ask_ts", 0.0))

    allowed, reason, message, cooldown_rem, remaining_today = _evaluate_quota_record(
        used_count, last_ask_ts, daily_limit, cooldown_sec, now
    )

    return {
        "ok": True,
        "enabled": True,
        "daily_limit": daily_limit,
        "used_today": used_count,
        "remaining_today": remaining_today,
        "cooldown_sec": cooldown_sec,
        "cooldown_remaining_sec": cooldown_rem,
        "allowed": allowed,
        "reason": reason,
        "message": message,
    }


def check_and_consume_quota(request: Request, session=None) -> dict:
    """
    질문 시작 시 쿼터와 쿨다운을 검증하고, 통과 시 '선제적으로 1회 차감' 및 시각을 갱신한다.
    실패 시 HTTP 429(COOLDOWN_ACTIVE 또는 DAILY_QUOTA_EXCEEDED)를 발생시킨다.
    [단일 락 원자성 - G-1]: 조회 ➔ 판정 ➔ 차감 ➔ 저장을 단일 _quota_lock 안에서 처리하여 TOCTOU를 제거한다.
    """
    if not _is_rate_limit_enabled():
        return {"allowed": True}

    key, ip, dev = extract_client_key(request)
    today = _today_kst()
    daily_limit = _get_daily_limit()
    cooldown_sec = _get_cooldown_sec()
    now = time.time()

    with _quota_lock:
        today_records = _quotas_cache.setdefault(today, {})
        client_rec = today_records.setdefault(key, {"count": 0, "last_ask_ts": 0.0})
        used_count = int(client_rec.get("count", 0))
        last_ask_ts = float(client_rec.get("last_ask_ts", 0.0))

        allowed, reason, msg, cooldown_rem, remaining_today = _evaluate_quota_record(
            used_count, last_ask_ts, daily_limit, cooldown_sec, now
        )

        if not allowed:
            session_id = getattr(session, "session_id", None) if session else None
            retry_after = cooldown_rem if reason == "COOLDOWN_ACTIVE" else None
            if session is not None:
                session.write_event("quota_rejected", reason=reason, message=msg)
            raise http_error(
                reason,
                msg,
                session_id=session_id,
                status=429,
                retry_after_sec=retry_after,
            )

        # 검증 통과 ➔ 단일 락 안에서 즉시 원자적 1회 선제 차감 (TOCTOU 방지)
        client_rec["count"] = used_count + 1
        client_rec["last_ask_ts"] = now

        # 7일 이상 지난 오래된 날짜 데이터 정리
        cutoff_date = (datetime.now(KST) - timedelta(days=7)).strftime("%Y-%m-%d")
        for old_day in list(_quotas_cache.keys()):
            if old_day < cutoff_date:
                del _quotas_cache[old_day]

        _save_quotas_to_disk()

        used = client_rec["count"]
        rem = max(0, daily_limit - used)

    if session is not None:
        session.write_event("quota_consumed", used=used, remaining=rem, client_key=key)

    logger.info(f"[GATE] 쿼터 1회 차감 완료: key={key[:12]}... used={used}/{daily_limit} rem={rem}")
    return {"allowed": True, "used_today": used, "remaining_today": rem}


def init_gate(max_concurrent_user: int = 4, max_waiting: int = 5):
    """서버 시작 시 입장 세마포어와 로컬 LLM 세마포어를 초기화하고 쿼터 원장을 로드한다."""
    global _local_sem, _ask_sem, _max_waiting, _max_concurrent_user
    _max_waiting = max(0, int(max_waiting))
    _max_concurrent_user = max(1, int(max_concurrent_user))
    # 기동 시 항상 현재 상한으로 다시 만든다. 이미 있으면 크기가 안 바뀌던 문제를 막는다.
    _local_sem = asyncio.Semaphore(1)
    _ask_sem = asyncio.Semaphore(_max_concurrent_user)
    with _quota_lock:
        _load_quotas_from_disk()


def get_local_llm_status() -> dict:
    """웹에서 조회할 로컬 LLM 상태"""
    return {
        "busy": _local_sem.locked() if _local_sem else False,
        "waiting": _local_waiting,
        "max_waiting": _max_waiting,
        "available": (_local_waiting < _max_waiting),
    }


def get_gate_status() -> dict:
    return {
        "max_concurrent_user": _max_concurrent_user,
        "ask_waiting": _ask_waiting,
        "max_waiting": _max_waiting,
        "local_busy": _local_sem.locked() if _local_sem else False,
        "local_waiting": _local_waiting,
    }


async def _put_rag_notice(yield_queue: asyncio.Queue, message: str, *, kind: str = "log") -> None:
    session = get_current_session()
    if session is not None:
        session.record_client_text(message, event=kind)
    await yield_queue.put((kind, message))


async def _put_gate_full(yield_queue: asyncio.Queue, message: str) -> None:
    session = get_current_session()
    env = build_envelope(
        "GATE_FULL",
        message,
        request_id=getattr(session, "request_id", None) if session else None,
        session_id=getattr(session, "session_id", None) if session else None,
        terminal=True,
        retry_after_sec=30,
    )
    if session is not None:
        session.write_event("gate_full", message=message)
    await yield_queue.put((SSE_KIND_ERROR, json.dumps(env, ensure_ascii=False)))


async def _acquire_sem(
    sem: asyncio.Semaphore,
    waiting_holder: str,
    request: Request,
    yield_queue: asyncio.Queue,
    full_msg: str,
    wait_msg: str | None,
) -> bool:
    """세마포어를 얻는다. 대기 한도 초과면 False. 연결이 끊기면 락을 붙잡지 않는다."""
    global _local_waiting, _ask_waiting
    if sem is None:
        return False

    if sem.locked():
        if waiting_holder == "local":
            if _local_waiting >= _max_waiting:
                await _put_gate_full(yield_queue, full_msg)
                return False
        elif _ask_waiting >= _max_waiting:
            await _put_gate_full(yield_queue, full_msg)
            return False

    if waiting_holder == "local":
        _local_waiting += 1
        queued = _local_waiting
    else:
        _ask_waiting += 1
        queued = _ask_waiting

    if wait_msg and sem.locked():
        await _put_rag_notice(yield_queue, wait_msg.replace("{n}", str(max(queued - 1, 0))), kind=SSE_KIND_SYSTEM)

    if request is not None and await request.is_disconnected():
        return False

    acquire_task = asyncio.create_task(sem.acquire())
    acquired = False
    try:
        while not acquire_task.done():
            if request is not None and await request.is_disconnected():
                return False
            try:
                await asyncio.wait_for(asyncio.shield(acquire_task), timeout=0.05)
                break
            except asyncio.TimeoutError:
                continue
        acquired = True
        return True
    except asyncio.CancelledError:
        return False
    finally:
        if waiting_holder == "local":
            _local_waiting -= 1
        else:
            _ask_waiting -= 1
        if not acquired:
            if acquire_task.done() and not acquire_task.cancelled():
                sem.release()
            else:
                acquire_task.cancel()


async def acquire_ask_slot(request: Request, yield_queue: asyncio.Queue) -> bool:
    """동시에 처리할 /ask 상한. 초과 대기는 입구컷."""
    return await _acquire_sem(
        _ask_sem,
        "ask",
        request,
        yield_queue,
        "현재 서버가 혼잡하여 대기열이 꽉 찼습니다. 잠시 후 다시 시도해주세요. (입구컷)",
        "앞에 {n}명이 대기 중입니다. 차례가 오면 자동으로 시작됩니다...",
    )


def release_ask_slot():
    if _ask_sem:
        _ask_sem.release()


async def acquire_local_llm_lock(request: Request, yield_queue: asyncio.Queue) -> bool:
    """로컬 GGUF는 한 명만 추론한다."""
    return await _acquire_sem(
        _local_sem,
        "local",
        request,
        yield_queue,
        "현재 서버가 혼잡하여 대기열이 꽉 찼습니다. 잠시 후 다시 시도해주세요. (입구컷)",
        "앞에 {n}명이 대기 중입니다. 차례가 오면 자동으로 시작됩니다...",
    )


def release_local_llm_lock():
    """로컬 LLM 락 반납"""
    if _local_sem:
        _local_sem.release()
