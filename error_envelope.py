# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""오류 봉투(error envelope) 빌더의 유일한 소유자.

사전 거부 HTTP 와 스트림 내 error 프레임이 같은 JSON 구조를 쓴다.
FastAPI 는 http_error() 안에서만 import 한다.
"""
from __future__ import annotations

from typing import Any

DOCS_PATH = "GET /options/schema"

STREAM_ONLY_CODES = frozenset({
    "GATE_FULL",
    "RESOURCE_UNAVAILABLE",
    "UPSTREAM_OVERLOAD",
    "UPSTREAM_ERROR",
    "PIPELINE_ERROR",
    "CANCELLED",
})

HTTP_BY_CODE = {
    "INVALID_OPTION_VALUE": 400,
    "UNKNOWN_MODEL": 400,
    "INCOMPATIBLE_OPTIONS": 400,
    "INVALID_BODY": 400,
    "PROMPT_INVALID": 400,
    "MISSING_SESSION": 400,
    "AUTH_REQUIRED": 401,
    "AUTH_FORBIDDEN": 403,
    "SESSION_BUSY": 409,
    "SESSION_CAPACITY": 503,
    "COOLDOWN_ACTIVE": 429,
    "DAILY_QUOTA_EXCEEDED": 429,
    "RATE_LIMITED": 429,
    "ADMIN_FORBIDDEN_IP": 403,
    "ADMIN_INVALID_BODY": 400,
    "ADMIN_DUPLICATE_ID": 409,
    "ADMIN_NO_SUCH_ID": 404,
    "ADMIN_SESSION_BUSY": 409,
    "INTERNAL_ERROR": 500,
}

EXIT_BY_CODE = {
    "INVALID_OPTION_VALUE": 2,
    "UNKNOWN_MODEL": 2,
    "INCOMPATIBLE_OPTIONS": 2,
    "INVALID_BODY": 2,
    "PROMPT_INVALID": 2,
    "MISSING_SESSION": 3,
    "AUTH_REQUIRED": 3,
    "AUTH_FORBIDDEN": 3,
    "SESSION_BUSY": 3,
    "SESSION_CAPACITY": 3,
    "COOLDOWN_ACTIVE": 3,
    "DAILY_QUOTA_EXCEEDED": 3,
    "RATE_LIMITED": 3,
    "GATE_FULL": 3,
    "RESOURCE_UNAVAILABLE": 3,
    "UPSTREAM_OVERLOAD": 4,
    "UPSTREAM_ERROR": 4,
    "PIPELINE_ERROR": 4,
    "CANCELLED": 130,
    "ADMIN_FORBIDDEN_IP": 3,
    "ADMIN_INVALID_BODY": 2,
    "ADMIN_DUPLICATE_ID": 3,
    "ADMIN_NO_SUCH_ID": 3,
    "ADMIN_SESSION_BUSY": 3,
    "INTERNAL_ERROR": 4,
}

SEVERITY_BY_CODE = {
    "INVALID_OPTION_VALUE": "reject",
    "UNKNOWN_MODEL": "reject",
    "INCOMPATIBLE_OPTIONS": "reject",
    "INVALID_BODY": "reject",
    "PROMPT_INVALID": "reject",
    "MISSING_SESSION": "reject",
    "AUTH_REQUIRED": "reject",
    "AUTH_FORBIDDEN": "reject",
    "SESSION_BUSY": "reject",
    "SESSION_CAPACITY": "retryable",
    "GATE_FULL": "retryable",
    "RESOURCE_UNAVAILABLE": "retryable",
    "UPSTREAM_OVERLOAD": "upstream",
    "UPSTREAM_ERROR": "upstream",
    "PIPELINE_ERROR": "upstream",
    "CANCELLED": "cancelled",
    "ADMIN_FORBIDDEN_IP": "reject",
    "ADMIN_INVALID_BODY": "reject",
    "ADMIN_DUPLICATE_ID": "reject",
    "ADMIN_NO_SUCH_ID": "reject",
    "ADMIN_SESSION_BUSY": "reject",
    "INTERNAL_ERROR": "upstream",
}


class PipelineFault(Exception):
    """파이프라인 실행 중 학생이 설정으로 고칠 수 없는 실패."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retry_after_sec: float | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retry_after_sec = retry_after_sec


def clip_given(value: Any, *, limit: int = 200) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    return value


def violation(
    field: str,
    given: Any,
    reason: str,
    expected: dict[str, Any],
    default: Any,
    hint: str = "",
) -> dict[str, Any]:
    item = {
        "field": field,
        "given": clip_given(given),
        "reason": reason,
        "expected": expected,
        "default": default,
    }
    if hint:
        item["hint"] = hint
    return item


def build_envelope(
    code: str,
    message: str,
    violations: list | tuple = (),
    ignored: list | tuple = (),
    *,
    request_id: str | None = None,
    session_id: str | None = None,
    terminal: bool | None = None,
    retry_after_sec: float | None = None,
    docs: str = DOCS_PATH,
) -> dict[str, Any]:
    err: dict[str, Any] = {
        "code": code,
        "message": message,
        "violations": list(violations or []),
        "ignored_fields": list(ignored or []),
        "request_id": request_id,
        "session_id": session_id,
        "docs": docs,
        "exit_hint": EXIT_BY_CODE.get(code, 1),
    }
    if code in SEVERITY_BY_CODE:
        err["severity"] = SEVERITY_BY_CODE[code]
    if terminal is not None:
        err["terminal"] = bool(terminal)
    if retry_after_sec is not None:
        err["retry_after_sec"] = retry_after_sec
    return {"ok": False, "error": err}


def unwrap_envelope(body: Any) -> dict | None:
    """HTTP JSON 에서 봉투를 꺼낸다. FastAPI 는 detail 로 한 겹 감쌀 수 있다."""
    if not isinstance(body, dict):
        return None
    if "error" in body and isinstance(body.get("error"), dict):
        return body
    detail = body.get("detail")
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        return detail
    return None


extract_envelope = unwrap_envelope


def http_error(
    code: str,
    message: str,
    *,
    violations: list | tuple = (),
    ignored: list | tuple = (),
    request_id: str | None = None,
    session_id: str | None = None,
    status: int | None = None,
    retry_after_sec: float | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    if code in STREAM_ONLY_CODES:
        raise AssertionError(f"{code} 는 SSE 전용 코드다")
    from fastapi import HTTPException

    env = build_envelope(
        code,
        message,
        violations,
        ignored,
        request_id=request_id,
        session_id=session_id,
        retry_after_sec=retry_after_sec,
    )
    resp_headers = dict(headers or {})
    if retry_after_sec is not None and "Retry-After" not in resp_headers:
        resp_headers["Retry-After"] = str(int(retry_after_sec))
    return HTTPException(
        status_code=status or HTTP_BY_CODE.get(code, 400),
        detail=env,
        headers=resp_headers or None,
    )


def envelope_code(env: dict | None) -> str:
    if not env:
        return ""
    err = env.get("error") if isinstance(env, dict) else None
    if isinstance(err, dict):
        return str(err.get("code") or "")
    return ""


def exit_code_for(code: str | None, *, http_status: int | None = None) -> int:
    if code and code in EXIT_BY_CODE:
        return EXIT_BY_CODE[code]
    if http_status in (400, 422):
        return 2
    if http_status in (401, 403, 409, 503):
        return 3
    if http_status:
        return 4
    return 1


def exit_code_from_envelope(env: dict | None) -> int:
    if not env:
        return 1
    err = env.get("error") or {}
    if isinstance(err, dict) and "exit_hint" in err:
        return int(err["exit_hint"])
    return exit_code_for(envelope_code(env))


def exit_code_from_done(done: dict | None) -> int:
    if not done:
        return 1
    status = done.get("status")
    reason = done.get("reason")
    if status == "completed":
        return 0
    if status == "cancelled":
        return EXIT_BY_CODE.get("CANCELLED", 130)
    if status in ("rejected", "error"):
        return exit_code_for(reason)
    return 1


def errors_codes_payload() -> dict[str, Any]:
    codes = sorted(
        set(HTTP_BY_CODE) | set(EXIT_BY_CODE) | set(STREAM_ONLY_CODES),
    )
    rows = []
    for code in codes:
        rows.append({
            "code": code,
            "http": HTTP_BY_CODE.get(code),
            "exit": EXIT_BY_CODE.get(code),
            "severity": SEVERITY_BY_CODE.get(code),
            "stream_only": code in STREAM_ONLY_CODES,
        })
    return {"codes": rows}
