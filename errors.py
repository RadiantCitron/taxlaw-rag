# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""일시적 LLM/인프라 오류 판정의 유일한 소유자.

서빙 경로와 평가 하네스가 같은 키워드를 써야 '재시도할 것'과 '모델 오답'이
갈리지 않는다. 무거운 의존성을 import 하지 않는다.
"""
from __future__ import annotations

import re

TRANSIENT_KEYS = (
    "429",
    "quota",
    "rate limit",
    "resource exhausted",
    "500",
    "502",
    "503",
    "504",
    "unavailable",
    "overloaded",
    "high demand",
    "try again",
    "retry",
    "timeout",
    "deadline",
)


def is_transient(msg: str) -> bool:
    """잠깐 기다리면 풀리는 서버/할당량 오류인가.

    'rate' 단독 매칭은 쓰지 않는다. 'generate' 안에 'rate' 가 들어 있어
    평범한 생성 실패를 과부하로 오인하기 때문이다.
    """
    low = (msg or "").lower()
    return any(k in low for k in TRANSIENT_KEYS)


OVERLOAD_KEYS = (
    "429",
    "quota",
    "rate limit",
    "resource exhausted",
    "overloaded",
    "high demand",
)


def is_overload(msg: str) -> bool:
    """사용자에게 과부하로 안내할 업스트림 오류인가. is_transient 보다 좁다."""
    low = (msg or "").lower()
    return any(k in low for k in OVERLOAD_KEYS)


def retry_after_seconds(msg: str) -> float | None:
    """오류 메시지의 'retry in 56s' / 'retry after 41' 힌트를 초로 읽는다."""
    m = re.search(
        r"retry\s*(?:in|after)?\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*s",
        (msg or "").lower(),
    )
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None
