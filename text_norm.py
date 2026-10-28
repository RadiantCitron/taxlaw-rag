# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""충실성 픽스처 생성·채점이 공유하는 문자열 정규화의 유일한 소유자.

build_faithfulness_testset 과 evaluate_faithfulness 가 같은 규칙을 쓴다.
생성기와 채점기가 서로 import 하지 않도록 제3 모듈로 분리했다.
"""
from __future__ import annotations

import re

NUM_UNIT_RE = re.compile(
    r"(\d[\d,]*)\s*(퍼센트|%|일|개월|년|월|주|배|억\s*원|만\s*원|천\s*원|원|"
    r"분의\s*\d+|호|항|세|제곱미터|㎡|미터)"
)


def norm(s: str) -> str:
    """공백을 모두 제거한 비교용 문자열."""
    return re.sub(r"\s+", "", s or "")


def contains(text: str, token: str) -> bool:
    """공백 무시 부분 일치."""
    return norm(token) in norm(text)
