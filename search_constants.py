# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""하이브리드 키워드 추출 상수의 유일한 소유자.

retriever.py 와 retriever_kiwi_patch.py 가 이 값을 복사하지 말고 import 한다.
"""
from __future__ import annotations

import re

SEARCH_KEYWORD_CAP = 12
KEYWORD_SKIP = {"원인", "따른", "적용", "있는", "대한", "그리고", "또는", "후"}
EMPTY_CONTEXT_MSG = "관련된 문서를 찾을 수 없습니다."


def keyword_bind_values(keywords: list[str] | None) -> tuple[str, str]:
    """하이브리드 SQL 에 넣을 (정규식 OR, trigram 비교 문자열).

    `full_text ~ pattern` 은 키워드를 `|` 로 이은 POSIX 정규식이다.
    `similarity(full_text, text)` 는 같은 파이프 문자열을 넣으면 안 된다.
    파이프는 trigram 유사도에서 메타문자가 아니라 비교 대상 문자다.
    """
    cleaned = [str(k).strip() for k in (keywords or []) if str(k).strip()]
    if not cleaned:
        return "", ""
    pattern = "|".join(re.escape(k) for k in cleaned)
    text = " ".join(cleaned)
    return pattern, text


def keywords_from_text(text: str, fallback: str) -> list[str]:
    keywords = []
    for word in re.findall(r"\b\w{2,}\b", text or ""):
        if word in KEYWORD_SKIP or word.isdigit():
            continue
        if word not in keywords:
            keywords.append(word)
        if len(keywords) >= SEARCH_KEYWORD_CAP:
            break
    if not keywords:
        keywords = [(fallback or text or "")[:20] or "법령"]
    return keywords
