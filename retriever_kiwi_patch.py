# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
# retriever_kiwi_patch.py
"""
[구현 2/2] 한국어 형태소 기반 키워드 추출 (Kiwi).

이전 버전(setup_korean_search.py, pg_bigm 서버 확장 + tsvector 컬럼 방식)을
대체한다. pg_bigm 경로는 서버 확장 설치가 필요했고, 실제 함수/타입명이
버전마다 다를 수 있어 검증이 불확실했다.

이번 방법은 그 불확실성이 없다:
    - PostgreSQL 서버를 전혀 안 건드린다. 확장 설치, 컬럼 추가, 인덱스
      재구축이 전부 필요 없다.
    - kiwipiepy 는 순수 파이썬 패키지라 'pip install kiwipiepy' 한 줄로 끝난다.
    - retriever.py 의 기존 함수 _keywords_from_text() 는 "키워드 문자열
      리스트"를 반환하고, 그걸 호출부가 "|".join() 해서 content ~ %s
      (PostgreSQL 정규식)에 그대로 넣는다. 즉 이 함수의 '반환 형식'만
      지키면 내부 구현을 정규식에서 형태소 분석으로 바꿔도 검색 SQL,
      get_hybrid_assembled_context(), 그 어떤 다른 코드도 안 건드린다.
      이것이 '드롭인(drop-in) 교체'다.

왜 이게 더 나은가 (구체적 예):
    질문: "소득세법상 공제 한도는?"
    - 기존(정규식 \\b\\w{2,}\\b): "소득세법상", "공제", "한도" 를 그대로 키워드로 씀.
      DB content 에 "소득세법"만 있고 "소득세법상"(조사 '상'이 붙은 형태)은
      없으면, content ~ '소득세법상|공제|한도' 매칭에서 "소득세법상" 조각이
      헛스윙한다(다른 두 키워드로는 매칭되지만, 이건 우연히 되는 것).
    - Kiwi: "소득세법상"을 형태소로 분석해 "소득세법"(명사) + "상"(접사/조사)
      으로 분리한다. 그러면 키워드가 "소득세법"이 되어 DB content 와
      정확히 맞아떨어진다.

적용 방법:
    1) pip install kiwipiepy
    2) 이 파일의 get_keywords_kiwi() 함수를 retriever.py 에 복사해 넣는다
       (또는 이 파일을 그대로 import 해도 된다 — 아래 '통합 방법 B' 참고)
    3) retriever.py 안에서 _keywords_from_text(...) 를 호출하는 두 곳
       (231번째 줄, 236번째 줄 부근)을 get_keywords_kiwi(...) 로 바꾼다.
       인자 순서와 개수가 동일하므로 이름만 바꾸면 된다.

    예)
        # 기존
        keywords = _keywords_from_text(expanded_query, query_text)
        # 교체 후
        keywords = get_keywords_kiwi(expanded_query, query_text)

    통합 방법 B (retriever.py 를 아예 안 건드리고 싶다면):
        retriever.py 맨 위에 이 한 줄만 추가한다.
            from retriever_kiwi_patch import get_keywords_kiwi as _keywords_from_text
        이러면 기존 _keywords_from_text 정의를 이 파일의 것으로 덮어써서
        호출부 코드를 단 한 줄도 안 고쳐도 된다. (단, 이 경우 원래
        _keywords_from_text 정의보다 import 문이 파일 아래에 있어야
        마지막 정의가 이긴다 — 파이썬은 나중에 정의된 이름이 유효하다.
        더 안전한 건 3)번처럼 호출부 두 곳의 이름을 바꾸는 것이다.)

실패 안전장치:
    kiwipiepy 가 설치되지 않았거나 로드에 실패하면, 예외를 삼키지 않고
    명확한 ImportError 메시지로 즉시 알린다 — 조용히 정규식으로
    되돌아가지 않는다. 조용한 폴백은 "왜 검색이 여전히 안 되지?"를
    디버깅할 때 원인을 숨기는 나쁜 습관이기 때문이다. 폴백이 필요하면
    호출부에서 명시적으로 try/except 로 감쌀 것(아래 예시 참고).
"""
from __future__ import annotations

import re

import threading

from search_constants import KEYWORD_SKIP, SEARCH_KEYWORD_CAP

# Kiwi 가 명사/고유명사/동사 어간 등으로 뽑아내는 품사 태그 중,
# 검색 키워드로 쓸 만한 것만 고른다. 조사(J*)·어미(E*)·구두점(S*) 등은 제외.
_USEFUL_POS = {"NNG", "NNP", "NNB", "SL", "SH", "SN", "VV", "VA", "XR"}

_kiwi_instance = None
_kiwi_lock = threading.Lock()


def _get_kiwi():
    """Kiwi 인스턴스를 프로세스당 한 번만 로드해 재사용한다(로드 비용이 있음)."""
    global _kiwi_instance
    if _kiwi_instance is None:
        with _kiwi_lock:
            if _kiwi_instance is None:
                try:
                    from kiwipiepy import Kiwi
                except ImportError as e:
                    raise ImportError(
                        "[FATAL] kiwipiepy 가 설치되어 있지 않습니다. "
                        "'pip install kiwipiepy' 로 설치하세요. "
                        "(조용히 정규식으로 폴백하지 않습니다 — 왜 검색 품질이 "
                        "바뀌었는지 원인을 숨기지 않기 위함입니다)"
                    ) from e
                _kiwi_instance = Kiwi()
    return _kiwi_instance


def get_keywords_kiwi(text: str, fallback: str, cap: int = SEARCH_KEYWORD_CAP) -> list[str]:
    """
    Kiwi 형태소 분석으로 검색용 키워드를 뽑는다.
    반환 형식은 기존 _keywords_from_text() 와 완전히 동일하다
    (문자열 리스트, 최대 cap 개, 중복 제거, 숫자 단독 토큰 제외).

    text 가 비어 있거나 형태소 분석 결과가 없으면 fallback 앞부분을 쓴다
    (기존 함수와 동일한 안전장치).
    """
    kiwi = _get_kiwi()
    keywords: list[str] = []

    for token in kiwi.tokenize(text or ""):
        form = token.form
        pos = token.tag
        if pos not in _USEFUL_POS:
            continue
        if len(form) < 2:  # 한 글자 토큰은 잡음이 많아 기존과 동일하게 제외
            continue
        if form in KEYWORD_SKIP or form.isdigit():
            continue
        if form not in keywords:
            keywords.append(form)
        if len(keywords) >= cap:
            break

    if not keywords:
        keywords = [(fallback or text or "")[:20] or "법령"]
    return keywords


def get_keywords_with_fallback(text: str, fallback: str, cap: int = SEARCH_KEYWORD_CAP) -> list[str]:
    """
    Kiwi 를 우선 시도하고, kiwipiepy 자체가 설치 안 됐을 때만 기존 정규식
    방식으로 자동 전환하고 싶다면 이 함수를 대신 쓴다(명시적 선택 사항).
    이 함수를 쓰면 '조용한 폴백'이 되므로, 어느 방식이 실제로 쓰였는지
    로그로 반드시 확인할 것(첫 호출 시 한 번만 경고를 찍는다).
    """
    global _fallback_warned
    try:
        return get_keywords_kiwi(text, fallback, cap)
    except ImportError:
        if not globals().get("_fallback_warned"):
            print("[WARN] kiwipiepy 미설치 → 정규식 키워드 추출로 폴백합니다. "
                  "(이 경고는 프로세스당 한 번만 출력됩니다)")
            globals()["_fallback_warned"] = True
        return _keywords_from_text_regex(text, fallback, cap)


def _keywords_from_text_regex(text: str, fallback: str, cap: int = SEARCH_KEYWORD_CAP) -> list[str]:
    """기존 retriever.py 의 _keywords_from_text() 와 동일한 구현(폴백용으로 보존)."""
    keywords = []
    for word in re.findall(r"\b\w{2,}\b", text or ""):
        if word in KEYWORD_SKIP or word.isdigit():
            continue
        if word not in keywords:
            keywords.append(word)
        if len(keywords) >= cap:
            break
    if not keywords:
        keywords = [(fallback or text or "")[:20] or "법령"]
    return keywords
