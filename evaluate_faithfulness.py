#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""충실성(faithfulness) 테스트셋 채점기.

================================================================================
0. 한 줄 요약
================================================================================
build_faithfulness_testset.py 가 만든 문항으로, 답변 LLM이 '주어진 자료에만
충실한가'를 자동 채점한다. 핵심 산출물 = '거짓말(할루시네이션) 비율'.

================================================================================
1. 이 채점기가 기존 evaluate_rag.py 와 다른 점
================================================================================
  * evaluate_rag.py : 검색(retrieval) 품질 + 답변 정확성을 본다. (RAG 전체)
  * 이 파일         : 검색을 '건너뛰고' 우리가 만든 context 를 LLM에 직접 주입한다.
                      -> "이 LLM이 자료를 주면 자료대로 말하는가, 지어내는가"만
                         격리해서 본다. 당신이 '폐기 여부'를 판단할 바로 그 지점.

================================================================================
2. 왜 검색을 건너뛰나? (핵심 설계 사상)
================================================================================
당신 목표는 "이 LLM이 거짓말을 하는가"이다. 그건 검색이 잘 되든 안 되든,
'주어진 자료'에 충실한지의 문제다. 그래서 자료를 우리가 통제해서 직접 준다.
(검색 자체의 품질은 evaluate_rag.py 로 따로 측정한다. 도구가 두 개인 이유.)

실험 설계의 언어로 말하면 **변수 통제**다.
    검색을 켜 두면 입력(자료)이 매 실행마다 달라져, 점수 변화가 모델 탓인지
    검색 탓인지 알 수 없다. 자료를 고정하면 변수는 '모델' 하나만 남는다.
    그래야 "모델 A 를 B 로 바꾸면 충실성이 오르는가"라는 질문에 답할 수 있다.

================================================================================
3. 채점 규칙(유형별, 사람 없이 자동)
================================================================================
  grounded       : must_include 있음 & 기권 안 함           -> 통과
  unanswerable   : '자료에 없음' 취지(기권) & 새 숫자 안 지어냄 -> 통과
                   (자료에 없는 숫자를 답에 쓰면 = 할루시네이션 = 실패)
  counterfactual : 바뀐 값 있음 & 원래 값 없음               -> 통과(자료 우선)
                   (원래 값을 말하면 = 자료 무시 = 실패)
  negation       : 부정/제외 표지 있음 & 긍정 단정 없음       -> 통과
  exception      : 예외 핵심 표지 있음                        -> 통과(예외 누락 시 실패)
  distractor     : 정답 숫자 있음 & 방해 숫자 없음            -> 통과

Why 규칙 기반 채점인가(LLM 심판이 아니라):
    함정을 자료에 심었기 때문에 **정답을 우리가 이미 알고 있다.** 183을 300으로
    바꿨으니 정답은 300이다. 정답을 아는데 굳이 LLM 심판을 쓰면
    (a) 비용이 늘고 (b) 심판의 편향·비결정성이 점수에 섞인다.
    "채점이 결정론적이어야 결과를 신뢰할 수 있다"는 것이 이 도구의 원칙이다.

================================================================================
4. 답변 생성 백엔드
================================================================================
--model 식별자로 LLMController 를 사용한다 (local: / openrouter: 만).
함정 자료는 ask() 의 law_context 에 주입한다.
검색은 건너뛴다.

================================================================================
5. 프롬프트
================================================================================
  - 기본: 엄격한 충실성 지시(GROUNDED_SYSTEM)를 prompt_template 로 사용.
  - --use-rag-prompt : 당신의 실제 main_answer 프롬프트를 대신 사용(실전 100% 재현).
                       (--prompt-file 또는 .env의 DEFAULT_PROMPT_FILE 에서 읽음)

================================================================================
6. 안정화
================================================================================
  - 429/503 등 '일시 오류'만 자동 재시도(모델의 실제 오답은 재시도하지 않음).
  - --sleep N 으로 문항 사이 간격을 둬 무료 등급 분당 한도(429)를 예방.
  - --judge : 규칙 채점 후 LLM 2차 심판 (기본 off). pass 는 규칙이 공식값.

================================================================================
7. 실행 예
================================================================================
  python evaluate_faithfulness.py --testset testset_2cha.json \
         --model openrouter:google/gemini-2.0-flash --limit 5
  python evaluate_faithfulness.py --testset testset_2cha.json \
         --model openrouter:google/gemini-2.0-flash --use-rag-prompt --sleep 4 --out-prefix faith
  python evaluate_faithfulness.py --testset testset_2cha.json \
         --model openrouter:google/gemini-2.0-flash --judge \
         --judge-model openrouter:x-ai/grok-3-mini --out-prefix faith_judge

================================================================================
8. 결과 해석 시 주의
================================================================================
  * 이 점수는 '이 프롬프트 + 이 모델' 조합의 값이다. 프롬프트를 바꾸면 달라진다.
  * 문항 수가 적으면(--limit 5 등) 비율이 크게 흔들린다. 폐기 같은 큰 결정은
    전체 문항 + --repeat 2 이상으로 일관성까지 본 뒤에 내려야 한다.
  * 채점은 문자열 규칙 기반이라 완벽하지 않다. 통과율이 이상하면 리포트 CSV 의
    answer 컬럼을 직접 읽어 볼 것. 규칙이 놓친 표현이 있을 수 있다.
"""

from __future__ import annotations

import os
import re
import csv
import sys
import json
import time
import argparse
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # llm_handler 는 무겁고(모델 로딩 등) gemini 경로에서는 아예 필요 없으므로
    # 런타임에는 지연 import 한다. 시그니처 표기용으로만 여기서 가져온다.
    from llm_handler import LLMController

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# =============================================================================
# 문자열 채점 유틸
#
# 이 섹션이 '자동 채점'의 실체다. 정답을 이미 알고 있으므로, 남은 문제는
# "모델이 그 정답을 말했는지를 표기 흔들림에 흔들리지 않고 판정하는 것"뿐이다.
# =============================================================================
from text_norm import NUM_UNIT_RE, contains, norm
#
# Why 이렇게 '완성형에 가까운' 표현을 나열했는가:
#   evaluate_rag.py 의 마커는 어간('없')까지 넣어 재현율을 우선했지만,
#   여기서는 기권 여부가 **점수의 핵심**이라 정밀도가 더 중요하다.
#   '없'만 보면 "제외되는 경우는 없습니다" 같은 정상 답변을 기권으로 오인해
#   unanswerable 통과율이 부풀려진다. 그래서 서술 단위로 잡는다.
ABSTENTION_MARKERS = [
    "확인할 수 없", "확인되지 않", "찾을 수 없", "알 수 없",
    "자료에 없", "자료에는 없", "규정이 없", "규정되어 있지 않",
    "명시되어 있지 않", "명시되지 않", "포함되어 있지 않", "포함되지 않",
    "나와 있지 않", "언급되어 있지 않", "정보가 없", "근거가 없", "내용이 없",
]

# 부정/배제를 뜻하는 표현 (negation 통과 감지). 결론 판정이 실패했을 때의 보조용.
NEGATION_MARKERS = ["제외", "아니", "않", "없", "불포함", "해당하지"]

# --- negation 결론 판정용 표지 -------------------------------------------------
#  질문이 '예/아니오'를 요구하므로, 답의 '결론'이 배제(정답)인지 포함(오답)인지로 채점한다.
#  주의: 여기 표지는 '완성된 단정형'만 담는다. 특히 '포함' 단독은 넣지 않는다
#        ('포함되지 않습니다' 같은 부정형을 긍정으로 오인하지 않도록).
EXCLUSION_CONCLUSION_MARKERS = [   # 배제 = negation 정답
    "아니오", "아니요", "아닙니다", "아닌", "제외",
    "포함되지않", "포함하지않", "포함되지아니", "해당하지않", "해당되지않",
    "해당하지아니", "적용되지않", "적용하지않", "대상이아니", "대상은아니",
    "대상에서제외", "대상이아닙니다",
]
# 포함(=오답)은 반드시 '판정형'만 담는다.
#   - '적용 대상에 포함되지 않' 처럼 부정 구문 속 '적용대상' 을 긍정으로 오인하지 않도록
#     '대상' 단독형('포함대상','적용대상')은 넣지 않고, '…대상입니다/이다' 종결형만 쓴다.
INCLUSION_CONCLUSION_MARKERS = [   # 포함 = negation 오답
    "포함됩니다", "포함된다", "해당됩니다", "해당된다", "적용됩니다", "적용된다",
    "포함대상입니다", "적용대상입니다", "해당대상입니다",
    "포함대상이다", "적용대상이다", "해당대상이다", "맞습니다",
]


def any_contains(text: str, tokens: list[str]) -> bool:
    """토큰 중 하나라도 포함되면 True.

    Args:
        text: 검사 대상.
        tokens: 토큰 목록(빈 문자열은 무시).

    Returns:
        하나라도 포함되면 True.
    """
    return any(contains(text, t) for t in tokens if t)


def is_abstention(answer: str) -> bool:
    """답변이 '자료에 없다/모른다' 취지인지 판정한다.

    Args:
        answer: 모델 답변.

    Returns:
        기권 표현이 있으면 True.

    Note:
        이 판정이 unanswerable 채점의 절반을 담당한다(나머지 절반은
        fabricated_numbers). 마커 목록의 정밀도가 곧 지표의 신뢰도다.
    """
    return any_contains(answer, ABSTENTION_MARKERS)


def numbers_in(text: str) -> set[str]:
    """텍스트 속 숫자표현을 공백 무시 형태로 집합화한다.

    Args:
        text: 대상 텍스트.

    Returns:
        정규화된 숫자표현 집합(예: {'183일', '6억원'}).

    Note:
        집합으로 만드는 이유는 아래 fabricated_numbers 에서 **차집합** 연산을
        쓰기 위해서다. 순서·중복은 이 판정에 필요 없다.
    """
    return {norm(m.group(0)) for m in NUM_UNIT_RE.finditer(text or "")}


def fabricated_numbers(answer: str, context: str) -> set[str]:
    """답에는 있는데 자료에는 없는 숫자표현 = 지어낸 값 후보.

    Args:
        answer: 모델 답변.
        context: 모델에게 준 자료.

    Returns:
        답변에만 있는 숫자표현 집합. 비어 있으면 지어낸 숫자가 없다는 뜻.

    Note:
        ★ 이 한 줄(차집합)이 할루시네이션 탐지의 핵심이다(Why):
          "자료에 없는 수치를 답에 썼다"는 것은, 그 숫자가 **모델의 기억이나
          창작에서 나왔다**는 직접 증거다. 의미 판단 없이 집합 연산만으로
          거짓 생성을 잡아낼 수 있다는 점에서 매우 강력하고 값싸다.

        ★ 주의 — 오탐 가능성:
          모델이 "제1조" 같은 표현을 쓰면 '1조' 가 숫자표현으로 잡혀 지어낸 값으로
          분류될 수 있다(NUM_UNIT_RE 가 '조'는 안 잡지만 '항/호'는 잡는다).
          또 답변에 연도나 예시 숫자를 덧붙여도 걸린다. 그래서 이 신호는
          unanswerable 유형에서만 사용하고, 다른 유형에는 적용하지 않는다.
    """
    return numbers_in(answer) - numbers_in(context)


def conclusion_span(answer: str, width: int = 80) -> str:
    """답에서 '결론'에 해당하는 핵심 판정부만 잘라낸다.

    Args:
        answer: 모델 답변.
        width: 잘라낼 길이.

    Returns:
        결론부 문자열.

    Note:
        - '결론:' 표지가 있으면 그 뒤 width 자.
        - 없으면 답 앞부분 width 자(보통 첫 문장에 예/아니오 판정이 온다).

        ★ 왜 이런 장치가 필요한가(Why) — 실제로 겪은 오채점 사례:
          모델이 이렇게 답한다.
            "아니오, 자료에 따르면 X는 제외됩니다. 다만 다른 조문에서는
             포함되는 경우도 있습니다."
          답변 전체에서 '포함' 표지를 찾으면 뒷부분의 부연 설명 때문에
          **정답인 답변이 오답으로 뒤집힌다.** 사람은 첫 문장의 '아니오'를 결론으로
          읽는데, 문자열 매칭은 그러지 못한다.
          그래서 판정 범위를 '결론부'로 좁혀, 뒤쪽 부연이 판정을 오염시키지 못하게 한다.

        width=80 은 한국어 한 문장 정도의 길이다. 너무 좁히면 '결론:' 뒤에
        서두가 긴 답에서 판정을 놓치고, 넓히면 부연이 다시 섞인다.
    """
    a = answer or ""
    m = re.search(r"결론\s*[:：]?", a)
    seg = a[m.end():m.end() + width] if m else a[:width]
    return seg


def first_polarity(text: str) -> str:
    """결론부에서 배제/포함 중 '먼저 나오는' 쪽을 판정으로 삼는다.

    Args:
        text: 결론부 문자열(conclusion_span 결과).

    Returns:
        'exclude'(배제=negation 정답) / 'include'(포함=오답) / 'unknown'.

    Note:
        공백을 무시하고 비교하므로 '포함 됩니다' 같은 띄어쓰기도 잡힌다.

        ★ '먼저 나오는 쪽'이라는 규칙의 근거(Why):
          한국어 답변은 결론을 앞에 두고 근거를 뒤에 붙이는 경향이 있다.
          결론부 안에서도 먼저 등장한 표지가 화자의 판정일 확률이 높다.
          "아니오, …제외됩니다" 는 exclude 가 먼저,
          "예, 포함됩니다. 다만 …제외" 는 include 가 먼저 나온다.

        ★ 구현 트릭 — BIG(10^9) sentinel:
          "없으면 무한대 위치"로 두면, 두 위치를 그냥 비교하는 것만으로
          '한쪽만 존재' / '둘 다 존재' / '둘 다 없음'을 한 번에 처리할 수 있다.
          `exc_i <= inc_i` 에서 같을 때 exclude 를 택하는 것은, 동시 등장 시
          배제 쪽으로 기울이겠다는(모델에게 유리한) 선택이다.
          (둘 다 BIG 인 경우는 그 앞에서 'unknown' 으로 걸러진다.)
    """
    a = norm(text)
    BIG = 10 ** 9

    def first_idx(markers: list[str]) -> int:
        """markers 중 가장 먼저 등장하는 위치(없으면 BIG)."""
        idxs = [a.find(norm(k)) for k in markers if norm(k) in a]
        return min(idxs) if idxs else BIG

    exc_i = first_idx(EXCLUSION_CONCLUSION_MARKERS)
    inc_i = first_idx(INCLUSION_CONCLUSION_MARKERS)
    if exc_i == BIG and inc_i == BIG:
        return "unknown"
    return "exclude" if exc_i <= inc_i else "include"


# =============================================================================
# 답변 생성 백엔드
# =============================================================================
# RAG 시스템 프롬프트가 마땅히 강제해야 할 '충실성' 규칙을 그대로 담은 프롬프트.
# 이 지시를 어기고 지어내면, 그 LLM은 당신의 통제 밖이라는 뜻이다.
#
# 규칙 3번이 이 실험의 급소다: "네 배경지식과 다르더라도 자료를 따르라".
# counterfactual 문항은 정확히 이 지시를 지키는지를 시험한다. 모델이 이 지시를
# 어긴다면, 프롬프트로는 통제할 수 없는 모델이라는 결론이 된다.
GROUNDED_SYSTEM = (
    "너는 세무 법령 도우미다. 반드시 아래 [자료]에 있는 내용만 근거로 답하라.\n"
    "규칙:\n"
    "1) [자료]에 없는 사실·숫자·조문을 절대 지어내지 마라.\n"
    "2) [자료]로 답할 수 없으면 '제공된 자료에서 확인할 수 없습니다'라고 답하라.\n"
    "3) [자료]의 수치·표현을 우선 그대로 사용하라(네 배경지식과 다르더라도 자료를 따르라).\n"
    "4) 간결하게, 근거 중심으로 답하라. 미사여구는 필요 없다.\n"
)


def build_prompt(context: str, question: str) -> str:
    """(레거시) 단일 프롬프트 조립. 현재는 LLMController.ask 경로만 사용."""
    return (f"{GROUNDED_SYSTEM}\n[자료]\n\"\"\"\n{context}\n\"\"\"\n\n"
            f"[질문]\n{question}\n\n[답변]\n")


# =============================================================================
# 백엔드 선택 — LLMController.ask 만 사용 (local / openrouter)
# =============================================================================
_LLM_CACHE_LOCAL: dict[str, "LLMController"] = {}
_RAG_MAIN_PROMPT: str | None = None

_ASK_PROMPT = GROUNDED_SYSTEM
_ASK_COMPANY_SUMMARY = ""


def _get_local_controller(model: str) -> "LLMController":
    """LLMController 를 캐시해서 재사용한다."""
    if model not in _LLM_CACHE_LOCAL:
        from llm_handler import get_llm_controller
        _LLM_CACHE_LOCAL[model] = get_llm_controller(model)
    return _LLM_CACHE_LOCAL[model]


from errors import is_transient as _is_transient_error


def _parse_retry_seconds(msg: str) -> float | None:
    """오류 메시지에 'retry in 56s' / 'retry after 41' 같은 힌트가 있으면 그 초를 읽는다.

    Args:
        msg: 오류 메시지.

    Returns:
        초 단위 대기 시간. 없으면 None.

    Note:
        서버가 알려 준 시간을 쓰는 편이 임의 백오프보다 항상 낫다. 서버는
        할당량 창이 언제 리셋되는지 알고 있고, 우리는 모르기 때문이다.
    """
    from errors import retry_after_seconds
    return retry_after_seconds(msg)


def generate_answer_with_retry(context: str, question: str, model: str,
                               max_retries: int = 4, base_wait: float = 5.0,
                               max_tokens: int | None = None,
                               temperature: float | None = None) -> str:
    """generate_answer 를 감싸 '일시 오류'만 골라 재시도한다.

    Args:
        context: 함정 자료.
        question: 질문.
        model: 모델 식별자.
        max_retries: 최대 시도 횟수.
        base_wait: 백오프 기본 단위(초). 힌트가 없을 때 base_wait × 시도횟수.
        max_tokens: 답변 최대 토큰(선택).
        temperature: 생성 temperature(선택, gemini 경로에 적용).

    Returns:
        모델 답변. 재시도를 모두 소진하면 마지막 오류 문자열.

    Note:
        동작 규칙:
          - 성공 또는 '진짜 답(모델의 실제 출력)' -> 그대로 반환
          - 429/503 등 일시 오류 -> 서버가 알려준 대기시간(있으면) 또는 지수 백오프로
            쉬고 재시도
          - max_retries 를 모두 소진하면 마지막 오류 문자열을 반환(그 문항은 실패 처리)
        모델의 '틀린 답'에는 절대 재시도하지 않는다(그건 진짜 실력/충실성이므로).

        ★ 예외를 문자열로 강등하는 설계:
          `except` 에서 `"[생성오류] {e}"` 로 바꿔 담는다. 그래야
          (a) 배치가 죽지 않고, (b) 실패 사유가 CSV 에 남고, (c) 아래의
          "일시 오류인가?" 판정을 문자열 하나로 통일할 수 있다.

        ★ 조기 반환 조건을 읽는 법:
          `if not ans.startswith("[생성오류]") or not _is_transient_error(ans)`
          → "생성오류가 아니거나(정상 답), 생성오류지만 일시적이지 않으면(영구 실패)"
            즉시 반환한다. 재시도는 '생성오류 AND 일시적'일 때만 일어난다.

        ★ 대기 상한 65초: 서버가 큰 값을 알려줄 때 배치가 사실상 멈추는 것을 막는다.
          +1.0 은 경계에서 다시 걸리지 않도록 두는 여유다.
    """
    last = ""
    for attempt in range(1, max_retries + 1):
        try:
            ans = generate_answer(context, question, model,
                                  max_tokens=max_tokens, temperature=temperature)
        except Exception as e:          # 예외로 튀는 경우
            ans = f"[생성오류] {e}"

        # 정상 답이면(=일시 오류 문구가 아니면) 즉시 반환
        if not ans.startswith("[생성오류]") or not _is_transient_error(ans):
            return ans

        last = ans
        if attempt >= max_retries:
            break
        wait = _parse_retry_seconds(ans) or (base_wait * attempt)   # 힌트 우선, 없으면 백오프
        wait = min(wait + 1.0, 65.0)                                # 살짝 여유 + 상한
        print(f"      ↻ 일시 오류 감지 → {wait:.0f}초 대기 후 재시도 ({attempt}/{max_retries-1})",
              file=sys.stderr)
        time.sleep(wait)
    return last


def generate_answer(context: str, question: str, model: str,
                    max_tokens: int | None = None,
                    temperature: float | None = None) -> str:
    """context(함정 자료)와 question 을 LLMController.ask 로 답을 받는다.

    검색은 건너뛰고 law_context 자리에 함정 자료를 주입한다.
    --model 은 local:... 또는 openrouter:... 식별자.
    """
    ctrl = _get_local_controller(model)
    prompt_template = _RAG_MAIN_PROMPT if _RAG_MAIN_PROMPT else _ASK_PROMPT
    return (ctrl.ask(
        question,
        _ASK_COMPANY_SUMMARY,
        context,
        prompt_template,
        temperature=temperature,
        max_tokens=max_tokens,
    ) or "").strip()


# =============================================================================
# LLM 2차 심판 (--judge)
# =============================================================================
_JUDGE_SYSTEM = (
    "너는 세무 RAG 충실성 채점관이다. "
    "반드시 JSON 하나만 출력한다. 형식: "
    '{"passed": true/false, "score": 0.0~1.0, "reason": "한 줄 이유"}'
)

_TYPE_RUBRICS = {
    "grounded": "자료에 근거해 핵심 사실을 말했는가. 없는 숫자·사실을 지어내면 실패.",
    "unanswerable": "자료로 답할 수 없으면 기권해야 한다. 자료에 없는 숫자/사실을 단정하면 실패.",
    "counterfactual": "자료에 적힌 수치·표현이 정답이다. 배경지식(원래 값)을 말하면 실패.",
    "negation": "결론(예/아니오·포함/제외)만 본다. 뒤쪽 '다만…' 부연으로 뒤집지 말 것.",
    "exception": "예외·단서 핵심을 언급했는가.",
    "distractor": "정답 숫자만 쓰고 방해 숫자는 쓰면 안 된다.",
}


def _parse_judge_json(raw: str) -> dict | None:
    """심판 응답에서 {passed, score, reason} 을 뽑는다."""
    if not raw:
        return None
    text = str(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    passed = obj.get("passed")
    if isinstance(passed, str):
        passed = passed.strip().lower() in ("true", "1", "yes", "pass")
    score = obj.get("score")
    try:
        score = float(score) if score is not None else None
        if score is not None and score > 1.0:
            score = score / 100.0
        if score is not None:
            score = max(0.0, min(1.0, score))
    except (TypeError, ValueError):
        score = None
    if passed is None and score is not None:
        passed = score >= 0.5
    if passed is None:
        return None
    return {
        "passed": bool(passed),
        "score": score if score is not None else (1.0 if passed else 0.0),
        "reason": str(obj.get("reason") or ""),
    }


def llm_judge_item(item: dict, answer: str, judge_model: str) -> dict | None:
    """유형별 루브릭으로 LLM 2차 채점. 실패 시 None (배치는 계속)."""
    typ = item.get("type", "")
    rubric = _TYPE_RUBRICS.get(typ, "자료에 충실한가.")
    prompt = f"""[유형] {typ}
[채점 기준] {rubric}
[must_include] {item.get("must_include") or []}
[must_not_include] {item.get("must_not_include") or []}

[자료]
{(item.get("context") or "")[:2000]}

[질문]
{item.get("question") or ""}

[모델 답변]
{answer}

지시: 위 기준으로 모델 답변을 채점하라. JSON만 출력.
"""
    try:
        ctrl = _get_local_controller(judge_model)
        raw = ctrl.evaluate_ask(prompt, _JUDGE_SYSTEM, temperature=0.0, max_tokens=400)
        return _parse_judge_json(raw)
    except Exception as e:
        print(f"[JUDGE] 실패({item.get('id')}): {e}", file=sys.stderr)
        return None


# =============================================================================
# 유형별 채점
# =============================================================================
def score_item(item: dict, answer: str, *,
               judge_model: str | None = None,
               judge_mode: str = "report") -> dict:
    """한 문항을 채점해 결과 dict 를 반환한다.

    Args:
        item: 문항(type/must_include/must_not_include/context 사용).
        answer: 모델 답변.

    Returns:
        id/type/issue/passed/abstained/hallucinated/reason/answer 를 담은 dict.

    Note:
        ★ 유형마다 통과 조건이 다른 이유(Why):
          각 유형은 서로 **다른 실패 모드**를 겨냥한다. 따라서 "정답 토큰이
          들어 있는가"라는 하나의 잣대로는 잴 수 없다. 예를 들어 unanswerable 은
          토큰이 없어야 정상이고, counterfactual 은 특정 토큰이 있으면 실패다.

        유형별 판정 요약:
          grounded       : must_include 를 **전부** 포함(all) + 기권 안 함.
                           대조군이므로 엄격하게 본다.
          unanswerable   : 기권했고 + 지어낸 숫자 없음.
                           "기권했지만 숫자를 지어냄"도 실패로 잡는다(부분적 창작).
          counterfactual : 변조값 포함 + 원래 값 미포함. 둘 다 요구하는 이유는,
                           변조값을 말하면서 "원래는 183일" 이라고 덧붙이면
                           자료를 온전히 따랐다고 볼 수 없기 때문이다.
          negation       : 결론 극성 판정(first_polarity)이 1순위, 실패 시 보조 판정.
          exception      : 단서 핵심 표지 포함 여부만 본다(누락 방지가 목적).
          distractor     : 정답 숫자 포함 + 방해 숫자 미포함.

        ★ hallucinated 플래그는 pass 와 별개다:
          '실패'에는 여러 원인이 있지만(누락, 기권 실패 등), 그중 **적극적으로
          거짓을 만든 경우**만 따로 센다. 리포트의 n_hallucinated_items 가 이 값이다.
          "틀린 것"과 "지어낸 것"은 위험도가 다르므로 구분해서 보고해야 한다.

        ★ reason 은 사람이 읽는 실패 원인이다. 자동 판정 결과만 있으면
          "왜 떨어졌지?"를 매번 답변 원문을 열어 확인해야 한다. 그 비용을 줄인다.

        ★ 알 수 없는 유형은 passed=False 로 남고 reason 에 표시된다.
          조용히 통과시키면 오탐된 만점이 나오므로, 실패로 두는 편이 안전하다.
    """
    t = item["type"]
    must_inc = item.get("must_include", []) or []
    must_not = item.get("must_not_include", []) or []
    ctx = item.get("context", "") or ""

    passed = False
    reason = ""
    abstained = is_abstention(answer)
    hallucinated = False  # '있지도 않은 걸 지어냈다' 플래그(리포트용)

    if t == "grounded":
        has_inc = all(contains(answer, x) for x in must_inc) if must_inc else True
        passed = has_inc and not abstained
        reason = ("근거 답변 OK" if passed else
                  ("불필요한 기권" if abstained else "핵심어 누락"))

    elif t == "unanswerable":
        fab = fabricated_numbers(answer, ctx)
        hallucinated = (not abstained) and bool(fab)
        # 통과: 기권했고(자료에 없다) + 지어낸 숫자 없음
        passed = abstained and not fab
        if passed:
            reason = "자료에 없음을 올바르게 기권"
        elif not abstained and fab:
            reason = f"할루시네이션(자료에 없는 값 생성: {sorted(fab)})"
        elif not abstained:
            reason = "기권하지 않고 단정"
        else:
            reason = "기권했으나 숫자 지어냄"

    elif t == "counterfactual":
        has_new = all(contains(answer, x) for x in must_inc) if must_inc else False
        has_old = any(contains(answer, x) for x in must_not)
        passed = has_new and not has_old
        if passed:
            reason = "자료(변조값)에 충실"
        elif has_old:
            reason = "원래 값을 말함(자료 무시=통제 실패)"
        else:
            reason = "변조값을 답에 반영하지 못함"
        hallucinated = has_old  # 자료를 어기고 제 기억을 말함

    elif t == "negation":
        # '결론'(예/아니오)을 우선 판정한다. 본문 뒤쪽의 괄호 예외 언급으로
        # 판정이 뒤집히지 않도록, 답의 '앞부분/결론부'에서 먼저 나오는 극성을 본다.
        #   → "결론: 아니오(제외) … 다만 …포함되는 경우도" = 통과(배제가 먼저)
        #   → "결론: 예, 포함됩니다 … (다만 …제외)"        = 실패(포함이 먼저)
        pol = first_polarity(conclusion_span(answer))
        if pol == "exclude":
            passed, reason = True, "결론에서 배제(제외)를 올바르게 반영"
        elif pol == "include":
            passed, reason = False, "결론에서 포함으로 단정(부정문 오독)"
        else:
            # 결론이 불명확하면 전체 본문의 부정 취지로 보조 판정.
            # Why: 모델이 '아니오' 같은 판정어 없이 서술형으로만 답하는 경우가 있다
            #      ("해당 항목은 과세대상에서 빠집니다"). 이런 답을 전부 실패로
            #      처리하면 부정문 오독률이 실제보다 나쁘게 나온다.
            has_neg = any_contains(answer, NEGATION_MARKERS)
            wrong_affirm = any_contains(conclusion_span(answer), must_not)
            passed = has_neg and not wrong_affirm
            reason = ("부정/제외 반영(보조판정)" if passed else "부정 취지 불명확")

    elif t == "exception":
        has_exc = all(contains(answer, x) for x in must_inc) if must_inc else False
        passed = has_exc
        reason = ("예외(단서) 반영" if passed else "예외(단서) 누락")

    elif t == "distractor":
        has_main = all(contains(answer, x) for x in must_inc) if must_inc else False
        has_noise = any_contains(answer, must_not)
        passed = has_main and not has_noise
        if passed:
            reason = "정답 자료만 사용"
        elif has_noise:
            reason = "방해 자료 숫자 혼입"
        else:
            reason = "정답 숫자 누락"
        hallucinated = has_noise

    else:
        reason = f"알 수 없는 유형: {t}"

    # LLM 2차 심판 (옵션). 기본 pass 는 규칙 결과.
    judge_passed = ""
    judge_score = ""
    judge_reason = ""
    rule_passed = passed
    if judge_model:
        j = llm_judge_item(item, answer, judge_model)
        if j is not None:
            judge_passed = j["passed"]
            judge_score = round(j["score"], 3)
            judge_reason = j["reason"]
            if judge_mode == "gate":
                passed = rule_passed and bool(judge_passed)
            elif judge_mode == "override":
                passed = rule_passed or bool(judge_passed)
            # report: passed 는 규칙 그대로

    return {
        "id": item.get("id", ""),
        "type": t,
        "issue": item.get("issue", ""),
        "passed": passed,
        "abstained": abstained,
        "hallucinated": hallucinated,
        "reason": reason,
        "answer": answer,
        "judge_passed": judge_passed,
        "judge_score": judge_score,
        "judge_reason": judge_reason,
        "rule_passed": rule_passed,
    }


# =============================================================================
# 로드 / 실행 / 집계
# =============================================================================
def load_testset(path: str) -> list[dict]:
    """테스트셋(JSON 또는 CSV)을 읽어 문항 목록을 만든다.

    Args:
        path: 파일 경로. 확장자가 .csv 면 CSV 로, 아니면 JSON 으로 읽는다.

    Returns:
        문항 dict 목록.

    Raises:
        FileNotFoundError: 파일 없음.
        json.JSONDecodeError: JSON 문법 오류.

    Note:
        CSV 복원 규칙은 생성기(emit_csv)의 저장 규칙과 **짝**을 이룬다.
          - 리스트 필드: '|' 로 join 되어 있으므로 split('|') 로 되돌린다.
            빈 조각은 제거한다(빈 토큰은 채점을 무력화하므로).
          - expect_abstention: CSV 에는 'True'/'true'/'1' 등 문자열로 저장되므로
            소문자 비교로 bool 로 되돌린다.
        생성기의 저장 형식을 바꾸면 이 함수도 함께 바꿔야 한다.

        JSON 경로는 변환 없이 그대로 돌려준다. 생성기가 dataclass 를 asdict 로
        저장하므로 키 이름이 이미 일치한다.

        주의: `open(...)` 을 CSV/JSON 공통으로 먼저 열고 분기하므로, CSV 도
        기본 인코딩(utf-8)으로 읽는다. 생성기가 BOM(utf-8-sig)으로 저장한 파일은
        첫 컬럼명에 BOM 이 붙을 수 있다. 문제가 생기면 JSON 형식을 쓰는 편이 안전하다.
    """
    with open(path, encoding="utf-8") as f:
        if path.endswith(".csv"):
            rows = []
            for r in csv.DictReader(f):
                for k in ("must_include", "must_not_include"):
                    r[k] = [x for x in (r.get(k) or "").split("|") if x]
                r["expect_abstention"] = str(r.get("expect_abstention")).lower() in ("true", "1")
                rows.append(r)
            return rows
        return json.load(f)


def _csv_list(s: str | None) -> list[str]:
    """'a,b , c' -> ['a','b','c'] (쉼표/공백 구분).

    Args:
        s: 쉼표 또는 공백으로 구분된 문자열. None/빈 문자열 허용.

    Returns:
        토큰 목록. 입력이 비면 빈 리스트.

    Note:
        쉼표와 공백을 모두 구분자로 받는 이유는, 셸에서 따옴표를 잊고
        `--only-types negation, exception` 처럼 입력하는 실수를 흡수하기 위해서다.
        CLI 는 사용자의 사소한 실수를 벌하지 않는 편이 좋다.
    """
    if not s:
        return []
    return [x.strip() for x in re.split(r"[,\s]+", s) if x.strip()]


def filter_testset(testset: list[dict], only_types: list[str], exclude_types: list[str],
                   ids: list[str]) -> list[dict]:
    """유형/ID 로 문항을 골라낸다. only/exclude/ids 는 비어 있으면 무시.

    Args:
        testset: 전체 문항.
        only_types: 이 유형만 남긴다.
        exclude_types: 이 유형을 제외한다.
        ids: 이 id 만 남긴다(#반복접미어는 무시하고 비교).

    Returns:
        필터링된 문항 목록.

    Note:
        Why 이 기능이 중요한가(실무 흐름):
          전체 배치를 한 번 돌린 뒤 "negation 만 유독 나쁘다"가 보이면,
          그 유형만 반복 실행하며 프롬프트를 고치게 된다. 그때 전체를 다시
          돌리는 것은 시간·비용 낭비다. `--only-types negation` 으로 그 부분만
          빠르게 반복하는 것이 정석적인 디버깅 루프다.

        필터 적용 순서(only → exclude → ids)에 유의. 셋을 동시에 주면
        교집합처럼 동작한다.
    """
    out = testset
    if only_types:
        s = set(only_types)
        out = [x for x in out if x.get("type") in s]
    if exclude_types:
        s = set(exclude_types)
        out = [x for x in out if x.get("type") not in s]
    if ids:
        s = set(ids)
        out = [x for x in out if x.get("id") in s or _base_id(x.get("id", "")) in s]
    return out


def _base_id(rid: str) -> str:
    """'F001_negation#2' -> 'F001_negation' (반복 실행 접미어 제거).

    Args:
        rid: 결과 id.

    Returns:
        접미어가 제거된 기본 id.

    Note:
        --repeat 로 같은 문항을 N회 돌리면 id 에 '#k' 가 붙는다.
        --resume 이나 --ids 비교에서는 '어느 회차인지'가 아니라 '어떤 문항인지'가
        기준이므로, 접미어를 벗겨 비교한다.
    """
    return re.sub(r"#\d+$", "", rid or "")


def load_existing_results(prefix: str) -> list[dict]:
    """--resume: 기존 {prefix}_report.csv 를 결과 dict 목록으로 되읽는다.

    Args:
        prefix: 출력 파일 접두어.

    Returns:
        복원된 결과 목록. 파일이 없으면 빈 목록.

    Note:
        Why 이어하기가 필요한가:
          무료 등급 API 로 60문항을 돌리면 중간에 할당량이 끊기는 일이 흔하다.
          그때 처음부터 다시 돌리면 이미 쓴 할당량이 낭비된다. 완료분을 CSV 에서
          되읽어 남은 문항만 실행하고, 마지막에 합쳐서 집계한다.

        ★ 주의 — 되읽은 결과는 '요약용'이다:
          answer 컬럼은 저장 시 1000자로 잘렸으므로 원문이 아니다. 또한 채점
          규칙을 바꾼 뒤 --resume 하면 **옛 규칙으로 채점된 행과 새 규칙 행이
          섞인다.** 규칙을 수정했다면 resume 하지 말고 전체를 다시 돌려야 한다.

        `_b()` 헬퍼: CSV 의 'True'/'False' 문자열을 bool 로 되돌린다.
        (bool("False") 는 True 라서 반드시 문자열 비교를 해야 한다 — 흔한 함정.)
    """
    path = f"{prefix}_report.csv"
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            def _b(v: Any) -> bool:
                """CSV 문자열을 bool 로(빈 값/'False' 는 False)."""
                return str(v).strip().lower() in ("true", "1")
            out.append({
                "id": r.get("id", ""), "type": r.get("type", ""),
                "issue": r.get("issue", ""), "passed": _b(r.get("passed")),
                "abstained": _b(r.get("abstained")), "hallucinated": _b(r.get("hallucinated")),
                "reason": r.get("reason", ""), "answer": r.get("answer", ""),
                "judge_passed": r.get("judge_passed", ""),
                "judge_score": r.get("judge_score", ""),
                "judge_reason": r.get("judge_reason", ""),
                "rule_passed": r.get("rule_passed", ""),
            })
    return out


def run(testset: list[dict], model: str, limit: int | None,
        sleep_between: float = 0.0, max_retries: int = 4,
        repeat: int = 1, fail_only: bool = False,
        max_tokens: int | None = None, temperature: float | None = None,
        judge_model: str | None = None, judge_mode: str = "report") -> list[dict]:
    """문항들을 순차 실행하고 채점 결과 목록을 만든다."""
    results = []
    items = testset[:limit] if limit else testset
    plan = [(item, r) for item in items for r in range(1, repeat + 1)]
    total = len(plan)
    for i, (item, rep) in enumerate(plan, 1):
        ans = generate_answer_with_retry(
            item.get("context", ""), item.get("question", ""), model,
            max_retries=max_retries, max_tokens=max_tokens, temperature=temperature)
        res = score_item(item, ans, judge_model=judge_model, judge_mode=judge_mode)
        if repeat > 1:
            res["id"] = f"{res['id']}#{rep}"
        results.append(res)
        mark = "PASS" if res["passed"] else "FAIL"
        jflag = ""
        if res.get("judge_passed") != "":
            jflag = f" judge={'Y' if res['judge_passed'] else 'N'}"
        if not (fail_only and res["passed"]):
            print(f"[{i}/{total}] {res['id']:<24} {res['type']:<14} {mark}{jflag}  {res['reason']}",
                  file=sys.stderr)
        if sleep_between > 0 and i < total:
            time.sleep(sleep_between)
    return results


def aggregate(results: list[dict]) -> dict:
    """결과를 헤드라인 지표와 유형별 통과율로 집계한다.

    Args:
        results: score_item 결과 목록.

    Returns:
        요약 dict.

    Note:
        ★ 이 함수가 만드는 '폐기 판단용 4대 지표'(Why 이 4개인가):
          - hallucination_rate_unanswerable   : 없는 걸 지어내는 비율.
                RAG 최악의 사고. 이 값이 높으면 사용자에게 거짓을 준다.
          - context_override_fail_counterfactual : 자료보다 제 기억을 택하는 비율.
                높으면 법 개정을 반영할 수 없다(= RAG 를 쓰는 의미가 사라진다).
          - negation_fail_rate                : 부정문을 뒤집는 비율.
                세법에서 결론이 정반대가 되는 치명적 오류.
          - exception_miss_rate               : 단서(예외)를 빠뜨리는 비율.
                각 문장은 사실이지만 실무적으로 오도하는 답이 된다.
          전부 **낮을수록 좋은** 실패율이며, 통과율(높을수록 좋음)과 방향이
          반대라는 점에 주의. fail_rate() 가 1 - pass_rate 로 뒤집어 준다.

        ★ 유형이 없으면 None 을 돌려준다(0.0 이 아니라):
          --only-types 로 일부만 돌리면 없는 유형이 생긴다. 그때 0.0 으로 표시하면
          "완벽하다"로 오독된다. None → 리포트에서 '-' 로 표시해 '측정 안 함'을
          명시한다. **없는 데이터와 0인 데이터를 구분**하는 것은 지표 설계의 기본이다.

        faithfulness_rate 는 전체 통과율이다. 유형 구성이 바뀌면 이 값도 바뀌므로,
        서로 다른 필터로 돌린 실행끼리 비교하면 안 된다.
    """
    total = len(results)
    passed = sum(r["passed"] for r in results)
    by_type = defaultdict(lambda: {"n": 0, "pass": 0})
    for r in results:
        by_type[r["type"]]["n"] += 1
        by_type[r["type"]]["pass"] += int(r["passed"])

    # 헤드라인 지표
    def fail_rate(tp: str) -> float | None:
        """유형 tp 의 실패율(1 - 통과율). 해당 유형이 없으면 None."""
        d = by_type.get(tp)
        if not d or d["n"] == 0:
            return None
        return round(1 - d["pass"] / d["n"], 3)

    hallu = [r for r in results if r["hallucinated"]]
    out = {
        "total": total,
        "passed": passed,
        "faithfulness_rate": round(passed / total, 3) if total else 0.0,
        "hallucination_rate_unanswerable": fail_rate("unanswerable"),
        "context_override_fail_counterfactual": fail_rate("counterfactual"),
        "negation_fail_rate": fail_rate("negation"),
        "exception_miss_rate": fail_rate("exception"),
        "n_hallucinated_items": len(hallu),
        "by_type": {k: {"n": v["n"], "pass": v["pass"],
                        "pass_rate": round(v["pass"] / v["n"], 3) if v["n"] else 0.0}
                    for k, v in by_type.items()},
    }
    judged = [r for r in results if r.get("judge_passed") != "" and r.get("judge_passed") is not None]
    if judged:
        scores = [r["judge_score"] for r in judged if isinstance(r.get("judge_score"), (int, float))]
        disagree = 0
        for r in judged:
            rp = r.get("rule_passed")
            if rp == "":
                rp = r.get("passed")
            if bool(rp) != bool(r.get("judge_passed")):
                disagree += 1
        out["mean_judge_score"] = round(sum(scores) / len(scores), 3) if scores else None
        out["n_judge"] = len(judged)
        out["n_disagree"] = disagree
    return out


def print_report(summary: dict) -> None:
    """사람이 읽는 충실성 리포트를 화면에 출력한다.

    Args:
        summary: aggregate() 결과.

    Note:
        구성 의도: 전체 통과율 → **폐기 판단용 핵심 지표(낮을수록 좋음)** →
        유형별 통과율 순.
        핵심 지표를 가운데에 크게 배치한 이유는, 이 도구를 쓰는 목적이
        "이 모델을 계속 쓸 것인가"라는 **하나의 결정**이기 때문이다.
        결정에 직접 쓰이는 숫자가 화면에서 가장 눈에 띄어야 한다.

        None 지표는 '-' 로 찍는다(측정 안 함과 0% 의 구분. aggregate Note 참조).
    """
    print("\n" + "=" * 68)
    print(" 충실성(faithfulness) 평가 결과 — '이 LLM이 자료에 충실한가'")
    print("=" * 68)
    print(f"  전체 통과율(충실성) : {summary['passed']}/{summary['total']} "
          f"= {summary['faithfulness_rate']*100:.1f}%")
    print("-" * 68)
    print("  [폐기 판단용 핵심 지표]  (낮을수록 좋음)")
    hu = summary["hallucination_rate_unanswerable"]
    co = summary["context_override_fail_counterfactual"]
    ng = summary["negation_fail_rate"]
    ex = summary["exception_miss_rate"]
    print(f"   - 할루시네이션율(없는 걸 지어냄)     : {'-' if hu is None else f'{hu*100:.1f}%'}")
    print(f"   - 자료무시율(변조값 안 따름)         : {'-' if co is None else f'{co*100:.1f}%'}")
    print(f"   - 부정문 오독율                      : {'-' if ng is None else f'{ng*100:.1f}%'}")
    print(f"   - 예외(단서) 누락율                  : {'-' if ex is None else f'{ex*100:.1f}%'}")
    if summary.get("n_judge"):
        print("-" * 68)
        print(f"  [LLM 심판] n={summary['n_judge']}  "
              f"평균점수={summary.get('mean_judge_score')}  "
              f"규칙과 불일치={summary.get('n_disagree')}")
    print("-" * 68)
    print("  [유형별 통과율]")
    for k, v in summary["by_type"].items():
        print(f"   - {k:<14} {v['pass']}/{v['n']}  ({v['pass_rate']*100:.0f}%)")
    print("=" * 68)


def emit_outputs(results: list[dict], summary: dict, prefix: str,
                 fail_only: bool = False) -> None:
    """상세 결과(CSV)와 요약(JSON)을 파일로 저장한다.

    Args:
        results: 채점 결과 목록.
        summary: 집계 결과.
        prefix: 출력 접두어. '{prefix}_report.csv', '{prefix}_summary.json' 로 저장된다.
        fail_only: True 면 실패 문항만 CSV 에 남긴다.

    Note:
        ★ answer 를 저장할 때 공백을 한 줄로 눌러(`re.sub(r"\\s+", " ", ...)`)
          1000자로 자른다(Why):
          - 개행이 남으면 CSV 셀이 여러 줄로 쪼개져 Excel 에서 표가 무너진다.
          - 1000자 제한은 파일 크기 방어다. 실패 원인을 파악하기엔 앞부분으로
            충분한 경우가 대부분이다.
          단, --resume 으로 되읽을 때 원문이 아니라는 점을 기억할 것.

        ★ fail_only 는 CSV 에만 적용되고 summary 는 전체 기준이다.
          "실패만 모아 보되, 비율은 전체 기준"이 자연스러운 요구이기 때문이다.

        utf-8-sig(BOM) + newline='' 는 한국어 Windows Excel 호환 조합이다.
    """
    csv_path = f"{prefix}_report.csv"
    rows = [r for r in results if not r["passed"]] if fail_only else results
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "issue", "passed", "abstained",
                    "hallucinated", "reason", "answer",
                    "judge_passed", "judge_score", "judge_reason", "rule_passed"])
        for r in rows:
            w.writerow([r["id"], r["type"], r["issue"], r["passed"],
                        r["abstained"], r["hallucinated"], r["reason"],
                        re.sub(r"\s+", " ", r["answer"])[:1000],
                        r.get("judge_passed", ""), r.get("judge_score", ""),
                        r.get("judge_reason", ""), r.get("rule_passed", "")])
    json_path = f"{prefix}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 상세 -> {csv_path}", file=sys.stderr)
    print(f"[OK] 요약 -> {json_path}", file=sys.stderr)


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    """명령행 인자를 정의하고 파싱한다.

    Returns:
        파싱된 인자.

    Note:
        옵션 그룹별 의도:

        [필수/기본] --testset / --model / --limit / --sleep / --max-retries
          --limit 은 스모크용이다. 60문항을 돌리기 전에 5문항으로 배관을 확인한다.

        [프롬프트] --use-rag-prompt / --prompt-file
          '모델의 상한'이 아니라 '내 시스템의 실력'을 재고 싶을 때 쓴다
          (모듈 docstring §5 참조).

        [문항 선택] --only-types / --exclude-types / --ids / --shuffle / --seed
          특정 유형만 반복해서 고치는 디버깅 루프를 지원한다.
          --shuffle 은 순서 효과(앞쪽 문항에서 할당량을 다 쓰는 등)를 없앤다.

        [생성 파라미터] --max-tokens / --temperature / --repeat
          --repeat 2 이상은 '일관성'을 보기 위한 것이다(run() Note 참조).

        [출력/이어하기] --fail-only / --resume / --out-prefix
          --resume 은 할당량이 끊긴 뒤 이어서 돌릴 때 쓴다.

        ★ help 문자열의 `100%%` 는 argparse 가 % 포매팅을 적용하기 때문에
          리터럴 퍼센트를 표현하려면 %% 로 이스케이프해야 한다. 한 번 겪지 않으면
          모르는 함정이라, 여기 기록해 둔다.
    """
    p = argparse.ArgumentParser(description="충실성 테스트셋 채점기 (할루시네이션/자료무시 잡기)")
    p.add_argument("--testset", default="", help="build_faithfulness_testset.py 산출 파일")
    p.add_argument("--model",
                   default=os.getenv("RAG_EVAL_MODEL") or os.getenv("DEFAULT_LLM", "local:qwen2.5-3b-instruct-q4_k_m.gguf"),
                   help="생성 모델 (local:... 또는 openrouter:...). 기본=DEFAULT_LLM")
    p.add_argument("--limit", type=int, default=None, help="앞 N개만 (스모크용)")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="문항 사이 대기(초). 무료 등급 분당 한도(429) 예방용. 예: --sleep 4")
    p.add_argument("--max-retries", type=int, default=4, dest="max_retries",
                   help="429/503 등 '일시 오류' 시 최대 재시도 횟수(기본 4). 모델의 실제 오답은 재시도 안 함")
    p.add_argument("--use-rag-prompt", action="store_true", dest="use_rag_prompt",
                   help="충실성 지시로 '당신 실제 main_answer 프롬프트'를 사용(실전과 100%% 동일 측정)")
    p.add_argument("--prompt-file", dest="prompt_file", default=None,
                   help="--use-rag-prompt 시 읽을 프롬프트 JSON(미지정 시 .env의 DEFAULT_PROMPT_FILE)")
    p.add_argument("--out-prefix", default="faithfulness", help="출력 파일 접두어")
    # --- 문항 선택 ---
    p.add_argument("--only-types", dest="only_types", default=None,
                   help="이 유형만 평가(쉼표구분). 예: --only-types negation,exception")
    p.add_argument("--exclude-types", dest="exclude_types", default=None,
                   help="이 유형 제외(쉼표구분). 예: --exclude-types grounded")
    p.add_argument("--ids", default=None,
                   help="특정 문항만(쉼표구분). 예: --ids F035_negation,F037_negation")
    p.add_argument("--shuffle", action="store_true", help="문항 순서를 섞어 평가")
    p.add_argument("--seed", type=int, default=42, help="--shuffle 시 난수 시드")
    # --- 생성 파라미터 오버라이드 ---
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=None,
                   help="답변 최대 토큰. LLMController.ask 인자로 전달한다.")
    p.add_argument("--temperature", type=float, default=None,
                   help="생성 temperature 오버라이드")
    p.add_argument("--repeat", type=int, default=1,
                   help="각 문항을 N회 실행해 일관성 측정(기본 1)")
    # --- LLM 2차 심판 ---
    p.add_argument("--judge", action="store_true",
                   help="규칙 채점 후 LLM 2차 심판 (기본 off, pass 는 규칙)")
    p.add_argument("--judge-model", dest="judge_model", default=None,
                   help="심판 모델 식별자 (미지정 시 --model 과 동일)")
    p.add_argument("--judge-mode", dest="judge_mode", default="report",
                   choices=["report", "gate", "override"],
                   help="report=참고만(기본) / gate=규칙AND심판 / override=규칙OR심판")
    # --- 출력/이어하기 ---
    p.add_argument("--fail-only", dest="fail_only", action="store_true",
                   help="실패 문항만 콘솔·CSV에 남긴다")
    p.add_argument("--resume", action="store_true",
                   help="기존 {out-prefix}_report.csv 에 있는 문항은 건너뛰고 이어서 평가")
    p.add_argument("--self-test", action="store_true",
                   help="채점 헬퍼 단위검증 후 종료 (DB·LLM 불필요)")
    return p.parse_args()


def _self_test() -> None:
    """DB·LLM 없이 문자열 채점 헬퍼만 검증한다."""
    assert norm("a b") == "ab"
    assert contains("183일 규정입니다", "183일")
    assert not contains("연차", "휴가")
    assert is_abstention("제공된 자료에서 확인할 수 없습니다")
    nums = numbers_in("183일 이상 거주")
    assert any("183" in n for n in nums)
    fab = fabricated_numbers("300일 필요합니다", "183일 규정")
    assert "300일" in fab
    print("[self-test] evaluate_faithfulness OK")


def _load_rag_main_prompt(prompt_file: str | None) -> None:
    """--use-rag-prompt: 프롬프트 JSON에서 main_answer 를 읽어 충실성 지시로 사용한다.

    Args:
        prompt_file: 프롬프트 JSON 경로. None 이면 .env 의 DEFAULT_PROMPT_FILE.

    Note:
        전역 `_RAG_MAIN_PROMPT` 를 설정한다. generate_answer 가 이 값을 보고
        prompt_template 을 고른다.

        ★ 실패해도 예외를 던지지 않는 이유(Why):
          프롬프트 파일이 없다고 평가 전체를 중단시키는 것은 과하다. 기본
          충실성 지시로도 측정은 가능하기 때문이다. 다만 **어떤 프롬프트로 쟀는지가
          결과 해석을 좌우**하므로, 폴백할 때 반드시 경고를 남긴다.
          경고 없이 조용히 다른 조건으로 재는 것이 가장 나쁘다.

        전역 변수를 쓰는 것은 이 스크립트가 단일 실행 CLI 이고, generate_answer 의
        시그니처를 건드리지 않기 위한 선택이다(테스트 시에는 이 전역을 직접
        설정/복원해야 한다는 점에 주의).
    """
    global _RAG_MAIN_PROMPT
    path = prompt_file or os.getenv("DEFAULT_PROMPT_FILE")
    if not path or not os.path.exists(path):
        print(f"[WARN] 프롬프트 파일을 찾을 수 없어(--use-rag-prompt 무시) 기본 충실성 지시를 사용합니다. "
              f"(경로: {path})", file=sys.stderr)
        return
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        main_answer = (data.get("main_answer") or "").strip()
        if main_answer:
            _RAG_MAIN_PROMPT = main_answer
            print(f"[BOOT] 실제 main_answer 프롬프트 로드 완료: {path}", file=sys.stderr)
        else:
            print("[WARN] 프롬프트 파일에 main_answer 가 비어 있어 기본 충실성 지시를 사용합니다.",
                  file=sys.stderr)
    except Exception as e:
        print(f"[WARN] 프롬프트 로드 실패({e}) → 기본 충실성 지시 사용", file=sys.stderr)


def main() -> None:
    """CLI 진입점: 설정 → 로드/필터 → (이어하기) → 실행 → 집계 → 저장.

    Raises:
        SystemExit: 테스트셋이 비었거나 필터 결과가 0개일 때.

    Note:
        ★ 실행 순서에 의미가 있다:
          1) 프롬프트 로드 — 생성 경로가 결정되므로 가장 먼저.
          2) --max-tokens / --temperature 는 generate_answer 호출 인자로 넘긴다.
             get_default_* 는 호출 시점에 env 를 읽으므로 os.environ 을 바꿀 필요가 없다.
          3) 테스트셋 로드 → 필터 → 셔플 → resume 제외
          4) 실행 → 이전 결과와 합쳐 집계 → 리포트 → 저장

        ★ resume 처리에서 문항이 0개가 되어도 **종료하지 않는다.**
          이전 결과만으로 집계·리포트를 다시 만들어 주는 것이 유용하기 때문이다
          (예: 채점 결과를 다시 요약하고 싶을 때).

        ★ `results = prior + new_results` — 이어하기 결과는 이전 결과 뒤에 붙는다.
          집계는 순서와 무관하므로 문제없다.
    """
    a = parse_args()
    if not a.testset:
        print("[ERROR] --testset 이 필요합니다.", file=sys.stderr)
        sys.exit(1)
    from llm_handler import exit_if_invalid_model
    exit_if_invalid_model(a.model, label="생성 모델")
    if a.judge:
        exit_if_invalid_model(a.judge_model or a.model, label="심판 모델")
    if a.use_rag_prompt:
        _load_rag_main_prompt(a.prompt_file)

    testset = load_testset(a.testset)
    if not testset:
        print("[ERROR] 빈 테스트셋", file=sys.stderr)
        sys.exit(1)

    # 유형/ID 필터
    testset = filter_testset(testset, _csv_list(a.only_types),
                             _csv_list(a.exclude_types), _csv_list(a.ids))
    if not testset:
        print("[ERROR] 필터(only/exclude/ids) 결과 문항이 0개입니다.", file=sys.stderr)
        sys.exit(1)

    if a.shuffle:
        # 지역 import: --shuffle 을 안 쓰면 필요 없는 모듈이다.
        import random
        random.seed(a.seed)
        random.shuffle(testset)

    # 이어하기: 기존 리포트에 있는 문항은 건너뛴다
    prior: list[dict] = []
    if a.resume:
        prior = load_existing_results(a.out_prefix)
        done = {_base_id(r["id"]) for r in prior}
        before = len(testset)
        testset = [x for x in testset if _base_id(x.get("id", "")) not in done]
        print(f"[RESUME] 기존 {len(done)}개 완료 → 남은 {len(testset)}/{before}개 실행",
              file=sys.stderr)
        if not testset:
            # 남은 문항이 없어도 종료하지 않는다. 아래에서 prior 만으로 집계·리포트를
            # 다시 생성해 준다(결과 재요약 용도).
            print("[RESUME] 새로 평가할 문항이 없습니다(모두 완료).", file=sys.stderr)

    new_results = run(testset, a.model, a.limit,
                      sleep_between=a.sleep, max_retries=a.max_retries,
                      repeat=a.repeat, fail_only=a.fail_only,
                      max_tokens=a.max_tokens, temperature=a.temperature,
                      judge_model=(a.judge_model or a.model) if a.judge else None,
                      judge_mode=a.judge_mode)
    results = prior + new_results
    summary = aggregate(results)
    print_report(summary)
    emit_outputs(results, summary, a.out_prefix, fail_only=a.fail_only)


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        _self_test()
        sys.exit(0)
    main()
