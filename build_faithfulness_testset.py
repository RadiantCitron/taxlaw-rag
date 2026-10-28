#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""충실성(faithfulness) 테스트셋 생성기.

================================================================================
0. 한 줄 요약
================================================================================
"이 RAG(정확히는 답변 LLM)가 '거짓말'을 하는가?"를 잡아내는 빡센 테스트셋을
DB에서 자동으로 만든다.

================================================================================
1. 왜 이게 기존 build_testset.py 보다 센가?
================================================================================
기존 build_testset.py 는 "정답을 맞히는가(정확성)"를 주로 봤다.
이 파일은 "주어진 자료에만 충실한가(충실성, faithfulness)"를 본다.

  * 정확성(correctness) : 실제 세상 기준으로 맞는 답인가?
  * 충실성(faithfulness): '내가 준 자료'에 충실한 답인가? 자료에 없으면 없다고
                          하는가? 자료가 틀렸으면 (틀린 채로) 자료를 따르는가?

당신의 요구는 후자다. "화려한 문장력이 아니라, 지어내지 않고 주어진 정보대로
말하는 것." 그래서 이 생성기는 정답이 코퍼스 안에 있는지 없는지를 '계산'해서,
LLM이 지어내기 쉬운 함정을 의도적으로 만든다.

이 구분이 왜 결정적인가(Why):
    정확성만 재면, 모델이 자기 사전학습 지식으로 답을 맞혀도 만점이 나온다.
    그러면 "이 시스템은 잘 돌아간다"고 착각하게 되는데, 실제로는 **검색 결과를
    무시하고 기억으로 답하는** 상태일 수 있다. 그런 시스템은 법이 개정되는 순간
    조용히 틀린 답을 내기 시작한다(내 자료는 바뀌었는데 모델의 기억은 안 바뀌므로).
    충실성 테스트는 바로 그 '통제 가능성'을 측정한다.

================================================================================
2. 만들어내는 함정(문항 유형) 5+1 가지
================================================================================
  1) grounded      : 자료에 답이 있음      -> 그대로 답해야 통과 (기본기 확인)
  2) unanswerable  : 자료에 답이 없음      -> "자료에 없다"고 해야 통과.
                                              지어내면 실패(= 할루시네이션)
  3) counterfactual: 자료의 숫자를 일부러 바꿈(183일->300일)
                     -> '바뀐 값'을 답해야 통과. 원래 값(183)을 답하면 실패
                        (= 자료를 무시하고 제 기억을 말함 = 통제 불능)
  4) negation      : 자료가 "X는 제외한다"고 함
                     -> "X는 포함되나?" 에 "아니오/제외"라고 해야 통과
                        (모델은 부정문을 자주 뒤집는다)
  5) exception     : 자료에 "다만 ..." 단서(예외)가 있음
                     -> 그 예외를 빠뜨리지 않고 말해야 통과
  6) distractor    : 관련 조문 + 무관한 조문을 섞어 줌
                     -> 엉뚱한 조문의 숫자를 끌어오면 실패

각 유형이 겨냥하는 실패 모드(Why 이 6가지인가):
    unanswerable   → 할루시네이션(없는 걸 지어냄). RAG 최악의 사고.
    counterfactual → 컨텍스트 무시(파라메트릭 기억 우선). 개정 대응 불가의 원인.
    negation       → 부정문 오독. LLM의 고질적 약점이며 세법에서 '제외 규정'은
                     결론을 정반대로 뒤집는다.
    exception      → 단서 누락. 세법 답변에서 '다만' 이하를 빠뜨리면 실무상
                     완전한 오답이 된다(원칙만 말하고 예외를 숨긴 답).
    distractor     → 근거 혼입. 검색이 여러 문서를 주는 실제 상황의 축소판.
    grounded       → 위 함정들이 '너무 어려워서' 못 맞힌 게 아님을 증명하는 대조군.
                     grounded 조차 못 맞히면 함정 성적은 해석할 가치가 없다.

================================================================================
3. 핵심 설계 원칙
================================================================================
  * 질문 문장은 '평범'해도 된다. 함정은 '질문'이 아니라 '자료(context)'에 심는다.
    그래서 이 생성기는 LLM 없이(결정론적으로) 돌아간다 -> 재현 가능, 무비용.
    → 이것이 이 파일의 가장 중요한 발상이다. 함정을 자료에 심으면, 정답이
      '계산 가능'해진다(내가 183을 300으로 바꿨으니 정답은 300이다). 정답을
      계산할 수 있으면 채점에 LLM 심판이 필요 없다.
  * 각 문항은 LLM에게 그대로 먹일 context 를 통째로 들고 다닌다.
    (평가기 evaluate_faithfulness.py 가 이 context 를 LLM에 주입해서 채점)
  * 채점은 사람 없이 되도록 must_include / must_not_include / expect_abstention
    같은 '기계 채점용 힌트'를 문항마다 붙인다.

================================================================================
4. 주의사항 (전역)
================================================================================
  * 이 생성기는 tax_laws_master(법령) 만 사용한다. 판례/심판례는 현재 검색기가
    다루지 않으므로 공정한 테스트를 위해 제외한다(기존 결정과 동일).
  * counterfactual 로 만든 context 는 **사실이 아닌 변조 텍스트**다. 실수로
    이 파일의 산출물을 학습·인덱싱 데이터로 쓰면 코퍼스가 오염된다.
    산출 JSON 은 반드시 '평가용'으로만 취급할 것.
  * 문항 생성기(make_*)는 실패 시 예외 대신 None 을 돌려준다. 상위 배분 함수가
    '조용히 건너뛰고 다음 조문으로' 넘어가기 위해서다. 그 결과 **요청한 개수보다
    적게 생성될 수 있고**, 그건 버그가 아니라 품질 방어의 결과다(main 이 경고한다).
"""

from __future__ import annotations

import os
import re
import csv
import sys
import json
import random
import argparse
from dataclasses import dataclass, asdict, field
from typing import TYPE_CHECKING, Any, Callable, NoReturn

if TYPE_CHECKING:
    # 런타임에 psycopg2 가 없을 수도 있으므로(아래 try/except 참조),
    # 커넥션 타입은 '타입 검사 시점에만' 가져온다.
    # `from __future__ import annotations` 로 어노테이션이 지연 평가되기에 안전하다.
    from psycopg2.extensions import connection as PgConnection

# --- .env 로드(있으면) -------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- DB 드라이버 -------------------------------------------------------------
# Why 이렇게 감싸는가:
#   이 모듈의 텍스트 유틸(숫자 추출, 배제 대상 파싱 등)은 DB 없이도 단위 테스트가
#   가능해야 한다. psycopg2 가 없다고 import 단계에서 죽으면 그 테스트조차 못 돈다.
#   그래서 드라이버는 '없으면 None' 으로 두고, 실제 DB 접근 시점(get_conn)에만
#   친절한 오류로 안내한다.
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:  # 셀프테스트 때는 psycopg2 가 없을 수 있음
    psycopg2 = None
    RealDictCursor = None

# DATABASE_URL 을 폴백으로 둔 이유: Heroku/도커 계열 관례 이름과의 호환.
TAX_DB_URI: str | None = os.getenv("TAX_DB_URI") or os.getenv("DATABASE_URL")


# =============================================================================
# 쟁점 분류 (기존 build_testset.py 와 동일한 도메인 축)
#
# Why 같은 축을 쓰는가: 두 시험지(정확성/충실성)의 결과를 쟁점별로 나란히 비교할
# 수 있어야, "부가가치세 쪽은 검색이 약하고, 상증세 쪽은 충실성이 약하다" 같은
# 진단이 가능해진다. 축이 다르면 교차 분석이 불가능하다.
# =============================================================================
ISSUE_KEYWORDS: dict[str, list[str]] = {
    "거주자_비거주자": ["거주자", "비거주자", "거소", "주소", "183일"],
    "양도소득":       ["양도소득", "양도차익", "양도소득세", "비사업용"],
    "상속_증여":      ["상속", "증여", "시가", "평가"],
    "부가가치세":     ["사업자", "과세대상", "면세", "영세율", "세금계산서"],
    "종합부동산세":   ["종합부동산세", "과세기준일", "합산", "공제"],
    "조세특례":       ["특례", "감면", "세액공제", "과세특례"],
}

# 부정/배제 표지 (negation 문항 재료 찾기용).
# 한국 법령의 배제 규정은 어미가 몇 가지로 정형화되어 있어, 이 목록만으로도
# 재현율이 꽤 높다. 순서에 의미는 없지만, 뒤로 갈수록 일반적('아니한다')이라
# 오탐이 늘어난다는 점만 기억할 것.
EXCLUSION_MARKERS = [
    "제외한다", "제외하며", "제외하고", "포함하지 아니한다", "포함되지 아니한다",
    "해당하지 아니한다", "적용하지 아니한다", "보지 아니한다", "아니한다",
]

# 예외(단서) 표지. 한국 법령에서 예외는 거의 항상 '다만' 으로 시작하고,
# 드물게 '단,' 을 쓴다. 이 두 개면 실질적으로 충분하다.
EXCEPTION_MARKERS = ["다만", "단,"]

# 숫자성 표현(값을 바꾸거나 뽑을 때 사용). text_norm.NUM_UNIT_RE 와 동일.
from text_norm import NUM_UNIT_RE, norm

STOPWORDS = {
    "대통령령", "경우", "다음", "각", "호의", "또는", "이하", "이상", "미만",
    "초과", "해당", "관련", "포함", "제외", "따라", "위하", "규정", "사항",
    "내용", "제도", "다만", "관하", "대하", "때에", "그러", "여기",
}


# =============================================================================
# 데이터 모델 : 한 문항의 스키마
# =============================================================================
@dataclass
class FaithItem:
    """충실성 문항 하나.

    Attributes:
        id: 'F001_negation' 형식의 문항 식별자(`F{연번}_{유형}`).
        type: grounded|unanswerable|counterfactual|negation|exception|distractor.
        issue: 쟁점 키(결과를 쟁점별로 쪼개 보기 위한 라벨).
        question: LLM 에게 물을 질문.
        context: **LLM 에게 그대로 먹일 자료.** counterfactual 이면 이미 변조된 상태다.
        expected_behavior: 사람이 읽는 '충실한 답'의 모습(자동 채점에는 쓰이지 않는
            설명 필드). 실패 문항을 사람이 검수할 때 판단 근거가 된다.
        must_include: 답에 있어야 하는 표지.
        must_not_include: 있으면 실패(할루시/누출).
        expect_abstention: True 면 "자료에 없음" 취지가 나와야 통과.
        source_doc_id: 원본 조문 DB id(추적용).
        source_law: 원본 법령명.
        source_article: 원본 조문 번호.
        mutation_from: counterfactual 에서 바꾸기 전의 원래 값.
        mutation_to: counterfactual 에서 바꾼 뒤의 값.

    Note:
        ★ context 를 문항이 통째로 들고 다니는 것이 이 설계의 핵심이다(Why).
          평가 시점에 검색을 돌리면 검색 결과가 매번 달라져, '충실성'을 재려던
          실험에 '검색 변동성'이라는 잡음이 섞인다. 자료를 문항에 고정해 두면
          같은 문항 = 항상 같은 입력이 보장되고, 모델 간 비교가 공정해진다.
          (대가: JSON 파일이 커진다. 조문 전문을 문항 수만큼 저장하므로.)

        ★ mutation_from / mutation_to 는 사고 조사용이다. 채점이 이상할 때
          "무엇을 무엇으로 바꿨는지"를 즉시 확인할 수 있어야 한다.

        field(default_factory=list) 를 쓰는 이유: dataclass 에서 가변 기본값
        (`= []`)은 모든 인스턴스가 같은 리스트를 공유하는 고전적 버그를 낳으므로
        파이썬이 아예 금지한다. default_factory 가 정석이다.
    """

    id: str
    type: str                       # grounded|unanswerable|counterfactual|negation|exception|distractor
    issue: str
    question: str                   # LLM 에게 물을 질문
    context: str                    # LLM 에게 '그대로' 먹일 자료(반사실이면 변조된 상태)
    expected_behavior: str          # 사람이 읽는 '충실한 답'의 모습
    must_include: list[str] = field(default_factory=list)      # 답에 있어야 하는 표지
    must_not_include: list[str] = field(default_factory=list)  # 있으면 실패(할루시/누출)
    expect_abstention: bool = False  # True면 "자료에 없음" 취지가 나와야 통과
    # --- 출처/추적용 ---
    source_doc_id: str = ""
    source_law: str = ""
    source_article: str = ""
    mutation_from: str = ""         # counterfactual: 원래 값
    mutation_to: str = ""           # counterfactual: 바꾼 값


def die(msg: str, code: int = 1) -> NoReturn:
    """치명적 오류를 stderr 에 찍고 종료한다.

    Args:
        msg: 오류 메시지.
        code: 종료 코드(0이 아니어야 CI 가 실패로 인식).

    Raises:
        SystemExit: 항상.

    Note:
        stdout 이 아닌 stderr 를 쓰는 이유는 `--format stdout` 일 때 결과 JSON 이
        stdout 으로 나가기 때문이다. 리다이렉트한 파일에 오류 문구가 섞이면 안 된다.
    """
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


# =============================================================================
# DB 접근
# =============================================================================
def get_conn() -> "PgConnection":
    """tax_rag_db(PostgreSQL) 커넥션을 연다.

    Returns:
        psycopg2 커넥션.

    Raises:
        SystemExit: 드라이버가 없거나 접속 URI 가 없을 때(die 경유).
        psycopg2.OperationalError: 실제 접속 실패.

    Note:
        드라이버 유무를 **여기서** 검사하는 이유: 모듈 import 시점에 죽이지 않고,
        DB 가 실제로 필요한 순간까지 미루기 위해서다(모듈 상단 import 주석 참조).
        덕분에 텍스트 유틸만 쓰는 셀프테스트가 psycopg2 없이도 돌아간다.
    """
    if psycopg2 is None:
        die("psycopg2 가 설치되어 있지 않습니다. (pip install psycopg2-binary)")
    if not TAX_DB_URI:
        die("TAX_DB_URI 환경변수가 없습니다. (.env 확인)")
    return psycopg2.connect(TAX_DB_URI)


def fetch_articles(conn: "PgConnection", keywords: list[str] | None,
                   must_contain: list[str] | None,
                   limit: int, laws: list[str] | None = None) -> list[dict]:
    """tax_laws_master 에서 조문을 가져온다.

    Args:
        conn: 열려 있는 DB 커넥션.
        keywords: 이 중 하나라도 본문/제목에 있으면 후보 (쟁점 필터). None 이면 무시.
        must_contain: 이 중 하나라도 본문에 반드시 있어야 함 (예: '다만', '제외한다').
            None 이면 무시.
        limit: 최대 행 수.
        laws: 지정 시 law_name 이 이 중 하나를 **포함**하는 조문만 (법령 필터, 부분일치).

    Returns:
        id/law_name/article_number/article_title/full_text 를 가진 dict 목록.

    Raises:
        psycopg2.Error: SQL 실행 실패.

    Note:
        keywords/must_contain 이 None 이면 '본문이 충분히 긴' 조문을 폭넓게 가져온다.

        ★ 이 함수가 이 파일의 '재료 조달' 전략 전체를 담당한다(Why 이렇게 일반화했나):
          유형마다 필요한 재료가 다르다.
            - negation  → 본문에 '제외한다' 류가 **반드시** 있어야 한다.
            - exception → 본문에 '다만' 이 **반드시** 있어야 한다.
            - 나머지    → 숫자만 있으면 된다.
          이 "반드시 포함" 조건을 SQL 로 미리 걸어야, 파이썬에서 수천 건을 훑으며
          버리는 낭비를 없앨 수 있다. 그래서 must_contain 파라미터가 존재한다.

        ★ 두 필터의 논리 구조 차이에 주의:
          keywords 와 must_contain 은 **각각 내부적으로 OR**, 서로는 AND 로 결합된다.
          즉 (kw1 OR kw2) AND (mk1 OR mk2) AND (law1 OR law2).
          "must_contain 의 모든 항목을 포함"이 아니라 "하나라도 포함"이라는 뜻이다.

        ★ `length(full_text) > 60` : build_testset(40자)보다 기준이 높다.
          충실성 문항은 자료 안에서 숫자·단서·배제 표현을 찾아내야 해서 더 긴
          본문이 필요하기 때문이다.

        ★ `ORDER BY length(full_text) DESC` 는 '정보량 많은 조문 우선'이라는 뜻이지만,
          그대로 쓰면 매번 같은 장문 조문만 뽑힌다. 그래서 호출부(build)가
          가져온 뒤 random.shuffle 로 다양성을 확보한다.

        ★ params 순서는 SQL 의 %s 등장 순서와 정확히 일치해야 한다.
          여기서는 (keywords → must_contain → laws → limit) 순이며,
          WHERE 절 조립 순서와 같다. 조건을 추가할 때 이 순서를 깨면
          "타입 오류" 또는 더 나쁘게 "조용한 오조회"가 발생한다.
    """
    # 공통 전제: 본문이 있고 충분히 긴 조문만.
    where = ["full_text IS NOT NULL", "length(full_text) > 60"]
    params: list = []

    if keywords:
        subs = []
        for kw in keywords:
            subs.append("(full_text ILIKE %s OR article_title ILIKE %s)")
            params.extend([f"%{kw}%", f"%{kw}%"])
        where.append("(" + " OR ".join(subs) + ")")

    if must_contain:
        subs = []
        for m in must_contain:
            subs.append("full_text ILIKE %s")
            params.append(f"%{m}%")
        where.append("(" + " OR ".join(subs) + ")")

    if laws:
        subs = []
        for lw in laws:
            subs.append("law_name ILIKE %s")
            params.append(f"%{lw}%")
        where.append("(" + " OR ".join(subs) + ")")

    params.append(limit)
    sql = f"""
        SELECT id, law_name, article_number,
               COALESCE(article_title,'') AS article_title, full_text
        FROM tax_laws_master
        WHERE {' AND '.join(where)}
        ORDER BY length(full_text) DESC
        LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


# =============================================================================
# 텍스트 유틸 (LLM 없이 함정을 '계산'하는 부분)
#
# 이 영역이 이 파일의 심장이다. "함정을 계산한다"는 말의 실체가 여기 있다.
# 예: 본문에서 '183일'을 찾아(find_number_tokens) 값인지 확인하고
#     (is_structural_ref), 가장 의미 있는 것을 골라(pick_salient_number)
#     '300일'로 바꾸면(mutate_number) 정답이 '300일'인 문항이 만들어진다.
#     이 과정에 LLM 은 한 번도 등장하지 않는다.
# =============================================================================


def format_article_display(art: str) -> str:
    """'1','1의4' -> '제1조','제1조의4' 로 표시형을 만든다.

    Args:
        art: DB 원형 조문 번호.

    Returns:
        표시용 문자열. 입력이 비면 빈 문자열.

    Note:
        질문 문장에 자연스러운 한국어 조문 표기를 쓰기 위한 변환이다.
        DB 저장형(정렬·비교용)과 표시형(사람이 읽는 형태)을 분리하는 원칙.
    """
    art = str(art or "").strip()
    if not art:
        return ""
    if "의" in art:
        base, sub = art.split("의", 1)
        return f"제{base}조의{sub}"
    return f"제{art}조"


def first_sentences(text: str, n: int = 3, max_chars: int = 600) -> str:
    """본문 앞의 문장 n개를 잘라 반환한다.

    Args:
        text: 원문.
        n: 취할 문장 수.
        max_chars: 길이 상한.

    Returns:
        요약 문자열. 분리에 실패하면 원문을 그대로 잘라 돌려준다.

    Note:
        법령이 두괄식이라는 점을 이용한 근사 요약이다.
        `[ \\t]+` 만 정규화(개행은 보존)하는 점이 build_testset 판과 다르다.
        여기서는 개행을 문장 경계로 활용해야 하므로 개행을 미리 지우면 안 된다.
    """
    text = re.sub(r"[ \t]+", " ", (text or "").strip())
    parts = re.split(r"(?<=[.。])\s+|\n+", text)
    out = " ".join(p for p in parts[:n] if p).strip() or text
    return out[:max_chars].strip()


def find_number_tokens(text: str) -> list[tuple[str, str]]:
    """본문에서 (원문숫자표현, 숫자만) 목록을 등장 순서대로 반환한다.

    Args:
        text: 검사할 텍스트.

    Returns:
        (raw, digits) 튜플 목록. 예: '183일' -> ('183일','183'),
        '6억 원' -> ('6억 원','6').

    Note:
        raw 를 함께 돌려주는 이유(Why):
            변조(counterfactual)와 채점(must_include)에는 **원문 표기 그대로**가
            필요하다. 본문 치환은 `full.replace(raw, new_raw)` 로 하므로 raw 가
            원문과 정확히 일치해야 하고, 답변 채점도 사람이 쓰는 표기('183일')로
            해야 의미가 있다. digits 는 비교·연산용이다.

        `re.sub(r"[^\\d]", "", ...)` 로 콤마를 제거한다('1,000' → '1000').
        int() 변환이 가능해야 mutate_number 가 동작하기 때문이다.

        ★ 주의: '6억 원' 의 digits 는 '6' 이다. 억 단위가 digits 에 반영되지 않으므로
          pick_salient_number 의 '값이 큰 것 우선' 비교에서 '6억원'(6)이
          '183일'(183)보다 작게 평가된다. 알려진 한계이며, 어차피 '어느 것을
          함정으로 쓸까'의 휴리스틱일 뿐이라 정확성 요구가 낮다.
    """
    out = []
    for m in NUM_UNIT_RE.finditer(text or ""):
        raw = m.group(0)
        digits = re.sub(r"[^\d]", "", m.group(1))
        if digits:
            out.append((raw, digits))
    return out


# 조문 구조 참조 단위(값이 아니라 '다른 조항을 가리키는 번호'). 정답/변조 대상에서 제외.
STRUCTURAL_UNITS = ("조", "항", "호", "목", "장", "절", "관")


def is_structural_ref(raw: str) -> bool:
    """'15항','4118호','3조'처럼 구조 참조인가(값이 아님)를 판정한다.

    Args:
        raw: find_number_tokens 가 돌려준 원문 숫자 표현.

    Returns:
        구조 참조면 True.

    Note:
        ★ 이 함수가 없으면 문항 품질이 무너진다(Why):
          법령 본문에는 '제3호에 따라', '제15항의 규정' 같은 **상호참조 번호**가
          숫자 표현으로 잔뜩 잡힌다. 이걸 값으로 오인하면
            - counterfactual : '제3호'를 '제8호'로 바꿔 버려 문장이 **말이 안 되게**
              된다(존재하지 않는 호를 가리키게 됨). 이런 자료로는 충실성을 잴 수 없다.
            - grounded/distractor : "'제3호' 관련 기준은?" 같은 무의미한 질문이 된다.
          즉 '숫자'와 '값'을 구분하는 이 한 줄이 문항의 의미를 지킨다.

        판정 방식: 숫자 뒤 단위가 정확히 구조 단위 중 하나인지 본다.
        '세'(나이), '일', '년' 등은 값이므로 통과된다.
    """
    m = re.match(r"^\d[\d,]*\s*(.+)$", (raw or "").strip())
    unit = (m.group(1).strip() if m else "")
    return unit in STRUCTURAL_UNITS


def value_number_tokens(text: str) -> list[tuple[str, str]]:
    """숫자 토큰 중 '값'인 것만 반환한다(구조 참조 호/항/조 등은 제외).

    Args:
        text: 검사할 텍스트.

    Returns:
        (raw, digits) 목록.

    Note:
        find_number_tokens + is_structural_ref 의 조합을 한 번에 쓰기 위한 편의 함수.
        counterfactual/distractor 처럼 '값'이 반드시 필요한 유형이 이걸 쓴다.
    """
    return [(raw, dig) for raw, dig in find_number_tokens(text) if not is_structural_ref(raw)]


def pick_salient_number(tokens: list[tuple[str, str]]) -> tuple[str, str] | None:
    """숫자 토큰들 중 '가장 의미 있는' 것을 고른다.

    Args:
        tokens: (raw, digits) 목록.

    Returns:
        선택된 (raw, digits). 후보가 없으면 None.

    Note:
        - 구조 참조(호/항/조 등)는 후보에서 제외한다(그건 값이 아니라 참조번호).
        - 남은 것 중 값이 가장 큰 숫자를 택한다(날짜 '1월'보다 '183일/9억' 같은
          기준을 선호).

        Why '가장 큰 값'이라는 단순 규칙인가:
            법령에서 '1월', '2년' 같은 작은 숫자는 시행일·경과규정 등 부수적 문구인
            경우가 많고, '183일', '9억원', '50퍼센트' 처럼 큰 값이 그 조문의
            **핵심 기준**인 경향이 있다. 완벽하진 않지만, 규칙이 단순해 예측
            가능하고 디버깅이 쉽다는 점이 중요하다(문항 품질 문제를 추적할 때
            "왜 이 숫자가 뽑혔지?"에 즉답할 수 있어야 한다).

        int 변환 실패는 조용히 건너뛴다. 정규식상 거의 발생하지 않지만,
        예외로 배치 전체를 죽일 이유가 없다.
    """
    best, best_val = None, -1
    for raw, digits in tokens:
        if is_structural_ref(raw):
            continue
        try:
            v = int(digits)
        except ValueError:
            continue
        if v > best_val:
            best, best_val = (raw, digits), v
    return best


def mutate_number(digits: str) -> str:
    """반사실(counterfactual) 문항용으로 숫자를 '확실히 다른' 값으로 바꾼다.

    Args:
        digits: 숫자 문자열(콤마 없는 상태).

    Returns:
        변조된 숫자 문자열. 변환 불가하면 입력 그대로.

    Note:
        규칙은 단순하지만 결과가 원본과 겹치지 않게만 보장하면 된다.

        Why 자릿수대별로 다르게 더하는가:
            변조값이 **그럴싸해야** 한다. 183일을 '10000일'로 바꾸면 모델이
            "말이 안 되는데?" 하고 자기 기억으로 되돌아갈 빌미를 준다.
            반대로 183 → 184 처럼 미세하게 바꾸면 답변에서 원래 값과 변조값을
            구분하기 어려워 채점이 흔들린다.
            그래서 "크기 대비 눈에 띄지만 현실감은 유지"되는 증분을 자릿수별로 뒀다.
              < 10   : +5     (5년 → 10년)
              < 100  : +40    (50% → 90%)
              < 1000 : +200   (183일 → 383일)
              그 이상 : ×1.7  (금액 등 큰 수는 비율로)

        마지막 `if new == n` 는 이론상 도달하지 않지만(모든 분기가 값을 바꾼다),
        규칙을 나중에 고칠 때를 대비한 안전장치다. '변조했는데 값이 같다'면
        정답과 오답이 같아져 문항이 완전히 망가지기 때문이다.
    """
    try:
        n = int(digits)
    except ValueError:
        return digits
    if n < 10:
        new = n + 5
    elif n < 100:
        new = n + 40
    elif n < 1000:
        new = n + 200
    else:
        new = int(n * 1.7)
    if new == n:
        new = n + 1
    return str(new)


def extract_salient_noun(text: str, k: int = 1) -> list[str]:
    """제목/본문에서 distinctive 한 한글 명사(2~6자)를 뽑는다.

    Args:
        text: 대상 텍스트.
        k: 뽑을 개수.

    Returns:
        길이 내림차순으로 정렬된 후보 중 앞 k개.

    Note:
        형태소 분석기 없이 정규식으로 근사한다. 긴 단어를 우선하는 이유는
        복합명사(예: '종합부동산세')가 더 구체적이고 변별력이 높기 때문이다.

        주의: 조사가 붙은 형태('거주자가')가 그대로 뽑힐 수 있다.
        unanswerable 문항에서 이 토큰이 질문의 주제어가 되므로, 어색한 주제어가
        나오면 이 함수의 한계 때문이다. STOPWORDS 로 최악만 거른다.
    """
    cand = []
    for w in re.findall(r"[가-힣]{2,6}", text or ""):
        if w in STOPWORDS or w in cand:
            continue
        cand.append(w)
    cand.sort(key=lambda x: -len(x))
    return cand[:k]


def clause_after(text: str, marker: str, max_chars: int = 160) -> str:
    """marker('다만' 등) 뒤의 한 문장(대략)을 잘라 반환한다.

    Args:
        text: 원문.
        marker: 시작 표지('다만', '단,').
        max_chars: 문장 끝을 못 찾았을 때의 최대 길이.

    Returns:
        marker 를 **포함한** 단서 구간. marker 가 없으면 빈 문자열.

    Note:
        동작: marker 위치부터 시작해, marker 이후 구간에서 첫 문장 종료(마침표 뒤
        공백 또는 개행)를 찾아 그 지점까지 자른다. 못 찾으면 max_chars 로 자른다.

        Why marker 이후부터 문장 끝을 찾는가:
            '다만' 바로 앞에 마침표가 있는 경우가 많은데, marker 부터 검색하면
            그 마침표에 걸려 길이 0짜리 구간이 나올 수 있다. `tail[len(marker):]`
            로 오프셋을 준 뒤 다시 marker 길이를 더해 원래 좌표계로 되돌린다.
            (end = len(marker) + m.start())

        주의: 단서가 여러 문장에 걸쳐 있으면 첫 문장만 잡는다. exception 문항은
        '예외를 언급했는가'만 보므로 첫 문장으로 충분하다는 판단이다.
    """
    i = text.find(marker)
    if i < 0:
        return ""
    tail = text[i:]
    m = re.search(r"(?<=[.。])\s|\n", tail[len(marker):])
    end = (len(marker) + m.start()) if m else min(len(tail), max_chars)
    return tail[:end].strip()


# =============================================================================
# 문항 생성기 (유형별). 각각 실패하면 None 을 돌려주고 상위에서 건너뛴다.
#
# 공통 계약(중요):
#   - 반환이 None = "이 조문으로는 품질 좋은 문항을 만들 수 없다"는 뜻이다.
#     예외를 던지지 않는 이유는, 실패가 '정상적이고 빈번한' 사건이기 때문이다.
#     (숫자 없는 조문, 배제 대상을 깔끔히 못 뽑는 조문 등)
#   - 문항 수보다 **문항 품질**을 우선한다. 애매하면 버린다.
# =============================================================================
def make_grounded(idx: int, issue: str, row: dict) -> FaithItem | None:
    """자료에 답이 있는 기본 문항. 특정 숫자 기준을 묻는다.

    Args:
        idx: 문항 연번.
        issue: 쟁점 키.
        row: DB 행.

    Returns:
        FaithItem. 이 유형은 사실상 항상 생성에 성공한다(숫자가 없어도 명사로 폴백).

    Note:
        Why 대조군이 필요한가:
            counterfactual/negation 점수가 낮을 때, 그것이 '함정이 어려워서'인지
            '모델이 애초에 자료를 못 읽어서'인지 구분해야 한다. grounded 는
            함정이 전혀 없는 기본 문항이므로, 여기 점수가 낮으면 함정 유형의
            점수는 해석할 가치가 없다(모델이 바닥이라는 뜻).

        질문에 '제공된 자료에 따르면' 을 못 박는 이유:
            충실성 측정이 목적이므로, 모델이 배경지식으로 답할 여지를 문장에서부터
            차단한다. 이 문구는 모든 유형에서 일관되게 쓰인다.

        must 가 raw 숫자 표현 하나인 이유: 채점이 가장 명확한 토큰이기 때문이다.
        숫자를 못 찾았을 때만 명사로 폴백한다(채점 정밀도는 떨어진다).
    """
    full = row["full_text"] or ""
    law, art = row["law_name"] or "", str(row["article_number"] or "")
    disp = format_article_display(art)
    salient = pick_salient_number(find_number_tokens(full))  # 구조 참조 제외됨
    if salient:
        raw, _ = salient
        q = f"제공된 자료에 따르면, 「{law}」 {disp}에서 정한 '{raw.strip()}' 관련 기준은 무엇인가요? 자료에 근거해 답하세요."
        must = [raw.strip()]
    else:
        # 숫자가 없으면 주제 명사로 폴백. 채점은 느슨해지지만 문항은 살린다.
        topic = (extract_salient_noun(row.get("article_title") or full) or ["해당 규정"])[0]
        q = f"제공된 자료에 따르면, 「{law}」 {disp}의 '{topic}'에 관한 내용은 무엇인가요?"
        must = [topic]
    return FaithItem(
        id=f"F{idx:03d}_grounded", type="grounded", issue=issue,
        question=q, context=full,
        expected_behavior="자료 본문에 근거해 해당 내용을 답한다.",
        must_include=must, must_not_include=[], expect_abstention=False,
        source_doc_id=row["id"], source_law=law, source_article=art,
    )


def make_counterfactual(idx: int, issue: str, row: dict) -> FaithItem | None:
    """자료 속 숫자를 몰래 바꾼 뒤 그 값을 묻는다.

    Args:
        idx: 문항 연번.
        issue: 쟁점 키.
        row: DB 행.

    Returns:
        변조된 context 를 가진 FaithItem. 변조할 만한 숫자를 못 찾으면 None.

    Note:
        충실한 LLM은 '바뀐 값'을 답해야 한다(자료를 따르므로).
        제 기억으로 '원래 값'을 답하면 실패 = 자료 통제 실패.

        ★ 이 유형이 측정하는 것(Why 이게 가장 중요한 지표인가):
          RAG 의 존재 이유는 "내 자료로 모델을 통제한다"이다. 모델이 자료와
          자기 기억이 충돌할 때 기억을 택한다면, 그 시스템은 **법 개정을 반영할 수
          없다.** 개정 전 값이 사전학습에 박혀 있기 때문이다.
          counterfactual 실패율은 곧 '개정 대응 불가 위험도'다.

        ★ 후보 선정의 3중 방어(전부 문항 품질을 위한 것):
          1) `full.count(digits) == 1` — 본문에 **딱 한 번만** 나오는 숫자만 고른다.
             여러 번 나오면 `replace(raw, new_raw, 1)` 로 첫 번째만 바꿨을 때
             자료 안에 원래 값이 남아 자기모순 자료가 된다(채점 불가).
          2) 달력연도 제외(기본) — '2024년'을 '3440년'으로 바꾸면 자료가 명백히
             비현실적이 되어, 모델이 자료를 불신하고 기억으로 돌아갈 명분을 준다.
             --allow-year-mutation 으로 열 수 있다.
          3) `norm(new_raw) in norm(full)` — 변조값이 우연히 본문 다른 곳에 이미
             있으면 스킵. 그렇지 않으면 must_not_include 판정이 오염된다.

        ★ `cand.sort(key=lambda rd: -int(rd[1]))` : 큰 값부터 시도한다.
          그 조문의 핵심 기준일 확률이 높기 때문이다(pick_salient_number 와 같은 철학).

        ★ 주의 — GEN_CFG 전역을 읽는다:
          이 함수는 순수 함수가 아니다. build() 가 CLI 값으로 GEN_CFG 를 먼저
          갱신해야 의도대로 동작한다. 단위 테스트 시 GEN_CFG 를 직접 세팅할 것.
    """
    full = row["full_text"] or ""
    law, art = row["law_name"] or "", str(row["article_number"] or "")
    disp = format_article_display(art)
    # 본문에 딱 한 번만 나오는 '값' 숫자 중에서(구조 참조 제외), 값이 큰 것부터 변조 시도
    cand = [(raw, digits) for raw, digits in value_number_tokens(full)
            if full.count(digits) == 1]
    # 달력연도(2024→3440 같은 비현실적 변조 방지): 기본은 연도 토큰을 후보에서 뺀다
    if not GEN_CFG["allow_year_mutation"]:
        cand = [(raw, digits) for raw, digits in cand
                if not (raw.strip().endswith("년") and digits.isdigit()
                        and 1900 <= int(digits) <= 2099)]
    cand.sort(key=lambda rd: -int(rd[1]))
    for raw, digits in cand:
        new_digits = mutate_number(digits)
        if new_digits == digits:
            continue
        new_raw = raw.replace(digits, new_digits, 1)
        mutated = full.replace(raw, new_raw, 1)
        # 바뀐 값이 우연히 원래 본문에 이미 있었다면 스킵
        if norm(new_raw) in norm(full):
            continue
        q = (f"제공된 자료에 따르면, 「{law}」 {disp}의 해당 기준 수치는 정확히 얼마인가요? "
             f"제공된 자료의 표현을 그대로 사용해 답하세요.")
        return FaithItem(
            id=f"F{idx:03d}_counterfactual", type="counterfactual", issue=issue,
            question=q, context=mutated,   # ★ 변조된 자료를 준다
            expected_behavior=f"자료에 적힌 '{new_raw.strip()}'를 그대로 답한다(원래 값 '{raw.strip()}'을 말하면 안 됨).",
            must_include=[new_raw.strip()],     # 변조값을 말해야 통과
            must_not_include=[raw.strip()],     # 원래 값을 말하면 실패
            expect_abstention=False,
            source_doc_id=row["id"], source_law=law, source_article=art,
            mutation_from=raw.strip(), mutation_to=new_raw.strip(),
        )
    return None


# =============================================================================
# 생성 옵션(런타임 설정) — build()에서 CLI 값으로 채운다.
#   negation_mode : parenthetical(가장 깨끗, 괄호형만) | strict(파편·표 가드) | loose(옛 동작)
#   skip_tables   : 표/수식/이미지 블록 안의 '제외' 무시(대상 추출이 깨지므로)
#   allow_year_mutation : counterfactual 에서 달력연도(2024→3440 등) 변조 허용 여부
#
# Why 모듈 전역 dict 인가(설계 트레이드오프):
#   make_negation / make_counterfactual 은 build() → take_* → maker(idx, issue, row)
#   형태로 **고정된 시그니처**를 통해 호출된다. 옵션을 인자로 넘기려면 그 시그니처와
#   중간 배분 함수들을 전부 바꿔야 해서, 전역 설정으로 우회했다.
#   대가: 함수가 순수하지 않고, 테스트 시 전역을 세팅해야 한다. 규모가 더 커지면
#   dataclass 설정 객체를 만들어 maker 에 partial 로 묶는 편이 낫다.
# =============================================================================
GEN_CFG: dict[str, Any] = {
    "negation_mode": "parenthetical",
    "skip_tables": True,
    "min_subject_len": 2,
    "max_subject_len": 25,
    "allow_year_mutation": False,
}

# 표/수식/이미지 블록 표지(이 안의 '제외'는 대상 추출이 깨진다)
# 크롤링·PDF 파싱 산물에는 이런 잔재가 남는데, 그 주변 텍스트는 문장 구조가
# 무너져 있어 '무엇을 제외하는지'를 신뢰성 있게 뽑을 수 없다.
_TABLE_CHARS = set("│┌┐└┘├┤┬┴┼─━┃╌╍┄┅")
_TABLE_TAGS = ("<img", "</img", "<table", "flDownload")


def _in_table_region(full: str, pos: int, radius: int = 200) -> bool:
    """해당 위치가 표/수식/이미지 블록 근처인지 판정한다.

    Args:
        full: 전체 본문.
        pos: 검사할 위치(배제 표지의 인덱스).
        radius: 앞뒤로 살펴볼 문자 수.

    Returns:
        표 영역으로 보이면 True.

    Note:
        정확한 구조 파싱 대신 '주변 ±200자에 표 문자/태그가 있는가'라는
        휴리스틱을 쓴다(Why): 표의 정확한 경계를 알아내려면 파서가 필요한데,
        우리에게 필요한 건 "여기서 대상을 뽑으면 위험한가"라는 **위험 신호**뿐이다.
        보수적으로 넓게 잡아 의심스러우면 문항을 포기하는 편이 이득이다.
    """
    window = full[max(0, pos - radius): pos + radius]
    if any(c in _TABLE_CHARS for c in window):
        return True
    return any(t in window for t in _TABLE_TAGS)


# 배제 대상(제외 subject) 파싱 도우미 ------------------------------------------
#  '제외한다' 에 결합된 명사구('X은/는/을/를 제외한다')를 뽑는다.
#  - 문자 클래스에 마침표/쉼표/줄바꿈/인용부호를 넣지 않아 그 기호들이 '왼쪽 경계'가 된다.
#    → 앞 문장/단서 접두/인용부호 잔재를 자동으로 잘라 파편을 줄인다.
#  - 괄호형 '(X은 제외한다)' 이면 X 가 파여 나온 '상위 범주'까지 함께 뽑는다.
#
#  정규식 해설:
#    ([가-힣A-Za-z0-9ㆍ·() ]+?)  : 대상 명사구(최소 매칭)
#    ([은는을를])               : 주격/목적격 조사 — 이게 있어야 '무엇을' 제외하는지 확신 가능
#    ' *$'                      : 표지 직전까지(공백 허용)
#  최소 매칭(+?)과 '$' 앵커의 조합으로 '표지 바로 앞의 가장 짧은 명사구'를 잡는다.
_JOSA_RE = re.compile(r"([가-힣A-Za-z0-9ㆍ·() ]+?)([은는을를]) *$")

# 선두에 오면 '깨끗한 배제 대상'이 아닌 대명사/접속/어미류(파편 신호)
_GENERIC_SUBJ_PREFIX = (
    "다음", "해당", "각 ", "위 ", "같은", "그러", "이와", "그 ", "이 ", "저 ",
    "및 ", "또는", "하여", "에서", "으로", "와 ", "과 ", "의 ", "에 ", "도 ",
    "는 ", "은 ", "을 ", "를 ", "이나", "거나",
)
# 조건절/상호참조로 끝나 '깨끗한 범주'가 아닌 대상(엄격/괄호형 모드에서 제외)
_MECHANICAL_SUFFIX = ("경우", "사유", "때", "부분")
_CROSSREF_HEAD_RE = re.compile(r"^(제?\s?\d+\s?[조항호목]|[가나다라마바사아자차카타파하]목)")
_JOSA_HEAD_RE = re.compile(r"^[은는을를이가와과의에도및]")


def _looks_like_fragment(text: str) -> bool:
    """선두가 조사/어미/접속이거나 인용부호 잔재를 물고 있으면 파편으로 본다.

    Args:
        text: 후보 명사구.

    Returns:
        파편(= 문항 재료로 부적합)이면 True.

    Note:
        Why 파편을 걸러야 하는가:
            "…에 따라 계산한 금액은 제외한다" 에서 '따라 계산한 금액' 같은 조각이
            잡히면, 질문이 "'따라 계산한 금액'은 포함되나요?" 가 되어 사람이 봐도
            무슨 말인지 모르는 문항이 된다. 그런 문항으로 매긴 점수는 모델의
            충실성이 아니라 **문항의 조악함**을 측정한 것이다.

        인용부호(「」"…)를 파편 신호로 보는 이유: 법령명 인용의 잔재가 붙어 있다는
        뜻이라 명사구 경계가 잘못 잡힌 상태다.
    """
    if not text:
        return True
    if text.startswith(_GENERIC_SUBJ_PREFIX):
        return True
    if any(ch in text for ch in "「」\"”“"):
        return True
    if _JOSA_HEAD_RE.match(text):
        return True
    return False


def _extract_excluded_subject(full: str, mpos: int) -> tuple[str | None, str | None]:
    """배제 표지 바로 앞의 '배제 대상 명사구'를 뽑는다.

    Args:
        full: 전체 본문.
        mpos: 배제 표지('제외한다' 등)가 시작하는 인덱스.

    Returns:
        (배제 대상, 상위 범주). 괄호형 '(X은 제외한다)' 이면 상위 범주가 함께
        채워지고, 그렇지 않으면 상위 범주는 None. 신뢰할 만하게 못 뽑으면
        (None, None).

    Note:
        ★ 반환 순서 주의: (subject, governing) 이다. 호출부도
          `cand, gov = _extract_excluded_subject(...)` 로 받는다.

        동작 원리(Why 이 순서인가):
          1) 표지 앞부분(before)에서 `_JOSA_RE` 로 'X은/는/을/를' 패턴을 찾는다.
             조사가 없으면 **아예 포기한다.** 조사는 "이것이 배제의 목적어"라는
             문법적 증거이고, 그게 없으면 어떤 추측도 신뢰할 수 없기 때문이다.
          2) 괄호가 있으면 그 앞을 '상위 범주', 뒤를 '대상'으로 분리한다.
             법령의 전형적 표현 "주택(부수토지는 제외한다)" 에서
             상위=주택, 대상=부수토지 를 얻는다. 이 쌍이 있으면
             "주택에 부수토지가 포함되나요?" 라는 **가장 자연스럽고 명확한 질문**이
             만들어진다. parenthetical 모드가 이것만 채택하는 이유다.
          3) 단서 접두('다만,')와 리스트 표지('1.', '가)')를 제거해 노이즈를 턴다.

        주의: 문자 클래스에 마침표/쉼표/개행을 넣지 않은 것이 핵심 장치다.
        그 기호들이 자연스러운 왼쪽 경계가 되어, 앞 문장을 통째로 끌어오는 사고를
        막는다. 클래스에 문자를 추가할 때는 이 성질이 깨지지 않는지 반드시 확인할 것.
    """
    before = full[:mpos].rstrip()
    m = _JOSA_RE.search(before)
    if not m:
        return None, None                       # 'X은/는 제외' 꼴이 아니면 신뢰 불가 → 포기
    subj = m.group(1).strip()
    governing = None
    p = subj.rfind("(")                          # 괄호형이면 상위범주/대상 분리
    if p >= 0:
        governing = subj[:p].strip(" ㆍ·,")
        subj = subj[p + 1:].strip()
    subj = re.sub(r"^(다만|단)\s*,?\s*", "", subj)   # 단서 접두 제거
    subj = re.sub(r"^[가-힣0-9]{1,2}[.)]\s*", "", subj)  # 리스트 표지 제거
    subj = subj.strip(" ()ㆍ·,")
    return (subj or None), (governing or None)


def make_negation(idx: int, issue: str, row: dict) -> FaithItem | None:
    """배제 규정에서 '실제 배제 대상'을 뽑아, 그 대상이 포함되는지 묻는다.

    Args:
        idx: 문항 연번.
        issue: 쟁점 키.
        row: DB 행.

    Returns:
        FaithItem. 배제 대상을 깨끗하게 특정하지 못하면 None(문항 생성 포기).

    Note:
        '제외한다/포함하지 아니한다' 같은 배제 규정에서 '실제 배제 대상'을 정확히
        뽑아, 그 대상이 포함(적용)되는지 묻는다. 충실한 LLM은 '아니오/제외된다'
        라고 해야 한다.

        ★ 엄격(strict) 정책: 배제 대상을 깨끗하게 특정할 수 있는 경우에만 문항을
          만든다. 애매하면(파편/문장 통째/대명사류/조문 핵심주제 오인 등) None 을
          돌려 '건너뛴다'.
          → 예전처럼 '포함되는 항목'을 배제 대상으로 오인해 불량 문항을 찍는 일을 막는다.

        ★ 왜 이렇게까지 방어적인가(이 함수의 역사):
          초기 버전은 '제외한다' 앞 텍스트를 대충 잘라 썼고, 그 결과
            - 문장 조각이 주어가 되거나,
            - 조문의 **핵심 주제**(제목에 있는 단어)를 배제 대상으로 오인하거나
              (예: '주택' 조문에서 '주택은 제외' 로 잘못 읽음 — 이른바 F037형),
            - 표 안의 깨진 텍스트를 끌어오는
          불량 문항이 다수 생성됐다. 불량 문항은 모델을 억울하게 떨어뜨리고,
          그 점수를 믿고 모델을 폐기하면 **판단 자체가 틀린 것**이 된다.
          그래서 "수량을 잃더라도 불량 문항은 만들지 않는다"로 정책을 뒤집었다.

        ★ 가드 순서(위에서부터 값싼 검사 → 비싼 검사):
          1) 표 영역 스킵
          2) 조사 패턴으로 대상 추출(실패 시 포기)
          3) 길이 범위(파편·문장 통째 차단)
          4) 숫자/기호만인 대상 차단
          5) 파편 신호(조사·인용부호 선두) 차단
          6) **조문 제목과 겹치는 대상 차단** ← F037형 방지의 핵심
          7) 조건절('~경우')·상호참조('제N호') 차단
          8) parenthetical 모드면 '상위 범주가 깨끗한 괄호형'만 채택
          9) 괄호형이 아닌데 본문에 대상이 여러 번 나오면 차단(역할이 애매)

        ★ 모드별 강도:
          parenthetical(기본) : 8번까지 통과한 괄호형만. 문항 수는 적지만 품질 최상.
          strict              : 괄호형이 아니어도 통과 가능.
          loose               : 5·7번 가드를 끈 옛 동작(호환용).

        ★ 주의 — must_include 가 `mk` 에 의존한다:
          루프 변수 mk 는 '마지막으로 검사한 표지'다. break 로 빠져나오면 채택된
          표지가 맞지만, 표현이 '제외'가 아니면(예: '포함하지 아니한다')
          must_include 가 ['아니'] 가 된다. 평가기(evaluate_faithfulness)는
          결론 극성 판정을 우선하므로 실제 채점 영향은 제한적이다.
    """
    full = row["full_text"] or ""
    law, art = row["law_name"] or "", str(row["article_number"] or "")
    disp = format_article_display(art)
    title = row.get("article_title") or ""

    # 모든 배제 표지의 '모든' 등장 위치를 앞에서부터 훑어, 처음으로 '깨끗한' 대상을 쓴다.
    # Why 전수 수집인가: 첫 번째 '제외한다'가 표 안이거나 파편일 수 있다. 한 번
    # 실패했다고 조문을 통째로 버리면 아까우므로, 같은 조문 안에서 다음 후보를 본다.
    positions = []
    for mk in EXCLUSION_MARKERS:
        start = 0
        while True:
            i = full.find(mk, start)
            if i < 0:
                break
            positions.append((i, mk))
            start = i + len(mk)
    positions.sort()   # 본문 등장 순서대로(앞쪽이 대체로 원칙 규정이라 품질이 좋다)

    mode = GEN_CFG["negation_mode"]
    lo, hi = GEN_CFG["min_subject_len"], GEN_CFG["max_subject_len"]

    subj = governing = mk = None
    for i, mk in positions:
        if GEN_CFG["skip_tables"] and _in_table_region(full, i):
            continue                                  # 표/수식/이미지 블록 → 스킵
        cand, gov = _extract_excluded_subject(full, i)
        if not cand:
            continue
        # ── 엄격 가드: 불량 문항 방지 ────────────────────────────────────
        if not (lo <= len(cand) <= hi):
            continue                                  # 파편(잘림)·문장 통째 → 스킵
        if re.fullmatch(r"[\d,.\s]+", cand):
            continue                                  # 숫자/기호만 → 스킵
        if mode != "loose" and _looks_like_fragment(cand):
            continue                                  # 조사/어미 선두·인용부호 잔재 → 스킵
        head = re.split(r"[ ㆍ·]", cand)[-1]
        if head and head in title:
            # 조문 제목에 있는 단어 = 그 조문의 '주제'다. 주제가 스스로에게서
            # 제외되는 일은 없으므로, 이건 파싱 오류일 확률이 매우 높다.
            continue                                  # 조문 핵심주제 오인(F037형) → 스킵
        if mode != "loose" and (cand.endswith(_MECHANICAL_SUFFIX)
                                or _CROSSREF_HEAD_RE.match(cand)):
            continue                                  # 조건절('~경우')·상호참조('제N호') → 스킵
        # 괄호형 우선 모드: 상위범주가 깨끗한 '(X은 제외한다)'만 채택
        if mode == "parenthetical":
            if not (gov and lo <= len(gov) <= 40 and not _looks_like_fragment(gov)):
                continue
        if gov is None and full.count(cand) > 1:
            continue                                  # 본문서 여러 역할 → 애매 → 스킵
        subj, governing = cand, gov
        break
    if not subj:
        return None                                   # 깨끗한 배제 대상 없음 → 문항 생성 포기

    # 상위 범주가 있으면 "A에 B가 포함되나?" 라는 가장 명확한 질문을 만든다.
    if governing and lo <= len(governing) <= 40 and not _looks_like_fragment(governing):
        q = (f"제공된 자료(「{law}」 {disp})에 따르면, '{governing}'에 "
             f"'{subj}'이(가) 포함(적용)되나요? 자료에 근거해 '예/아니오'와 이유를 답하세요.")
        eb = f"자료에서 '{subj}'은(는) '{governing}'에서 제외되므로 '아니오/제외' 취지로 답한다."
    else:
        q = (f"제공된 자료(「{law}」 {disp})에 따르면, '{subj}'은(는) 포함(적용) 대상인가요? "
             f"자료에 근거해 '예/아니오'와 이유를 답하세요.")
        eb = f"'{subj}'은(는) 제외/적용되지 않는다는 취지로 답한다."

    return FaithItem(
        id=f"F{idx:03d}_negation", type="negation", issue=issue,
        question=q, context=full,
        expected_behavior=eb,
        must_include=["제외"] if "제외" in mk else ["아니"],
        # 긍정 단정형만 금지어로 둔다. '포함' 단독을 넣으면 '포함되지 않습니다'가
        # 오탐되므로 반드시 종결형이어야 한다.
        must_not_include=["포함됩니다", "해당됩니다", "적용됩니다"],
        expect_abstention=False,
        source_doc_id=row["id"], source_law=law, source_article=art,
    )


def make_exception(idx: int, issue: str, row: dict) -> FaithItem | None:
    """'다만 ...' 단서(예외)를 찾아, 그 예외 내용을 묻는다.

    Args:
        idx: 문항 연번.
        issue: 쟁점 키.
        row: DB 행.

    Returns:
        FaithItem. 쓸 만한 단서 문장을 못 찾으면 None.

    Note:
        충실한 LLM은 예외를 빠뜨리지 않고 말해야 한다.

        Why 세법에서 예외 누락이 치명적인가:
            "원칙은 A다"만 말하고 "다만 B인 경우는 아니다"를 빠뜨린 답은,
            문장 하나하나는 사실이지만 **실무적으로는 완전한 오답**이다.
            사용자는 자기 사례가 예외에 해당하는지 알 길이 없어진다.
            즉 이 유형은 '거짓말'이 아니라 '누락에 의한 오도'를 잡는다.

        채점 토큰(key) 선택:
            단서 문장에서 숫자를 최우선으로 뽑고(가장 명확), 없으면 명사,
            그것도 없으면 단서 시작 부분 6자를 쓴다. 마지막 폴백은 조악하지만
            빈 must_include(=무조건 통과)만은 피하려는 것이다.

        주의: `len(clause) < 6` 인 단서는 버린다. '다만,' 뒤가 사실상 비어 있으면
        물어볼 내용이 없기 때문이다.
    """
    full = row["full_text"] or ""
    law, art = row["law_name"] or "", str(row["article_number"] or "")
    disp = format_article_display(art)
    for mk in EXCEPTION_MARKERS:
        clause = clause_after(full, mk)
        if not clause or len(clause) < 6:
            continue
        # 예외 문장에서 채점용 핵심 토큰(숫자 우선, 없으면 명사) 하나 뽑기
        nums = find_number_tokens(clause)
        if nums:
            key = nums[0][0].strip()
        else:
            noun = extract_salient_noun(clause.replace(mk, ""))
            key = noun[0] if noun else clause[len(mk):len(mk) + 6].strip()
        q = (f"제공된 자료(「{law}」 {disp})에는 원칙 외에 '다만' 이하의 예외(단서) 규정이 있습니다. "
             f"그 예외의 내용은 무엇인가요? 자료에 근거해 답하세요.")
        return FaithItem(
            id=f"F{idx:03d}_exception", type="exception", issue=issue,
            question=q, context=full,
            expected_behavior=f"단서(예외) 내용을 언급하며 핵심 표지 '{key}'가 답에 들어가야 한다.",
            must_include=[key], must_not_include=[], expect_abstention=False,
            source_doc_id=row["id"], source_law=law, source_article=art,
        )
    return None


def make_unanswerable(idx: int, issue: str, row: dict, foreign_row: dict) -> FaithItem | None:
    """자료에 '없는' 주제를 물어 기권 능력을 시험한다.

    Args:
        idx: 문항 연번.
        issue: 쟁점 키.
        row: 자료로 제공할 DB 행.
        foreign_row: **다른** 조문. 여기서 주제어를 빌려 온다.

    Returns:
        FaithItem. 자료에 없는 주제어를 못 찾으면 None.

    Returns 설명:
        자료(row) 는 주되, 그 자료에 '없는' 주제(foreign_row 에서 가져온)를 묻는다.
        충실한 LLM은 '자료에 없다'고 해야 한다. 지어내면 할루시네이션.

    Note:
        ★ 왜 완전히 지어낸 단어가 아니라 '다른 조문의 주제어'를 쓰는가(Why):
          '아무말대잔치' 같은 무의미어를 물으면 모델이 질문의 이상함만 보고 거부해,
          진짜 실력을 못 잰다. 반면 **세법에 실제로 존재하지만 이 자료에는 없는**
          주제를 물으면, 모델은 자기 배경지식으로 답하고 싶은 강한 유혹을 받는다.
          그 유혹을 이기고 "제공된 자료에는 없습니다"라고 말하는지가 핵심이다.
          이것이 RAG 에서 가장 현실적인 할루시네이션 시나리오이기도 하다.

        ★ 검증 조건 `c not in full and c not in law`:
          주제어가 자료 본문이나 법령명에 이미 있으면 '답할 수 있는 질문'이 되어
          문항이 성립하지 않는다. 후보를 5개 뽑아 첫 번째로 조건을 만족하는 것을
          쓴다.

        ★ must_include 가 비고 expect_abstention=True 인 이유:
          기권 문구의 표현은 무한히 다양해서 토큰 매칭으로 잡을 수 없다.
          평가기가 ABSTENTION_MARKERS 목록으로 '기권 여부'를 판정하고,
          덤으로 '자료에 없는 숫자를 지어냈는지'까지 함께 본다.
    """
    full = row["full_text"] or ""
    law, art = row["law_name"] or "", str(row["article_number"] or "")
    disp = format_article_display(art)
    # foreign 주제어가 현재 자료에 실제로 '없는지' 확인
    cands = extract_salient_noun(foreign_row.get("article_title") or foreign_row.get("full_text") or "", k=5)
    topic = None
    for c in cands:
        if c and c not in full and c not in law:
            topic = c
            break
    if not topic:
        return None
    q = (f"제공된 자료(「{law}」 {disp})에 따르면, '{topic}'에 대한 기준(또는 규정)은 무엇인가요? "
         f"자료에 근거해 답하세요.")
    return FaithItem(
        id=f"F{idx:03d}_unanswerable", type="unanswerable", issue=issue,
        question=q, context=full,
        expected_behavior=f"'{topic}'은(는) 제공된 자료에 없으므로 '확인할 수 없다'고 답한다.",
        must_include=[], must_not_include=[], expect_abstention=True,
        source_doc_id=row["id"], source_law=law, source_article=art,
    )


def make_distractor(idx: int, issue: str, row: dict, noise_row: dict) -> FaithItem | None:
    """관련 자료와 무관한 자료를 섞어 주고, 관련 자료의 숫자를 묻는다.

    Args:
        idx: 문항 연번.
        issue: 쟁점 키.
        row: 정답 근거가 되는 자료.
        noise_row: 방해용(무관한) 자료.

    Returns:
        FaithItem. 정답 자료에 값 숫자가 없으면 None.

    Note:
        엉뚱한 자료의 숫자를 끌어오면 실패.

        Why 이 유형이 현실적인가:
            실제 RAG 는 top-k 로 여러 문서를 한꺼번에 준다. 그중 정답 문서는
            보통 하나뿐이고 나머지는 노이즈다. 모델이 근거를 구분하지 못하면
            "A법의 기준"을 묻는데 "B법의 숫자"로 답하는 사고가 난다.
            이 유형은 그 상황을 통제된 형태(자료 2개)로 축소해 재현한 것이다.

        ★ 금지어(noise_forbid) 선정의 조건 `norm(raw) not in norm(full)`:
          방해 자료의 숫자가 정답 자료에도 있으면, 모델이 그 숫자를 말했을 때
          '방해에 낚였다'인지 '정답을 말했다'인지 구분할 수 없다. 그래서
          **정답 자료에 없는** 숫자만 금지어로 삼는다. 없으면 금지어 없이 진행한다
          (그때는 정답 숫자 포함 여부만 본다).

        ★ context 조립 형식('[자료 1]', '[자료 2]')이 중요하다:
          질문이 "「법령」 조문에 근거하여"라고 지정하므로, 모델은 두 자료 중
          어느 쪽이 지정된 것인지 스스로 판별해야 한다. 구획 표시가 없으면
          자료 경계 자체를 못 읽어 실패 원인이 흐려진다.
    """
    full = row["full_text"] or ""
    law, art = row["law_name"] or "", str(row["article_number"] or "")
    disp = format_article_display(art)
    nums_main = value_number_tokens(full)       # 값 숫자만(구조 참조 제외)
    if not nums_main:
        return None
    nums_noise = value_number_tokens(noise_row["full_text"] or "")
    main_raw = (pick_salient_number(nums_main) or nums_main[0])[0].strip()
    # 방해용 자료에서 '본문에 없는' 숫자를 골라 금지어로 (겹치면 채점이 애매)
    noise_forbid = None
    for raw, _ in nums_noise:
        if norm(raw) not in norm(full):
            noise_forbid = raw.strip()
            break
    context = (f"[자료 1]\n{full}\n\n"
               f"[자료 2]\n{noise_row['full_text']}")
    q = (f"위 자료들 중 「{law}」 {disp}에 근거하여, 해당 조문의 '{main_raw}' 관련 기준을 답하세요. "
         f"관련 없는 자료의 내용을 섞지 마세요.")
    return FaithItem(
        id=f"F{idx:03d}_distractor", type="distractor", issue=issue,
        question=q, context=context,
        expected_behavior=f"「{law}」 {disp}의 '{main_raw}'만 근거로 답하고, 다른 자료 숫자는 쓰지 않는다.",
        must_include=[main_raw],
        must_not_include=[noise_forbid] if noise_forbid else [],
        expect_abstention=False,
        source_doc_id=row["id"], source_law=law, source_article=art,
    )


# =============================================================================
# 생성 오케스트레이션
# =============================================================================
def build(n: dict, issues: list[str], seed: int,
          gen_opts: dict | None = None) -> tuple[list[FaithItem], dict]:
    """유형별 목표 개수만큼 문항을 생성하고 통계를 함께 돌려준다.

    Args:
        n: 유형별 목표 개수 dict. 키는 grounded/unanswerable/counterfactual/
            negation/exception/distractor.
        issues: 사용할 쟁점 키 목록.
        seed: 난수 시드(풀 셔플에 사용. 같은 시드 = 같은 시험지).
        gen_opts: CLI 에서 온 생성 옵션. negation_mode/skip_tables/
            min_subject_len/max_subject_len/allow_year_mutation 은 GEN_CFG 로
            반영되고, laws/pool_limit/base_pool_limit 은 여기서 직접 쓴다.

    Returns:
        (문항 목록, 통계 dict). 통계에는 요청 수/생성 수/풀 크기가 담긴다.

    Raises:
        SystemExit: DB 접속 실패(get_conn 경유).

    Note:
        ── 3개의 후보 풀을 따로 만드는 이유(Why) ───────────────────────────
        유형마다 필요한 재료가 다르므로 SQL 을 세 번 던진다.
          base_pool : 숫자가 있는 일반 조문 (grounded/counterfactual/
                      unanswerable/distractor 가 공유)
          neg_pool  : 본문에 배제 표지가 있는 조문
          exc_pool  : 본문에 '다만' 이 있는 조문
        한 번에 다 가져와 파이썬에서 나누면, 배제 표지가 있는 조문이 우연히
        하나도 안 뽑히는 사태가 생긴다(길이순 정렬 때문에). SQL 로 조건을 걸어야
        각 유형의 **재료 수급이 보장**된다.

        ── used 집합 분리 전략 ────────────────────────────────────────────
        base_used : 숫자 기반 4유형(grounded/counterfactual/unanswerable/
                    distractor)이 '공유' → 같은 조문이 이 4유형 중 둘로
                    중복되는 것만 방지(질문이 비슷해지는 걸 막음).
        negation / exception : 각자 '독립' 풀 + 독립 used
                    → base 유형이 조문을 다 써도 굶지 않는다(중요 유형 보호).

        Why 이런 비대칭인가: negation/exception 은 재료 조건이 까다로워 생성
        수율이 낮다. 이들이 base 유형과 used 집합을 공유하면, 앞서 배정된
        base 유형이 좋은 조문을 선점해 negation 이 0개가 될 수 있다. 가장
        만들기 어려운 유형을 보호하는 것이 전체 시험지의 균형에 유리하다.

        ── 배정 순서 ──────────────────────────────────────────────────────
        숫자 단독(grounded→counterfactual) → 보조행 필요(unanswerable→distractor)
        → 독립 마커 유형(negation→exception).
        보조행이 필요한 유형을 뒤에 두는 이유는, 그들이 풀 전체를 훑을 수 있어
        선택지가 넓기 때문이다(먼저 배정하면 다른 유형의 선택지를 빼앗는다).

        ── 주의 ───────────────────────────────────────────────────────────
        * GEN_CFG 는 모듈 전역이다. 이 함수가 호출되면 프로세스 전체 설정이 바뀐다.
        * `finally: conn.close()` 로 커넥션을 반드시 닫는다.
        * 반환 통계의 pool_sizes 는 '가져온 후보 수'이지 '쓸 수 있는 수'가 아니다.
          negation 은 가드 때문에 후보 200개 중 몇 개만 통과하는 일이 흔하다.
    """
    gen_opts = gen_opts or {}
    # 런타임 생성 설정 반영(negation 모드/표 스킵/길이/연도변조)
    for k in ("negation_mode", "skip_tables", "min_subject_len",
              "max_subject_len", "allow_year_mutation"):
        if k in gen_opts and gen_opts[k] is not None:
            GEN_CFG[k] = gen_opts[k]
    laws = gen_opts.get("laws") or None
    pool_limit = gen_opts.get("pool_limit") or 200
    base_pool_limit = gen_opts.get("base_pool_limit") or 300

    random.seed(seed)
    conn = get_conn()
    items: list[FaithItem] = []
    idx = 1
    produced: dict = {}

    try:
        # --- 유형별로 필요한 후보 풀을 DB에서 확보 ---
        kw_all = [k for iss in issues for k in ISSUE_KEYWORDS.get(iss, [])]

        # 숫자 있는 일반 조문 풀 (grounded / counterfactual / distractor / unanswerable 기반)
        # SQL 로는 '숫자 포함'을 걸 수 없어(정규식 조건은 인덱스를 못 탄다)
        # 파이썬에서 한 번 더 거른다.
        base_pool = fetch_articles(conn, kw_all, None, limit=base_pool_limit, laws=laws)
        base_pool = [r for r in base_pool if find_number_tokens(r["full_text"] or "")]
        random.shuffle(base_pool)

        # 부정/배제 조문 풀 (negation)
        neg_pool = fetch_articles(conn, kw_all, EXCLUSION_MARKERS, limit=pool_limit, laws=laws)
        random.shuffle(neg_pool)

        # 단서(예외) 조문 풀 (exception)
        exc_pool = fetch_articles(conn, kw_all, EXCEPTION_MARKERS, limit=pool_limit, laws=laws)
        random.shuffle(exc_pool)

        def issue_of(row: dict) -> str:
            """조문 본문/제목에 등장하는 키워드로 쟁점 라벨을 역추정한다.

            Args:
                row: DB 행.

            Returns:
                가장 먼저 매칭된 쟁점 키. 아무것도 안 걸리면 첫 쟁점(또는 '일반').

            Note:
                풀을 쟁점별로 나누지 않고 한꺼번에 조회했기 때문에, 라벨은 사후에
                붙인다. issues 순서상 앞선 쟁점이 우선하므로 라벨이 겹치는 조문은
                앞쪽 쟁점으로 분류된다(분석용 근사값일 뿐 채점에는 영향 없음).
            """
            blob = (row.get("article_title") or "") + (row.get("full_text") or "")
            for iss in issues:
                if any(k in blob for k in ISSUE_KEYWORDS.get(iss, [])):
                    return iss
            return issues[0] if issues else "일반"

        # ── used 집합 분리 전략 ────────────────────────────────────────────
        # base_used : 숫자 기반 4유형(grounded/counterfactual/unanswerable/
        #             distractor)이 '공유' → 같은 조문이 이 4유형 중 둘로
        #             중복되는 것만 방지(질문이 비슷해지는 걸 막음).
        # negation / exception : 각자 '독립' 풀 + 독립 used
        #             → base 유형이 조문을 다 써도 굶지 않는다(중요 유형 보호).
        base_used: set[str] = set()

        def take_base(maker: Callable[[int, str, dict], FaithItem | None], count: int) -> int:
            """숫자만 필요한 유형(grounded/counterfactual) 배정.

            Args:
                maker: (idx, issue, row) -> FaithItem | None 형태의 생성 함수.
                count: 목표 개수.

            Returns:
                실제로 만든 개수(목표에 못 미칠 수 있다).

            Note:
                maker 가 None 을 주면 그 조문은 그냥 넘어간다. base_used 에도
                넣지 않으므로 다른 유형이 그 조문을 다시 시도할 수 있다.
                `nonlocal idx` 로 문항 번호를 전 유형이 공유해 F001~ 이 연속된다.
            """
            nonlocal idx
            made = 0
            for r in base_pool:
                if made >= count:
                    break
                if r["id"] in base_used:
                    continue
                it = maker(idx, issue_of(r), r)
                if it:
                    base_used.add(r["id"]); items.append(it); idx += 1; made += 1
            return made

        def take_paired(maker2: Callable[[int, str, dict, dict], FaithItem | None],
                        count: int) -> int:
            """보조 조문이 필요한 유형(unanswerable=무관주제, distractor=방해자료).

            Args:
                maker2: (idx, issue, row, other) -> FaithItem | None.
                count: 목표 개수.

            Returns:
                실제로 만든 개수.

            Note:
                짝(other)을 풀 전체에서 훑어 **첫 성공을 채택**한다(수율 ↑).
                unanswerable 은 '자료에 없는 주제어'를 가진 짝이 필요하고,
                distractor 는 '겹치지 않는 숫자'를 가진 짝이 필요해서, 첫 번째
                후보로는 실패하는 일이 잦기 때문이다.

                `primaries` 폴백(`or base_pool`): base_used 가 풀을 다 덮으면
                빈 목록이 되어 0개가 나온다. 그때는 사용된 조문이라도 재사용해
                문항 수를 확보한다(품질보다 수량을 택하는 유일한 지점).

                주의: 이중 루프라 최악의 경우 O(n²)다. base_pool 이 300 정도라
                실측상 문제없지만, 풀을 크게 키우면 체감될 수 있다.
            """
            nonlocal idx
            made = 0
            primaries = [r for r in base_pool if r["id"] not in base_used] or base_pool
            for r in primaries:
                if made >= count:
                    break
                # 짝이 될 조문을 풀 전체에서 찾아 첫 성공을 채택(수율 ↑)
                for other in base_pool:
                    if other["id"] == r["id"]:
                        continue
                    it = maker2(idx, issue_of(r), r, other)
                    if it:
                        base_used.add(r["id"]); items.append(it)
                        idx += 1; made += 1
                        break
            return made

        def take_marker(pool: list[dict],
                        maker: Callable[[int, str, dict], FaithItem | None],
                        count: int) -> int:
            """독립 풀 유형(negation/exception) 배정.

            Args:
                pool: 이 유형 전용 후보 풀.
                maker: 생성 함수.
                count: 목표 개수.

            Returns:
                실제로 만든 개수.

            Note:
                base_used 를 쓰지 않고 로컬 `seen` 만 쓴다. 이것이 '중요 유형 보호'
                전략의 구현이다(위 build() Note 참조). 같은 조문이 negation 과
                grounded 양쪽에 쓰일 수는 있지만, 질문의 성격이 완전히 달라
                중복으로 보지 않는다.
            """
            nonlocal idx
            made = 0
            seen: set[str] = set()
            for r in pool:
                if made >= count:
                    break
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                it = maker(idx, issue_of(r), r)
                if it:
                    items.append(it); idx += 1; made += 1
            return made

        # 배정 순서: 숫자 단독 → 보조행 필요 → 독립 마커 유형
        produced["grounded"] = take_base(make_grounded, n["grounded"])
        produced["counterfactual"] = take_base(make_counterfactual, n["counterfactual"])
        produced["unanswerable"] = take_paired(make_unanswerable, n["unanswerable"])
        produced["distractor"] = take_paired(make_distractor, n["distractor"])
        produced["negation"] = take_marker(neg_pool, make_negation, n["negation"])
        produced["exception"] = take_marker(exc_pool, make_exception, n["exception"])

    finally:
        conn.close()

    stats = {"requested": dict(n), "produced": produced,
             "pool_sizes": {"base": len(base_pool), "negation": len(neg_pool),
                            "exception": len(exc_pool)}}
    return items, stats


# =============================================================================
# 출력
#
# 데이터는 stdout/파일, 로그는 stderr 로 분리한다(리다이렉트 오염 방지).
# =============================================================================
def emit_json(items: list[FaithItem], path: str | None) -> None:
    """문항 목록을 JSON 으로 저장하거나 화면에 출력한다.

    Args:
        items: 문항 목록.
        path: 파일 경로. None 이면 stdout.

    Note:
        `ensure_ascii=False` 는 필수다. 한글이 \\uXXXX 로 저장되면 사람이 시험지를
        검수할 수 없다. 이 JSON 이 evaluate_faithfulness.py 의 입력이 된다.
    """
    data = [asdict(x) for x in items]
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[OK] {len(items)}문항 -> {path}", file=sys.stderr)
    else:
        print(text)


def emit_csv(items: list[FaithItem], path: str | None) -> None:
    """문항 목록을 CSV 로 저장하거나 화면에 출력한다.

    Args:
        items: 문항 목록.
        path: 파일 경로. None 이면 stdout.

    Note:
        ★ 리스트 필드는 '|' 로 join 한다(JSON 이 아니다).
          평가기의 load_testset 이 CSV 를 읽을 때 `split("|")` 로 되돌리므로,
          **토큰 안에 '|' 가 들어가면 깨진다.** 법령 텍스트에는 거의 없지만,
          형식의 한계로 알고 있어야 한다. 무손실이 필요하면 JSON 형식을 쓸 것.

        ★ utf-8-sig(BOM) : 한국어 Windows Excel 이 한글을 깨지 않게 하는 실무 필수 설정.
        ★ newline='' : Windows 에서 빈 줄이 끼는 문제를 막는 csv 모듈 표준 관용구.

        주의: path 가 None 이면 sys.stdout 을 쓰고 닫지 않는다(닫으면 이후 출력이
        전부 죽는다). 조건부 close 가 의도된 동작이다.
    """
    cols = ["id", "type", "issue", "question", "context", "expected_behavior",
            "must_include", "must_not_include", "expect_abstention",
            "source_doc_id", "source_law", "source_article",
            "mutation_from", "mutation_to"]
    f = open(path, "w", newline="", encoding="utf-8-sig") if path else sys.stdout
    w = csv.writer(f)
    w.writerow(cols)
    for it in items:
        d = asdict(it)
        row = []
        for c in cols:
            v = d[c]
            row.append("|".join(v) if isinstance(v, list) else v)
        w.writerow(row)
    if path:
        f.close()
        print(f"[OK] {len(items)}문항 -> {path}", file=sys.stderr)


def emit_stdout(items: list[FaithItem]) -> None:
    """문항을 사람이 검수하기 좋은 형태로 화면에 출력한다.

    Args:
        items: 문항 목록.

    Note:
        context 는 앞 120자만 보여 준다. 조문 전문을 다 찍으면 화면이 넘쳐
        정작 봐야 할 '질문과 기대 동작'의 대비가 묻히기 때문이다.

        `re.sub(chr(10), ' ', ...)` 에서 chr(10) 은 개행이다. 문자열 리터럴 안에
        이스케이프를 넣지 않으려는 표현 방식이며, 한 줄로 눌러 미리보기 정렬을
        유지하려는 것이다.
    """
    for it in items:
        print(f"\n=== {it.id} [{it.type}] ({it.issue}) ===")
        print(f"Q : {it.question}")
        print(f"기대: {it.expected_behavior}")
        if it.expect_abstention:
            print("채점: '자료에 없음' 취지가 나와야 통과")
        if it.must_include:
            print(f"must_include    : {it.must_include}")
        if it.must_not_include:
            print(f"must_not_include: {it.must_not_include}")
        print(f"context(앞 120자): {re.sub(chr(10),' ', it.context)[:120]} ...")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    """명령행 인자를 정의하고 파싱한다.

    Returns:
        파싱된 인자 네임스페이스.

    Note:
        기본 개수 배분의 의도(grounded 6 / unanswerable 10 / counterfactual 10 /
        negation 8 / exception 8 / distractor 4):
            가장 중요한 지표인 **할루시네이션(unanswerable)** 과 **자료 무시
            (counterfactual)** 에 표본을 가장 많이 배정했다. 통계적으로 의미 있는
            비율을 보려면 표본이 필요하기 때문이다.
            grounded 는 대조군이라 적어도 되고, distractor 는 재료 조건이 까다로워
            보수적으로 잡았다.

        `--negation-mode parenthetical` 이 기본인 이유:
            문항 수가 줄더라도 불량 문항을 만들지 않는 쪽이 안전하다. 개수가
            부족하면 --pool-limit 을 올리는 것이 정석이며, 모드를 loose 로
            내리는 것은 최후의 수단이다.

        `action="store_false"` 계열(--no-skip-tables) 은 dest 와 default 를
        함께 지정해야 의도대로 동작한다. 여기서는 default=True + store_false 로
        "옵션을 주면 끈다" 의미가 된다.

        도움말의 `%%` 이스케이프: argparse 는 help 문자열에 %-포매팅을 적용하므로
        리터럴 퍼센트는 %% 로 써야 한다(이 파일에는 해당 사례가 없지만
        evaluate_faithfulness.py 에는 있다).
    """
    p = argparse.ArgumentParser(
        description="충실성(faithfulness) 테스트셋 생성기 - 지어내기/부정문/예외를 정조준")
    p.add_argument("--grounded", type=int, default=6, help="자료에 답 있는 기본 문항 수")
    p.add_argument("--unanswerable", type=int, default=10, help="자료에 답 없는 문항 수(할루시 함정)")
    p.add_argument("--counterfactual", type=int, default=10, help="숫자 변조 문항 수(자료 우선 검증)")
    p.add_argument("--negation", type=int, default=8, help="부정/배제 문항 수")
    p.add_argument("--exception", type=int, default=8, help="예외(단서) 문항 수")
    p.add_argument("--distractor", type=int, default=4, help="방해 자료 혼입 문항 수")
    p.add_argument("--issues", nargs="*", default=list(ISSUE_KEYWORDS.keys()),
                   help="사용할 쟁점(기본: 전체)")
    p.add_argument("--laws", nargs="*", default=None,
                   help="법령 필터(부분일치). 예: --laws 소득세법 조세특례제한법")
    p.add_argument("--format", choices=["json", "csv", "stdout"], default="json")
    p.add_argument("--out", default="testset_faithfulness.json", help="출력 파일 경로")
    p.add_argument("--seed", type=int, default=42, help="난수 시드(재현성)")
    # --- negation 품질/수량 제어 ---
    p.add_argument("--negation-mode", dest="negation_mode",
                   choices=["parenthetical", "strict", "loose"], default="parenthetical",
                   help="negation 생성 강도: parenthetical(가장 깨끗) | strict | loose(옛 동작)")
    p.add_argument("--no-skip-tables", dest="skip_tables", action="store_false", default=True,
                   help="표/수식/이미지 블록 안의 '제외'도 대상 후보로 허용(기본은 건너뜀)")
    p.add_argument("--min-subject-len", dest="min_subject_len", type=int, default=2,
                   help="배제 대상 최소 길이(기본 2)")
    p.add_argument("--max-subject-len", dest="max_subject_len", type=int, default=25,
                   help="배제 대상 최대 길이(기본 25). 짧을수록 깔끔")
    # --- counterfactual/풀 제어 ---
    p.add_argument("--allow-year-mutation", dest="allow_year_mutation", action="store_true",
                   default=False, help="counterfactual 에서 달력연도(2024→…) 변조 허용(기본 차단)")
    p.add_argument("--pool-limit", dest="pool_limit", type=int, default=200,
                   help="negation/exception 후보 조문 수(기본 200). 개수 많이 뽑으려면 늘려라")
    p.add_argument("--base-pool-limit", dest="base_pool_limit", type=int, default=300,
                   help="grounded/counterfactual/unanswerable/distractor 후보 조문 수(기본 300)")
    p.add_argument("--report", action="store_true",
                   help="유형별 요청/생성 개수와 부족분을 표로 출력")
    return p.parse_args()


def _self_test() -> None:
    """DB 없이 순수 텍스트 유틸만 검증한다."""
    assert format_article_display("1의4") == "제1조의4"
    assert norm("6억 원") == "6억원"
    print("[self-test] build_faithfulness_testset OK")


def main() -> None:
    """CLI 진입점: 인자 검증 → 생성 → 경고/리포트 → 출력.

    Raises:
        SystemExit: 알 수 없는 쟁점이거나 생성 문항이 0개일 때.

    Note:
        ★ '부족분 경고'가 이 CLI 의 중요한 UX 다(Why):
          이 생성기는 품질 가드 때문에 요청보다 적게 만드는 것이 **정상 동작**이다.
          그런데 사용자는 "--negation 8 을 줬는데 왜 3개지?"에서 버그를 의심하게 된다.
          그래서 부족한 유형마다 이유(깨끗한 후보 부족)와 처방(--pool-limit↑ 또는
          모드 완화)을 즉시 알려 준다. 침묵하는 축소가 가장 나쁜 설계다.

        ★ 출력 경로 처리: `--format stdout` 이면 out 을 None 으로 바꿔 화면 출력으로
          보낸다. csv 분기의 `out if out else None` 은 이미 None 인 값을 한 번 더
          방어한 것이다(동작상 동일).
    """
    a = parse_args()
    for iss in a.issues:
        if iss not in ISSUE_KEYWORDS:
            die(f"알 수 없는 쟁점: {iss} (가능: {list(ISSUE_KEYWORDS)})")

    n = dict(grounded=a.grounded, unanswerable=a.unanswerable,
             counterfactual=a.counterfactual, negation=a.negation,
             exception=a.exception, distractor=a.distractor)

    gen_opts = dict(
        negation_mode=a.negation_mode, skip_tables=a.skip_tables,
        min_subject_len=a.min_subject_len, max_subject_len=a.max_subject_len,
        allow_year_mutation=a.allow_year_mutation, laws=a.laws,
        pool_limit=a.pool_limit, base_pool_limit=a.base_pool_limit,
    )
    items, stats = build(n, a.issues, a.seed, gen_opts)
    if not items:
        die("생성된 문항이 0개입니다. DB 연결/데이터를 확인하세요.")

    # 유형별 개수 요약(stderr)
    from collections import Counter
    c = Counter(it.type for it in items)
    print(f"[요약] 총 {len(items)}문항 " + ", ".join(f"{k}:{v}" for k, v in c.items()),
          file=sys.stderr)

    # 부족분 경고: 요청보다 적게 생성된 유형(특히 parenthetical negation)
    short = [(k, n[k], stats["produced"].get(k, 0))
             for k in n if stats["produced"].get(k, 0) < n[k]]
    for k, req, got in short:
        print(f"[주의] {k}: 요청 {req}개 중 {got}개만 생성됨(깨끗한 후보 부족). "
              f"더 필요하면 --pool-limit↑ 또는 --negation-mode strict 를 고려하세요.",
              file=sys.stderr)

    if a.report:
        print("\n[생성 리포트] 유형별 요청/생성", file=sys.stderr)
        print(f"  {'type':<15}{'요청':>6}{'생성':>6}", file=sys.stderr)
        for k in ("grounded", "counterfactual", "unanswerable",
                  "distractor", "negation", "exception"):
            print(f"  {k:<15}{n.get(k,0):>6}{stats['produced'].get(k,0):>6}",
                  file=sys.stderr)
        ps = stats["pool_sizes"]
        print(f"  후보 풀 크기: base={ps['base']} negation={ps['negation']} "
              f"exception={ps['exception']} (negation-mode={a.negation_mode})",
              file=sys.stderr)

    out = None if a.format == "stdout" else a.out
    if a.format == "json":
        emit_json(items, out)
    elif a.format == "csv":
        emit_csv(items, out if out else None)
    else:
        emit_stdout(items)


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
        sys.exit(0)
    main()
