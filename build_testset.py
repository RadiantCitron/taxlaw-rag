#!/usr/bin/env python
# -*- coding: utf-8 -*-
# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""세무 RAG 테스트셋(골든셋) 생성기.

================================================================================
0. 이 파일을 읽는 법 (학습 가이드)
================================================================================
이 모듈은 "RAG 시스템을 채점하기 위한 시험지(골든셋)"를 자동으로 찍어내는
공장이다. 코드를 위에서 아래로 읽으면 다음 순서로 이야기가 전개된다.

    (1) 상수 영역   : 어떤 '쟁점'을 시험 범위로 삼을 것인가(ISSUE_TAXONOMY)
    (2) 데이터 모델 : 한 문항이 어떤 필드를 갖는가(GoldenItem)
    (3) DB 접근     : 시험 재료(법령 조문)를 어디서 어떻게 긁어오는가
    (4) 텍스트 유틸 : 원문에서 '채점 가능한 토큰'을 어떻게 뽑아내는가
    (5) 엔진 A/B    : 질문 문장을 템플릿으로 만들 것인가, LLM으로 만들 것인가
    (6) 오케스트레이션 : 유형별 개수를 맞춰 문항을 배분하는 루프
    (7) 출력/CLI    : JSON/CSV/화면 출력과 명령행 옵션

================================================================================
1. 목적
================================================================================
tax_laws_master(법령 코퍼스)에서 쟁점별 조문을 샘플링하여
(질문 / 정답 / 정답 컨텍스트) 3종 세트를 자동 생성한다.

================================================================================
2. 핵심 원칙 (Why) — RAG 평가의 생명은 'Ground Truth 고정'이다
================================================================================
RAG 평가에서 가장 흔하고 가장 치명적인 실패는 **정답 자체를 LLM이 만들게 하는 것**
이다. LLM이 질문도 만들고 정답도 만들면, 그 정답이 틀렸을 때 우리는 영원히
"모델이 틀린 건지, 시험지가 틀린 건지" 구분할 수 없다. 채점자가 오염되면 점수는
숫자일 뿐 의미가 사라진다.

그래서 이 생성기는 역할을 엄격히 쪼갠다.

    * '정답 컨텍스트(gold_context)'는 절대 LLM이 지어내지 않는다.
      DB 원문(full_text)과 그 row 의 id(gold_doc_id)를 **그대로 고정**한다.
      → 이것이 나중에 검색 정확도(hit@k / MRR)를 계산할 수 있는 유일한 근거다.
    * LLM(Gemini)은 '질문 문장'만 만든다. 정답과 근거는 DB에서 고정된다.
    * 즉 "검색기가 gold_doc_id 를 실제로 끌어왔는가", "답변이 정답 컨텍스트에
      부합하는가"를 사후에 기계적으로 채점할 수 있게 만드는 것이 목표다.

이 원칙 하나 때문에 아래의 설계가 전부 따라 나온다.
GoldenItem 이 gold_doc_id / gold_context 를 반드시 들고 다니는 이유,
template 엔진이 존재할 수 있는 이유(질문만 만들면 되므로 LLM이 필요 없다),
must_include 를 '조문 번호'가 아니라 '본문 속 실질 토큰'에서 뽑는 이유가 모두
여기서 나온다.

================================================================================
3. 기존 파일(generate_goldenset_from_db.py) 대비 변경점
================================================================================
    1) 버그 수정: ISSUE_TAXONOMY 의 "law_ Prefer"(오타) → "law_prefer"
       - 오타 키는 meta.get("law_prefer") 에서 조용히 None 이 되어, '선호 법령
         정렬'이 통째로 죽어 있었다. 예외가 안 나기 때문에 더 위험한 종류의 버그다.
    2) 무(無)-API `template` 엔진 추가:
       - Gemini 키 없이도 결정론적(재현 가능)으로 테스트셋 생성
       - 질문은 조문 제목/본문에서 파생, 정답/컨텍스트는 DB 원문 고정
       - 답변 채점이 의미 있도록 must_include 를 '본문 속 핵심 토큰(숫자·용어)'
         에서 추출 (조문번호가 아니라 '정답에 등장해야 할 실질 내용')
    3) 통일된 산출 스키마(evaluate_rag.py 가 그대로 먹는 형태) + 구/신 필드 동시 출력
    4) --seed 로 재현성, --limit 등 옵션

================================================================================
4. 산출 스키마 (JSON 리스트, 항목 1개)
================================================================================
    {
      "id":                 "G001_exact",
      "type":               "exact | criteria | hybrid | absence",
      "issue":              "거주자_비거주자",
      "query":              "실무자가 물을 법한 질문",
      "expected_answer":    "정답(모범답) - DB 원문 기반",
      "must_include":       ["정답에 반드시 있어야 할 토큰"],
      "must_not_include":   ["나오면 감점(할루시/오해)"],
      "gold_doc_id":        "소득세법_제1조의4",   ← 검색이 맞혀야 할 정답 문서 id
      "gold_law_name":      "소득세법",
      "gold_article_number":"1의4",
      "gold_context":       "DB full_text 원문(정답 컨텍스트)",
      "engine":             "template | gemini",
      # --- 하위 호환(구 스키마) alias ---
      "input":              동일값(=query),
      "expected_output":    동일값(=expected_answer),
      "source_id":          동일값(=gold_doc_id)
    }

문항 유형(type)의 의도:
    exact    : 조문 번호를 질문에 노출한다. 검색 난이도가 가장 낮아 **회귀
               베이스라인** 역할을 한다. 여기서 실패하면 파이프라인 기본기가 깨진 것.
    criteria : 조문 번호를 숨기고 주제어만 준다. **의미 검색(임베딩/리랭커)** 능력
               을 본다.
    hybrid   : 특정 수치·용어를 콕 집어 유사 조문과의 혼동을 유발한다.
    absence  : DB 에 존재하지 않는 조문을 묻는다. **모르면 모른다고 말하는가**
               (기권 능력)를 본다. 이 유형은 gold_doc_id 가 빈 문자열이다.

================================================================================
5. 사용 예
================================================================================
    # (무료) 템플릿 엔진으로 40문항
    python build_testset.py --engine template \
        --exact 12 --criteria 12 --hybrid 10 --absence 6 \
        --format json --out testset.json

    # (유료) Gemini 로 더 교묘한 질문
    python build_testset.py --engine gemini \
        --exact 5 --criteria 5 --hybrid 5 --absence 3 \
        --format json --out testset.json

    # 쟁점 지정 + 표준출력 미리보기
    python build_testset.py --engine template \
        --issues 거주자_비거주자,위임_대통령령 --exact 4 --format stdout

================================================================================
6. .env
================================================================================
    TAX_DB_URI=postgresql://user:pw@localhost:5432/tax_rag_db
    GEMINI_API_KEY=...            # engine=gemini 일 때만 필요

================================================================================
7. 전체적인 주의사항
================================================================================
    * 이 생성기는 tax_laws_master(법령)만 사용한다. 판례/심판례는 현재 검색기가
      다루지 않으므로 공정한 시험을 위해 제외한다.
    * template 엔진은 '결정론적'이다. 같은 --seed 와 같은 DB 상태면 같은 시험지가
      나온다. 반대로 말하면 **DB 가 바뀌면 시험지도 바뀐다.** 모델 A/B 를 비교할
      때는 반드시 '같은 testset.json 파일'을 재사용해야 하며, 매번 새로 생성하면
      비교 자체가 무의미해진다.
    * gemini 엔진은 비결정론적이다(temperature=0.4). 재현성이 필요하면 생성된
      JSON 을 버전관리에 넣어라. 시드로는 고정되지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import TYPE_CHECKING, Any, NoReturn

import psycopg2
from psycopg2.extras import RealDictCursor

if TYPE_CHECKING:
    # 타입 검사 전용 import.
    # Why: 런타임에는 필요 없는 심볼(psycopg2 의 커넥션 타입)을 시그니처에서
    #      쓰기 위해서다. `from __future__ import annotations` 덕분에 어노테이션은
    #      문자열로만 평가되므로, 이 블록은 실행 시 절대 import 되지 않는다.
    #      → 런타임 비용 0, 순환 import 위험 0.
    from psycopg2.extensions import connection as PgConnection

try:
    # .env 로딩은 '있으면 좋고 없어도 도는' 선택 의존성으로 취급한다.
    # Why: 이 스크립트는 CI/도커/로컬 등 여러 환경에서 돌 수 있고, 환경변수가
    #      이미 주입된 환경(예: 컨테이너)에서는 python-dotenv 가 없을 수 있다.
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# 주의: 모듈 로드 시점에 환경변수를 '한 번' 읽어 상수로 굳힌다.
#       따라서 프로세스 실행 중에 os.environ 을 바꿔도 여기에는 반영되지 않는다.
TAX_DB_URI: str | None = os.getenv("TAX_DB_URI")
# GOOGLE_API_KEY 를 폴백으로 둔 이유: google-generativeai 계열 도구가 환경마다
# 서로 다른 키 이름을 쓰는 역사가 있어, 둘 다 받아준다.
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


# =============================================================================
# 쟁점 사전 (키워드로 tax_laws_master 검색)
#   * 기존 파일의 "law_ Prefer" 오타를 "law_prefer" 로 통일
#
# 설계 배경(Why):
#   조문을 무작위로 뽑으면 부칙·별표·정의규정 같은 '시험 가치가 낮은' 조문이
#   대량으로 섞인다. 그래서 실무 쟁점(거주자 판정, 양도소득, 상증세 …)을 축으로
#   삼아, 각 쟁점의 대표 키워드로 코퍼스를 훑는 방식을 택했다.
#   - keywords   : ILIKE 부분일치용. 형태소 분석기 없이도 동작하도록 '조사가 붙어도
#                  살아남는 어간'을 골라 둔 것이 포인트다(예: '거주자', '183일').
#   - law_prefer : 같은 키워드가 여러 법령에 걸릴 때 어느 법령을 앞세울지.
#                  ORDER BY 의 CASE 절로 쓰인다(0=선호, 1=그 외).
#
# 주의:
#   키워드를 늘릴수록 SQL 의 OR 절이 길어지고 인덱스를 못 타 느려진다.
#   또 '위임_대통령령'처럼 문구 자체가 흔한 쟁점은 후보가 수천 건이 되므로,
#   fetch_issue_rows 의 LIMIT 과 정렬 기준(본문 길이 DESC)이 실질적인 필터다.
# =============================================================================
ISSUE_TAXONOMY: dict[str, dict] = {
    "거주자_비거주자": {
        "keywords": ["거주자", "비거주자", "거소", "주소", "183일"],
        "law_prefer": ["소득세법", "소득세법 시행령"],
    },
    "양도소득": {
        "keywords": ["양도소득", "양도차익", "양도소득세", "비사업용"],
        "law_prefer": ["소득세법", "소득세법 시행령"],
    },
    "상속_증여": {
        "keywords": ["상속", "증여", "시가", "평가"],
        "law_prefer": ["상속세 및 증여세법", "상속세 및 증여세법 시행령"],
    },
    "부가가치세": {
        "keywords": ["사업자", "과세대상", "면세", "영세율", "세금계산서"],
        "law_prefer": ["부가가치세법", "부가가치세법 시행령"],
    },
    "종합부동산세": {
        "keywords": ["종합부동산세", "과세기준일", "합산", "공제"],
        "law_prefer": ["종합부동산세법", "종합부동산세법 시행령"],
    },
    "조세특례": {
        "keywords": ["특례", "감면", "세액공제", "과세특례"],
        "law_prefer": ["조세특례제한법", "조세특례제한법 시행령"],
    },
    "위임_대통령령": {
        "keywords": ["대통령령으로 정한다", "대통령령", "위임"],
        "law_prefer": ["소득세법", "부가가치세법", "상속세 및 증여세법"],
    },
}

# absence(존재하지 않는 근거) 유형에 쓸 '가짜 조문' 후보.
#
# Why 이런 번호들인가:
#   실제로 존재할 가능성이 거의 없으면서(999조, 1234조), 표기법 자체는 완벽히
#   자연스러운 번호를 골랐다. 형식이 이상하면(예: '제-1조') 모델이 질문의 형식만
#   보고 거부할 수 있어 '기권 능력'을 제대로 못 잰다. 형식은 멀쩡한데 존재만
#   하지 않아야 좋은 함정이 된다.
# 주의:
#   법 개정으로 실제 조문이 생길 수 있으므로, generate() 는 사용 직전에
#   article_exists() 로 DB 를 조회해 '진짜 없는지' 확인한다.
FAKE_ARTICLE_PAIRS = [
    ("소득세법", "999"),
    ("소득세법", "1의99"),
    ("부가가치세법", "888"),
    ("종합부동산세법", "777"),
    ("조세특례제한법", "1234"),
    ("상속세 및 증여세법", "555"),
]

# 핵심어 추출 시 걸러낼 흔한 조사/일반어.
#
# Why 필요한가:
#   한국어 형태소 분석기(konlpy 등)를 쓰지 않고 정규식 [가-힣]{2,6} 으로 명사를
#   근사 추출하기 때문에, '경우/해당/규정' 같은 기능어가 상위로 올라온다.
#   이런 단어가 must_include 에 들어가면 **아무 답변이나 통과**시키는 무의미한
#   채점 기준이 되어 버린다. 그래서 수동 스톱워드로 방어한다.
# 주의:
#   형태소 분석기를 넣지 않은 것은 의도적 트레이드오프다(설치 부담·속도·OS 의존성).
#   대신 스톱워드를 늘려가며 품질을 관리한다.
STOPWORDS = {
    "대통령령", "경우", "다음", "각", "호의", "여기", "그러", "또는", "이하",
    "이상", "미만", "초과", "해당", "관련", "포함", "제외", "따라", "위하",
    "규정", "사항", "내용", "제도", "이때", "다만", "관하", "대하", "때에",
}


# =============================================================================
# 데이터 모델
# =============================================================================
@dataclass
class GoldenItem:
    """골든셋 한 문항의 전체 스키마.

    dataclass 를 쓴 이유(Why):
        문항 필드가 13개가 넘어 dict 로 다루면 오타(`gold_doc_id` vs `gold_docid`)를
        런타임까지 못 잡는다. dataclass 로 고정하면 IDE 자동완성과 정적 분석이
        먹히고, `asdict()` 한 방으로 JSON 직렬화까지 끝난다.

    Attributes:
        id: 문항 식별자. 'G001_exact' 처럼 `G{연번}_{유형}` 형식이라 파일명 정렬과
            눈으로 훑기가 동시에 된다.
        type: 문항 유형. exact | criteria | hybrid | absence.
        issue: 이 문항이 속한 쟁점 키(ISSUE_TAXONOMY 의 키). absence 는 "absence".
        query: 모델에게 실제로 던질 질문 문장.
        expected_answer: 모범답. DB 원문에서 추출하거나(template) LLM이 원문 범위
            안에서 작성한다(gemini).
        must_include: 답변에 반드시 등장해야 하는 토큰. 채점의 주 기준(kw_recall).
        must_not_include: 등장하면 감점되는 토큰(흔한 오해·할루시네이션 신호).
        gold_doc_id: **검색기가 맞혀야 할 정답 문서의 DB id.** hit@k/MRR 의 기준이며,
            absence 유형에서는 정답 문서가 없으므로 빈 문자열이다.
        gold_law_name: 정답 조문의 법령명.
        gold_article_number: 정답 조문 번호(DB 원형. 예: '1', '1의4').
        gold_context: 정답 컨텍스트. DB full_text 원문 그대로(가공 금지).
        engine: 이 문항을 만든 엔진('template' | 'gemini'). 사후 분석 시 엔진별
            난이도 차이를 분리해서 보기 위한 메타데이터.
        input: 하위 호환 alias(= query). finalize() 가 채운다.
        expected_output: 하위 호환 alias(= expected_answer). finalize() 가 채운다.
        source_id: 하위 호환 alias(= gold_doc_id). finalize() 가 채운다.

    Note:
        하위 호환 alias 3개를 남긴 이유는, 예전 평가 스크립트/노트북이 아직
        input/expected_output/source_id 키를 읽고 있기 때문이다. 값을 **중복 저장**
        하는 대신 property 로 만들 수도 있었지만, `asdict()` 로 그대로 JSON 이
        되어야 하므로(=property 는 asdict 에 안 실린다) 실제 필드로 두고
        finalize() 에서 동기화하는 방식을 택했다.

        따라서 **GoldenItem 을 만들면 반드시 finalize() 를 호출해야 한다.**
        빠뜨리면 alias 필드가 빈 문자열로 나가서 구 스크립트가 조용히 오작동한다.
    """

    id: str
    type: str                 # exact | criteria | hybrid | absence
    issue: str
    query: str
    expected_answer: str
    must_include: list[str]
    must_not_include: list[str]
    gold_doc_id: str
    gold_law_name: str
    gold_article_number: str
    gold_context: str
    engine: str
    # 하위 호환(구 스키마) alias
    input: str = ""
    expected_output: str = ""
    source_id: str = ""

    def finalize(self) -> "GoldenItem":
        """하위 호환 alias 필드를 채우고 자기 자신을 돌려준다.

        Returns:
            alias 가 동기화된 자기 자신. 반환값을 주는 이유는
            `return item.finalize()` 형태로 한 줄에 쓰기 위해서다(메서드 체이닝).

        Note:
            이 메서드는 **멱등(idempotent)** 하다. 여러 번 불러도 결과가 같다.
            다만 finalize() 이후 query/expected_answer 를 바꾸면 alias 가 어긋나므로,
            수정이 필요하면 수정 후 다시 finalize() 를 호출해야 한다.
        """
        self.input = self.query
        self.expected_output = self.expected_answer
        self.source_id = self.gold_doc_id
        return self


def die(msg: str, code: int = 1) -> NoReturn:
    """치명적 오류를 stderr 에 찍고 프로세스를 종료한다.

    Args:
        msg: 사용자에게 보여줄 오류 메시지.
        code: 프로세스 종료 코드. 0 이 아니어야 CI/셸 스크립트가 실패로 인식한다.

    Raises:
        SystemExit: 항상 발생한다(정상 흐름).

    Note:
        반환 타입 `NoReturn` 은 "이 함수는 절대 정상 반환하지 않는다"를 타입 체커에
        알려 준다. 덕분에 `die(...)` 뒤의 코드가 '도달 불가'로 인식되고,
        `x = a or die(...)` 같은 패턴에서 x 의 타입이 좁혀진다.

        stdout 이 아니라 **stderr** 로 찍는 이유: 이 스크립트는 `--format stdout`
        일 때 결과 데이터를 stdout 으로 흘린다. 오류를 stdout 에 섞으면
        `python build_testset.py > out.json` 같은 파이프가 오염된다.
    """
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


# =============================================================================
# DB 접근
# =============================================================================
def get_conn() -> "PgConnection":
    """tax_rag_db(PostgreSQL) 커넥션을 연다.

    Returns:
        psycopg2 커넥션. 호출부가 `try/finally` 로 반드시 close() 해야 한다.

    Raises:
        SystemExit: TAX_DB_URI 환경변수가 없을 때.
        psycopg2.OperationalError: 접속 실패(호스트/인증/DB명 오류 등).

    Note:
        커넥션 풀을 쓰지 않고 매번 단일 커넥션을 여는 이유(Why):
        이 스크립트는 배치성 CLI 로, 프로세스 수명 동안 커넥션 1개만 쓰고 끝난다.
        풀은 동시성이 있을 때 이득인데 여기선 동시성이 0 이므로 복잡도만 늘어난다.
        (반대로 api_server 는 요청이 동시에 들어오므로 PostgresRepo 로 풀을 쓴다.)
    """
    if not TAX_DB_URI:
        die("TAX_DB_URI 환경변수가 없습니다. (.env 확인)")
    return psycopg2.connect(TAX_DB_URI)


def fetch_issue_rows(conn: "PgConnection", issue: str, limit: int = 12) -> list[dict]:
    """쟁점 키워드로 tax_laws_master 에서 본문 있는 조문 후보를 뽑는다.

    Args:
        conn: 열려 있는 DB 커넥션.
        issue: ISSUE_TAXONOMY 의 키. 존재하지 않으면 KeyError.
        limit: 가져올 최대 행 수. 문항 수보다 넉넉해야 중복 없이 배분할 수 있다.

    Returns:
        각 행이 id/law_name/article_number/article_title/full_text 를 갖는 dict 목록.
        RealDictCursor 를 쓰므로 인덱스가 아니라 **컬럼명으로 접근**한다.

    Raises:
        KeyError: `issue` 가 ISSUE_TAXONOMY 에 없을 때.
        psycopg2.Error: SQL 실행 실패.

    Note:
        설계 의도(Why) — SQL 하나로 '필터 + 우선순위 + 표본 크기'를 동시에 끝낸다.

        1) `length(full_text) > 40` : 본문이 사실상 비어 있는 껍데기 조문(제목만
           있거나 '삭제' 처리된 조문)을 걸러낸다. 이런 행으로 문항을 만들면
           must_include 를 뽑을 재료가 없어 채점이 무의미해진다.
        2) 키워드 OR 절 : `full_text ILIKE %s OR article_title ILIKE %s`.
           ILIKE 는 대소문자 무시 부분일치인데, 한글에는 대소문자가 없으므로
           사실상 LIKE 와 같다. 영문 약어가 섞인 조문 때문에 ILIKE 를 쓴다.
        3) `CASE WHEN law_name = ANY(%s) THEN 0 ELSE 1 END` : 선호 법령을 앞으로.
           ANY(배열) 를 쓰면 psycopg2 가 파이썬 list 를 그대로 Postgres 배열로
           변환해 주므로, IN 절을 문자열로 조립할 필요가 없다(= SQL 인젝션 회피).
        4) `length(full_text) DESC` : 본문이 긴 조문 우선. 긴 조문일수록 숫자·요건이
           많아 '채점 가능한 토큰'이 잘 나온다.

        ★ 파라미터 순서 주의 :
          params 리스트에 쌓는 순서가 SQL 안의 %s 등장 순서와 **정확히** 일치해야
          한다. 이 함수에서는 (키워드들) → (선호 법령 배열) → (limit) 순이다.
          SELECT 절에는 %s 가 없고, ORDER BY 의 CASE 가 WHERE 뒤에 오기 때문에
          이 순서가 성립한다. WHERE 에 조건을 추가할 때는 반드시 이 순서를
          다시 확인해야 한다(가장 흔한 회귀 버그 지점).

        ★ f-string 으로 SQL 을 조립하지만 **값은 절대 문자열로 넣지 않는다.**
          where_kw / prefer_sql 은 코드가 만든 고정 구조(placeholder 만 포함)이고,
          실제 값은 전부 params 로 바인딩된다. 이 경계를 흐리면 즉시 인젝션이 된다.
    """
    meta = ISSUE_TAXONOMY[issue]
    keywords = meta["keywords"]
    prefer = meta.get("law_prefer") or []

    # 키워드마다 (본문 OR 제목) 쌍을 만들어 OR 로 잇는다.
    # 파라미터가 키워드당 2개씩 쌓이는 점에 주의.
    kw_clauses, params = [], []
    for kw in keywords:
        kw_clauses.append("(full_text ILIKE %s OR article_title ILIKE %s)")
        params.extend([f"%{kw}%", f"%{kw}%"])
    where_kw = " OR ".join(kw_clauses)

    # 선호 법령이 있으면 ORDER BY 맨 앞에 CASE 절을 끼워 넣는다.
    # 끝에 쉼표가 붙어 있어 아래 f-string 에서 `{prefer_sql} length(...)` 로 이어진다.
    # (선호 법령이 없으면 빈 문자열이라 ORDER BY 가 자연스럽게 length 만 남는다.)
    prefer_sql = ""
    if prefer:
        prefer_sql = "CASE WHEN law_name = ANY(%s) THEN 0 ELSE 1 END,"
        params.append(prefer)

    params.append(limit)
    sql = f"""
        SELECT id, law_name, article_number,
               COALESCE(article_title, '') AS article_title,
               full_text
        FROM tax_laws_master
        WHERE full_text IS NOT NULL AND length(full_text) > 40
          AND ({where_kw})
        ORDER BY {prefer_sql} length(full_text) DESC
        LIMIT %s
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def article_exists(conn: "PgConnection", law_name: str, article_number: str) -> bool:
    """해당 법령·조문이 DB 에 실제로 존재하는지 확인한다.

    Args:
        conn: 열려 있는 DB 커넥션.
        law_name: 법령명(정확 일치. 부분일치 아님).
        article_number: 조문 번호(DB 원형. 예: '999', '1의99').

    Returns:
        존재하면 True.

    Note:
        absence 문항의 **안전장치**다(Why). FAKE_ARTICLE_PAIRS 는 "없을 것 같은"
        조문일 뿐, 법 개정으로 실제 생길 수 있다. 실제로 존재하는 조문을 두고
        "없다고 답해야 통과"로 채점하면 시험지 자체가 틀린 것이 된다.

        `LIMIT 1` 과 `SELECT 1` 을 쓴 이유: 존재 여부만 알면 되므로 행 내용을
        가져올 필요가 없고, 첫 행을 찾는 즉시 스캔을 멈추게 하려는 것이다.

        주의: law_name 은 정확 일치라서 '소득세법 시행령' 같은 표기 차이가 있으면
        '없음'으로 판정된다. FAKE_ARTICLE_PAIRS 의 법령명 표기를 DB 와 맞춰야 한다.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM tax_laws_master WHERE law_name=%s AND article_number=%s LIMIT 1",
            (law_name, article_number),
        )
        return cur.fetchone() is not None


# =============================================================================
# 텍스트 유틸 / 핵심어 추출 (template 엔진용)
#
# 이 영역의 존재 이유(Why):
#   template 엔진은 LLM 없이 '채점 가능한 문항'을 만들어야 한다. 그러려면
#   원문에서 (a) 모범답으로 쓸 요약, (b) 답변에 반드시 나와야 할 토큰을
#   규칙만으로 뽑아내야 한다. 아래 함수들이 그 규칙의 전부다.
# =============================================================================
def first_sentences(text: str, n: int = 2, max_chars: int = 320) -> str:
    """본문 앞부분에서 문장 n개를 잘라 모범답 후보를 만든다.

    Args:
        text: 원문(법령 full_text). None/빈 문자열도 안전하게 처리한다.
        n: 취할 문장 수.
        max_chars: 최종 길이 상한. 초과분은 잘라낸다.

    Returns:
        공백이 정규화된 요약 문자열. 입력이 비면 빈 문자열.

    Note:
        Why '앞 n문장'인가:
            한국 법령 조문은 대체로 **두괄식**이다. 제1항 첫 문장에 원칙이 오고,
            뒤로 갈수록 위임·단서·별표 참조가 붙는다. 그래서 앞부분만 잘라도
            '이 조문이 무엇을 규정하는가'라는 질문의 모범답으로 쓸 만하다.

        문장 분리 정규식 `(?<=[.。])\\s+|\\n+` 의 의미:
            - `(?<=[.。])\\s+` : 마침표(반각/전각) **뒤**의 공백에서 자른다.
              lookbehind 를 쓰므로 마침표 자체는 앞 문장에 남는다.
            - `|\\n+` : 줄바꿈도 문장 경계로 본다. 법령 텍스트는 항·호가 줄바꿈으로
              구분되는 경우가 많아 이게 실질적으로 더 자주 걸린다.

        주의(한계):
            한국어 법령 문장은 '…한다.' 로 끝나지만, 중간에 '제1호.' 같은 표기나
            소수점('0.5')이 있으면 오분할된다. 완벽한 문장 분리는 목표가 아니다.
            어차피 모범답은 **참고용**이고, 실제 채점의 주력은 must_include 이기
            때문에 이 정도 근사로 충분하다는 판단이다.

            또 `parts[:n]` 이 전부 빈 문자열이면 out 이 비므로, 그때는 원문 전체를
            쓰도록 폴백을 둔다(빈 모범답이 나가는 사고 방지).
    """
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""
    parts = re.split(r"(?<=[.。])\s+|\n+", text)
    out = " ".join(p for p in parts[:n] if p).strip()
    if not out:
        # 문장 분리가 실패해도 빈 모범답은 내보내지 않는다.
        out = text
    return out[:max_chars].strip()


def extract_salient_tokens(title: str, full_text: str, k: int = 3) -> list[str]:
    """'정답에 등장해야 할 실질 토큰'을 뽑는다.

    Args:
        title: 조문 제목(article_title). 없으면 빈 문자열.
        full_text: 조문 본문.
        k: 뽑을 토큰 개수 상한.

    Returns:
        최대 k개의 토큰 목록. 아무것도 못 뽑으면 제목(또는 본문 앞 8자) 1개를
        담은 목록을 돌려준다(빈 리스트를 절대 반환하지 않는다).

    Note:
        Why 숫자를 최우선으로 두는가:
            채점은 문자열 포함 검사다. 그래서 **검증이 명확한 토큰**일수록 좋다.
            '183일', '6억원', '10%' 같은 수치는 답변에 있거나 없거나 둘 중 하나라
            판정이 흔들리지 않는다. 반면 '거주자' 같은 일반 명사는 문맥 없이도
            우연히 등장할 수 있어 '통과인데 사실은 오답'을 만들어낸다.

            즉 이 함수의 우선순위는 곧 **채점 신뢰도 순위**다.
            숫자 > 제목 명사 > 본문 앞부분 명사.

        2단계(명사) 처리 상세:
            - 후보 수집 범위를 `full_text[:160]` 으로 자른 이유: 뒤로 갈수록
              위임 문구·별표 참조가 많아 조문 주제와 무관한 단어가 섞인다.
            - `cand.sort(key=lambda x: -len(x))` : 긴 단어 우선.
              한국어 복합명사(예: '종합부동산세')는 길수록 구체적이고 변별력이 높다.
              짧은 단어는 다른 조문에도 흔해 변별력이 떨어진다.

        주의(한계와 함정):
            1) 이 함수는 형태소 분석을 하지 않는다. '거주자가'에서 정규식이
               '거주자가'를 통째로 뽑을 수 있다. STOPWORDS 로 최악만 막는다.
            2) 숫자 패턴은 `\\d+\\s*호`, `\\d+\\s*항` 같은 **구조 참조**를 걸러내지
               않는다. 즉 '제3호'의 3이 토큰이 될 수 있다. (참고: 형제 파일
               build_faithfulness_testset.py 는 이 문제를 is_structural_ref() 로
               해결했다. 이 파일은 exact/criteria 문항 위주라 영향이 작아 단순한
               규칙을 유지한다.)
            3) 반환 순서에 의미가 있다. 호출부가 `salient[:2]` 처럼 앞에서 자르므로,
               앞쪽일수록 채점 기준으로 강하게 쓰인다.
    """
    tokens: list[str] = []
    blob = f"{title}\n{full_text}"

    # 1) 숫자성 표현 — 가장 채점하기 좋은 토큰이므로 최우선.
    #    패턴 순서가 곧 선호 순서다(기간 → 비율 → 금액 → 분수).
    num_patterns = [
        r"\d+\s*일", r"\d+\s*년", r"\d+\s*개월", r"\d+\s*%", r"\d+\s*퍼센트",
        r"\d+\s*억\s*원?", r"\d+\s*만\s*원", r"\d+\s*원", r"\d+\s*분의\s*\d+",
    ]
    for pat in num_patterns:
        for m in re.findall(pat, blob):
            # '6억 원' 처럼 내부 공백이 섞인 표기를 '6억원' 으로 정규화한다.
            # Why: 답변 LLM 의 띄어쓰기는 예측 불가라, 저장할 때부터 공백을 없애
            #      두는 편이 채점 측(공백 무시 비교)과 일관된다.
            t = re.sub(r"\s+", "", m)
            if t not in tokens:
                tokens.append(t)
            if len(tokens) >= k:
                return tokens[:k]

    # 2) 제목 우선 명사 — 숫자가 부족할 때의 대체재.
    cand: list[str] = []
    for src in (title, full_text[:160]):
        for w in re.findall(r"[가-힣]{2,6}", src or ""):
            if w in STOPWORDS or w in cand:
                continue
            cand.append(w)
    # 길이가 긴(=구체적) 단어 우선
    cand.sort(key=lambda x: -len(x))
    for w in cand:
        if w not in tokens:
            tokens.append(w)
        if len(tokens) >= k:
            break
    # 최후의 보루: 그래도 비면 제목(또는 본문 앞 8자)이라도 넣는다.
    # 빈 must_include 는 '무조건 통과'를 뜻해 채점이 무력화되므로 반드시 피한다.
    return tokens[:k] if tokens else [(title or full_text[:8]).strip()]


def format_article_display(art: str) -> str:
    """DB 의 article_number('1','1의4')를 자연스러운 표기('제1조','제1조의4')로 바꾼다.

    Args:
        art: DB 원형 조문 번호. None/빈 값도 허용한다.

    Returns:
        '제N조' 또는 '제N조의M'. 입력이 비면 빈 문자열.

    Note:
        Why 저장형과 표시형을 분리하는가:
            DB 는 정렬·조인·비교를 위해 '1의4' 같은 **정규형**을 저장한다.
            반면 질문 문장에 '소득세법 1의4는…' 이라고 쓰면 사람이 쓰지 않는
            부자연스러운 한국어가 되어, 모델이 질문 자체를 오해할 수 있다.
            평가 문항의 질문은 최대한 **실무자가 실제로 묻는 문장**이어야 하므로
            표시형으로 변환한다.

        주의: '의' 가 들어간 조문만 처리한다. '1의2의3' 같은 삼중 표기가 DB 에
        있다면 split(maxsplit=1) 특성상 '제1조의2의3' 이 되는데, 이는 의도한
        동작이다(원형을 보존한 채 접두사만 붙임).
    """
    art = str(art or "").strip()
    if not art:
        return ""
    if "의" in art:
        base, sub = art.split("의", 1)
        return f"제{base}조의{sub}"
    return f"제{art}조"


def pick_topic_word(title: str, full_text: str) -> str:
    """질문 문장에 쓸 핵심 주제어 하나를 고른다.

    Args:
        title: 조문 제목.
        full_text: 조문 본문(제목이 쓸모없을 때의 폴백 재료).

    Returns:
        주제어 한 개. 최후에는 '해당 규정' 이라는 무난한 문구를 돌려준다.

    Note:
        Why 제목을 1순위로 두는가:
            법령 조문 제목(예: '거주자의 주소와 거소의 판정')은 사람이 이미
            요약해 둔 최고 품질의 주제어다. 자동 추출한 명사보다 항상 낫다.

        길이 조건 `2 <= len(t) <= 20` 의 의미:
            너무 짧으면(1자) 변별력이 없고, 너무 길면 질문 문장이 어색해진다.
            법령 제목은 가끔 한 문장급으로 길기 때문에 상한이 필요하다.

        `strip("()（）")` : 제목이 '(정의)' 처럼 괄호로 감싸인 사례가 있어
        반각·전각 괄호를 모두 벗겨낸다.
    """
    t = (title or "").strip().strip("()（）")
    if t and 2 <= len(t) <= 20 and t not in STOPWORDS:
        return t
    toks = extract_salient_tokens(title, full_text, k=1)
    return toks[0] if toks else "해당 규정"


# =============================================================================
# 엔진 A: template (무 API, 결정론적)
#
# 이 엔진의 존재 이유(Why):
#   1) 비용 0 · 재현성 100%. 같은 시드·같은 DB 면 언제나 같은 시험지가 나온다.
#      회귀 테스트(어제보다 나빠졌는가?)에는 결정론이 필수다.
#   2) 네트워크·API 키 없이 CI 에서 돌릴 수 있다.
#   3) '질문만 만들면 된다'는 설계 원칙(§2) 덕분에, 질문 품질이 조금 투박해도
#      평가의 타당성은 훼손되지 않는다. 정답과 근거는 어차피 DB 고정이다.
#
# 한계(주의):
#   질문 문장이 정형화되어 있어 '표현 다양성에 대한 강건성'은 못 잰다.
#   그 목적이라면 gemini 엔진을 쓴다.
# =============================================================================
def make_template_item(
    idx: int,
    type_name: str,
    issue: str,
    row: dict,
    engine: str = "template",
) -> GoldenItem:
    """DB 행 하나로 exact/criteria/hybrid 문항을 결정론적으로 만든다.

    Args:
        idx: 문항 연번(1부터). id 문자열에 3자리로 zero-pad 되어 들어간다.
        type_name: 'exact' | 'criteria' | 그 외(=hybrid 로 취급).
        issue: 쟁점 키. 결과 분석 시 쟁점별 정답률을 보기 위한 라벨이다.
        row: fetch_issue_rows() 가 돌려준 행(id/law_name/article_number/
            article_title/full_text).
        engine: 산출물에 기록할 엔진 이름. 기본 'template'.

    Returns:
        finalize() 까지 끝난 GoldenItem.

    Raises:
        KeyError: row 에 'law_name' / 'article_number' / 'id' 키가 없을 때.

    Note:
        ★ type_name 의 분기 구조가 곧 '난이도 사다리'다.

        exact:
            질문에 조문 번호를 그대로 노출한다. 검색기 입장에서는 키워드 매칭만
            해도 맞힐 수 있는 가장 쉬운 문항이다. 그래서 **베이스라인**으로 쓴다.
            여기서 실패하면 임베딩 품질이 아니라 인덱싱·파싱 같은 기본 배관이
            깨진 것이므로, 원인 추적 범위를 크게 좁혀 준다.
            모범답은 3문장/400자로 넉넉히 잡는다(내용 설명형 질문이므로).

        criteria:
            조문 번호를 **숨기고** 주제어만 준다. 어휘가 정확히 일치하지 않아도
            의미로 찾아내야 하므로 임베딩·리랭커의 실력이 드러난다.
            모범답은 1문장/200자로 짧게 잡는다. 채점 기준을 좁혀 '핵심 요지를
            말했는가'만 보기 위해서다.

        hybrid(else):
            salient 의 첫 토큰(주로 수치)을 앵커로 삼아 질문에 박아 넣는다.
            비슷한 수치를 가진 다른 조문과 혼동을 유발하는 것이 목적이다.
            must_include 를 3개까지 요구해 채점을 가장 엄격하게 건다.

        ★ 주의 1 — `else` 가 hybrid 를 삼킨다:
            type_name 에 오타('hybird')가 들어와도 예외 없이 hybrid 로 처리된다.
            호출부(generate 의 plan 리스트)가 유일한 진입점이므로 현재는 안전하지만,
            새 유형을 추가할 때는 이 분기 구조를 먼저 손봐야 한다.

        ★ 주의 2 — must_not_include 가 항상 비어 있다:
            template 엔진은 '무엇이 오답인지'를 규칙만으로 알 수 없기 때문이다.
            오답 함정이 필요하면 gemini 엔진이나 build_faithfulness_testset.py 를
            쓴다. 빈 must_not_include 는 채점 시 '할루시네이션 없음'으로 처리된다.
    """
    law = row["law_name"] or ""
    art = str(row["article_number"] or "")
    title = row.get("article_title") or ""
    full = row.get("full_text") or ""

    topic = pick_topic_word(title, full)
    salient = extract_salient_tokens(title, full, k=3)

    art_disp = format_article_display(art)

    if type_name == "exact":
        # 조문 번호를 명시(검색 난이도 낮음, 회귀 베이스라인)
        query = f"「{law}」 {art_disp}는 어떤 내용을 규정하고 있나요? 핵심만 알려주세요."
        expected = first_sentences(full, n=3, max_chars=400)
        # salient 가 비는 일은 없지만(위 함수가 폴백을 보장), 방어적으로 law 를 둔다.
        must_inc = salient[:2] if salient else [law]
        must_not: list[str] = []

    elif type_name == "criteria":
        # 조문 번호 미노출(의미 검색 유도) + 채점 기준 명확
        query = f"세법상 '{topic}'에 관해 어떤 기준·요건이 정해져 있나요?"
        expected = first_sentences(full, n=1, max_chars=200)
        must_inc = salient[:2] if salient else [topic]
        must_not = []

    else:  # hybrid
        # 특정 용어/수치를 콕 집어 혼동 유발
        anchor = salient[0] if salient else topic
        query = f"「{law}」에서 {anchor} 관련 요건(또는 기준)은 구체적으로 무엇인가요?"
        expected = first_sentences(full, n=2, max_chars=300)
        must_inc = salient[:3] if salient else [anchor]
        must_not = []

    item = GoldenItem(
        id=f"G{idx:03d}_{type_name}",
        type=type_name,
        issue=issue,
        query=query,
        expected_answer=expected,
        # str() 캐스팅 + 빈 값 제거: 토큰에 None/숫자가 섞여도 JSON 스키마가 깨지지
        # 않게 한다. 빈 문자열 토큰은 '항상 포함'이라 채점을 무력화하므로 반드시 제거.
        must_include=[str(x) for x in must_inc if x],
        must_not_include=[str(x) for x in must_not if x],
        gold_doc_id=row["id"],
        gold_law_name=law,
        gold_article_number=art,
        # ★ 원문을 통째로 보존한다. 자르거나 요약하면 검색 채점(context_hit)의
        #   기준이 흔들린다. 여기가 이 파일 전체 설계의 핵심 지점이다.
        gold_context=full,
        engine=engine,
    )
    return item.finalize()


def make_template_absence(idx: int, law: str, art: str) -> GoldenItem:
    """DB 에 없는 조문을 묻는 absence 문항을 만든다.

    Args:
        idx: 문항 연번.
        law: 법령명(실존 법령이어야 질문이 자연스럽다).
        art: 존재하지 않는 조문 번호.

    Returns:
        finalize() 까지 끝난 GoldenItem. gold_doc_id 는 빈 문자열이다.

    Note:
        Why 이 유형이 필요한가:
            RAG 의 가장 위험한 실패는 '틀린 답'이 아니라 **'모르면서 아는 척하는
            답'** 이다. 세무 도메인에서는 없는 조문을 그럴듯하게 지어내는 순간
            실무자가 잘못된 신고를 하게 된다. 그래서 "근거가 없으면 없다고 말하는가"
            를 별도 유형으로 반드시 측정한다.

        must_include=["없", "확인"] 의 의도:
            '확인할 수 없습니다', '규정이 없습니다', '찾을 수 없습니다' 등
            기권 표현의 변주를 어간 수준에서 잡기 위한 최소 집합이다.
            (평가 측 evaluate_rag.is_absence_answer 도 같은 취지의 마커를 본다.)

        ★ 주의 — 이 방식은 느슨하다:
            "관련 내용이 **없**지는 않으나…" 처럼 '없'이 다른 맥락으로 들어가도
            통과된다. 즉 재현율은 높지만 정밀도가 낮은 채점 기준이다.
            더 엄격한 판정이 필요하면 evaluate_faithfulness.py 의
            ABSTENTION_MARKERS(완성형 표현 목록) 방식을 참고할 것.

        ★ 주의 — gold_doc_id 가 빈 문자열인 이유:
            정답 문서가 존재하지 않으므로 hit@k/MRR 계산에서 이 문항은 제외되어야
            한다. 평가 측은 `type == "absence"` 를 보고 검색 지표에서 빼낸다.
    """
    art_disp = format_article_display(art)
    query = f"「{law}」 {art_disp}에 규정된 내용이 무엇인지 알려주세요."
    expected = (
        f"해당 조문(「{law}」 {art_disp})은 제공된 자료에서 확인할 수 없습니다. "
        f"근거가 없어 내용을 단정할 수 없습니다."
    )
    item = GoldenItem(
        id=f"G{idx:03d}_absence",
        type="absence",
        issue="absence",
        query=query,
        expected_answer=expected,
        must_include=["없", "확인"],       # '없다/확인할 수 없다' 취지
        must_not_include=[],
        gold_doc_id="",
        gold_law_name=law,
        gold_article_number=art,
        # 실제 원문이 없으므로 '왜 없는지'를 사람이 알아볼 수 있게 표식을 남긴다.
        gold_context=f"[DB 미존재] {law} 제{art}조",
        engine="template",
    )
    return item.finalize()


# =============================================================================
# 엔진 B: gemini (질문만 생성, 정답/근거는 DB 고정)
#
# 이 엔진의 존재 이유(Why):
#   template 엔진의 질문은 문형이 고정되어 있어, 모델이 '문형에 과적합'되면
#   실제보다 점수가 좋게 나올 수 있다. 사람처럼 다양하고 교묘한 질문이 필요할 때
#   LLM 을 쓴다. 단, §2 원칙에 따라 **질문만** 맡긴다.
#
# 주의:
#   비결정론적이다. 같은 입력에도 매번 다른 질문이 나오므로, 모델 A/B 비교에는
#   반드시 '한 번 생성해 파일로 굳힌 시험지'를 재사용해야 한다.
# =============================================================================
# 프롬프트의 '헌법' 역할. 모든 생성 요청 앞에 붙는다.
# 규칙 1(원문 밖 사실 금지)이 가장 중요하다. 이것이 무너지면 시험지가 오염되어
# 평가 결과 전체가 쓰레기가 된다.
SYSTEM_RULES = """
당신은 세무 법령 RAG 평가용 골든셋 출제자다.
절대 규칙:
1) 제공된 EVIDENCE 원문 밖의 사실·조문·숫자를 만들어내지 마라.
2) 정답 문장은 EVIDENCE에 있는 표현을 우선 사용하라. 의역은 최소화.
3) 질문은 한국어, 실무자가 물을 법한 톤.
4) 함정 질문이면 흔한 오해(예: 1년 거주, 제1조와 제1조의2 혼동)를 이용하되,
   정답은 반드시 EVIDENCE에 근거해야 한다.
5) 출력은 JSON 객체 하나만. 설명 문장 금지.
"""


def _prompt(kind: str, issue: str, row: dict) -> str:
    """유형별 문항 생성 프롬프트를 조립한다.

    Args:
        kind: 'exact' | 'criteria' | 그 외(=hybrid).
        issue: 쟁점 키(프롬프트에 문맥으로 제공).
        row: DB 행(id/law_name/article_number/full_text 사용).

    Returns:
        SYSTEM_RULES + 근거(EVIDENCE) + 유형별 요구사항이 합쳐진 프롬프트 문자열.

    Note:
        `[:1800]` 으로 근거를 자르는 이유(Why):
            토큰 비용과 지연을 억제하고, 무엇보다 **긴 컨텍스트에서 모델이 뒷부분에
            집중력을 잃는 현상(lost-in-the-middle)** 을 피하기 위해서다.
            앞부분에 원칙이 오는 법령의 두괄식 구조와도 맞는다.

        주의: 이때 프롬프트에 들어가는 근거는 1800자로 잘리지만,
        **gold_context 로 저장되는 원문은 자르지 않는다**(make_gemini_item 참조).
        질문 생성용 입력과 채점용 정답 근거는 별개라는 점을 혼동하지 말 것.

        JSON 스키마를 프롬프트 끝에 예시로 못 박은 이유: 모델이 설명문을 덧붙이는
        습관을 억제하기 위해서다. 그럼에도 코드펜스를 붙이는 일이 잦아,
        extract_json_block() 이 뒷정리를 맡는다(방어적 파싱).
    """
    ev = (row["full_text"] or "")[:1800]
    head = f'''{SYSTEM_RULES}

유형: {kind}
쟁점: {issue}
SOURCE_ID: {row['id']}
법령: {row['law_name']} 제{row['article_number']}조
EVIDENCE:
"""
{ev}
"""
'''
    if kind == "exact":
        tail = """
요구:
- 이 EVIDENCE만으로 답할 수 있는 질문 1개
- expected_output: EVIDENCE 핵심을 2~4문장으로 명확히
- must_include: 정답에 반드시 등장할 핵심어/수치 2~5개
- must_not_include: 이 근거로 말하면 안 되는 흔한 오해 1~3개
JSON: {"input":"질문","expected_output":"모범답","must_include":["..."],"must_not_include":["..."]}
"""
    elif kind == "criteria":
        tail = """
요구:
- 채점 기준이 명확한 질문 1개
- expected_output: 한 줄 핵심 요지
- must_include: 답이 반드시 포함해야 할 근거 표지(수치/핵심 문구)
- must_not_include: 나오면 감점인 할루시·오해
JSON: {"input":"질문","expected_output":"한 줄 요지","must_include":["..."],"must_not_include":["..."]}
"""
    else:  # hybrid
        tail = """
요구:
- 비슷해 보이는 다른 조문과 혼동하기 쉽거나, 단서/위임/정의를 찌르는 질문
- expected_output: EVIDENCE 근거 답
- must_include: 올바른 핵심어·수치
- must_not_include: 혼동하기 쉬운 오답 요소
JSON: {"input":"질문","expected_output":"모범답","must_include":["..."],"must_not_include":["..."]}
"""
    return head + tail


def _prompt_absence(law: str, art: str) -> str:
    """absence 문항용 프롬프트를 만든다.

    Args:
        law: 법령명.
        art: 존재하지 않는 조문 번호.

    Returns:
        '없는 조문을 묻는 질문'을 요청하는 프롬프트.

    Note:
        여기서는 EVIDENCE 를 주지 않는다. 줄 근거가 없기 때문이다. 대신 "이 조문은
        DB 에 없다"는 **사실을 선언**해 주고, 그 전제 위에서 질문을 만들게 한다.

        ★ 문법 주의: 이 함수의 f-string 안에서 JSON 예시의 중괄호는 `{{ }}` 로
        이스케이프해야 한다(_prompt 는 f-string 이 아닌 일반 문자열에 JSON 예시를
        넣으므로 이스케이프가 필요 없다). 두 함수의 처리 방식이 다른 이유가
        바로 이것이며, 수정 시 가장 자주 깨지는 지점이다.
    """
    return f'''{SYSTEM_RULES}

유형: absence (없는 근거)
사실: 아래 조문은 우리 DB에 존재하지 않는다.
- {law} 제{art}조

요구:
- 이 조문/제도가 있는지 묻는 질문 1개
- expected_output: 반드시 '없다' 또는 '확인할 수 없다' 취지
- must_include: ["없"]
- must_not_include: 존재하지 않는 조문 내용을 지어낸 표현
JSON: {{"input":"질문","expected_output":"모범답","must_include":["없"],"must_not_include":["..."]}}
'''


def call_gemini(prompt: str, model: str) -> str:
    """Gemini 공식 SDK 로 프롬프트를 한 번 호출한다.

    Args:
        prompt: 완성된 프롬프트 문자열.
        model: 모델 이름(예: 'gemini-2.0-flash').

    Returns:
        모델의 텍스트 응답. 응답이 비면 빈 문자열.

    Raises:
        SystemExit: API 키가 없거나 SDK 가 설치되지 않았을 때(die 경유).
        Exception: SDK 호출 자체의 오류(할당량/네트워크 등)는 그대로 전파된다.
            상위 generate() 루프가 문항 단위로 잡아 [SKIP] 처리한다.

    Note:
        Why import 를 함수 안에서 하는가:
            google-generativeai 는 무겁고, template 엔진만 쓰는 사용자에게는
            아예 필요 없는 의존성이다. 모듈 최상단에서 import 하면 키도 없고 패키지도
            없는 환경에서 **스크립트가 시작조차 못 한다.** 지연 import 로 "필요할 때만
            요구"하는 구조를 만든다.

        temperature=0.4 의 의도:
            0에 가까우면 질문이 천편일률이 되어 template 엔진과 다를 바 없어지고,
            높이면 규칙(원문 밖 금지)을 어길 확률이 올라간다. 그 사이의 절충값이다.

        `getattr(resp, "text", None)` 를 쓰는 이유:
            안전 필터에 걸리면 응답 객체에 text 속성이 아예 없거나 접근 시 예외가
            날 수 있다. 방어적으로 읽어 빈 문자열로 떨어뜨리고, 이후
            extract_json_block() 이 ValueError 를 내 상위에서 [SKIP] 되게 한다.
    """
    if not GEMINI_API_KEY:
        die("engine=gemini 에는 GEMINI_API_KEY(또는 GOOGLE_API_KEY)가 필요합니다.")
    try:
        import google.generativeai as genai
    except ImportError:
        die("pip install google-generativeai 후 다시 실행하세요.")
    genai.configure(api_key=GEMINI_API_KEY)
    m = genai.GenerativeModel(model)
    resp = m.generate_content(
        prompt, generation_config={"temperature": 0.4, "max_output_tokens": 1024}
    )
    return (getattr(resp, "text", None) or "").strip()


def extract_json_block(text: str) -> dict[str, Any]:
    """LLM 응답 문자열에서 JSON 객체 하나를 뽑아 파싱한다.

    Args:
        text: 모델 원문 응답(코드펜스·설명문이 섞여 있을 수 있다).

    Returns:
        파싱된 dict.

    Raises:
        ValueError: 중괄호 블록을 못 찾았을 때(앞 200자를 메시지에 담아 디버깅 지원).
        json.JSONDecodeError: 블록은 찾았지만 JSON 문법이 깨졌을 때.

    Note:
        Why 이런 방어적 파싱이 필요한가:
            "JSON만 출력하라"고 아무리 강하게 지시해도 LLM 은
            (a) ```json 코드펜스로 감싸거나, (b) "다음과 같습니다:" 같은 서두를 붙이거나,
            (c) 뒤에 설명을 덧붙이는 일이 흔하다. 이건 모델 결함이라기보다
            챗 형식으로 학습된 결과라서, 프롬프트로 100% 막을 수 없다.
            따라서 **파서가 관대해야 한다.**

        처리 순서:
            1) 앞뒤 코드펜스 제거,
            2) `\\{[\\s\\S]*\\}` 로 **가장 바깥 중괄호 구간**을 통째로 잡는다.
               `[\\s\\S]` 는 개행을 포함하는 '모든 문자'라서 여러 줄 JSON 도 잡힌다.

        ★ 주의 — greedy 매칭의 부작용:
            정규식이 greedy 이므로 텍스트에 중괄호가 여러 번 나오면 '첫 { 부터
            마지막 } 까지'를 한 덩어리로 잡는다. 모델이 JSON 을 두 개 출력하면
            파싱에 실패한다. 실무상 이 편이(=조용히 앞 것만 취하는 것보다) 낫다.
            실패하면 상위에서 [SKIP] 되어 그 문항만 버려지기 때문이다.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"JSON 없음: {text[:200]}")
    return json.loads(m.group(0))


def make_gemini_item(idx: int, type_name: str, issue: str, row: dict, model: str) -> GoldenItem:
    """LLM 으로 질문/모범답을 받아 문항을 만든다(근거는 DB 고정).

    Args:
        idx: 문항 연번.
        type_name: 'exact' | 'criteria' | 'hybrid'.
        issue: 쟁점 키.
        row: DB 행.
        model: Gemini 모델 이름.

    Returns:
        finalize() 까지 끝난 GoldenItem.

    Raises:
        ValueError: 응답에 JSON 이 없거나, input/expected_output 이 비었을 때.
        Exception: API 호출 실패는 그대로 전파(상위에서 [SKIP]).

    Note:
        ★ 이 함수가 §2 원칙이 구현되는 지점이다.
          LLM 이 만든 값은 query / expected_answer / must_include / must_not_include
          뿐이고, gold_doc_id·gold_law_name·gold_article_number·gold_context 는
          **전부 row(DB) 에서 그대로 가져온다.** 즉 '검색이 맞혀야 할 정답'은
          LLM 이 절대 건드릴 수 없다.

        빈 값 검증(`if not q or not a`)을 두는 이유:
            질문이나 모범답이 빈 문항은 채점이 불가능한 불량품이다. 조용히
            통과시키면 리포트의 분모만 늘려 점수를 왜곡한다. 그래서 예외를 던져
            상위 루프가 그 문항을 버리고 다른 row 로 재시도하게 만든다.

        as_list() 헬퍼: 모델이 must_include 를 문자열 하나로 줄 때가 있어
        (예: `"must_include": "183일"`) 리스트로 정규화한다.
    """
    raw = call_gemini(_prompt(type_name, issue, row), model=model)
    obj = extract_json_block(raw)

    def as_list(v: Any) -> list[Any]:
        """None/스칼라/리스트를 모두 리스트로 정규화한다."""
        if v is None:
            return []
        return v if isinstance(v, list) else [str(v)]

    q = str(obj.get("input") or "").strip()
    a = str(obj.get("expected_output") or "").strip()
    if not q or not a:
        raise ValueError("input/expected_output 빈 값")

    must_inc = [str(x) for x in as_list(obj.get("must_include"))]
    must_not = [str(x) for x in as_list(obj.get("must_not_include"))]

    item = GoldenItem(
        id=f"G{idx:03d}_{type_name}",
        type=type_name,
        issue=issue,
        query=q,
        expected_answer=a,
        must_include=must_inc,
        must_not_include=must_not,
        # ↓↓↓ 여기부터는 전부 DB 고정값. LLM 의 출력이 절대 끼어들지 않는다.
        gold_doc_id=row["id"],
        gold_law_name=row["law_name"] or "",
        gold_article_number=str(row["article_number"] or ""),
        gold_context=row["full_text"] or "",   # 프롬프트와 달리 자르지 않는다
        engine="gemini",
    )
    return item.finalize()


def make_gemini_absence(idx: int, law: str, art: str, model: str) -> GoldenItem:
    """LLM 으로 absence 문항(없는 조문 질문)을 만든다.

    Args:
        idx: 문항 연번.
        law: 법령명.
        art: 존재하지 않는 조문 번호.
        model: Gemini 모델 이름.

    Returns:
        finalize() 까지 끝난 GoldenItem.

    Note:
        ★ 방어 로직의 핵심 — must_include 에 '없' 강제 주입:
            absence 문항의 채점은 '기권했는가'를 보는 것인데, LLM 이 must_include 를
            엉뚱하게(예: ["소득세법"]) 채워 오면 **기권하지 않아도 통과**하는
            망가진 문항이 된다. 그래서 '없' 이 없으면 코드가 직접 추가해
            채점 의미를 보장한다.

            "시험지를 LLM 에게 맡기되, 채점 기준의 최저선은 코드가 지킨다"는
            이 파일의 일관된 태도가 여기에도 적용된 것이다.

        각 필드에 폴백 문자열을 둔 이유: 모델 응답에 키가 빠져도 문항이 완성되게
        해 생성 수율을 지킨다(absence 는 근거가 없어 재시도 가치가 낮다).
    """
    raw = call_gemini(_prompt_absence(law, art), model=model)
    obj = extract_json_block(raw)
    mi = obj.get("must_include") or []
    mi = mi if isinstance(mi, list) else [str(mi)]
    if not any("없" in str(x) for x in mi):
        # 채점 기준의 최저선 보장(위 Note 참조).
        mi.append("없")
    item = GoldenItem(
        id=f"G{idx:03d}_absence",
        type="absence",
        issue="absence",
        query=str(obj.get("input") or f"「{law}」 제{art}조 내용을 알려주세요.").strip(),
        expected_answer=str(obj.get("expected_output") or "해당 조문은 확인할 수 없습니다.").strip(),
        must_include=[str(x) for x in mi],
        must_not_include=[str(x) for x in (obj.get("must_not_include") or [])],
        gold_doc_id="",
        gold_law_name=law,
        gold_article_number=art,
        gold_context=f"[DB 미존재] {law} 제{art}조",
        engine="gemini",
    )
    return item.finalize()


# =============================================================================
# 생성 오케스트레이션
# =============================================================================
def generate(
    engine: str,
    n_exact: int,
    n_criteria: int,
    n_hybrid: int,
    n_absence: int,
    issues: list[str],
    model: str,
    sleep_s: float,
    seed: int,
) -> list[GoldenItem]:
    """유형별 목표 개수를 채우도록 문항을 생성·배분한다.

    Args:
        engine: 'template' | 'gemini'.
        n_exact: exact 문항 목표 개수.
        n_criteria: criteria 문항 목표 개수.
        n_hybrid: hybrid 문항 목표 개수.
        n_absence: absence 문항 목표 개수.
        issues: 사용할 쟁점 키 목록.
        model: gemini 엔진일 때 쓸 모델 이름.
        sleep_s: gemini 호출 간 대기(초). 분당 한도(429) 예방용.
        seed: 샘플링 시드. 같은 시드 + 같은 DB = 같은 시험지(template 한정).

    Returns:
        생성된 GoldenItem 목록. 목표 개수에 못 미칠 수 있다(후보 부족·API 실패).

    Note:
        ── 설계 1: '쟁점 라운드로빈' 배분 ──────────────────────────────────
        쟁점별로 후보 풀(pools)을 미리 만들어 두고, next_row() 가 쟁점을 **돌아가며**
        하나씩 꺼낸다.
        Why: 한 쟁점을 몰아 쓰면 시험지가 '거주자 판정 전문 시험'처럼 편향된다.
        라운드로빈은 쟁점 커버리지를 자동으로 균등하게 만든다.

        ── 설계 2: used_ids 로 중복 방지, 고갈 시 재사용 허용 ───────────────
        같은 조문으로 여러 문항을 만들면 사실상 같은 문제를 반복 채점하게 되어
        점수 신뢰도가 떨어진다. 그래서 1차로는 안 쓴 row 만 고른다.
        다만 풀이 마르면 **문항 수를 채우는 것**을 우선해 random.choice 로 재사용한다.
        (품질보다 수량을 택한 의도적 트레이드오프. 재사용된 문항은 gold_doc_id 가
        겹치므로 사후 분석 시 중복을 확인할 수 있다.)

        ── 설계 3: guard 카운터로 무한루프 차단 ────────────────────────────
        `while made < count and guard < count * 6` — gemini 엔진은 API 실패로
        문항 생성이 계속 실패할 수 있다. 목표치의 6배까지만 시도하고 포기한다.
        이 상한이 없으면 API 가 죽었을 때 스크립트가 영원히 돌 수 있다.

        ── 설계 4: 예외를 문항 단위로 삼킨다 ───────────────────────────────
        try/except 가 루프 **안쪽**에 있다. 한 문항 실패가 전체 배치를 죽이지 않게
        하려는 것이다. 40문항 중 3개가 실패하면 37개짜리 시험지를 얻는 편이,
        아무것도 못 얻는 것보다 낫다.

        ── 주의 ────────────────────────────────────────────────────────────
        * `finally: conn.close()` 로 커넥션 누수를 막는다. 중간에 예외가 나도 닫힌다.
        * pools 의 limit 계산 `max(12, n_exact+n_criteria+n_hybrid)` 은
          '쟁점 하나가 전체 문항을 감당할 수 있을 만큼'을 뜻한다. 쟁점 수가 많으면
          과다 조회가 되지만, 조회 1회 비용이 작아 단순함을 택했다.
        * absence 는 FAKE_ARTICLE_PAIRS 개수(현재 6개)가 상한이다. 그보다 많이
          요청해도 그만큼만 나온다.
        * cycle 변수는 계속 증가하며 next_row 의 시작 쟁점을 회전시킨다.
          유형이 바뀌어도 초기화하지 않으므로, 유형 간에도 쟁점이 골고루 섞인다.
    """
    random.seed(seed)
    conn = get_conn()
    items: list[GoldenItem] = []
    idx = 1

    # 쟁점별 후보 미리 확보 (중복 소비 방지용 풀)
    pools: dict[str, list[dict]] = {}
    for iss in issues:
        rows = fetch_issue_rows(conn, iss, limit=max(12, (n_exact + n_criteria + n_hybrid)))
        # 시드 고정 셔플: '본문 길이 DESC' 순서를 그대로 쓰면 매번 같은 장문 조문만
        # 뽑힌다. 셔플로 다양성을 주되, seed 로 재현성은 유지한다.
        random.shuffle(rows)
        pools[iss] = rows
        if not rows:
            print(f"[WARN] 쟁점 '{iss}' 검색 0건", file=sys.stderr)

    used_ids: set[str] = set()

    def next_row(iss_cycle_start: int) -> tuple[dict | None, str]:
        """쟁점을 돌아가며 아직 안 쓴 row 를 하나 집는다.

        Args:
            iss_cycle_start: 라운드로빈 시작 오프셋(호출마다 1씩 증가한다).

        Returns:
            (행, 쟁점키). 모든 풀이 비어 있으면 (None, "").

        Note:
            1차 루프는 '미사용 row'만, 2차 루프는 '재사용 허용'으로 돈다.
            2차에서 random.choice 를 쓰므로 재사용 문항은 무작위로 겹친다.
        """
        for step in range(len(issues)):
            iss = issues[(iss_cycle_start + step) % len(issues)]
            for r in pools.get(iss, []):
                if r["id"] not in used_ids:
                    used_ids.add(r["id"])
                    return r, iss
        # 다 썼으면 재사용 허용
        for step in range(len(issues)):
            iss = issues[(iss_cycle_start + step) % len(issues)]
            if pools.get(iss):
                return random.choice(pools[iss]), iss
        return None, ""

    plan = [("exact", n_exact), ("criteria", n_criteria), ("hybrid", n_hybrid)]
    cycle = 0
    try:
        for type_name, count in plan:
            made, guard = 0, 0
            while made < count and guard < count * 6:
                guard += 1
                row, iss = next_row(cycle)
                cycle += 1
                if row is None:
                    print(f"[SKIP] {type_name}: 후보 row 없음", file=sys.stderr)
                    break
                try:
                    if engine == "template":
                        item = make_template_item(idx, type_name, iss, row)
                    else:
                        item = make_gemini_item(idx, type_name, iss, row, model)
                        # 호출 직후 쉬어 분당 한도를 예방한다(성공한 뒤에만 쉰다).
                        time.sleep(sleep_s)
                    items.append(item)
                    idx += 1
                    made += 1
                    print(f"[OK] {item.id} issue={iss} src={row['id']}", file=sys.stderr)
                except Exception as e:
                    # 문항 단위 실패 격리: 이 row 는 버리고 다음 row 로 넘어간다.
                    print(f"[SKIP] {type_name}/{iss}: {e}", file=sys.stderr)

        # absence: DB 에 '정말로 없는' 조문만 골라 만든다.
        abs_made = 0
        for law, art in FAKE_ARTICLE_PAIRS:
            if abs_made >= n_absence:
                break
            if article_exists(conn, law, art):
                # 법 개정 등으로 실제 생긴 조문 → 함정으로 쓸 수 없으므로 건너뛴다.
                continue
            try:
                if engine == "template":
                    item = make_template_absence(idx, law, art)
                else:
                    item = make_gemini_absence(idx, law, art, model)
                    time.sleep(sleep_s)
                items.append(item)
                idx += 1
                abs_made += 1
                print(f"[OK] {item.id} absence {law} {art}", file=sys.stderr)
            except Exception as e:
                print(f"[SKIP] absence {law} {art}: {e}", file=sys.stderr)
    finally:
        # 예외가 나든 정상 종료든 커넥션은 반드시 닫는다.
        conn.close()

    return items


# =============================================================================
# 출력
#
# 출력 채널 설계(Why):
#   - 데이터(JSON/CSV/미리보기)는 stdout 또는 파일로,
#   - 진행 로그/경고/요약은 stderr 로 분리한다.
#   덕분에 `python build_testset.py --format json > testset.json` 처럼 리다이렉트해도
#   로그가 데이터에 섞이지 않는다. 유닉스 파이프라인의 기본 예절이자, 이 스크립트가
#   다른 도구와 조합될 수 있게 하는 핵심 규칙이다.
# =============================================================================
def emit_json(items: list[GoldenItem], path: str | None) -> None:
    """문항 목록을 JSON 으로 저장하거나 화면에 출력한다.

    Args:
        items: 저장할 문항 목록.
        path: 파일 경로. None 이면 stdout 으로 출력한다.

    Note:
        `ensure_ascii=False` 는 필수다. 기본값(True)이면 한글이 전부
        \\uXXXX 이스케이프로 저장되어 사람이 시험지를 검수할 수 없게 된다.
        indent=2 도 같은 이유(diff 가독성 — 시험지는 버전관리에 넣어 관리한다).
    """
    data = [asdict(x) for x in items]
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[WRITE] {path} ({len(items)}건)", file=sys.stderr)
    else:
        print(text)


def emit_csv(items: list[GoldenItem], path: str | None) -> None:
    """문항 목록을 CSV 로 저장하거나 화면에 출력한다.

    Args:
        items: 저장할 문항 목록. 비어 있으면 종료한다.
        path: 파일 경로. None 이면 stdout.

    Raises:
        SystemExit: items 가 비었을 때(die 경유).

    Note:
        ★ encoding='utf-8-sig' 를 쓰는 이유:
            BOM 이 없으면 한국어 Windows 의 Excel 이 CSV 를 CP949 로 오인해
            한글이 전부 깨진다. 실무자가 시험지를 Excel 로 검수하는 워크플로를
            지원하려면 BOM 이 사실상 필수다.

        ★ newline='' 를 쓰는 이유:
            csv 모듈은 자체적으로 개행을 제어한다. Windows 에서 이 인자를 빼면
            개행이 이중(\\r\\r\\n)으로 들어가 빈 줄이 생긴다. csv 사용 시의 표준 관용구다.

        ★ 리스트 필드(must_include 등)는 json.dumps 로 직렬화한다.
            CSV 에는 리스트 타입이 없다. JSON 문자열로 넣어 두면 나중에
            json.loads 로 무손실 복원이 가능하다(evaluate_rag._as_list 가 이를 처리).

        주의: path 가 None 이면 sys.stdout 을 쓰는데, finally 에서 **닫지 않는다.**
        stdout 을 닫으면 이후 print 가 전부 실패하기 때문이다. 조건부 close 가
        의도된 동작이다.
    """
    if not items:
        die("생성된 항목이 없습니다.")
    # 첫 항목의 필드 순서를 헤더로 삼는다(dataclass 필드 선언 순서와 동일).
    fieldnames = list(asdict(items[0]).keys())
    rows = []
    for it in items:
        d = asdict(it)
        for k in ("must_include", "must_not_include"):
            d[k] = json.dumps(d[k], ensure_ascii=False)
        rows.append(d)
    dst = open(path, "w", encoding="utf-8-sig", newline="") if path else sys.stdout
    try:
        w = csv.DictWriter(dst, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    finally:
        if path:
            dst.close()
            print(f"[WRITE] {path} ({len(items)}건)", file=sys.stderr)


def resolve_output(fmt: str, out: str) -> tuple[str, str | None]:
    """저장 형식과 경로를 결정한다(저장 비대칭 제거).

    Args:
        fmt: --format 값('json' | 'csv' | 'stdout').
        out: --out 값. 빈 문자열이면 파일 저장 안 함.

    Returns:
        (형식, 경로). 경로가 None 이면 화면 출력.

    Note:
        규칙:
          - --out 이 주어지면 '파일로 저장'한다.
              · --format 이 json/csv 로 명시됐으면 그 형식을 따름.
              · --format 이 기본값(stdout)이면 --out 확장자로 추론(.csv→csv, 그 외→json).
          - --out 이 없으면 --format 을 그대로(기본 stdout=화면).

        Why 이 함수가 따로 존재하는가('저장 비대칭' 문제):
            예전에는 `--out testset.json` 만 주면 --format 이 stdout 이라 파일이
            생기지 않고 화면에만 뿌려졌다. 사용자는 저장됐다고 믿고 다음 단계로
            넘어가 "테스트셋이 없다"는 오류를 만났다.
            **사용자의 의도(경로를 줬다 = 저장하고 싶다)를 우선**하도록 규칙을
            한곳에 모아 버그를 구조적으로 없앤 것이다.

            이런 '의도 추론' 로직은 흩어지면 반드시 어긋나므로, 순수 함수 하나로
            격리해 두는 편이 테스트하기도 쉽다.
    """
    if out:
        if fmt in ("json", "csv"):
            return fmt, out
        ext = os.path.splitext(out)[1].lower()
        return ("csv" if ext == ".csv" else "json"), out
    return fmt, None


def emit_stdout(items: list[GoldenItem]) -> None:
    """문항을 사람이 읽기 좋은 형태로 화면에 출력한다(검수용).

    Args:
        items: 출력할 문항 목록.

    Note:
        시험지를 파일로 굳히기 전에 **눈으로 품질을 확인**하는 용도다.
        gold_context 는 160자만 잘라 보여 준다. 조문 전문을 다 찍으면 화면이
        넘쳐 정작 봐야 할 질문/정답 대비가 안 보이기 때문이다.
    """
    for it in items:
        print("=" * 66)
        print(f"[{it.id}] type={it.type} issue={it.issue} engine={it.engine}")
        print(f"Q : {it.query}")
        print(f"A : {it.expected_answer}")
        print(f"must_include     : {it.must_include}")
        print(f"must_not_include : {it.must_not_include}")
        print(f"gold_doc_id      : {it.gold_doc_id}  ({it.gold_law_name} 제{it.gold_article_number}조)")
        print(f"gold_context     : {(it.gold_context or '')[:160]}...")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    """명령행 인자를 정의하고 파싱한다.

    Returns:
        파싱된 인자 네임스페이스.

    Note:
        기본값 설계 의도:
          - --engine template : 키 없이도 즉시 돌아가야 '처음 써 보는 사람'이
            막히지 않는다. 유료 경로는 명시적으로 선택하게 한다.
          - --format stdout : 실수로 파일을 덮어쓰지 않도록 '보기'가 기본이다.
            (--out 을 주면 resolve_output 이 저장으로 승격시킨다.)
          - --seed 42 : 재현성 확보. 시드를 바꾸면 다른 표본이 나온다.
          - --sleep 0.8 : 무료 등급 Gemini 의 분당 한도를 피하기 위한 경험값.
    """
    p = argparse.ArgumentParser(description="세무 RAG 테스트셋(골든셋) 생성기")
    p.add_argument("--engine", choices=["template", "gemini"], default="template",
                   help="template=무API/결정론적(기본), gemini=질문만 LLM 생성")
    p.add_argument("--exact", type=int, default=6, help="정답형 개수")
    p.add_argument("--criteria", type=int, default=6, help="기준형 개수")
    p.add_argument("--hybrid", type=int, default=6, help="혼합형(교묘) 개수")
    p.add_argument("--absence", type=int, default=4, help="absence(없는 근거) 개수")
    p.add_argument("--issues", type=str, default="all", help="쟁점 키 콤마구분 또는 all")
    p.add_argument("--format", choices=["json", "csv", "stdout"], default="stdout",
                   help="출력 형식(기본 stdout=화면). --out 지정 시 확장자로 자동 판단됨")
    p.add_argument("--out", type=str, default="",
                   help="파일 경로. 지정하면 파일로 저장(확장자 .json/.csv 로 형식 자동 판단, 기본 json)")
    p.add_argument("--model", type=str, default="gemini-2.0-flash", help="engine=gemini 모델")
    p.add_argument("--sleep", type=float, default=0.8, help="Gemini 호출 간격(초)")
    p.add_argument("--seed", type=int, default=42, help="샘플링 재현 시드")
    return p.parse_args()


def _self_test() -> None:
    """DB 없이 순수 텍스트 유틸만 검증한다."""
    assert format_article_display("59") == "제59조"
    assert format_article_display("1의4") == "제1조의4"
    tokens = extract_salient_tokens("연차", "근로자는 183일 이상 거주해야 합니다")
    assert tokens
    print("[self-test] build_testset OK")


def main() -> None:
    """CLI 진입점: 인자 파싱 → 생성 → 출력 → 요약.

    Raises:
        SystemExit: 알 수 없는 쟁점이거나 생성 결과가 0건일 때.

    Note:
        쟁점 검증을 **생성 전에** 하는 이유: 오타('거주자_비거주')로 몇 분간 DB 를
        긁고 나서 실패하면 시간이 아깝다. 실패는 가능한 한 빨리, 그리고
        '가능한 값 목록'과 함께 알려 주는 것이 좋은 CLI 다.

        `from collections import Counter` 를 함수 안에서 하는 이유는 특별한
        의도라기보다, 요약 출력에서만 쓰이는 국소 의존성이기 때문이다.

        요약을 stderr 로 보내는 것도 의도적이다(§출력 채널 설계 참조).
    """
    args = parse_args()
    if args.issues.strip() == "all":
        issues = list(ISSUE_TAXONOMY.keys())
    else:
        issues = [x.strip() for x in args.issues.split(",") if x.strip()]
        for i in issues:
            if i not in ISSUE_TAXONOMY:
                die(f"알 수 없는 쟁점: {i}. 가능: {list(ISSUE_TAXONOMY)}")

    items = generate(
        engine=args.engine,
        n_exact=args.exact, n_criteria=args.criteria,
        n_hybrid=args.hybrid, n_absence=args.absence,
        issues=issues, model=args.model, sleep_s=args.sleep, seed=args.seed,
    )
    if not items:
        die("생성된 항목이 없습니다. (DB 연결/쟁점 키워드 확인)")

    fmt, out = resolve_output(args.format, args.out)
    if fmt == "json":
        emit_json(items, out)
    elif fmt == "csv":
        emit_csv(items, out)
    else:
        emit_stdout(items)

    # 요약
    from collections import Counter
    c = Counter(it.type for it in items)
    print(f"\n[SUMMARY] 총 {len(items)}건 | " +
          " ".join(f"{k}={v}" for k, v in c.items()), file=sys.stderr)


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_test()
        sys.exit(0)
    main()
