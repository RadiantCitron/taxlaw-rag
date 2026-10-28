# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""검색 원본 테이블 이름·비교 테이블 명명·DDL 문의 유일한 소유자.

psycopg2 / FastAPI 를 import 하지 않는다. 생산자(build_compare_embedding_tax)
와 소비자(postgres_repo)가 이 파일만 본다.
"""
from __future__ import annotations

import re

# 원본 embedding 컬럼 차원(jhgan/ko-sroberta-multitask). 비교 모델은 다를 수 있다.
VECTOR_DIM = 768
CMP_INFIX = "_cmp_"
PG_IDENT_MAXLEN = 63

ALLOWED_SOURCE_TABLES = (
    "tax_laws_master",
    "tax_precedents_master",
    "tax_tribunal_master",
    "tax_amendments_2026",
)

# 검색 SELECT 는 항상 law_name / article_number 별칭을 쓴다.
SOURCE_LABEL_COLS = {
    "tax_laws_master": ("law_name", "article_number"),
    "tax_precedents_master": ("case_name", "case_number"),
    "tax_tribunal_master": ("case_name", "case_number"),
    "tax_amendments_2026": ("statute_area", "item_no"),
}


def normalize_sources(sources) -> list[str]:
    """요청된 검색 테이블을 허용 목록 순서로 정규화한다. 비거나 모르면 ValueError."""
    if sources is None:
        return list(ALLOWED_SOURCE_TABLES)
    if isinstance(sources, str):
        parts = [p.strip() for p in sources.split(",") if p.strip()]
    else:
        try:
            parts = [str(s).strip() for s in sources if str(s).strip()]
        except TypeError as exc:
            raise ValueError("검색 테이블 목록이 올바르지 않습니다.") from exc
    if not parts:
        raise ValueError("검색 테이블이 비어 있습니다.")
    seen: list[str] = []
    allowed = ", ".join(ALLOWED_SOURCE_TABLES)
    for name in parts:
        if name not in ALLOWED_SOURCE_TABLES:
            raise ValueError(f"허용되지 않은 검색 테이블: {name}. 허용: {allowed}")
        if name not in seen:
            seen.append(name)
    return seen


def require_source(source_table: str) -> str:
    name = (source_table or "").strip()
    if name not in ALLOWED_SOURCE_TABLES:
        raise ValueError(
            f"[안전장치] 허용되지 않은 원본 테이블: {source_table}. "
            f"허용: {', '.join(ALLOWED_SOURCE_TABLES)}"
        )
    return name


def safe_suffix(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (model_id or "").lower()).strip("_")


def cmp_table(source_table: str, model_id: str) -> str:
    """원본 테이블 + 모델 id 로 비교 임베딩 테이블명을 만든다.

    생산자(build_compare_embedding_tax)와 소비자(postgres_repo)가 같은 함수를
    써야 이름이 어긋나지 않는다. PostgreSQL 식별자 63바이트를 넘으면 거절한다.
    """
    table = require_source(source_table)
    name = f"{table}{CMP_INFIX}{safe_suffix(model_id)}"
    if len(name.encode("utf-8")) > PG_IDENT_MAXLEN:
        raise ValueError(
            f"[식별자 길이] 생성될 테이블명이 {len(name)}자로 PostgreSQL 제한"
            f"({PG_IDENT_MAXLEN})을 넘습니다:\n    {name}\n"
            f"모델 id 를 짧은 별칭으로 바꾸세요. 잘린 이름으로 만들면 검색기가 "
            f"테이블을 못 찾는 조용한 실패가 생깁니다."
        )
    return name


def extension_statements() -> list[str]:
    return [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        "CREATE EXTENSION IF NOT EXISTS pg_trgm;",
    ]


def trigram_index_statements() -> list[str]:
    """full_text 하이브리드 검색용 GIN(trigram) 인덱스.

    벡터 ANN 인덱스(ivfflat/hnsw)는 두지 않는다. 코퍼스가 테이블마다 다르고
    전수 스캔(<=>)이 작은 집합에서는 더 정확하다. ANN 이 필요해지면 측정 후에
    이 함수 옆에 추가한다 — 지금은 누락이 아니라 의도다.
    """
    stmts = []
    for table in ALLOWED_SOURCE_TABLES:
        stmts.append(
            f'CREATE INDEX IF NOT EXISTS idx_{table}_full_text_trgm '
            f'ON "{table}" USING gin (full_text gin_trgm_ops);'
        )
    return stmts
