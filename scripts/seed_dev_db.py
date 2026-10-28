# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""개발용 PostgreSQL(pgvector) 시드 스크립트.

로컬/클라우드 개발 환경에서 앱이 기동하고 하이브리드 검색이 동작하도록,
운영 덤프(수천 건) 대신 소량의 실제 세법 조문/판례/개정해설 샘플을
기본 임베딩 모델(jhgan/ko-sroberta-multitask, 768차원)로 임베딩해 적재한다.

이 스크립트는 멱등(idempotent)하다. 여러 번 실행해도 안전하다.
운영 데이터가 이미 들어 있는 테이블은 건드리지 않는다(행이 있으면 스킵).

사용:
    TAX_DB_URI=postgresql://myuser:****@127.0.0.1:5432/tax_rag_db \
        python scripts/seed_dev_db.py
"""
from __future__ import annotations

import os
import sys

import psycopg2
from psycopg2.extras import execute_values

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass

EMBED_MODEL_ID = os.getenv("RAG_EMBEDDING_MODEL_ID", "jhgan/ko-sroberta-multitask")
EMBED_DIM = 768

# (id, law_name, article_number, full_text)
LAWS = [
    (
        "소득세법_제12조",
        "소득세법",
        "제12조",
        "제12조(비과세소득) 다음 각 호의 소득에 대해서는 소득세를 과세하지 아니한다. "
        "1. 「공익신탁법」에 따른 공익신탁의 이익 2. 사업소득 중 논ㆍ밭을 작물 생산에 "
        "이용하게 함으로써 발생하는 소득 3. 대통령령으로 정하는 농어가부업소득 등.",
    ),
    (
        "소득세법_제59조",
        "소득세법",
        "제59조",
        "제59조(근로소득세액공제) 근로소득이 있는 거주자에 대해서는 그 근로소득에 대한 "
        "종합소득산출세액에서 대통령령으로 정하는 금액을 공제한다. 공제액은 산출세액 구간에 "
        "따라 차등 적용되며 총급여액 수준에 따라 공제 한도가 달라진다.",
    ),
    (
        "법인세법_제13조",
        "법인세법",
        "제13조",
        "제13조(과세표준) 내국법인의 각 사업연도의 소득에 대한 법인세의 과세표준은 각 "
        "사업연도의 소득의 범위에서 이월결손금, 비과세소득, 소득공제액을 차례로 공제한 "
        "금액으로 한다.",
    ),
    (
        "부가가치세법_제26조",
        "부가가치세법",
        "제26조",
        "제26조(재화 또는 용역의 공급에 대한 면세) 다음 각 호의 재화 또는 용역의 공급에 "
        "대하여는 부가가치세를 면제한다. 1. 가공되지 아니한 식료품 2. 수돗물 3. 의료보건 "
        "용역 4. 교육 용역 등 대통령령으로 정하는 것.",
    ),
    (
        "조세특례제한법_제99조의11",
        "조세특례제한법",
        "제99조의11",
        "제99조의11(월세액에 대한 세액공제) 과세기간 종료일 현재 무주택 세대의 세대주로서 "
        "총급여액이 일정 금액 이하인 근로소득자가 국민주택규모 주택을 임차하고 지급한 "
        "월세액의 일정 비율을 종합소득산출세액에서 공제한다.",
    ),
    (
        "상속세및증여세법_제53조",
        "상속세및증여세법",
        "제53조",
        "제53조(증여재산 공제) 거주자가 다음 각 호의 어느 하나에 해당하는 사람으로부터 "
        "증여를 받은 경우에는 증여세 과세가액에서 일정 금액을 공제한다. 배우자로부터의 "
        "증여, 직계존비속으로부터의 증여 등 관계에 따라 공제 한도가 다르다.",
    ),
]

# (id, case_name, case_number, full_text)
PRECEDENTS = [
    (
        "PREC_2018두12345",
        "종합소득세부과처분취소",
        "2018두12345",
        "대법원은 근로소득세액공제의 적용에 있어 총급여액 산정 시 비과세소득은 제외하여야 "
        "하며, 과세관청이 이를 포함하여 산출세액을 계산한 부과처분은 위법하다고 판단하였다.",
    ),
    (
        "PREC_2020두56789",
        "부가가치세경정거부처분취소",
        "2020두56789",
        "면세 대상인 의료보건 용역의 범위에 관하여, 부수적으로 제공되는 용역이 주된 면세 "
        "용역에 통상적으로 부수되는 경우에는 함께 면세 대상에 해당한다고 본 사례.",
    ),
]

# (id, case_name, case_number, full_text)
TRIBUNAL = [
    (
        "TT_2021_0001",
        "월세액세액공제경정청구기각",
        "조심2021서0001",
        "조세심판원은 무주택 세대주 요건과 국민주택규모 요건을 모두 충족하는 경우 월세액 "
        "세액공제가 적용되며, 임대차계약서상 주소와 주민등록상 주소가 일치하여야 한다고 "
        "결정하였다.",
    ),
]

# (id, statute_area, item_no, full_text)
AMENDMENTS = [
    (
        "AMEND_2026_소득세법_01",
        "소득세법",
        "1",
        "2026년 개정: 근로소득세액공제 한도가 총급여액 구간별로 조정되었으며, 자녀세액공제 "
        "대상 및 공제액이 확대되었다. 시행일은 2026년 1월 1일이다.",
    ),
]

TABLES = {
    "tax_laws_master": ("law_name", "article_number", LAWS),
    "tax_precedents_master": ("case_name", "case_number", PRECEDENTS),
    "tax_tribunal_master": ("case_name", "case_number", TRIBUNAL),
    "tax_amendments_2026": ("statute_area", "item_no", AMENDMENTS),
}


def _ddl(table: str, name_col: str, num_col: str) -> str:
    return (
        f'CREATE TABLE IF NOT EXISTS "{table}" ('
        f"  id text PRIMARY KEY,"
        f'  "{name_col}" text,'
        f'  "{num_col}" text,'
        f"  full_text text,"
        f"  embedding vector({EMBED_DIM})"
        f");"
    )


def main() -> int:
    db_uri = os.getenv("TAX_DB_URI")
    if not db_uri:
        print("[ERROR] TAX_DB_URI 환경변수가 없습니다.", file=sys.stderr)
        return 1

    print(f"[INIT] 임베딩 모델 로드: {EMBED_MODEL_ID}", flush=True)
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    embed = HuggingFaceEmbedding(model_name=EMBED_MODEL_ID)

    conn = psycopg2.connect(db_uri)
    conn.autocommit = True
    from database.postgres_repo import apply_search_support
    apply_search_support(conn, include_indexes=False)
    total_inserted = 0
    with conn.cursor() as cur:
        for table, (name_col, num_col, rows) in TABLES.items():
            cur.execute(_ddl(table, name_col, num_col))
            cur.execute(f'SELECT count(*) FROM "{table}"')
            existing = cur.fetchone()[0]
            if existing > 0:
                print(f"[SKIP] {table}: 이미 {existing}건 존재 (운영 데이터 보호)", flush=True)
                continue
            payload = []
            for rid, name, num, text in rows:
                vec = embed.get_text_embedding(text)
                payload.append((rid, name, num, text, "[" + ",".join(f"{x:.6f}" for x in vec) + "]"))
            execute_values(
                cur,
                f'INSERT INTO "{table}" (id, "{name_col}", "{num_col}", full_text, embedding) '
                f"VALUES %s ON CONFLICT (id) DO NOTHING",
                payload,
            )
            print(f"[PG] {table}: {len(payload)}건 적재", flush=True)
            total_inserted += len(payload)
    apply_search_support(conn, include_indexes=True)
    conn.close()
    print(f"[DONE] 시드 완료. 신규 {total_inserted}건.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
