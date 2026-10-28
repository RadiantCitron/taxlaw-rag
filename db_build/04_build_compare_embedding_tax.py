# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
# build_compare_embedding_tax.py
"""
[taxlaw-rag / 세무] 비교용 임베딩 테이블 생성기.

근로기준법 RAG 프로젝트의 build_compare_embedding.py 와 목적은 완전히 같다.
"원본은 절대 안 건드리고, 임베딩만 다른 복제 테이블을 옆에 만들어
옵션으로 골라 비교한다." 다만 이 프로젝트는 스키마와 구조가 달라서 그 부분만
맞췄으며, 실무 환경에 맞춰 제어 옵션을 대폭 확장했다.

[핵심 구조]
    - id 가 TEXT 다. 세무 조문 id 가 '소득세법_제55조__2' 같은 문자열이라
      비교 테이블의 doc_id 도 TEXT 로 만든다.
    - 본문 컬럼 이름이 full_text 다.
    - 임베딩 차원은 원본이 VECTOR(768). 새 모델은 차원이 다를 수 있어 자동 감지한다.
    - 데이터 무결성: 원본 테이블의 id를 참조(REFERENCES)하는 외래키(FK)를 걸어
      원본 데이터 삭제 시 연쇄 삭제(ON DELETE CASCADE)되도록 보장한다.

[확장된 통제 옵션]
    - 유연한 테이블 제어: 기존 4개 하드코딩 테이블뿐만 아니라,
      --source-table과 --target-table 옵션으로 커스텀 테이블 작업이 가능하다.
    - 타겟 DB 지정: --db-name 옵션으로 테스트 DB 등 타겟을 유연하게 바꿀 수 있다.
    - VRAM 제어: 무거운 모델 구동 시 OOM 방지를 위해 --batch-size 조절이 가능하다.

만드는 것 (예: 모델 BAAI/bge-m3, 원본 tax_laws_master):
    tax_laws_master_cmp_baai_bge_m3   (doc_id TEXT, embedding VECTOR(n))
    원문은 원본에서 조인해 가져오므로 여기엔 벡터만 담는다.

원본은 오직 SELECT 만 한다. embedding_models.json 에는 등록만 하고
indexed=false 로 둔다(비교 실험용 표시, 실서비스 안전장치와 무충돌).

멱등성: 몇 번을 다시 돌려도 안전(먼저 지우고 새로 만든다).

식별자 길이: PostgreSQL 은 테이블명이 63바이트로 제한된다.
    자동으로 네이밍될 때 길이를 63바이트로 안전하게 자르도록 처리되었다.

사용:
    # 1. 기본 화이트리스트 테이블 일괄 생성 (기본 배치 사이즈 64)
    python build_compare_embedding_tax.py --model BAAI/bge-m3

    # 2. 특정 커스텀 테이블 1:1 지정 생성 (메모리 제어 포함)
    python build_compare_embedding_tax.py --model BAAI/bge-m3 --source-table custom_cases_2026 --batch-size 32

    # 3. 테스트용 DB를 타겟으로 지정하여 생성
    python build_compare_embedding_tax.py --model BAAI/bge-m3 --db-name tax_db_test
"""


# =====================================================================
# 마스터 비교용 임베딩 테이블 생성기 (build_compare_embedding_tax.py)
# =====================================================================
# [설계 및 작동 원리]
# 1. 목적: 새로운 임베딩 모델(예: BAAI/bge-m3)의 검색 성능을 테스트하기 위해, 원본 데이터는 건드리지 않고 '문서 ID'와 '새 임베딩 벡터'만 담은 가벼운 비교용 복제 테이블을 생성합니다.
# 2. 데이터 무결성(FK): 새 테이블의 doc_id는 원본 테이블의 id를 참조(REFERENCES)합니다. 원본 데이터가 삭제되면 복제된 임베딩도 연쇄 삭제(CASCADE)되어 DB 찌꺼기가 남지 않습니다.
# 3. 유연성: schema.py의 하드코딩된 화이트리스트에 얽매이지 않고, 사용자가 --source-table을 통해 지정한 커스텀 테이블에도 자유롭게 임베딩 비교 테이블을 붙일 수 있습니다.
# 4. 메모리 통제: 무거운 최신 모델 사용 시 발생하는 GPU 메모리(VRAM) 부족 현상을 방지하기 위해 --batch-size 옵션으로 연산 부하를 직접 제어합니다.
# =====================================================================

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import re
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# [설정] 기본 화이트리스트 및 명명 규칙 (schema.py 연동)
# ---------------------------------------------------------
# 기존 파이프라인의 안전장치를 존중하되, 파일이 없어도 단독 실행되도록 예외 처리함.
try:
    from schema import ALLOWED_SOURCE_TABLES as _SEARCH_SOURCES
    from schema import CMP_INFIX, cmp_table as cmp_table_name
    ALLOWED_SOURCE_TABLES = set(_SEARCH_SOURCES)
except ImportError:
    ALLOWED_SOURCE_TABLES = {"tax_laws_master", "tax_precedents_master", "tax_tribunal_master", "tax_amendments_2026"}
    CMP_INFIX = "_cmp_"
    # PostgreSQL의 식별자 최대 길이(63바이트)를 넘지 않도록 안전하게 자름
    def cmp_table_name(src, model): return f"{src}{CMP_INFIX}{model.split('/')[-1].replace('-', '_')}"[:63]

CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedding_models.json")

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

def _safe_identifier(name: str, label: str) -> str:
    """f-string으로 SQL/접속 URI에 직접 삽입되는 값이 안전한 식별자인지 검증한다.
    --source-table 은 화이트리스트를 벗어난 커스텀 테이블도 허용하지만, SQL 인젝션을
    막기 위해 식별자 형식(영문/숫자/밑줄, 63바이트 이내) 자체는 항상 강제한다."""
    if not _IDENT_RE.match(name):
        raise SystemExit(f"[FATAL ERROR] {label} 값이 안전하지 않습니다: {name!r} "
                          f"(영문/숫자/밑줄만 허용, 63자 이내, 숫자로 시작 불가)")
    return name

# ---------------------------------------------------------
# [유틸리티] 차원 감지 및 카탈로그 등록
# ---------------------------------------------------------
def detect_dim(model_id: str) -> int:
    """새로운 모델의 출력 벡터 차원(예: 768, 1024)을 동적으로 알아내는 함수."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    print(f"[DIM] 차원 자동 감지 중... ({model_id})")
    
    # 모델을 메모리에 올리고 샘플 문장을 던져서 튀어나오는 리스트의 길이를 잼
    model = HuggingFaceEmbedding(model_name=model_id)
    dim = len(model.get_text_embedding("차원 확인용 샘플 문장"))
    print(f"[DIM] 감지된 차원: {dim}")
    return dim

def register_catalog(model_id: str) -> None:
    """프론트엔드(UI)나 검색 엔진에서 이 모델이 '비교 실험용'임을 인지하도록 JSON 카탈로그에 기록."""
    rows = []
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH, encoding="utf-8") as f: rows = json.load(f)
        
    # 이미 등록되어 있으면 무시 (중복 방지)
    if any(row.get("id") == model_id for row in rows): return
    
    label = model_id.split("/")[-1]
    # indexed: False로 두어 실서비스 메인 검색망에 엮이지 않도록 차단
    rows.append({"id": model_id, "label": f"{label} (비교용)", "indexed": False})
    
    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[CATALOG] '{model_id}' 등록 완료 (비교용 표시)")

# ---------------------------------------------------------
# [DB 로직] 테이블 생성 및 데이터 적재
# ---------------------------------------------------------
def create_cmp_table(conn, table_name: str, source_table: str, dim: int) -> None:
    """단일 DB 환경에 맞춰 외래키(FK)를 강제하여 비교 테이블을 생성하는 함수."""
    with conn.cursor() as cur:
        # 기존에 같은 이름의 실험 테이블이 있으면 깨끗하게 날림 (멱등성 보장)
        cur.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE;')
        
        # 핵심 로직: 원본(source_table)의 id를 참조하는 doc_id를 생성. 
        # 원본 레코드가 지워지면 이 테이블의 레코드도 같이 날아감(ON DELETE CASCADE).
        cur.execute(f"""
            CREATE TABLE "{table_name}" (
                doc_id TEXT PRIMARY KEY REFERENCES "{source_table}"(id) ON DELETE CASCADE,
                embedding VECTOR({dim})
            );
        """)
    conn.commit()
    print(f"[CREATE] {table_name} (dim={dim}, 원본 연쇄삭제(CASCADE) 외래키 적용 완료)")

def embed_and_insert(conn, table_name: str, rows: list[tuple[str, str]], model_id: str, batch_size: int) -> int:
    """텍스트를 벡터로 변환(임베딩)하고 새로 만든 테이블에 꽂아 넣는 함수."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    
    print(f"[LOAD] 모델 로드 중: {model_id} (배치 사이즈: {batch_size})")
    model = HuggingFaceEmbedding(model_name=model_id)
    has_batch = hasattr(model, "get_text_embedding_batch")

    inserted = 0
    t0 = time.time()
    with conn.cursor() as cur:
        # 튜닝 포인트: 전체 데이터를 한 번에 올리면 OOM이 나므로 batch_size만큼 끊어서 처리
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            texts = [ft for _, ft in batch]

            # 1. 임베딩 추출 (모델이 배치 처리를 지원하면 통째로, 아니면 한 줄씩)
            if has_batch:
                try: 
                    vectors = model.get_text_embedding_batch(texts)
                except Exception: 
                    # 배치에서 텐서 에러가 나면 안전하게 단건 처리로 우회
                    vectors = [model.get_text_embedding(t) for t in texts]
            else:
                vectors = [model.get_text_embedding(t) for t in texts]

            # 2. DB 적재용 페이로드 생성 (문서ID, 벡터문자열)
            payload = [(doc_id, str(vec)) for (doc_id, _), vec in zip(batch, vectors)]
            
            # PostgreSQL 확장인 pgvector 포맷(::vector)으로 캐스팅하여 일괄 삽입
            execute_values(cur, f'INSERT INTO "{table_name}" (doc_id, embedding) VALUES %s', payload, template="(%s, %s::vector)")
            
            inserted += len(payload)
            print(f"\r  └ [{inserted}/{len(rows)}] 적재 중... ({time.time() - t0:.1f}초 경과)", end="", flush=True)
            
    conn.commit()
    print()
    return inserted

def build_one(conn, src_table: str, tgt_table: str, model_id: str, batch_size: int) -> None:
    """하나의 원본 테이블을 읽어 비교용 임베딩 테이블 하나를 완성하는 통합 파이프라인."""
    print("=" * 60)
    print(f"[BUILD] 원본({src_table}) ➔ 타겟({tgt_table}) / 모델: {model_id}")
    
    with conn.cursor() as cur:
        # 원본에서 빈 값이 아닌 실제 텍스트 본문(full_text)만 가져옴
        cur.execute(f'SELECT id, full_text FROM "{src_table}" WHERE full_text IS NOT NULL;')
        rows = cur.fetchall()
        
    if not rows:
        print("[WARN] 원본 테이블이 비어있거나 존재하지 않습니다. 작업을 건너뜁니다.")
        return

    print(f"[FETCH] 원본에서 {len(rows)}건 데이터 로드 완료")
    
    # 1. 모델 차원 파악
    dim = detect_dim(model_id)
    # 2. 껍데기(테이블) 생성
    create_cmp_table(conn, tgt_table, src_table, dim)
    # 3. 알맹이(벡터 데이터) 채우기
    inserted = embed_and_insert(conn, tgt_table, rows, model_id, batch_size)
    
    print(f"[DONE] {tgt_table} : {inserted}건 적재 완료")
    
    # 4. 프론트엔드용 JSON 카탈로그에 기록
    register_catalog(model_id)

# ---------------------------------------------------------
# [메인 제어] CLI 옵션 파싱 및 실행
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="[세무 RAG] 유연한 비교용 임베딩 테이블 생성기 (단일 DB 특화)")
    
    # 1. 필수 모델 제어
    parser.add_argument("--model", required=True, help="임베딩 모델 ID (예: BAAI/bge-m3)")
    parser.add_argument("--batch-size", type=int, default=64, help="GPU VRAM 부족 방지용 연산 단위 (기본 64)")
    
    # 2. DB 및 테이블 제어
    parser.add_argument("--db-name", help="타겟팅할 DB 이름 (미지정시 TAX_DB_URI 환경변수의 기본 DB 사용)")
    parser.add_argument("--source-table", help="특정 원본 테이블 하나만 콕 집어서 돌릴 때 사용")
    parser.add_argument("--target-table", help="생성될 테이블 이름을 커스텀할 때 사용 (미지정시 자동 네이밍)")

    args = parser.parse_args()

    if args.db_name:
        _safe_identifier(args.db_name, "--db-name")
    if args.source_table:
        _safe_identifier(args.source_table, "--source-table")
    if args.target_table:
        _safe_identifier(args.target_table, "--target-table")

    # DB URI 세팅 로직
    base_uri = os.getenv("TAX_DB_URI") or os.getenv("DB_URI")
    if not base_uri:
        print("[FATAL ERROR] 환경변수에 TAX_DB_URI 또는 DB_URI가 없습니다.")
        sys.exit(1)

    # --db-name 옵션이 들어오면 기존 URI의 맨 마지막 경로(DB명)를 교체함
    db_uri = re.sub(r'/[^/]+$', f'/{args.db_name}', base_uri) if args.db_name else base_uri

    print(f"📡 단일 DB 접속 중... 타겟 DB: {db_uri.split('/')[-1]}")
    conn = psycopg2.connect(db_uri)

    try:
        # [모드 A] 사용자가 커스텀 테이블명을 직접 지시한 경우 (화이트리스트 족쇄 해제)
        if args.source_table:
            # 타겟 이름을 안 주면 자동으로 _cmp_ 꼬리표를 붙여서 만들어줌
            target = args.target_table if args.target_table else cmp_table_name(args.source_table, args.model)
            build_one(conn, args.source_table, target, args.model, args.batch_size)
            
        # [모드 B] 아무 지정 없이 돌린 경우 (schema.py의 기본 테이블 4개를 순회하며 일괄 작업)
        else:
            print("[INFO] --source-table 미지정: 기본 화이트리스트 테이블 일괄 작업을 시작합니다.")
            for src in ALLOWED_SOURCE_TABLES:
                # 해당 테이블이 현재 DB에 진짜로 존재하는지 검증 (없으면 튕기지 않고 스킵)
                with conn.cursor() as cur:
                    cur.execute("SELECT to_regclass(%s);", (src,))
                    if not cur.fetchone()[0]: 
                        print(f"[-] {src} 테이블이 존재하지 않아 건너뜁니다.")
                        continue
                        
                tgt = cmp_table_name(src, args.model)
                build_one(conn, src, tgt, args.model, args.batch_size)

    finally:
        conn.close()
        print("\n[+] 모든 비교 임베딩 구축 작업이 안전하게 완료되었습니다.")

if __name__ == "__main__":
    main()