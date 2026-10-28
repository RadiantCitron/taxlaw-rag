# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
# =====================================================================
# 마스터 DB 빌더 및 인제스터 (02_build_db.py)
# =====================================================================
# [설계 및 개편 히스토리]
# 1. 역할 분리: 외부 네트워크 환경에 종속된 수집기(01)와 내부 리소스 기반의 적재기(02)를 분리하여 시스템 안정성을 극대화했습니다.
# 2. 경로 및 명칭 매핑: 수집기의 통제 규칙에 따라 폴더명(cases, tribunal)을 맞췄습니다.
# 3. 유연한 DB 통제: 기존 스키마(VECTOR(768))[cite: 1]는 고정하되, --table-laws, --table-cases, --table-tribunal 옵션을 신설해 타겟 테이블 이름을 실무 환경에 맞게 커스텀할 수 있도록 설계했습니다.
# =====================================================================

from __future__ import annotations
import os
import re
import json
import time
import argparse
import psycopg2
from psycopg2.extras import execute_values, Json
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

# [검색 인덱스 안내] full_text 에 대한 pg_trgm GIN 인덱스는 database/postgres_repo.py의
# apply_search_support() + schema.trigram_index_statements() 가 실제 DDL을 갖고 있다.
# 과거엔 이 인덱스가 scripts/seed_dev_db.py(개발용 시드)에서만 호출돼서, 이 스크립트로
# 실제 운영 DB를 구축하면 인덱스가 전혀 안 생기는 상태였다(하이브리드 검색이 매번
# full_text 전수 스캔으로 느려짐). 그래서 ingest 완료 후 여기서 직접 호출한다.
# ※ 벡터(embedding) 컬럼에는 ivfflat/hnsw 같은 ANN 인덱스를 일부러 안 둔다 — ADR-010 참고.
try:
    from database.postgres_repo import apply_search_support
except ImportError:
    apply_search_support = None

def parse_arguments():
    parser = argparse.ArgumentParser(description="세무 RAG 마스터 적재기 (DB 빌더)")
    
    # 그룹 A: 실행 단계 제어
    parser.add_argument("--step", choices=["parse", "ingest", "all"], default="all")
    parser.add_argument("--target", choices=["laws", "cases", "tribunal", "all"], default="all")
    
    # 그룹 B: 경로 제어 (상위 폴더만 지정하면 하위는 강제 맵핑)
    parser.add_argument("--raw-dir", default="./data/01_raw_data")
    parser.add_argument("--processed-dir", default="./data/02_processed_chunks")
    
    # 그룹 C: DB 제어 (신규 커스텀 테이블 옵션 추가)
    parser.add_argument("--db-name", default="tax_db")
    parser.add_argument("--reset-db", action="store_true")
    parser.add_argument("--table-laws", default="tax_laws_master", help="법령 타겟 테이블")
    parser.add_argument("--table-cases", default="tax_precedents_master", help="판례 타겟 테이블")
    parser.add_argument("--table-tribunal", default="tax_tribunal_master", help="심판원 타겟 테이블")
    
    return parser.parse_args()

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

def _safe_identifier(name: str, label: str) -> str:
    """f-string으로 SQL에 직접 삽입되는 테이블명/DB명이 안전한 식별자인지 검증한다."""
    if not _IDENT_RE.match(name):
        raise SystemExit(f"[FATAL ERROR] {label} 값이 안전하지 않습니다: {name!r} "
                          f"(영문/숫자/밑줄만 허용, 63자 이내, 숫자로 시작 불가)")
    return name

args = parse_arguments()
_safe_identifier(args.table_laws, "--table-laws")
_safe_identifier(args.table_cases, "--table-cases")
_safe_identifier(args.table_tribunal, "--table-tribunal")
_safe_identifier(args.db_name, "--db-name")

load_dotenv()
TAX_DB_URI = os.getenv("TAX_DB_URI")
if TAX_DB_URI and args.db_name != "tax_db":
    TAX_DB_URI = re.sub(r'/[^/]+$', f'/{args.db_name}', TAX_DB_URI)
BATCH_SIZE = 100

def get_files(directory: str) -> list[str]:
    if not os.path.exists(directory) or not os.path.isdir(directory): return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")]

def to_text(v) -> str:
    if v is None: return ""
    if isinstance(v, list): return "\n".join(to_text(x) for x in v if x is not None).strip()
    if isinstance(v, dict):
        for k in ("조문내용", "항내용", "호내용", "목내용", "내용", "텍스트"):
            if k in v and v[k] is not None: return to_text(v[k])
        return ""
    return str(v).strip()

def clean_text(v) -> str:
    s = str(v).strip() if v is not None else ""
    return "" if s in ("(내용없음)", "내용없음", "None", "null") else s

def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def extract_hang_text(hang_list) -> str:
    parts = []
    for h in as_list(hang_list):
        if not isinstance(h, dict):
            continue
        hc = to_text(h.get("항내용"))
        if hc:
            parts.append(hc)
        for ho in as_list(h.get("호")):
            if not isinstance(ho, dict):
                continue
            hoc = to_text(ho.get("호내용"))
            if hoc:
                parts.append(hoc)
            for mok in as_list(ho.get("목")):
                if not isinstance(mok, dict):
                    continue
                mc = to_text(mok.get("목내용"))
                if mc:
                    parts.append(mc)
    return "\n".join(parts)

# ==========================================
# 1. DB 초기화 (동적 테이블명 대응)
# ==========================================
def setup_postgres_safe(conn, reset: bool = False):
    print("\n[INIT] PostgreSQL 테이블 세팅 중...")
    with conn.cursor() as cur:
        if reset:
            print("[INIT] --reset-db 적용: 기존 테이블 드롭")
            cur.execute(f"DROP TABLE IF EXISTS {args.table_laws}, {args.table_cases}, {args.table_tribunal} CASCADE;")
            
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {args.table_laws} (
                id TEXT PRIMARY KEY, law_name TEXT, article_number TEXT,
                article_title TEXT, full_text TEXT, children_json JSONB, embedding VECTOR(768)
            );
            CREATE TABLE IF NOT EXISTS {args.table_cases} (
                id TEXT PRIMARY KEY, court_level TEXT, case_number TEXT,
                case_name TEXT, summary TEXT, full_text TEXT, referenced_laws TEXT,
                related_lower_cases TEXT, embedding VECTOR(768)
            );
            CREATE TABLE IF NOT EXISTS {args.table_tribunal} (
                id TEXT PRIMARY KEY, tribunal_id TEXT, case_name TEXT,
                case_number TEXT, tax_item TEXT, summary TEXT, full_text TEXT,
                related_laws TEXT, embedding VECTOR(768)
            );
        """)
    conn.commit()
    print("[INIT] ✅ 마스터 테이블 세팅 완료")

# ==========================================
# 2. 데이터 변환기 (Transformer)
# ==========================================
def transform_laws(raw_dir: str) -> list[dict]:
    chunks = []
    target_dir = os.path.join(raw_dir, "laws")
    for path in get_files(target_dir):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        law_info = data.get("법령", {})
        if not law_info:
            continue
        basic = law_info.get("기본정보", {}) or {}
        law_name = to_text(basic.get("법령명_한글"))
        articles = as_list((law_info.get("조문") or {}).get("조문단위"))

        seen_ids = set()
        for art in articles:
            if not isinstance(art, dict):
                continue
            art_num = to_text(art.get("조문번호"))
            branch = to_text(art.get("조문가지번호"))  # 의2, 의3 등 가지번호
            content = to_text(art.get("조문내용"))

            # 장/절 목차 헤더 제외 (예: "제1장 총칙")
            head = content[:20] if content else ""
            if (
                content.startswith("제")
                and ("장 " in head or "절 " in head)
                and "조" not in head
                and not art.get("항")
            ):
                continue

            if not art_num and not content:
                continue

            # 조문 번호 조립 (예: 24의2)
            article_number = f"{art_num}의{branch}" if branch else art_num
            if not article_number:
                continue

            # 항/호/목 하위 텍스트 결합
            hang_text = extract_hang_text(art.get("항"))
            body_parts = []
            if content:
                body_parts.append(content)
            if hang_text:
                body_parts.append(hang_text)
            full_text = "\n".join(body_parts).strip()
            if not full_text:
                continue

            # ID 생성 및 중복 방지
            parent_id = f"{law_name}_제{art_num}조의{branch}" if branch else f"{law_name}_제{art_num}조"
            base_id = parent_id
            dup = 1
            while parent_id in seen_ids:
                dup += 1
                parent_id = f"{base_id}__{dup}"
            seen_ids.add(parent_id)

            # 조문제목 보정
            title = to_text(art.get("조문제목") or art.get("조제목"))
            if not title and content:
                m = re.search(r"\(([^)]+)\)", content[:60])
                if m:
                    title = m.group(1)

            # 항 하위 구조 보존
            children = []
            for h in as_list(art.get("항")):
                if not isinstance(h, dict):
                    continue
                p_num = to_text(h.get("항번호"))
                p_content = to_text(h.get("항내용"))
                if not p_content:
                    continue
                child_id = f"{parent_id}_{p_num}항" if p_num else f"{parent_id}_항"
                children.append({
                    "child_id": child_id,
                    "paragraph_number": p_num,
                    "text": p_content,
                })

            chunks.append({
                "id": parent_id,
                "law_name": law_name,
                "article_number": article_number,
                "article_title": title,
                "full_text": full_text,
                "children": children,
            })
    return chunks

# 여러 래핑 키를 순회하며 안전하게 본문 dict를 추출한다.
# 법제처 API는 응답이 루트에 바로 오거나, PrecService/판례/판례정보 등으로
# 한 겹 감싸져 오는 경우가 섞여 있어(과거에 특정 키 하나만 믿고 파싱하다가
# 값이 전부 비어버린 이력이 있음) 후보 키를 순서대로 시도한다.
def _unwrap(body: dict, candidate_keys: tuple[str, ...]) -> dict:
    if not isinstance(body, dict): return {}
    for key in candidate_keys:
        v = body.get(key)
        if isinstance(v, dict) and v: return v
    return body

# 판례 파싱 (경로 cases/precedents 자동 감지)
def transform_cases(raw_dir: str) -> list[dict]:
    chunks = []
    target_dir = os.path.join(raw_dir, "cases")
    if not os.path.exists(target_dir):
        alt = os.path.join(raw_dir, "precedents")
        if os.path.exists(alt):
            target_dir = alt
    for path in get_files(target_dir):
        with open(path, "r", encoding="utf-8") as f: data = json.load(f)
        body = data.get("body", data)
        info = _unwrap(body, ("PrecService", "판례", "판례정보"))

        prec_id = str(info.get("판례정보일련번호") or info.get("판례일련번호") or "").strip()
        case_num = info.get("사건번호", "")
        if not prec_id:
            m = re.search(r"ID(\d+)", os.path.basename(path))
            prec_id = m.group(1) if m else ""
        if not prec_id and not case_num:
            continue

        cid = f"PREC_{prec_id or re.sub(r'[^\\w\\s\\-\\[\\]()]', '', case_num).strip()}"
        full_text = info.get("판례내용") or info.get("이유") or ""

        chunks.append({
            "id": cid, "court_level": info.get("법원명", ""), "case_number": case_num,
            "case_name": info.get("사건명", ""), "summary": info.get("판결요지", ""), "full_text": full_text,
            "referenced_laws": info.get("참조조문", ""), "related_lower_cases": info.get("참조판례", "")
        })
    return chunks

# 심판원 파싱 (경로 tribunal/tax_tribunal 자동 감지)
def transform_tribunal(raw_dir: str) -> list[dict]:
    chunks = []
    target_dir = os.path.join(raw_dir, "tribunal")
    if not os.path.exists(target_dir):
        alt = os.path.join(raw_dir, "tax_tribunal")
        if os.path.exists(alt):
            target_dir = alt
    for path in get_files(target_dir):
        with open(path, "r", encoding="utf-8") as f: data = json.load(f)
        body = data.get("body", data)
        meta = data.get("meta", {}) if isinstance(data, dict) else {}
        info = _unwrap(body, ("SpecialDeccService", "TtSpecialDeccService", "특별행정심판재결례", "재결례"))

        # 분산되어 있던 ID 복원 및 추출 로직 통합
        tid = ""
        m = re.search(r"ID(\d+)", os.path.basename(path), re.IGNORECASE)
        if m: tid = m.group(1)
        if not tid:
            for src in (info, meta):
                for k in ("특별행정심판재결례일련번호", "특별행정심판재결례 일련번호", "일련번호"):
                    v = src.get(k)
                    if v and re.fullmatch(r"\d+", str(v).strip()): tid = str(v).strip()
        if not tid: tid = re.sub(r"[^\w\-]", "_", os.path.splitext(os.path.basename(path))[0])[:60]

        reason = clean_text(info.get("이유"))
        summary = clean_text(info.get("재결요지"))
        order = clean_text(info.get("주문"))
        case_name = clean_text(info.get("사건명")) or clean_text(meta.get("사건명"))

        chunks.append({
            "id": f"TT_{tid}", "tribunal_id": str(tid),
            "case_name": case_name[:80] + "..." if len(case_name) > 120 else case_name,
            "case_number": clean_text(info.get("사건번호")) or clean_text(info.get("청구번호")) or clean_text(meta.get("청구번호")) or clean_text(meta.get("사건번호")),
            "tax_item": clean_text(info.get("세목")),
            "summary": summary or reason[:300],
            "full_text": "\n\n".join([x for x in (summary, order, reason) if x]),
            "related_laws": clean_text(info.get("관련법령"))
        })
    return chunks

# ==========================================
# 3. 통합 적재기 (Loader) - 동적 테이블 맵핑
# ==========================================
def ingest_to_postgres(conn, chunks: list[dict], table_type: str, embed_model):
    if not chunks: return 0
    total = len(chunks)
    
    # 옵션으로 받은 테이블명 맵핑
    table_name = args.table_laws if table_type == "laws" else args.table_cases if table_type == "cases" else args.table_tribunal
    print(f"\n[{table_name.upper()}] 📖 DB 적재 시작 (총 {total}건)")
    
    batch = []
    inserted = 0
    for i, c in enumerate(chunks, 1):
        print(f"\r  └ 임베딩 중... ({i}/{total})", end="", flush=True)
        if table_type == "laws":
            vec = str(embed_model.get_text_embedding(c['full_text']))
            batch.append((c['id'], c['law_name'], c['article_number'], c['article_title'], c['full_text'], Json(c['children']), vec))
            query = f"INSERT INTO {table_name} VALUES %s ON CONFLICT (id) DO NOTHING"
            template = "(%s, %s, %s, %s, %s, %s, %s::vector)"
        elif table_type == "cases":
            vec = str(embed_model.get_text_embedding(c['summary'] or c['full_text']))
            batch.append((c['id'], c['court_level'], c['case_number'], c['case_name'], c['summary'], c['full_text'], c['referenced_laws'], c['related_lower_cases'], vec))
            query = f"INSERT INTO {table_name} VALUES %s ON CONFLICT (id) DO NOTHING"
            template = "(%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)"
        elif table_type == "tribunal":
            vec = str(embed_model.get_text_embedding(c['summary'] or c['full_text']))
            batch.append((c['id'], c['tribunal_id'], c['case_name'], c['case_number'], c['tax_item'], c['summary'], c['full_text'], c['related_laws'], vec))
            query = f"INSERT INTO {table_name} VALUES %s ON CONFLICT (id) DO NOTHING"
            template = "(%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)"

        if len(batch) >= BATCH_SIZE or i == total:
            with conn.cursor() as cur:
                execute_values(cur, query, batch, template=template)
            conn.commit()
            inserted += len(batch)
            batch = []
            
    print(f"\r  └ ✅ 적재 완료: {inserted}건          ")
    return inserted

def main():
    print("==================================================")
    print("🚀 [세무 RAG] 마스터 DB 빌더 가동")
    print("==================================================")
    
    if args.step in ["parse", "all"]:
        print("\n[PHASE 1] 원본 파싱 및 로컬 저장")
        os.makedirs(args.processed_dir, exist_ok=True)
        if args.target in ["laws", "all"]:
            chunks = transform_laws(args.raw_dir)
            with open(os.path.join(args.processed_dir, "laws_processed.json"), "w", encoding="utf-8") as f: json.dump(chunks, f, ensure_ascii=False)
        if args.target in ["cases", "all"]:
            chunks = transform_cases(args.raw_dir)
            with open(os.path.join(args.processed_dir, "cases_processed.json"), "w", encoding="utf-8") as f: json.dump(chunks, f, ensure_ascii=False)
        if args.target in ["tribunal", "all"]:
            chunks = transform_tribunal(args.raw_dir)
            with open(os.path.join(args.processed_dir, "tribunal_processed.json"), "w", encoding="utf-8") as f: json.dump(chunks, f, ensure_ascii=False)
        
        if args.step == "parse": return

    if args.step in ["ingest", "all"]:
        print("\n[PHASE 2] 모델 로드 및 DB 적재")
        embed_model = HuggingFaceEmbedding(model_name="jhgan/ko-sroberta-multitask")
        conn = psycopg2.connect(TAX_DB_URI)
        
        try:
            setup_postgres_safe(conn, reset=args.reset_db)
            if args.target in ["laws", "all"]:
                with open(os.path.join(args.processed_dir, "laws_processed.json"), "r", encoding="utf-8") as f:
                    ingest_to_postgres(conn, json.load(f), "laws", embed_model)
            if args.target in ["cases", "all"]:
                with open(os.path.join(args.processed_dir, "cases_processed.json"), "r", encoding="utf-8") as f:
                    ingest_to_postgres(conn, json.load(f), "cases", embed_model)
            if args.target in ["tribunal", "all"]:
                with open(os.path.join(args.processed_dir, "tribunal_processed.json"), "r", encoding="utf-8") as f:
                    ingest_to_postgres(conn, json.load(f), "tribunal", embed_model)

            # [PHASE 3] 검색 인덱스 적용: full_text 컬럼에 pg_trgm GIN 인덱스를 건다.
            # 데이터는 이미 커밋됐으므로, 인덱스 적용이 실패해도 적재 결과는 그대로 유지하고
            # 경고만 남긴다(재실행 시 CREATE INDEX IF NOT EXISTS라 안전하게 재시도 가능).
            print("\n[PHASE 3] 검색 인덱스(pg_trgm GIN) 적용 중...")
            if apply_search_support:
                try:
                    apply_search_support(conn, include_indexes=True)
                    print("  └ ✅ 인덱스 적용 완료")
                except Exception as e:
                    print(f"  └ ⚠️ 인덱스 적용 실패(데이터 적재는 정상 완료됨): {e}")
            else:
                print("  └ ⚠️ database.postgres_repo를 불러올 수 없어 건너뜁니다. "
                      "저장소 루트에서 실행 중인지 확인하거나, schema.trigram_index_statements()를 수동 실행하세요.")
        finally:
            conn.close()

if __name__ == "__main__":
    main()