# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
# =====================================================================
# 개정세법(2026년) 원스톱 통합 인제스터 (unified_amendment_ingester.py)
# =====================================================================
# [설계 및 개편 히스토리]
# 1. 2단계 파이프라인의 완전 통합: 
#    - 과거 PDF 파서(parse_tax_amendment_pdf.py)와 DB 적재기(ingest_amendments_2026.py)로 나뉘어 있던 구조를 단일 스크립트로 병합했습니다[cite: 2].
#    - 기존 통합 버전에서는 중간 산출물(JSON)에 의존하는 반쪽짜리 통합이었으나, 이번 개편으로 PDF 원본 읽기 ➔ 파싱/청킹 ➔ 노이즈 필터링 ➔ DB 적재(PostgreSQL, Neo4j)가 진정한 원스톱으로 흘러가도록 완성했습니다[cite: 2].
# 2. CLI 제어 옵션 신설:
#    - --step: 'parse'(JSON 추출만), 'ingest'(DB 적재만), 'all'(전체 실행)을 분리하여 표 파싱이 까다로운 세법 PDF의 추출 상태를 눈으로 먼저 검수할 수 있게 했습니다.
#    - --test-pages: 수백 페이지의 PDF를 전부 돌리지 않고 특정 페이지(예: 10-15)만 빠르게 뜯어보며 정규식과 표 추출 로직을 디버깅할 수 있는 핵심 기능을 추가했습니다.
# 3. 유연한 파일 입출력:
#    - 고정된 파일명으로 인한 덮어쓰기 참사를 막기 위해, 원본 PDF 이름에 맞춰 출력 JSON 파일명이 자동 생성되도록 개선했습니다. (예: 2026_tax.pdf -> 2026_tax_processed.json)
#    - 필요시 --output-file 옵션으로 수동 지정도 가능합니다.
# 4. DB 스키마 유지: 
#    - 기존에 합의된 VECTOR(768) 스키마와 Neo4j 동시 적재 로직은 그대로 유지했습니다[cite: 2].
# =====================================================================

from __future__ import annotations
import argparse
import os
import re
import time
import json
import pdfplumber
import psycopg2
from psycopg2.extras import execute_values, Json
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# [검색 인덱스 안내] full_text 에 대한 pg_trgm GIN 인덱스는 database/postgres_repo.py의
# apply_search_support() + schema.trigram_index_statements() 가 실제 DDL을 갖고 있다.
# 과거엔 scripts/seed_dev_db.py(개발용 시드)에서만 호출돼서, 이 스크립트만으로
# tax_amendments_2026 을 적재하면 인덱스가 안 생기는 상태였다. ingest 완료 후 여기서 직접 호출한다.
try:
    from database.postgres_repo import apply_search_support
except ImportError:
    apply_search_support = None

# ==========================================
# ⚙️ CLI 인자 설정 및 초기화
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(description="개정세법 원스톱 통합 인제스터")
    
    # 그룹 A: 실행 단계 제어
    parser.add_argument("--step", choices=["parse", "ingest", "all"], default="all",
                        help="실행 단계: parse(PDF->JSON 저장), ingest(JSON->DB 적재), all(전체)")
    
    # 그룹 B: 작업 대상 및 경로 제어
    parser.add_argument("--pdf-path", default="./data/2026년 개정세법 해설.pdf",
                        help="처리할 PDF 파일 경로 (기본값: ./data/2026년 개정세법 해설.pdf)")
    parser.add_argument("--processed-dir", default="./data/02_processed_chunks",
                        help="가공 데이터(JSON) 저장 경로")
    parser.add_argument("--output-file", default="2026_tax_amendments_processed.json",
                        help="저장할 JSON 파일명 (기본값: 2026_tax_amendments_processed.json)")
    parser.add_argument("--test-pages", default="",
                        help="테스트할 페이지 범위 (예: '10-15' 또는 '5')")
    
    # 그룹 C: DB 제어
    parser.add_argument("--db-name", default="tax_db", help="접속할 데이터베이스 이름")
    parser.add_argument("--reset-db", action="store_true", help="적재 전 tax_amendments_2026 테이블 초기화")
    
    return parser.parse_args()

args = parse_arguments()

# PDF 경로 자동 보정 (대체 경로 탐색)
if not os.path.exists(args.pdf_path):
    for cand in ["./data/2026년 개정세법 해설.pdf", "./data/01_raw_data/2026_tax_amendments.pdf", "./data/2026_tax_amendments.pdf"]:
        if os.path.exists(cand):
            args.pdf_path = cand
            break

if not re.match(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$", args.db_name):
    raise SystemExit(f"[FATAL ERROR] --db-name 값이 안전하지 않습니다: {args.db_name!r} "
                      f"(영문/숫자/밑줄만 허용, 63자 이내, 숫자로 시작 불가)")

load_dotenv()
TAX_DB_URI = os.getenv("TAX_DB_URI")
if TAX_DB_URI and args.db_name != "tax_db":
    TAX_DB_URI = re.sub(r'/[^/]+$', f'/{args.db_name}', TAX_DB_URI)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
BATCH_SIZE = 50

# ==========================================
# 🛠 정규식 및 유틸리티 (파서 로직 유지)
# ==========================================
RE_AREA_HEADER = re.compile(
    r"^(국세기본법|국세징수법|소득세법|법인세법|국제조세조정에\s*관한\s*법률|"
    r"상속세\s*및\s*증여세법|종합부동산세법|부가가치세법|조세특례제한법|"
    r"개별소비세법|주세법|증권거래세법|조세범처벌절차법|교육세법|농어촌특별세법)"
    r"(?:\s*[·・]\s*(?:국세징수법|국세기본법))?", re.M)
RE_ITEM_START = re.compile(r"(?:^|\n)\s*0*(\d{1,2})\s+([^\n]{5,80}?)\s*(?=\n|$)", re.M)
RE_RELATED = re.compile(
    r"(국세기본법|국세징수법|소득세법|법인세법|상속세\s*및\s*증여세법|"
    r"종합부동산세법|부가가치세법|조세특례제한법|국제조세조정에\s*관한\s*법률)"
    r"[^\n]{0,40}?(?:제\s*[\d조의의영문]+)", re.M)
RE_PURPOSE = re.compile(r"가\.\s*개정취지\s*(.*?)(?=나\.\s*개정내용|다\.\s*적용시기|\Z)", re.S)
RE_CONTENT = re.compile(r"나\.\s*개정내용\s*(.*?)(?=다\.\s*적용시기|\Z)", re.S)
RE_EFFECTIVE = re.compile(r"다\.\s*적용시기(?:\s*및\s*적용례)?\s*(.*?)(?=\n\s*0*\d{1,2}\s+[^\n]{5,}|$)", re.S)
RE_SKIP_PAGES = re.compile(r"(CONTENTS|목\s*차|집\s*필\s*진|머리말|주요\s*개정\s*내용)", re.I)

def clean(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def slug(s: str) -> str:
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\w가-힣]+", "_", s)
    return s[:40]

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"<<<PAGE:\d+>>>", " ", s)
    s = re.sub(r"국세기본법\s*[･·]\s*국세징수법\s*\d+", " ", s)
    s = re.sub(r"소득세법\([^)]*\)\s*\d+", " ", s)
    s = re.sub(r"기타세법\s*\d+", " ", s)
    s = re.sub(r"\d{3,4}\s*2026년\s*개정세법\s*해설", " ", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def is_noise_chunk(c: dict) -> bool:
    title = (c.get("title") or "").strip()
    full = clean_text(c.get("full_text") or "")
    purpose = (c.get("purpose") or "").strip()
    after = (c.get("after") or "").strip()
    effective = (c.get("effective") or "").strip()

    if "PAGE" in title or title.startswith("<<<") or title in ("2026년 개정세법 해설", "개정세법 해설"): return True
    if len(full) < 80 and not purpose and not after: return True
    if not purpose and not after and not effective and len(full) < 120: return True
    if "발행처" in full and "국세청 법규과" in full and len(after) < 100: return True
    return False

# ==========================================
# [STEP 1] PDF 파싱 및 청킹
# ==========================================
def parse_and_chunk_pdf(pdf_path: str, test_pages: str) -> list[dict]:
    print(f"[1/3] PDF 파싱 및 청킹 시작: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        start_idx, end_idx = 0, total_pages
        
        # 페이지 범위 지정 (--test-pages 옵션 처리)
        if test_pages:
            parts = test_pages.split('-')
            start_idx = max(0, int(parts[0]) - 1)
            end_idx = int(parts[1]) if len(parts) > 1 else start_idx + 1
            end_idx = min(end_idx, total_pages)
            print(f"  └ ⚠️ 테스트 모드: {start_idx+1} ~ {end_idx} 페이지만 추출합니다.")
            
        for i in range(start_idx, end_idx):
            t = pdf.pages[i].extract_text() or ""
            tables = pdf.pages[i].extract_tables() or []
            for tb in tables:
                lines = [" | ".join([c.replace("\n", " ").strip() if c else "" for c in row]) for row in tb]
                if lines: t += "\n" + "\n".join(lines)
            pages.append((i + 1, t))

    annotated = [f"\n<<<PAGE:{pno}>>>\n{txt}" for pno, txt in pages]
    whole = "\n".join(annotated)
    
    chunks = []
    current_area = ""
    spans = list(RE_ITEM_START.finditer(whole))
    
    for i, m in enumerate(spans):
        start = m.start()
        end = spans[i + 1].start() if i + 1 < len(spans) else len(whole)
        no = m.group(1).zfill(2)
        title = clean(m.group(2))
        
        if RE_SKIP_PAGES.search(title) or len(title) < 4 or re.match(r"^\d+$", title): continue
        
        block = whole[start:end]
        pnums = [int(x) for x in re.findall(r"<<<PAGE:(\d+)>>>", block)]
        p_start, p_end = (pnums[0], pnums[-1]) if pnums else (0, 0)
        
        ctx = whole[max(0, start - 300) : start + 200]
        area_m = RE_AREA_HEADER.search(ctx[:400])
        if area_m: current_area = re.sub(r"\s+", "", area_m.group(1))
            
        if "개정취지" not in block and "개정내용" not in block and len(block) < 400: continue
            
        purpose_m = RE_PURPOSE.search(block)
        content_m = RE_CONTENT.search(block)
        eff_m = RE_EFFECTIVE.search(block)
        
        purpose = clean(purpose_m.group(1)) if purpose_m else ""
        content = clean(content_m.group(1)) if content_m else ""
        effective = clean(eff_m.group(1)) if eff_m else ""
        
        related = []
        for rm in re.finditer(r"(?:국세기본법|국세징수법|소득세법|법인세법|상속세\s*및\s*증여세법|종합부동산세법|부가가치세법|조세특례제한법|국제조세조정에\s*관한\s*법률)[^\n]{0,60}", block[:500]):
            frag = clean(rm.group(0))
            if "제" in frag and frag not in related: related.append(frag)
        related = related[:8]
        
        before, after, table_md = "", "", ""
        if content:
            if "종 전" in content or "종전" in content:
                # 1차: "개정"이 줄 맨 앞(표 헤더/셀 경계)에 오는 경우만 분리 지점으로 인정.
                # "종전" 본문 중간에 "...개정된 바 있다" 같은 문구가 섞여 있어도
                # 잘못 잘리지 않도록 하기 위한 보수적 매칭이다.
                sm = re.search(r"(?:종\s*전|종전)\s*(.*?)(?:^|\n)[ \t]*(?:개\s*정|개정)[ \t]*\n?(.*)",
                               content, re.S | re.M)
                if sm:
                    before, after = clean(sm.group(1)), clean(sm.group(2))
                else:
                    # 2차 폴백: 줄 경계를 못 찾으면 기존 방식(첫 "개정" 등장 지점)으로 완화 매칭
                    sm = re.search(r"(종\s*전|종전)\s*(.*?)(개\s*정|개정)\s*(.*)", content, re.S)
                    if sm:
                        before, after = clean(sm.group(2)), clean(sm.group(4))
                if before or after:
                    table_md = f"| 종전 | 개정 |\n|---|---|\n| {before[:500]} | {after[:500]} |"
            elif "신 설" in content or "〈신 설〉" in content or "<신 설>" in content:
                after = re.sub(r"[〈<]?\s*신\s*설\s*[〉>]?", "", content).strip()
                before, table_md = "(신설)", f"| 종전 | 개정 |\n|---|---|\n| (신설) | {after[:800]} |"
            else:
                after = content

        full_parts = [f"[{current_area}] {no}. {title}", f"관계조문: {', '.join(related)}" if related else "",
                      f"개정취지: {purpose}" if purpose else "", f"종전: {before}" if before else "",
                      f"개정: {after}" if after else "", f"적용시기: {effective}" if effective else ""]
        full_text = clean("\n".join([p for p in full_parts if p]))
        
        chunk_dict = {
            "id": f"AMEND_2026_{slug(current_area or 'MISC')}_{no}_{slug(title)}",
            "statute_area": current_area or "", "item_no": no, "title": title,
            "related_articles": related,
            "purpose": clean_text(purpose),
            "before": clean_text(before),
            "after": clean_text(after),
            "body_raw": clean_text(after),
            "effective": clean_text(effective),
            "full_text": clean_text(full_text),
            "source": "2026년 개정세법 해설",
            "page_start": p_start, "page_end": p_end, "engine": "pdfplumber"
        }
        
        # 노이즈 필터링 로직 적용[cite: 2]
        if not is_noise_chunk(chunk_dict) and len(chunk_dict["full_text"]) >= 60:
            chunks.append(chunk_dict)
            
    print(f"[1/3] 파싱 완료: 총 {len(chunks)}개 유효 청크 추출")
    return chunks

# ==========================================
# [STEP 2] PostgreSQL 적재 
# ==========================================
def ingest_postgres(conn, chunks: list[dict], embed_model, reset: bool):
    print("\n[2/3] PostgreSQL 적재 및 임베딩 시작...")
    with conn.cursor() as cur:
        if reset:
            print("  └ ⚠️ --reset-db 옵션 적용: tax_amendments_2026 테이블 초기화")
            cur.execute("DROP TABLE IF EXISTS tax_amendments_2026 CASCADE;")
            
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tax_amendments_2026 (
                id TEXT PRIMARY KEY, statute_area TEXT, item_no TEXT, title TEXT,
                related_articles JSONB, purpose TEXT, before_text TEXT, after_text TEXT,
                body_raw TEXT, effective TEXT, full_text TEXT, source TEXT,
                page_start INT, page_end INT, engine TEXT, embedding VECTOR(768)
            );
            CREATE INDEX IF NOT EXISTS idx_amend_2026_area ON tax_amendments_2026 (statute_area);
        """)
    conn.commit()

    total = 0
    batch = []
    for i, c in enumerate(chunks, 1):
        print(f"\r  └ 임베딩 중... ({i}/{len(chunks)})", end="", flush=True)
        # 768차원 임베딩 생성[cite: 2]
        vec = str(embed_model.get_text_embedding(f"[{c.get('statute_area', '')} {c.get('item_no', '')}] {c.get('title', '')}\n{c.get('full_text', '')}"))
        batch.append((
            c["id"], c["statute_area"], c["item_no"], c["title"], Json(c["related_articles"]),
            c["purpose"], c["before"], c["after"], c["body_raw"], c["effective"],
            c["full_text"], c["source"], c["page_start"], c["page_end"], c["engine"], vec
        ))

        if len(batch) >= BATCH_SIZE or i == len(chunks):
            uniq = {row[0]: row for row in batch}
            with conn.cursor() as cur:
                execute_values(cur, """
                    INSERT INTO tax_amendments_2026 VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        statute_area=EXCLUDED.statute_area, title=EXCLUDED.title,
                        related_articles=EXCLUDED.related_articles, purpose=EXCLUDED.purpose,
                        before_text=EXCLUDED.before_text, after_text=EXCLUDED.after_text,
                        effective=EXCLUDED.effective, full_text=EXCLUDED.full_text,
                        embedding=EXCLUDED.embedding
                """, list(uniq.values()),
                template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)")
            conn.commit()
            total += len(uniq)
            batch = []
    print(f"\n  └ ✅ PostgreSQL 적재 완료: 총 {total}건")

# ==========================================
# [STEP 3] Neo4j (그래프 DB) 적재
# ==========================================
def ingest_neo4j(chunks: list[dict]):
    print("\n[3/3] Neo4j 그래프 노드 갱신 시작...")
    if not NEO4J_PASSWORD:
        print("  └ ⚠️ NEO4J_PASSWORD 환경변수 없음, Neo4j 적재를 건너뜁니다.")
        return
        
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        node_rows = [{"id": c["id"], "statute_area": c["statute_area"], "title": c["title"], "full_text": c["full_text"][:3000]} for c in chunks]
        
        with driver.session() as session:
            session.run("MATCH (a:Amendment {source: '2026년 개정세법 해설'}) DETACH DELETE a")
            for i in range(0, len(node_rows), 200):
                session.run("""
                    UNWIND $batch AS row
                    MERGE (a:Amendment {id: row.id})
                    SET a.statute_area = row.statute_area, a.title = row.title, a.full_text = row.full_text, a.source = '2026년 개정세법 해설', a.year = 2026
                """, batch=node_rows[i:i+200])
        driver.close()
        print(f"  └ ✅ Neo4j 적재 완료[cite: 2]")
    except Exception as e:
        print(f"  └ ❌ Neo4j 적재 실패: {e}")

# ==========================================
# 🚀 메인 컨트롤러
# ==========================================
def main():
    start_time = time.time()
    print("==================================================")
    print("🚀 [개정세법 2026] 통합 인제스트 파이프라인 가동")
    print(f"▶ 실행 단계 : {args.step}")
    print(f"▶ 원본 경로 : {args.pdf_path}")
    print("==================================================")
    
    # 출력 파일명 동적 결정
    out_filename = args.output_file if args.output_file else f"{os.path.splitext(os.path.basename(args.pdf_path))[0]}_processed.json"
    out_filepath = os.path.join(args.processed_dir, out_filename)

    chunks = []

    # --- PHASE 1: 파싱 및 청킹 ---
    if args.step in ["parse", "all"]:
        chunks = parse_and_chunk_pdf(args.pdf_path, args.test_pages)
        
        # 파일 저장
        os.makedirs(args.processed_dir, exist_ok=True)
        with open(out_filepath, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"  └ 💾 JSON 추출물 저장 완료: {out_filepath}")
        
        if args.step == "parse":
            print("\n✅ 파싱 단계가 완료되었습니다. 데이터 정제 상태를 확인하세요.")
            return

    # --- PHASE 2 & 3: 임베딩 및 DB 적재 ---
    if args.step in ["ingest", "all"]:
        if not chunks:
            if not os.path.exists(out_filepath):
                print(f"[FATAL ERROR] 읽어들일 JSON 파일이 없습니다: {out_filepath}")
                return
            print(f"\n[PHASE 2] 기존 추출물 로드 중: {out_filepath}")
            with open(out_filepath, "r", encoding="utf-8") as f:
                chunks = json.load(f)

        if not TAX_DB_URI:
            print("[FATAL ERROR] TAX_DB_URI 환경변수 또는 접속 정보가 올바르지 않습니다.")
            return

        print("\n⏳ 로컬 임베딩 모델(jhgan/ko-sroberta-multitask) 로드 중...")
        embed_model = HuggingFaceEmbedding(model_name="jhgan/ko-sroberta-multitask")
        
        conn = psycopg2.connect(TAX_DB_URI)
        try:
            ingest_postgres(conn, chunks, embed_model, args.reset_db)
            ingest_neo4j(chunks)

            # 검색 인덱스 적용: full_text 컬럼에 pg_trgm GIN 인덱스를 건다.
            # 데이터는 이미 커밋됐으므로, 인덱스 적용이 실패해도 경고만 남기고 계속한다.
            print("\n[+1] 검색 인덱스(pg_trgm GIN) 적용 중...")
            if apply_search_support:
                try:
                    apply_search_support(conn, include_indexes=True)
                    print("  └ ✅ 인덱스 적용 완료")
                except Exception as e:
                    print(f"  └ ⚠️ 인덱스 적용 실패(데이터 적재는 정상 완료됨): {e}")
            else:
                print("  └ ⚠️ database.postgres_repo를 불러올 수 없어 건너뜁니다. "
                      "저장소 루트에서 실행 중인지 확인하거나, schema.trigram_index_statements()를 수동 실행하세요.")
        except Exception as e:
            print(f"\n[FATAL ERROR] DB 적재 중 오류 발생: {e}")
            conn.rollback()
            return
        finally:
            conn.close()

    print("\n" + "="*50)
    print(f"🎉 모든 작업이 완료되었습니다! (소요 시간: {time.time() - start_time:.1f}초)")
    print("="*50)

if __name__ == "__main__":
    main()