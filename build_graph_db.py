# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
import os
import re
import time
import psycopg2
from dotenv import load_dotenv
from neo4j import GraphDatabase

# ==========================================
# ⚙️ 설정 및 초기화
# ==========================================
load_dotenv()
TAX_DB_URI = os.getenv("TAX_DB_URI")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

SIMILARITY_THRESHOLD = 0.85



def build_advanced_graph_clean(pg_conn):
    def neo4j_unwind(query, rows, batch_size=300):
        total = 0
        if not rows:
            return 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            result = session.run(query, batch=batch)
            total += result.consume().counters.relationships_created
        return total

    print("\n" + "=" * 50)
    print("🕸️ [GRAPH] Neo4j 데이터베이스 재구축 시작 (내용 기반)")
    print("=" * 50)

    try:
        neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        neo4j_driver.verify_connectivity()
    except Exception as e:
        print(f"[GRAPH ERROR] ❌ Neo4j 접속 실패: {e}")
        return

    start_time = time.time()

    with neo4j_driver.session() as session:
        # ==========================================
        # PHASE 0: 초기화
        # ==========================================
        print("[PHASE 0] 기존 그래프 데이터 초기화 중...")
        session.run("MATCH (n) DETACH DELETE n")
        print("   └ 🗑️ 기존 노드 및 엣지 삭제 완료.\n")

        # ==========================================
        # PHASE 1: 노드 생성
        # ==========================================
        print("[PHASE 1] 노드 생성 중...")
        with pg_conn.cursor() as cur:
            # 1) 법령
            cur.execute(
                """
                SELECT id, law_name, article_number, article_title, full_text
                FROM tax_laws_master;
                """
            )
            laws = cur.fetchall()
            if laws:
                session.run(
                    """
                    UNWIND $batch AS row
                    CREATE (l:Law {
                        id: row[0],
                        law_name: row[1],
                        article_number: row[2],
                        article_title: row[3],
                        full_text: row[4]
                    })
                    """,
                    batch=laws,
                )
            print(f"   └ ✅ 법령(Law) 노드 {len(laws)}개 생성 완료.")

            # 2) 판례
            cur.execute(
                """
                SELECT id, court_level, case_number, case_name,
                       COALESCE(related_lower_cases, ''),
                       COALESCE(summary, '')
                FROM tax_precedents_master;
                """
            )
            precs = cur.fetchall()
            if precs:
                session.run(
                    """
                    UNWIND $batch AS row
                    CREATE (p:Precedent {
                        id: row[0],
                        court_level: row[1],
                        case_number: row[2],
                        case_name: row[3],
                        related_lower_cases: row[4],
                        summary: row[5]
                    })
                    """,
                    batch=precs,
                )
            print(f"   └ ✅ 판례(Precedent) 노드 {len(precs)}개 생성 완료.")

            # 3) 조세심판원
            cur.execute(
                """
                SELECT id,
                       COALESCE(tribunal_id, ''),
                       COALESCE(case_number, ''),
                       COALESCE(case_name, ''),
                       COALESCE(tax_item, ''),
                       COALESCE(summary, ''),
                       COALESCE(related_laws, ''),
                       COALESCE(full_text, '')
                FROM tax_tribunal_master;
                """
            )
            tribunals = cur.fetchall()
            if tribunals:
                session.run(
                    """
                    UNWIND $batch AS row
                    CREATE (t:Tribunal {
                        id: row[0],
                        tribunal_id: row[1],
                        case_number: row[2],
                        case_name: row[3],
                        tax_item: row[4],
                        summary: row[5],
                        related_laws: row[6],
                        full_text: row[7]
                    })
                    """,
                    batch=tribunals,
                )
            print(f"   └ ✅ 조세심판원(Tribunal) 노드 {len(tribunals)}개 생성 완료.")

        print("   └ 🔑 id 인덱스/제약 생성 중...")
        session.run("CREATE CONSTRAINT law_id IF NOT EXISTS FOR (l:Law) REQUIRE l.id IS UNIQUE")
        session.run("CREATE CONSTRAINT prec_id IF NOT EXISTS FOR (p:Precedent) REQUIRE p.id IS UNIQUE")
        session.run("CREATE CONSTRAINT tt_id IF NOT EXISTS FOR (t:Tribunal) REQUIRE t.id IS UNIQUE")
        print("   └ ✅ 제약 생성 완료.")

        # ==========================================
        # PHASE 2: 명시적 인용 관계 (CITES)
        # ==========================================
        print("\n[PHASE 2] 명시적 인용 관계(CITES) 생성 중...")

        with pg_conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, law_name, article_number, full_text
                FROM tax_laws_master
                WHERE law_name LIKE '%시행령' OR law_name LIKE '%시행규칙'
                """
            )
            child_laws = cur.fetchall()

            cur.execute(
                """
                SELECT id, law_name, article_number
                FROM tax_laws_master
                WHERE law_name NOT LIKE '%시행%'
                """
            )
            parent_laws = cur.fetchall()

        cite_pattern = re.compile(
            r"(?:「([^」]+)」)?\s*(?:법\s*)?제\s*(\d+)\s*조(?:의\s*(\d+))?(?:\s*제\s*(\d+)\s*항)?"
        )

        parent_lookup = {}
        for pid, pname, pnum in parent_laws:
            clean_pnum = re.sub(r"[^0-9의]", "", str(pnum or ""))
            parent_lookup[(pname, clean_pnum)] = pid
            parent_lookup[(pname.replace("법", ""), clean_pnum)] = pid

        cite_pairs = []
        for cid, cname, cnum, ctext in child_laws:
            if not ctext:
                continue

            base_name = (
                cname.replace(" 시행령", "").replace(" 시행규칙", "").strip()
            )
            matches = cite_pattern.findall(ctext)

            for m in matches:
                cited_law_name = m[0] if m[0] else base_name
                cited_article = m[1]
                if m[2]:
                    cited_article += f"의{m[2]}"

                key = (cited_law_name, cited_article)
                key2 = (
                    (cited_law_name + "법", cited_article)
                    if not cited_law_name.endswith("법")
                    else None
                )

                parent_id = parent_lookup.get(key) or (
                    parent_lookup.get(key2) if key2 else None
                )
                if parent_id:
                    cite_pairs.append((cid, parent_id))

        cite_pairs = list(set(cite_pairs))

        if cite_pairs:
            created = neo4j_unwind(
                """
                UNWIND $batch AS row
                MATCH (child:Law {id: row[0]}), (parent:Law {id: row[1]})
                MERGE (child)-[:CITES]->(parent)
                """,
                cite_pairs,
                batch_size=50,
            )
            print(f"   └ ✅ 명시적 인용 관계(CITES) {created}개 생성 완료.")
        else:
            print("   └ ⚠️ 명시적 인용 패턴을 찾지 못했습니다.")

        # ==========================================
        # PHASE 3: 벡터 유사도 관계 (SIMILAR_TO)
        # ==========================================
        print("\n[PHASE 3] 벡터 유사도 관계(SIMILAR_TO) 생성 중...")
        with pg_conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT l1.id, l2.id,
                       (1.0 - (l1.embedding <=> l2.embedding)) AS sim_score
                FROM tax_laws_master l1
                JOIN tax_laws_master l2 ON l1.id <> l2.id
                WHERE l1.law_name NOT LIKE '%%시행%%'
                  AND (l2.law_name LIKE '%%시행령' OR l2.law_name LIKE '%%시행규칙')
                  AND (1.0 - (l1.embedding <=> l2.embedding)) >= {SIMILARITY_THRESHOLD}
                """
            )
            law_semantic_pairs = cur.fetchall()

            if law_semantic_pairs:
                created = neo4j_unwind(
                    """
                    UNWIND $batch AS row
                    MATCH (parent:Law {id: row[0]}), (child:Law {id: row[1]})
                    CREATE (child)-[:SIMILAR_TO {score: row[2]}]->(parent)
                    """,
                    law_semantic_pairs,
                    batch_size=50,
                )
                print(f"   └ ✅ 의미적 유사 관계(SIMILAR_TO) {created}개 생성 완료.")

        # ==========================================
        # PHASE 4: 최종 구체화 관계 (CONCRETIZES)
        # ==========================================
        print("\n[PHASE 4] 최종 구체화 관계(CONCRETIZES) 생성 중...")

        session.run(
            """
            MATCH (child:Law)-[:CITES]->(parent:Law)
            MERGE (child)-[:CONCRETIZES {method: 'citation'}]->(parent)
            """
        )

        session.run(
            """
            MATCH (child:Law)-[s:SIMILAR_TO]->(parent:Law)
            WHERE s.score >= $th
              AND NOT EXISTS {
                  MATCH (child)-[:CONCRETIZES]->(parent)
              }
            MERGE (child)-[:CONCRETIZES {
                method: 'high_similarity',
                score: s.score
            }]->(parent)
            """,
            th=SIMILARITY_THRESHOLD,
        )

        result = session.run(
            "MATCH ()-[r:CONCRETIZES]->() RETURN count(r) AS cnt"
        )
        cnt = result.single()["cnt"]
        print(f"   └ ✅ 최종 구체화 관계(CONCRETIZES) {cnt}개 생성 완료.")

        # ==========================================
        # PHASE 5: 판례 / 심판원 관계 (Python 파싱 + 배치)
        # ==========================================
        print("\n[PHASE 5] 판례 및 조세심판원 관계 생성 중...")

        # ---- 공통: 법령 lookup ----
        with pg_conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, law_name, article_number
                FROM tax_laws_master
                WHERE law_name IS NOT NULL AND article_number IS NOT NULL
                """
            )
            law_rows = cur.fetchall()

        law_lookup = {}  # (law_name, article_num) -> id
        law_names = set()
        for lid, lname, anum in law_rows:
            name = (lname or "").strip()
            num = re.sub(r"[^0-9의]", "", str(anum or ""))
            if not name or not num:
                continue
            law_lookup[(name, num)] = lid
            law_lookup[(name.replace(" ", ""), num)] = lid
            law_names.add(name)

        # 긴 법령명 우선 매칭
        law_names_sorted = sorted(law_names, key=len, reverse=True)

        cite_re = re.compile(
            r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?"
        )

        def extract_law_article_pairs(text: str):
            """텍스트에서 (법령명, 조번호) 후보 추출"""
            if not text:
                return set()
            pairs = set()
            # 법령명이 나타난 위치 근처의 제N조를 연결
            for name in law_names_sorted:
                if name not in text:
                    continue
                # 법령명 등장 구간마다 뒤쪽 200자에서 조문 탐색
                start = 0
                while True:
                    idx = text.find(name, start)
                    if idx < 0:
                        break
                    window = text[idx : idx + len(name) + 250]
                    for m in cite_re.finditer(window):
                        art = m.group(1)
                        if m.group(2):
                            art = f"{art}의{m.group(2)}"
                        pairs.add((name, art))
                    start = idx + len(name)
            return pairs

        def resolve_law_ids(pairs):
            ids = set()
            for name, art in pairs:
                art_c = re.sub(r"[^0-9의]", "", art)
                lid = law_lookup.get((name, art_c)) or law_lookup.get(
                    (name.replace(" ", ""), art_c)
                )
                if lid:
                    ids.add(lid)
            return ids

        def neo4j_batch_refers(pairs, start_label):
            """pairs: [(from_id, law_id), ...] → REFERS_TO 배치 생성"""
            if not pairs:
                return 0
            created = 0
            batch_size = 50
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                result = session.run(
                    f"""
                    UNWIND $batch AS row
                    MATCH (a:{start_label} {{id: row[0]}})
                    MATCH (l:Law {{id: row[1]}})
                    MERGE (a)-[:REFERS_TO]->(l)
                    """,
                    batch=batch,
                )
                created += result.consume().counters.relationships_created
            return created

        # ---- 5-1) 판례 상호 참조 (REFERENCES_CASE) ----
        # 전수 cartesian 대신 related_lower_cases 파싱
        with pg_conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, case_number, COALESCE(related_lower_cases, '')
                FROM tax_precedents_master
                """
            )
            prec_rows = cur.fetchall()

        case_to_id = {}
        for pid, cnum, _ in prec_rows:
            cnum = (cnum or "").strip()
            if cnum:
                case_to_id[cnum] = pid

        case_ref_pairs = []
        # 사건번호 패턴 (예: 2002두5351, 95누9174)
        case_pat = re.compile(
            r"\d{2,4}\s*[가-힣]{1,3}\s*\d{1,6}|\d{2,4}[가-힣]{1,3}\d{1,6}"
        )
        for pid, cnum, related in prec_rows:
            if not related:
                continue
            found = set(case_pat.findall(related))
            # 공백 제거 정규화
            for raw in found:
                norm = re.sub(r"\s+", "", raw)
                other = case_to_id.get(norm) or case_to_id.get(raw)
                if other and other != pid:
                    case_ref_pairs.append((pid, other))

        case_ref_pairs = list(set(case_ref_pairs))
        created_case = 0
        batch_size = 500
        for i in range(0, len(case_ref_pairs), batch_size):
            batch = case_ref_pairs[i : i + batch_size]
            result = session.run(
                """
                UNWIND $batch AS row
                MATCH (a:Precedent {id: row[0]})
                MATCH (b:Precedent {id: row[1]})
                MERGE (a)-[:REFERENCES_CASE]->(b)
                """,
                batch=batch,
            )
            created_case += result.consume().counters.relationships_created
        print(f"   └ ✅ 판례 상호 참조 엣지(REFERENCES_CASE) {created_case}개 생성.")

        # ---- 5-2) 판례 → 조문 REFERS_TO ----
        with pg_conn.cursor() as cur:
            cur.execute(
                """
                SELECT id,
                       COALESCE(summary, ''),
                       COALESCE(full_text, ''),
                       COALESCE(referenced_laws, '')
                FROM tax_precedents_master
                """
            )
            prec_text_rows = cur.fetchall()

        prec_ref_pairs = []
        for pid, summary, full_text, ref_laws in prec_text_rows:
            text = "\n".join([summary or "", ref_laws or "", full_text or ""])
            pairs = extract_law_article_pairs(text)
            for lid in resolve_law_ids(pairs):
                prec_ref_pairs.append((pid, lid))
        prec_ref_pairs = list(set(prec_ref_pairs))
        created_prec = neo4j_batch_refers(prec_ref_pairs, "Precedent")
        print(f"   └ ✅ 판례 참조 조문 엣지(REFERS_TO) {created_prec}개 생성.")

        # ---- 5-3) 심판원 → 조문 REFERS_TO ----
        with pg_conn.cursor() as cur:
            cur.execute(
                """
                SELECT id,
                       COALESCE(summary, ''),
                       COALESCE(full_text, ''),
                       COALESCE(related_laws, ''),
                       COALESCE(tax_item, '')
                FROM tax_tribunal_master
                """
            )
            tt_rows = cur.fetchall()

        tt_ref_pairs = []
        for tid, summary, full_text, rel, tax_item in tt_rows:
            text = "\n".join(
                [summary or "", rel or "", full_text or "", tax_item or ""]
            )
            pairs = extract_law_article_pairs(text)
            for lid in resolve_law_ids(pairs):
                tt_ref_pairs.append((tid, lid))
        tt_ref_pairs = list(set(tt_ref_pairs))
        created_tt = neo4j_batch_refers(tt_ref_pairs, "Tribunal")
        print(f"   └ ✅ 심판원 참조 조문 엣지(REFERS_TO) {created_tt}개 생성.")


def main():
    start_time = time.time()

    print("[진행 상황] ⏳ PostgreSQL 데이터베이스(tax_rag_db)에 접속 중...")
    pg_conn = psycopg2.connect(TAX_DB_URI)
    build_advanced_graph_clean(pg_conn)

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("📊 [그래프디비 적재 완료 보고서]")
    print("=" * 50)
    print(f"▶ 소요 시간: {elapsed:.2f}초")
    print("=" * 50)

    pg_conn.close()


if __name__ == "__main__":
    main()