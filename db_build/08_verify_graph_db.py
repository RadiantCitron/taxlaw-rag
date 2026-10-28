# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def check_graph_db():
    print("==================================================")
    print("🔍 [GRAPH SCAN] Neo4j 데이터베이스 전체 스캔 시작")
    print("==================================================")
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
    except Exception as e:
        print(f"[ERROR] Neo4j 접속 실패: {e}")
        return

    with driver.session() as session:
        # 1. 전체 노드 개수 요약 (Tribunal 추가)
        print("\n📊 [1. 노드(Node) 통계]")
        law_count = session.run("MATCH (n:Law) RETURN count(n) AS cnt").single()["cnt"]
        prec_count = session.run("MATCH (n:Precedent) RETURN count(n) AS cnt").single()["cnt"]
        tribunal_count = session.run("MATCH (n:Tribunal) RETURN count(n) AS cnt").single()["cnt"]
        print(f"   ▶ 법령(Law) 노드: {law_count}개")
        print(f"   ▶ 판례(Precedent) 노드: {prec_count}개")
        print(f"   ▶ 조세심판원(Tribunal) 노드: {tribunal_count}개")

        # 2. 전체 엣지 개수 요약 (IMPLEMENTS 추가)
        print("\n📊 [2. 엣지(Edge) 통계]")
        implements_count = session.run("MATCH ()-[r:IMPLEMENTS]->() RETURN count(r) AS cnt").single()["cnt"]
        appealed_count = session.run("MATCH ()-[r:APPEALED_FROM]->() RETURN count(r) AS cnt").single()["cnt"]
        refers_count = session.run("MATCH ()-[r:REFERS_TO]->() RETURN count(r) AS cnt").single()["cnt"]
        similar_count = session.run("MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) AS cnt").single()["cnt"]
        
        print(f"   ▶ 법령 위계(IMPLEMENTS): {implements_count}개")
        print(f"   ▶ 심급 연계(APPEALED_FROM): {appealed_count}개")
        print(f"   ▶ 참조 조문(REFERS_TO): {refers_count}개")
        print(f"   ▶ 의미적 연관(SIMILAR_TO): {similar_count}개")

        # 3. 엣지 연결 상태 상세 출력 (전체 출력)
        print("\n🔗 [3. 엣지(Edge) 연결 상세 내역 (전체 출력)]")
        edges = session.run("""
            MATCH (a)-[r]->(b)
            RETURN labels(a)[0] AS a_label, a.id AS a_id, type(r) AS r_type, labels(b)[0] AS b_label, b.id AS b_id
            ORDER BY r_type, a_id
        """)
        edge_list = list(edges)
        if not edge_list:
            print("   - 생성된 엣지가 없습니다.")
        else:
            print(f"   - 총 {len(edge_list)}개의 엣지가 발견되었습니다. 출력을 시작합니다...\n")
            for record in edge_list:
                print(f"   [{record['a_label']}] {record['a_id']}  --({record['r_type']})-->  [{record['b_label']}] {record['b_id']}")

    driver.close()
    print("\n[DONE] 스캔이 완료되었습니다.")

if __name__ == "__main__":
    check_graph_db()