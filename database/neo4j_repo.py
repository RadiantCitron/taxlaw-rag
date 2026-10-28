# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
import os
import logging
try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

logger = logging.getLogger("my_rag")

class Neo4jRepo:
    """ Neo4j 그래프 데이터베이스와의 통신을 전담하는 Repository 클래스입니다. """
    
    # 🌟 [추가됨] 클래스 변수(정적 변수): 프로그램 전체에서 공유할 단 하나의 드라이버(풀)
    driver = None

    # 🌟 [추가됨] 서버 시작 시 드라이버(풀)를 생성하는 클래스 메서드
    @classmethod
    def init_driver(cls):
        if cls.driver is None:
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD")
            try:
                cls.driver = GraphDatabase.driver(uri, auth=(user, password))
                cls.driver.verify_connectivity()
                logger.info("[GRAPH] Neo4j 드라이버(커넥션 풀) 생성 완료")
            except Exception as e:
                logger.error(f"[GRAPH ERROR] Neo4j 접속 실패 (GraphRAG 비활성화됨): {e}")
                cls.driver = None

    # 🌟 [추가됨] 서버 종료 시 드라이버를 닫는 클래스 메서드
    @classmethod
    def close_driver(cls):
        if cls.driver:
            cls.driver.close()
            logger.info("[GRAPH] Neo4j 드라이버 종료 완료")

    def __init__(self):
        # 🌟 [수정됨] 객체를 생성할 때마다 새로 드라이버를 만들지 않고, 클래스 변수를 그대로 씁니다.
        self.driver = Neo4jRepo.driver

    def close(self):
        # 🌟 [수정됨] 여기서 드라이버를 닫으면 안 됩니다! (다른 요청에서도 써야 하므로)
        # 기존 코드(retriever.py 등)에서 repo.close()를 호출해도 에러가 나지 않도록 pass 처리합니다.
        pass

    def get_graph_context(self, law_ids: list, limit_per_type: int = 2) -> str:
        """
        RDB에서 찾은 법령 ID 리스트를 받아, 연결된 시행령/규칙 및 판례를 찾아 문자열로 반환합니다.
        """
        if not self.driver or not law_ids:
            return ""

        graph_context = ""
        try:
            with self.driver.session() as session:
                # 1. 하위 법령 (시행령, 시행규칙) 찾기 - IMPLEMENTS 엣지 역추적
                impl_query = """
                    UNWIND $law_ids AS law_id
                    MATCH (child:Law)-[:IMPLEMENTS]->(parent:Law {id: law_id})
                    RETURN parent.law_name AS p_name, parent.article_number AS p_num,
                           child.law_name AS c_name, child.article_number AS c_num, child.full_text AS c_text
                    LIMIT $limit
                """
                impl_results = session.run(impl_query, law_ids=law_ids, limit=limit_per_type)
                
                impl_texts = []
                for record in impl_results:
                    impl_texts.append(f"[{record['p_name']} {record['p_num']} 관련 하위법령] {record['c_name']} {record['c_num']}\n{record['c_text']}")
                
                if impl_texts:
                    graph_context += "\n[GraphDB 보충: 관련 시행령/규칙]\n" + "\n".join(impl_texts) + "\n"

                # 2. 관련 판례 찾기 - REFERS_TO 엣지 역추적
                prec_query = """
                    UNWIND $law_ids AS law_id
                    MATCH (p:Precedent)-[:REFERS_TO]->(l:Law {id: law_id})
                    RETURN l.law_name AS l_name, l.article_number AS l_num,
                           p.court_level AS court, p.case_number AS case_num, p.summary AS summary
                    LIMIT $limit
                """
                prec_results = session.run(prec_query, law_ids=law_ids, limit=limit_per_type)
                
                prec_texts = []
                for record in prec_results:
                    prec_texts.append(f"[{record['l_name']} {record['l_num']} 관련 판례] {record['court']} {record['case_num']}\n{record['summary']}")
                
                if prec_texts:
                    graph_context += "\n[GraphDB 보충: 관련 판례 요지]\n" + "\n".join(prec_texts) + "\n"

        except Exception as e:
            logger.error(f"[GRAPH ERROR] 컨텍스트 검색 중 오류 발생: {e}")
            
        return graph_context