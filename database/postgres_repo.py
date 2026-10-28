# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
import logging

try:
    from psycopg2 import pool
except ImportError:
    pool = None

from schema import (
    ALLOWED_SOURCE_TABLES,
    SOURCE_LABEL_COLS,
    cmp_table,
    extension_statements,
    normalize_sources,
    require_source,
    trigram_index_statements,
)
from search_constants import keyword_bind_values
from settings import require_tax_db_uri

logger = logging.getLogger("my_rag")

# 하위 호환: 기존 import 경로 유지
__all__ = [
    "ALLOWED_SOURCE_TABLES",
    "PostgresRepo",
    "apply_search_support",
    "normalize_sources",
]


def apply_search_support(conn, *, include_indexes: bool = True) -> None:
    """pgvector / pg_trgm 확장과, 이미 있는 원본 테이블의 trigram GIN 인덱스를 만든다."""
    with conn.cursor() as cur:
        for sql in extension_statements():
            cur.execute(sql)
        if include_indexes:
            cur.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = current_schema()
                  AND table_name = ANY(%s)
                """,
                (list(ALLOWED_SOURCE_TABLES),),
            )
            existing = {row[0] for row in cur.fetchall()}
            for table, sql in zip(ALLOWED_SOURCE_TABLES, trigram_index_statements()):
                if table in existing:
                    cur.execute(sql)
    conn.commit()


class PostgresRepo:
    """ PostgreSQL 데이터베이스와의 통신을 전담하는 Repository 클래스입니다. """
    
    # 🌟 [추가됨] 클래스 변수(정적 변수): 프로그램 전체에서 공유할 단 하나의 커넥션 풀
    connection_pool = None 

    # 🌟 [추가됨] 서버 시작 시 풀을 생성하는 클래스 메서드
    @classmethod
    def init_pool(cls):
        if cls.connection_pool is None:
            if pool is None:
                raise ImportError("PostgreSQL 연결을 위해 psycopg2 패키지가 필요합니다.")
            uri = require_tax_db_uri()
            cls.connection_pool = pool.SimpleConnectionPool(
                1, 20,  # 최소 1개, 최대 20개 커넥션 유지
                uri,
            )
            logger.info("[DB] PostgreSQL 커넥션 풀 생성 완료")

    # 🌟 [추가됨] 서버 종료 시 풀을 닫는 클래스 메서드
    @classmethod
    def close_pool(cls):
        if cls.connection_pool:
            cls.connection_pool.closeall()
            logger.info("[DB] PostgreSQL 커넥션 풀 종료 완료")

    def __init__(self):
        # 🌟 [수정됨] 매번 새로 연결하지 않고, 만들어둔 풀에서 커넥션을 하나 빌려옵니다.
        if PostgresRepo.connection_pool is None:
            raise Exception("Postgres 커넥션 풀이 초기화되지 않았습니다. 서버 시작 시 init_pool()을 호출하세요.")
        self.conn = PostgresRepo.connection_pool.getconn()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(has_error=exc_type is not None)
        return False

    def close(self, has_error: bool = False):
        # 🌟 연결을 완전히 끊는 것이 아니라, 다 쓴 커넥션을 풀에 반납합니다.
        if self.conn:
            try:
                # 에러가 있었거나 트랜잭션이 열려있는 경우 롤백하여 풀 오염 방어
                if has_error or getattr(self.conn, "status", None) != 0:
                    self.conn.rollback()
            except Exception:
                pass
            try:
                if PostgresRepo.connection_pool:
                    PostgresRepo.connection_pool.putconn(self.conn)
            except Exception:
                pass
            self.conn = None

    def _require_source(self, source_table: str) -> str:
        return require_source(source_table)

    def _src_cte(self, source_table: str, embed_variant=None) -> str:
        """하이브리드 검색용 src CTE. 개정 해설은 컬럼 별칭만 쓴다."""
        table = self._require_source(source_table)
        name_col, num_col = SOURCE_LABEL_COLS[table]
        if embed_variant:
            variant = cmp_table(table, embed_variant)
            return (
                f'SELECT s.id, s.{name_col} AS law_name, s.{num_col} AS article_number, '
                f's.full_text, c.embedding '
                f'FROM "{table}" s JOIN "{variant}" c ON c.doc_id = s.id'
            )
        return (
            f'SELECT id, {name_col} AS law_name, {num_col} AS article_number, '
            f'full_text, embedding FROM "{table}"'
        )

    def search_legacy_rdb(self, keywords, limit=20, source_table="tax_laws_master"):
        keyword_pattern, keyword_text = keyword_bind_values(keywords)
        if not keyword_pattern:
            return []
        table = self._require_source(source_table)
        name_col, num_col = SOURCE_LABEL_COLS[table]
        sql = f"""
            SELECT id, {name_col} AS law_name, {num_col} AS article_number, full_text
            FROM "{table}"
            WHERE full_text ~ %s
            ORDER BY similarity(full_text, %s) DESC LIMIT %s
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (keyword_pattern, keyword_text, limit))
            return cur.fetchall()

    def search_legacy_vector(self, query_vector, limit=20, source_table="tax_laws_master"):
        table = self._require_source(source_table)
        name_col, num_col = SOURCE_LABEL_COLS[table]
        sql = (
            f'SELECT id, {name_col} AS law_name, {num_col} AS article_number, full_text '
            f'FROM "{table}" ORDER BY embedding <=> %s::vector ASC LIMIT %s'
        )
        with self.conn.cursor() as cur:
            cur.execute(sql, (str(query_vector), limit))
            return cur.fetchall()

    def search_hybrid_rrf(self, keywords, query_vector, base_limit=20,
                          source_table="tax_laws_master", embed_variant=None):
        """키워드+벡터 RRF. 5-튜플 (id, law_name, article_number, full_text, rrf_score)."""
        if not keywords:
            return []
        src_cte = self._src_cte(source_table, embed_variant)
        keyword_pattern, keyword_text = keyword_bind_values(keywords)
        if not keyword_pattern:
            return []
        sql = f"""
            WITH src AS ({src_cte}),
            keyword_search AS (
                SELECT id, law_name, article_number, full_text, rank
                FROM (
                    SELECT id, law_name, article_number, full_text,
                           ROW_NUMBER() OVER(ORDER BY similarity(full_text, %s) DESC) AS rank
                    FROM src WHERE full_text ~ %s
                ) ranked
                WHERE rank <= %s
            ),
            vector_search AS (
                SELECT id, law_name, article_number, full_text,
                       ROW_NUMBER() OVER(ORDER BY embedding <=> %s::vector ASC) AS rank
                FROM src ORDER BY embedding <=> %s::vector ASC LIMIT %s
            )
            SELECT COALESCE(k.id, v.id), COALESCE(k.law_name, v.law_name),
                   COALESCE(k.article_number, v.article_number),
                   COALESCE(k.full_text, v.full_text),
                   COALESCE(1.0 / (60 + k.rank), 0.0)
                     + COALESCE(1.0 / (60 + v.rank), 0.0) AS rrf_score
            FROM keyword_search k FULL OUTER JOIN vector_search v ON k.id = v.id
            ORDER BY rrf_score DESC LIMIT %s;
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                keyword_text, keyword_pattern, base_limit,
                str(query_vector), str(query_vector), base_limit, base_limit,
            ))
            return cur.fetchall()

    def search_hybrid_score_based(self, keywords, query_vector, alpha, base_limit=20,
                                  source_table="tax_laws_master", embed_variant=None):
        """키워드+벡터 점수 융합. 5-튜플 (id, law_name, article_number, full_text, final_score)."""
        if not keywords:
            return []
        src_cte = self._src_cte(source_table, embed_variant)
        keyword_pattern, keyword_text = keyword_bind_values(keywords)
        if not keyword_pattern:
            return []
        sql = f"""
            WITH src AS ({src_cte}),
            keyword_search AS (
                SELECT id, law_name, article_number, full_text,
                       similarity(full_text, %s) AS keyword_sim
                FROM src WHERE full_text ~ %s ORDER BY keyword_sim DESC LIMIT %s
            ),
            vector_search AS (
                SELECT id, law_name, article_number, full_text,
                       (1.0 - (embedding <=> %s::vector)) AS vector_sim
                FROM src ORDER BY embedding <=> %s::vector ASC LIMIT %s
            )
            SELECT COALESCE(k.id, v.id), COALESCE(k.law_name, v.law_name),
                   COALESCE(k.article_number, v.article_number),
                   COALESCE(k.full_text, v.full_text),
                   (%s * COALESCE(v.vector_sim, 0.0))
                     + ((1.0 - %s) * COALESCE(k.keyword_sim, 0.0)) AS final_score
            FROM keyword_search k FULL OUTER JOIN vector_search v ON k.id = v.id
            ORDER BY final_score DESC LIMIT %s;
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                keyword_text, keyword_pattern, base_limit,
                str(query_vector), str(query_vector), base_limit,
                alpha, alpha, base_limit,
            ))
            return cur.fetchall()

    def extract_db_vocabulary(self, source_table, limit=500):
        """해당 원본 테이블 full_text 에서만 단어장을 뽑는다."""
        table = self._require_source(source_table)
        logger.info(f"[DB] 단어장 추출: {table}")
        sql = f"""
            WITH words AS (
                SELECT unnest(regexp_matches(full_text, '[가-힣]{{2,}}', 'g')) AS word
                FROM "{table}"
            )
            SELECT word, count(*) as cnt
            FROM words
            GROUP BY word
            ORDER BY cnt DESC
            LIMIT %s;
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
        vocab_list = [row[0] for row in rows]
        logger.info(f"[DB] {table} 단어 {len(vocab_list)}개.")
        return ", ".join(vocab_list)

    def _table_exists(self, table_name: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = current_schema()
                  AND table_name = %s
                """,
                (table_name,),
            )
            return cur.fetchone() is not None

    def ensure_search_tables(self, sources, embed_variant=None) -> None:
        """고른 원본(과 비교 모델이면 그 cmp)이 없으면 짧은 ValueError."""
        tables = normalize_sources(sources)
        for src in tables:
            if not self._table_exists(src):
                raise ValueError(f"검색 테이블이 없습니다: {src}")
            if embed_variant:
                name = cmp_table(src, embed_variant)
                if not self._table_exists(name):
                    raise ValueError(
                        f"비교 테이블이 없습니다: {name}. "
                        f"build_compare_embedding_tax.py --model {embed_variant} --table {src}"
                    )

    def search_hybrid_rrf_compare(self, keywords, query_vector, base_limit=20,
                                   source_table="tax_laws_master", embed_variant=None):
        return self.search_hybrid_rrf(
            keywords, query_vector, base_limit,
            source_table=source_table, embed_variant=embed_variant,
        )

    def search_hybrid_score_based_compare(self, keywords, query_vector, alpha, base_limit=20,
                                           source_table="tax_laws_master", embed_variant=None):
        return self.search_hybrid_score_based(
            keywords, query_vector, alpha, base_limit,
            source_table=source_table, embed_variant=embed_variant,
        )