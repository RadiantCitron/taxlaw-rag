-- === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
-- 빈 PostgreSQL 에 검색 확장만 깐다. 테이블은 02_build_tax_db.py /
-- 03_build_tax_amendments.py / scripts/seed_dev_db.py 가 만든다.
-- trigram GIN 인덱스는 테이블/데이터 적재 뒤 apply_search_support()
-- (database/postgres_repo.py, 내부적으로 schema.trigram_index_statements() 사용)가
-- 만든다 — 02/03 스크립트가 ingest 마지막 단계에서 자동으로 호출한다.
--
--   psql "$TAX_DB_URI" -f database/init.sql
--
-- 벡터 ANN 인덱스(ivfflat/hnsw)는 만들지 않는다. 코퍼스 규모를 측정하기 전에는
-- 전수 스캔이 더 정확하다. 의도가 누락이 아님을 여기에 남긴다.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
