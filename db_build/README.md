# 데이터베이스 구축 및 검증 파이프라인 (db_build/)

본 디렉토리는 세무 RAG 시스템의 원천 데이터 수집, RDB(PostgreSQL) 마스터 구축, 비교 임베딩 생성, 데이터 무결성 검증, 그리고 Neo4j 지식그래프 구축/점검을 담당하는 표준 파이프라인 스크립트를 모아둔 곳입니다.
`01`~`06`은 `DB_SETUP_GUIDE.md`의 표준 체계이고, 지식그래프 파이프라인(`07`, `08`)은 그 뒤 단계로 이어집니다.

---

## 1. 파이프라인 조감도 및 실행 순서

```
[STEP 1] 01_collect_tax_data.py          (법제처 API 원천 데이터 수집)
   │
   ▼
[STEP 2] 02_build_tax_db.py              (PostgreSQL 법령·판례·심판원 RDB 구축 & ko-sroberta 임베딩)
   │
   ▼
[STEP 3] 03_build_tax_amendments.py      (2026 개정세법 PDF 파싱 & RDB 적재)
   │
   ▼
[STEP 4] 04_build_compare_embedding_tax.py (BGE-M3 등 비교 평가용 임베딩 테이블 증설)
   │
   ▼
[STEP 5] 05_verify_processed_chunks.py   (가공 청크 JSON 무결성 및 스키마 검증)
   │
   ▼
[STEP 6] 06_verify_db_migration.py       (신·구 PostgreSQL RDB 볼륨/MD5 정합성 대조 검증)
   │
   ▼
[STEP 7] 07_build_graph_db.py            (RDB 데이터 기반 Neo4j 지식그래프 노드/엣지 구축)
   │
   ▼
[STEP 8] 08_verify_graph_db.py           (Neo4j 지식그래프 노드/엣지 무결성 점검)
```

---

## 2. 스크립트 목록 및 분류 상세

| 단계 | 스크립트 파일명 | 분류 | 역할 및 처리 내용 |
| :---: | :--- | :---: | :--- |
| **01** | [`01_collect_tax_data.py`](01_collect_tax_data.py) | **수집** | 법제처 Open API 기반 법령·판례·심판원 원천 JSON 수집 |
| **02** | [`02_build_tax_db.py`](02_build_tax_db.py) | **RDB 구축** | PostgreSQL 마스터 테이블(3종) 생성 및 기본 임베딩(`jhgan/ko-sroberta-multitask`) 적재 |
| **03** | [`03_build_tax_amendments.py`](03_build_tax_amendments.py) | **RDB 구축** | 2026 개정세법 해설(PDF) 파싱 및 RDB 적재 (`tax_amendments_2026`) |
| **04** | [`04_build_compare_embedding_tax.py`](04_build_compare_embedding_tax.py) | **임베딩 증설** | BGE-M3, Multilingual-E5 등 비교 모델 테이블 생성 (`_cmp_{model}`) |
| **05** | [`05_verify_processed_chunks.py`](05_verify_processed_chunks.py) | **검증** | 가공 청킹 JSON 파일의 스키마 및 무결성 대조 검증 |
| **06** | [`06_verify_db_migration.py`](06_verify_db_migration.py) | **검증** | 신·구 PostgreSQL DB 간 행 수, 임베딩 수, MD5 해시 대조 |
| **07** | [`07_build_graph_db.py`](07_build_graph_db.py) | **그래프 구축** | RDB 데이터를 기반으로 Neo4j 노드/엣지(조문위계, 심급, 참조) 구축 |
| **08** | [`08_verify_graph_db.py`](08_verify_graph_db.py) | **그래프 검증** | Neo4j 지식그래프 노드/엣지 연결 및 통계 점검 |

---

## 3. DB 구축 (스크립트로 직접 구축)

이 저장소에는 **데이터와 DB 덤프가 없습니다.** 법제처 Open API로 원천 데이터를 직접 수집해 구축하세요. 인증키 발급, 이용 조건, 개인정보 주의는 루트 [`README.md`](../README.md)의 3절을 먼저 읽어 주세요. 기본 경로는 `./data/01_raw_data`(원천 데이터)와 `./data/02_processed_chunks`(가공 청크)입니다.

```bash
# 가상환경 활성화
source .venv/bin/activate

# 0) DB 준비: 루트의 deploy.sh 로 띄운 컨테이너라면 확장과 init.sql 이 이미 적용돼 있습니다.
#    직접 설치한 PostgreSQL 이라면 확장을 먼저 설치합니다.
psql -d tax_rag_db -f database/init.sql

# 1) 원천 데이터 수집 (.env 의 LAW_OC_KEY 필요)
python db_build/01_collect_tax_data.py --limit 3

# 2) 법령·판례·심판례 가공(청킹)·임베딩·적재
python db_build/02_build_tax_db.py --db-name tax_rag_db --reset-db

# 3) (선택) 개정세법 해설 적재 — 해설서 PDF 를 직접 구해야 합니다 (--pdf-path 로 위치 지정)
python db_build/03_build_tax_amendments.py --db-name tax_rag_db

# 4) (선택) 지식그래프 구축 (PostgreSQL + Neo4j 기동 상태)
python db_build/07_build_graph_db.py
```

- 스크립트는 `.env`의 `TAX_DB_URI`로 접속합니다. 컨테이너 DB를 쓰는 접속 정보 예시는 루트 `README.md` 3절의 「코퍼스 직접 구축」에 있습니다.
- 수집 규모와 시점에 따라 결과(건수·내용)가 달라집니다. 재현될 수 있는 문제는 루트 `README.md` 3절의 「직접 구축하면 재현될 수 있는 문제」를 보세요.

---

## 4. 데이터베이스 사용 가능 상태 검증 4단계 절차 (Verification SOP)

DB를 구축한 뒤, RAG 시스템이 즉시 질문을 처리할 수 있는 정상 상태인지 검사하는 표준 절차입니다:

```bash
# [1단계] RDB 스키마 및 볼륨/인덱스 상태 점검
python db_build/06_verify_db_migration.py --db-old tax_rag_db --db-new tax_rag_db --show-info

# [2단계] 원천 청크 JSON과 실제 DB 레코드 일치 여부 역대조
python db_build/05_verify_processed_chunks.py --mode db-dir --db-name tax_rag_db --processed-dir ./data/02_processed_chunks

# [3단계] Neo4j 지식그래프 노드/엣지 연결 무결성 점검
python db_build/08_verify_graph_db.py

# [4단계] 서버 부팅 시 자동 계약 검사 (bootstrap.py)
# - pgvector, pg_trgm 확장 설치 여부
# - 필수 마스터 테이블 4종(tax_laws, tax_precedents, tax_tribunal, tax_amendments_2026) 존재 여부
# - 각 테이블의 embedding 컬럼이 vector(768) 타입인지 검증
python -c "
import psycopg2, os
from dotenv import load_dotenv; load_dotenv()
from bootstrap import verify_contracts
conn = psycopg2.connect(os.getenv('TAX_DB_URI'))
problems = verify_contracts(conn)
conn.close()
if not problems:
    print('✅ DB 계약 검증 통과')
else:
    print('⚠️ 문제 발견:', problems)
"
```
