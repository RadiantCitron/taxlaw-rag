# 세법 RAG 데이터베이스 허브 (docs/DB/)

> **목적**: 세법 RAG 파이프라인이 쓰는 PostgreSQL(pgvector + trgm)과 Neo4j(지식그래프)의 **구축·검증 절차 문서**를 한곳에서 찾도록 안내하는 색인입니다.

---

## 1. 이 저장소의 데이터베이스

이 저장소에는 **데이터와 DB 덤프가 포함되어 있지 않습니다.** 환경 구성 스크립트(`deploy.sh` / `deploy.ps1`)는 빈 PostgreSQL과 빈 Neo4j를 띄우고, 데이터는 사용자가 `db_build/` 스크립트로 직접 구축합니다. 데이터 출처, 이용 조건, 개인정보 주의는 루트 [`README.md`](../../README.md)의 3절을 먼저 읽어 주세요.

| 구분 | 엔진 | 용도 |
| :--- | :--- | :--- |
| **RDB (관계형/벡터)** | PostgreSQL 16 (`pgvector`, `pg_trgm`) | 마스터 테이블 4종(`tax_laws_master`, `tax_precedents_master`, `tax_tribunal_master`, `tax_amendments_2026`), 선택적으로 비교 임베딩 테이블 |
| **지식그래프** | Neo4j | 조문 위계(`PARENT_OF`), 판례 인용/참조(`CITES`, `APPLIES_TO`) 관계. `use_graphdb=true` 일 때만 필요 |

개발자의 개발용 코퍼스 규모는 청크 29,403건, 그래프 엣지 164,540개였습니다. 직접 구축한 코퍼스는 수집 시점과 범위에 따라 다릅니다.

---

## 2. 구축 경로

```
[스크립트 구축]  법제처 Open API 수집 → 가공(청킹) → 임베딩 → PostgreSQL 적재 → (선택) Neo4j 그래프
  └── 문서: docs/DB/디비구축작업.md  (파이프라인: db_build/ 01~08)
  └── 요약: db_build/README.md
```

---

## 3. 세부 문서 색인

1. [**`디비구축작업.md`**](디비구축작업.md)
   - `db_build/` 8단계 스크립트(01 수집 ➔ 02 마스터구축 ➔ 03 개정세법 ➔ 04 비교임베딩 ➔ 05 가공청크검증 ➔ 06 마이그레이션검증 ➔ 07 지식그래프구축 ➔ 08 그래프검증) 상세 SOP
   - 마스터 4개 테이블 및 비교 모델 테이블 증설 규칙
   - 청크 JSON 데이터 무결성 검증 방법
2. [**`db_build/README.md`**](../../db_build/README.md)
   - `db_build/` 파이프라인 스크립트 전용 조감도 및 실행 명령어 모음
