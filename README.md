# taxlaw-rag — 대한민국 세법 하이브리드 RAG 시스템

> **기술 포트폴리오 (Architecture & Systems Engineering Showcase)**  
> **세무·법률 자문이 아닙니다** ([면책 전문](DEMO_DISCLAIMER.txt))

> ⚠️ **먼저 읽어 주세요**: 이 저장소에는 **법령·판례 등의 데이터와 DB 덤프가 포함되어 있지 않습니다.** 소스 코드와 환경 구성 스크립트만 제공하며, 데이터는 사용자가 공개 출처에서 직접 구축합니다. 법률 정보 시스템이므로 3절의 [데이터 출처와 이용 조건](#데이터-출처와-이용-조건), [개인정보 주의](#개인정보-주의), [면책](#면책)을 반드시 읽어 주세요.

대한민국 세법 4개 마스터 코퍼스(`tax_laws_master`, `tax_precedents_master`, `tax_tribunal_master`, `tax_amendments_2026`)를 대상으로, **근거 기반 질의응답, 자원 제어, 요청 단위 감사 가능성(Auditability)**을 갖추도록 설계한 하이브리드 RAG 시스템입니다.

---

### 💡 왜 이 프로젝트인가? (Problem → Solution → Value)

| 구분 | 당면한 문제 (Problem) | 아키텍처적 해결책 (Solution) | 엔지니어링 가치 (Value) |
|---|---|---|---|
| **검색 정밀도** | 법률 전문 용어와 일상 질의 간의 심각한 어휘 불일치 (키워드 검색의 한계 & 단일 벡터 검색의 정밀도 한계) | **하이브리드 검색 & RRF 융합**: `pgvector`(Dense) + `pg_trgm` GIN(Sparse) 상호 순위 융합 + `ko-reranker` CrossEncoder 재순위화 | 단일 검색보다 복합 세무 질의에서 검색 재현율(Recall)과 상위 순위 적중을 높이도록 설계 |
| **답변 신뢰성** | 세무 도메인 특성상 AI의 거짓 정보(Hallucination)가 치명적인 법적/재정적 리스크 초래 | **Fail-Closed 2단계 가드레일**: 입구(도메인 외 질문 차단) + 출구(인출된 법령·판례 조문 기반 팩트체크 검증) | 근거 없는 수치·조문 생성을 줄이도록 설계(완전히 막는다고 보장하지는 않음), DeepEval 연동 정량 벤치마크 하네스로 품질을 측정 |
| **자원 제어** | 단일 프로세스/제한된 자원 환경에서 로컬 LLM 동시 추론 시 OOM 및 세그폴트 발생 위험 | **동시성 게이팅 & 슬롯 관리**: `asyncio.Semaphore` 기반 진입 제한 + 모델 슬롯 `refcount + TTL` 메모리 수거 | 단일 프로세스(`workers=1`)에서도 자원 고갈 없는 안정적 동시 서빙 실현 |
| **관측 가능성** | RAG 파이프라인 내부(검색 후보, 프롬프트, 토큰, 시간)가 블랙박스화되어 디버깅 불가 | **요청 1건 전체 JSON 감사 덤프 (`AskRun`)**: 세션별 `ContextVar` 라우팅 로그 + 후보 문서·타이밍 전수 기록 | 요청 단위로 사후 감사(Auditability)가 가능하고 재현 가능한 평가 환경 제공 |
| **데이터 무결성** | 대규모 법령 DB 파싱 및 임베딩 과정의 데이터 손실 및 불일치 감지 어려움 | **01~06단계 무결성 파이프라인**: 가공 청크 사전 검증(`05`) + 신·구 DB 전수 MD5 해시 대조(`06`) | 데이터 적재부터 마이그레이션까지 불일치를 점검할 수 있는 검증 단계 제공 |

---

FastAPI 단일 프로세스로 동작하며, 세션마다 로그가 격리되고 요청 1건이 통째로 JSON 덤프로 남습니다.  
검색은 PostgreSQL 하이브리드(벡터 + trigram), 융합은 RRF, 생성은 OpenRouter 또는 로컬 GGUF입니다.  
웹 브라우저·CLI·Streamlit 모바일·평가 하네스가 **동일한 오케스트레이터 계약**을 공유합니다.

환경 기동·데이터 구축·이용 조건·고지는 아래 3절, 바로 실행하려면 [2. 빠른 시작](#2-빠른-시작)으로, 아키텍처 개요부터 보려면 아래 1절부터 이어집니다.

---

## 🌟 1. 세무 RAG 핵심 특징 (Core Features)

대한민국 세무 법률 및 판례의 특수성에 맞추어 서비스 운영을 염두에 두고 설계한 5대 핵심 특징입니다:

1. **정밀 하이브리드 검색 (Hybrid Search & RRF Fusion)**:
   - 대한민국 세법 4종 코퍼스(법령·판례·조세심판원 재결례·개정세법 해설)를 대상으로 합니다. 데이터는 포함되어 있지 않으며 사용자가 직접 구축합니다(개발용 코퍼스 기준 29,403건 청크, 164,540개 지식그래프 엣지).
   - 의미적 유사도를 포착하는 **Dense 벡터 검색 (`pgvector`)**과 정확한 조문 번호·어휘를 매칭하는 **Sparse 키워드 검색 (Kiwi 형태소 분석 + `pg_trgm` GIN)**의 상호 순위 융합(RRF).
   - 한국어 특화 `ko-reranker` CrossEncoder를 통한 정밀 재순위화.
2. **Fail-Closed 2단계 가드레일 (안전한 신뢰성 보장)**:
   - **입구 가드레일 (`guard_in`)**: 세무 도메인 외의 악의적 질문이나 공격성 프롬프트 사전 차단.
   - **출구 가드레일 (`guard_out`)**: 실제 인출된 법령 조문 및 판례 근거에 부합하는지 팩트체크 검증하여 할루시네이션(거짓 답변)을 줄입니다(완전히 막지는 못하므로 답변은 반드시 원문으로 확인하세요).
3. **비용 방어 쿼터 게이트웨이 & UI 타이머 (Production Cost Guard)**:
   - 악의적 호출 및 API 비용 폭탄을 줄이기 위해 **사용자당 일일 5회 질문 제한 및 15분(900초) 쿨다운 게이트** 적용.
   - Cloudflare IP (`CF-Connecting-IP`, `X-Forwarded-For`) 기반 1차 키 통제 및 KST 일자별 원자적 영속 원장(`logs/quotas.json`).
   - 업계 표준 UI/UX: 5회 소진 시 입력창 자동 비활성화 안내, 쿨다운 중 실시간 카운트다운 타이머 제공.
4. **실시간 SSE 스트리밍 & 단일 프로세스 자원 제어**:
   - 질문과 접속의 수명을 분리한 Server-Sent Events (SSE) 실시간 스트리밍 답변.
   - 단일 프로세스(`workers=1`) 설계로 임베딩/리랭커 슬롯 관리(`refcount + TTL`) 및 GGUF 세마포어를 통한 메모리 누수/OOM 차단.
5. **DeepEval 기반 정량적 품질 자동 평가 하네스**:
   - RAG Triad 중 충실도(Faithfulness)와 답변 관련성(Answer Relevancy)을 DeepEval로 정량 채점.
   - LLM 입출력 트레이스와 요청 1건 전체 감사 덤프(`AskRun`)를 영구 JSONL로 기록하여 사후 감사(Auditability)가 가능하도록 합니다.

---

## 🖥️ 2. 웹 UI 사용법 (How to Use)

코퍼스를 직접 구축하면(아래 3절 참고) 브라우저에서 쉽고 직관적으로 세무 법률 RAG를 체험할 수 있습니다:

1. **웹 브라우저 접속**:
   - 기동 후 브라우저에서 `http://localhost:8000` 접속.
2. **LLM 모델 선택**:
   - 상단 드롭다운에서 화이트리스트(`openrouter_models.json`)에 등록된 모델 중 원하는 모델 선택:
     * 예: `claude-haiku-5.5`, `gemini-3.8-flash`, `gemini-3.5-flash-lite`, `qwen3-32b` 등. 목록은 `GET /llm/models`로도 볼 수 있습니다.
3. **세무 질문 입력 및 실시간 답변 확인**:
   - 입력창에 자연어로 세무 관련 질문 입력 (예: *"1세대 1주택 비과세 요건과 장기보유특별공제율을 알려줘"*).
   - 실시간 SSE 스트리밍으로 생성되는 명확한 법률 해석과 관련 조문 카드 확인.
4. **인용 근거 조문 및 판례 검증**:
   - 답변 하단에 인용된 실제 법령 조문(`소득세법 제89조`, `소득세법 시행령 제154조` 등), 대법원 판례, 심판원 결정례의 원문 및 유사도 점수 직접 확인 가능.
5. **쿼터 및 쿨다운 UX**:
   - 일일 잔여 질문 횟수(최대 5회)가 표시되며, 질문 직후 15분 쿨다운 카운트다운 타이머가 실시간 동작합니다.

---

## 🚀 3. 시작하기 — 먼저 읽어 주세요

> **한 줄 요약**: 이 저장소는 **소스 코드와 환경 구성 스크립트**를 제공합니다. **데이터(코퍼스)와 DB 덤프는 포함되어 있지 않으며**, 사용자가 공개 출처에서 직접 구축합니다. 법률 정보를 다루는 시스템이므로 이 절의 [데이터 출처와 이용 조건](#데이터-출처와-이용-조건), [개인정보 주의](#개인정보-주의), [면책](#면책)을 반드시 읽어 주세요.

### 이 저장소가 제공하는 것과 제공하지 않는 것

- **제공하는 것**: 세법 RAG 시스템의 소스 코드, 환경을 띄우는 스크립트(Docker Compose), 코퍼스(데이터베이스)를 **직접 구축**하기 위한 수집·적재 스크립트(`db_build/`)와 안내 문서.
- **제공하지 않는 것**: **법령·판례·조세심판원 재결례·개정세법 해설 등의 데이터와 데이터베이스 덤프는 이 저장소 어디에도 포함되어 있지 않습니다.** 코퍼스는 각자 공개 출처에서 직접 수집해 구축합니다. (평가용 질의응답 `testset*.json`에는 법령·판례 원문을 발췌한 내용이 포함되어 있을 수 있습니다.)
- 이 프로젝트는 법제처, 국세청 등 데이터 출처 기관과 **관련이 없는 비공식 개인 프로젝트**이며, 기관의 보증·승인을 받지 않았습니다.

### 빠른 시작 (환경만 설치)

준비물: Docker(Compose v2 권장, v1 `docker-compose`도 인식), 헬스체크용 `curl` 또는 `wget`. Windows는 PowerShell을 씁니다.

```bash
git clone https://github.com/RadiantCitron/taxlaw-rag.git
cd taxlaw-rag
./deploy.sh    # Windows: .\deploy.ps1
```

> **⚠️ 처음 실행은 안내 메시지와 함께 멈추는 것이 정상입니다.**
> DB 비밀번호는 저장소에 들어 있지 않습니다. 첫 실행이 `.env`를 만든 뒤 멈추면, `.env`를 열어 아래 두 값을 **직접 정해** 적고 스크립트를 다시 실행하세요.
>
> ```
> POSTGRES_PASSWORD=<영문·숫자 비밀번호>
> NEO4J_PASSWORD=<8자 이상 비밀번호>
> ```
>
> 어떤 서비스의 계정 비밀번호가 아니라, **내 PC에 뜨는 DB 컨테이너에 쓸 새 비밀번호**입니다. `.env` 대신 환경변수로 넘겨도 됩니다(`POSTGRES_PASSWORD=… NEO4J_PASSWORD=… ./deploy.sh`).

- **호스트에는 Docker만 있으면 됩니다.** 스크립트는 Docker나 Python을 설치해 주지 않으며(Docker가 없거나 꺼져 있으면 안내만 하고 멈춥니다), Python은 API 컨테이너 안에서만 쓰이므로 호스트에 따로 설치하지 않아도 됩니다.
- 스크립트는 Docker 점검 → `.env` 준비(없으면 `.env.example`을 복사하고, 대화형 터미널이면 OpenRouter API Key 입력을 묻습니다. 건너뛰어도 됩니다) → **DB 비밀번호 확인(비어 있으면 안내하고 멈춤)** → 포트 충돌 점검 → `docker compose up -d` → `/health` 대기 순서로 진행합니다. 최초 실행은 API 이미지를 빌드하느라 수 분 걸립니다.
- PostgreSQL(pgvector), Neo4j, API 서버가 컨테이너로 **빈 DB 상태**로 뜹니다. 데이터가 없으므로 검색 결과는 비어 있고, `GET /health`의 `contracts.problems`에 "아직 없는 검색 테이블: …"이 보이는 것이 **정상**입니다.
- 웹 UI는 `http://localhost:8000`입니다. 질문 기능을 쓰려면 `.env`에 OpenRouter API Key(또는 로컬 GGUF 모델)가 필요합니다.
- **모든 포트는 기본적으로 `127.0.0.1`에만 열립니다.** 외부 접속이 필요하면 `API_BIND=0.0.0.0 ./deploy.sh`처럼 지정하되(DB 포트는 항상 `127.0.0.1`) 보안에 유의하세요.
- DB 데이터는 이름 있는 볼륨(`pgdata`, `neo4jdata`)에 남습니다. `docker compose down`으로는 지워지지 않고 `docker compose down -v`로 지워집니다.
- ⚠️ 스크립트는 `POSTGRES_PASSWORD`와 `NEO4J_PASSWORD`가 비어 있으면(`--dry-run`·`-DryRun` 포함) 안내를 출력하고 멈춥니다. Neo4j 비밀번호는 8자 미만이어도 멈춥니다. Postgres 비밀번호는 접속 주소에 들어가므로 영문·숫자만 쓰는 것을 권합니다. 위 안내 상자를 참고하세요.

| 플래그 (Linux/Mac) | 매개변수 (Windows) | 동작 및 설명 |
| :--- | :--- | :--- |
| *(인자 없음)* | *(인자 없음)* | 기동: Docker 점검 → `.env` 준비 → 포트 점검 → 컨테이너 기동 → `/health` 대기 |
| `--dry-run` | `-DryRun` | **가상 시뮬레이션**: 도커를 띄우지 않고 배포 흐름만 확인 |
| `--help`, `-h` | `-Help` | 사용법과 환경변수 안내 출력 |

호스트 포트는 환경변수로 바꿀 수 있습니다: `PG_PORT`(기본 5432), `NEO4J_HTTP_PORT`(7474), `NEO4J_BOLT_PORT`(7687), `API_PORT`(8000). `HEALTH_RETRIES`는 `/health` 대기 횟수(5초 간격, 기본 60)입니다. 호스트 포트가 이미 쓰이고 있으면 스크립트가 알려 줍니다.

### 코퍼스 직접 구축

- 순서와 옵션은 [`db_build/README.md`](db_build/README.md)와 [`DB_SETUP_GUIDE.md`](DB_SETUP_GUIDE.md), 그리고 아래 [2-2 설치](#2-2-설치)에 있습니다. 요약:

  ```bash
  python db_build/01_collect_tax_data.py --limit 3                  # 수집 (LAW_OC_KEY 필요)
  python db_build/02_build_tax_db.py --db-name tax_rag_db           # 법령·판례·심판례 가공·임베딩·적재
  python db_build/03_build_tax_amendments.py --db-name tax_rag_db   # (선택) 개정세법 해설 — PDF를 직접 구해야 함
  python db_build/07_build_graph_db.py                              # (선택) Neo4j 지식그래프 — use_graphdb=true 일 때만 필요
  ```

- 컨테이너 DB를 대상으로 호스트에서 위 스크립트를 실행하려면 `.env`에 접속 정보를 넣습니다(기본 포트 기준이며 비밀번호는 위에서 `.env`에 정한 값과 같아야 합니다. 호스트 포트를 바꿨다면 그 포트로):

  ```
  TAX_DB_URI=postgresql://myuser:<POSTGRES_PASSWORD>@127.0.0.1:5432/tax_rag_db
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=<NEO4J_PASSWORD>
  ```

- 필요한 것: 법제처 국가법령정보 공동활용 Open API 인증키(OC, **본인이 직접 신청** — 아래 [인증키 OC 발급 방법](#인증키-oc-발급-방법)), (개정세법 해설을 쓰려면) 해설서 PDF(**본인이 직접 입수**), 충분한 디스크·시간.
- 수집 범위(법령 이름 목록, 검색어)는 `db_build/01_collect_tax_data.py` 위쪽의 상수(`TARGET_TAX_LAWS`, `TAX_KEYWORDS`)에서 **원하는 대로 바꿀 수 있습니다.**
- 수집은 시간이 오래 걸릴 수 있고, **같은 명령을 써도 수집 시점·수집량에 따라 결과(건수·내용)가 달라집니다.** 재현될 수 있는 문제는 아래 [직접 구축하면 재현될 수 있는 문제](#직접-구축하면-재현될-수-있는-문제)에 적었습니다.

### 인증키 OC 발급 방법

코퍼스를 구축하려면 법제처 국가법령정보 공동활용 Open API의 **인증키(OC)** 가 필요합니다. 본인이 직접 신청해서 받으세요(이 저장소는 키를 제공하지 않습니다).

1. 국가법령정보 공동활용 사이트(<https://open.law.go.kr>)에 접속해 **회원가입·로그인**합니다.
2. **OPEN API 신청** 메뉴에서 사용 신청을 합니다. 이때 수집에 쓸 API를 선택합니다: 이 저장소는 **법령, 판례, 특별행정심판재결례(조세심판원 재결례)** 를 호출합니다. 신청하지 않은 API는 호출이 거부될 수 있습니다.
3. 호출하는 컴퓨터의 **공인 IP(또는 도메인)** 를 등록합니다(마이페이지의 신청 현황 메뉴). 등록하지 않은 곳에서 호출하면 인증에 실패할 수 있습니다.
4. 발급·확인한 OC 값을 프로젝트의 `.env` 파일에 `LAW_OC_KEY=...` 로 넣습니다(`.env.example`에 항목이 있습니다). **키를 커밋하거나 공유하지 마세요.**

> 위 절차는 공식 사이트의 메뉴 구성과 공개된 설명을 바탕으로 한 안내이며, **세부 절차와 이용 조건은 사이트의 최신 안내가 우선**합니다. 사용 신청과 이용 문의는 사이트에 안내된 방법을 따르세요.

### 데이터 출처와 이용 조건

**데이터는 사용자가 직접 수집합니다. 수집·저장·이용에 관한 책임은 수집하는 사용자에게 있습니다.**

| 데이터 | 출처 | 수집 방법 | 이 저장소에 포함 |
|---|---|---|---|
| 법령, 판례, 조세심판원 재결례 | 법제처 국가법령정보 공동활용 Open API (`law.go.kr`) | 본인의 인증키(OC)로 API 호출 | 포함하지 않음 |
| 개정세법 해설 | 국세청(법규과) 발행 「개정세법 해설」 PDF | 본인이 직접 입수 | 포함하지 않음 |

**법제처 Open API 사용 조건**

- Open API는 법제처 국가법령정보 공동활용 사이트(`open.law.go.kr`)에서 **사용 신청 후 발급받은 인증키(OC)** 로 호출합니다. 이용 조건·약관·호출 제한은 **법제처의 공식 안내가 우선**하며, 이 저장소는 그 내용을 보증하지 않습니다. 사용 신청과 이용 문의는 해당 사이트의 안내를 따르세요(사이트에 "사용신청 및 이용문의" 전화번호가 안내되어 있습니다).
- 법제처 공개 안내(조회일 2026-10-23, 원문은 URL에서 확인):
  - 저작권 정책(<https://easylaw.go.kr/CSP/InfoCopyright.laf>): 법령정보의 저작권은 저작권법의 보호 범위 안에서 법제처에 있고, 영리 목적을 포함한 자유로운 활용이 보장된다고 안내하면서 제3자 저작물은 예외로 하며, **이용자는 법령과 이용조건에 따른 의무를 지켜야 하고, 출처를 표기하지 않거나 다른 기관명으로 제공하는 행위 등은 제한·처벌 대상이 될 수 있다**고 안내합니다. (이 정책이 Open API 이용에 그대로 적용되는지는 이 저장소가 확인하지 못했습니다.)
  - 법제처 국민참여입법센터의 문의 답변(2026-06-10): 인증키(OC)는 아이디로 등록되고, IP는 공인 IP가 필수는 아니나 등록을 권장하며, 호출 빈도를 따로 검증하지는 않지만 **동시다발적인 대량 호출은 차단될 수 있고 이 경우 등록된 IP가 차단**된다고 답했습니다. (다른 서비스의 답변일 수 있어 Open API에 그대로 적용되는지는 확인하지 못했습니다.)
- **확인하지 못한 것**: 이용약관 본문, 상업적 이용·재배포 허용 범위, 호출 한도 수치, 출처 표시 방식. 이는 사용자가 법제처의 공식 안내에서 직접 확인해야 합니다.
- 이 저장소의 수집 스크립트는 호출 사이에 간격(0.35초)을 두고 순차로 호출하지만, **대량 수집 중 차단·중단될 수 있고 그에 대한 책임은 사용자에게 있습니다.** 수집 규모를 키우기 전에 반드시 이용 조건을 확인하세요.

**개정세법 해설(국세청 발행)**

- 해설서 PDF의 판권에는 "복사·출판하고자 하는 경우에는 발행처(법규과)에 사전문의하여 주시기 바랍니다"라는 안내가 있습니다. **PDF를 이 저장소에 올리거나 추출 결과를 공유·재배포하지 마세요.** 개인이 학습·개발용으로 자신의 환경에서 이용하는 범위를 넘는 이용은 발행처에 문의하세요. 이용 범위에 대한 판단은 사용자 책임입니다.

**출처 표시**

- 이 시스템이 보여 주는 법령·판례·결정례·해설은 위 출처 기관의 자료에서 가져온 것입니다. 이 시스템의 답변을 외부에 게시할 때에는 출처를 밝히고, 원문의 정확성은 반드시 원 제공처에서 확인하세요.

### 개인정보 주의

⚠️ **법령·판례·조세심판원 재결례 같은 공공정보에도 개인정보가 남아 있을 수 있습니다.**

- 판례와 조세심판원 재결례는 출처 기관이 이름 등을 가려(비실명 처리) 공개한 자료입니다. 그래도 **가려지지 않은 개인정보가 일부 남아 있습니다.** 개발 과정의 측정에서 정해진 형식(계좌번호 형태, 이메일, 주민번호 형태, 주소, 사업자등록번호 등)만 세어도 전체의 약 1%였고, 실명처럼 형식으로 찾을 수 없는 정보는 세지 못했습니다.
- **이 저장소는 데이터를 제공하지 않습니다.** 위 정보는 사용자가 직접 수집하는 데이터에 들어 있을 수 있습니다.
- **수집한 데이터를 외부에 공개하거나 재배포하지 마세요.** 이 시스템을 외부에 공개하는 서비스로 운영하면 검색 결과와 답변에 그 정보가 그대로 나올 수 있고(이 시스템에는 개인정보를 가리는 기능이 없습니다), 개인정보 보호 관련 법령을 지킬 책임은 **수집하고 운영하는 사용자**에게 있습니다.
- 개인 학습·개발 용도로 자신의 환경에서만 쓰는 경우에도 데이터를 다른 사람과 공유하지 않도록 주의하세요.
- 질문을 입력할 때 실명, 주민번호, 연락처 같은 개인정보를 넣지 마세요.

### 면책

본 저장소와 문서는 있는 그대로 제공되며, 정확성·완전성·최신성을 보장하지 않습니다. 이 시스템의 답변은 **법률 자문이 아니며** 법적 판단이나 실무에 사용하기 전에 반드시 원문(법령·판결·결정례)을 확인하고 전문가의 도움을 받으세요. 법령은 수집 시점의 스냅샷이어서 이후 개정이 반영되지 않을 수 있습니다. 이용으로 인한 손해에 대해 배포자는 책임을 지지 않습니다. 데이터 수집·이용이 출처 기관의 이용 조건이나 법령에 맞는지 확인하는 책임은 사용자에게 있습니다. ([면책 전문](DEMO_DISCLAIMER.txt))

### 라이선스

- **코드**: 별도의 라이선스 파일이 없으며, 코드의 사용·복제·수정·재배포에 관한 권리는 저작권자에게 있습니다. 사용자가 수집한 데이터에는 각 출처의 이용 조건이 적용됩니다.

### 직접 구축하면 재현될 수 있는 문제

개발자의 개발용 코퍼스(법령 2,903조문 청크, 판례 10,284건, 조세심판원 재결례 15,928건, 개정세법 288건)는 배포하지 않습니다. **같은 명령을 써도 사용자의 결과는 개발용 코퍼스와 같지 않습니다.** 아래는 개발 중 확인한 사실과, 라이브 API에 달려 있어 재현 여부를 단정할 수 없는 항목(**[추정]**)입니다.

| # | 문제 | 내용과 대처 |
|---|---|---|
| 1 | 수집량이 개발용의 약 10% 수준 | `01_collect_tax_data.py`의 `--limit` 기본값은 3(검색어당 3쪽, 쪽당 100건)이라 대상당 최대 약 2,100건입니다. 개발용 원본은 판례 20,238개·심판례 15,928개였고 개발용 원본을 만들 때 쓴 `--limit` 값은 기록이 남아 있지 않습니다. 원하는 규모에 맞춰 `--limit`을 정하세요(커질수록 오래 걸리고 차단 가능성이 큽니다). |
| 2 | 법령 이름 검색의 첫 결과만 사용 | 이름이 비슷한 법령이 먼저 나오면 그 법령이 저장됩니다. 개발 중 파일명이 `지방세법*`인 원본 3개의 실제 내용이 지방교부세법이었습니다. 사용자 환경에서 재현되는지는 [추정]. 아래 점검 스크립트로 확인하고, 불일치 파일은 지우고 다시 받거나 스크립트에서 법령명 정확 일치를 고르도록 고치세요. |
| 3 | 일부 법령이 조용히 누락될 수 있음 | 검색 응답에 기대한 키가 없으면 오류 메시지 없이 건너뜁니다. 개발용 원본에는 법인세법 계열이 없었고, 원인은 [추정]입니다. 수집 후 파일 목록을 대상 목록과 대조하세요. |
| 4 | 시행규칙이 목록에 없음 | 스크립트의 `TARGET_TAX_LAWS`(10개)에는 시행규칙이 없습니다. 필요한 법령(시행령·시행규칙 포함)을 직접 적으세요. |
| 5 | 시점에 따라 결과가 달라짐 | API는 실시간 자료를 줍니다. 개발용 코퍼스는 2026-07-30~08-01 스냅샷입니다. 수집일, `--limit`, 검색어, 법령 목록을 기록해 두세요. |
| 6 | 문서의 건수가 서로 다름 | 일부 문서의 판례 건수 표기(10,284 / 19,991 / 20,239)가 서로 다릅니다. 코드가 기준입니다. |
| 7 | 개정세법 해설은 PDF를 직접 구해야 함 | `03_build_tax_amendments.py`는 국세청 법규과 발행 「개정세법 해설」 PDF(개발 시 2026년판)를 `pdfplumber`로 파싱하며 테이블 이름도 `tax_amendments_2026`입니다. 다른 연도 PDF는 맞지 않을 수 있습니다([추정]). 필요 없으면 이 단계를 건너뛰어도 나머지는 구축됩니다. 위 [이용 조건](#데이터-출처와-이용-조건)을 지키세요. |
| 8 | 시간과 자원이 많이 듦 | 원천 데이터가 있을 때 파싱·임베딩·적재가 CPU 기준 약 30~50분(문서 기준)이며 개발용 규모(약 3.6만 건)의 수집 시간은 측정하지 않았습니다. 먼저 작은 `--limit`으로 전 과정을 돌려 보세요. |
| 9 | 인증키·IP 등록·호출 차단 | 위 [인증키 OC 발급 방법](#인증키-oc-발급-방법)과 [이용 조건](#데이터-출처와-이용-조건)을 보세요. 오류율이 오르거나 응답이 이상하면 즉시 중단하세요. |
| 10 | 개인정보가 남아 있을 수 있음 | 위 [개인정보 주의](#개인정보-주의)를 보세요. 이 앱에는 검색 결과와 답변에서 개인정보를 가리는 기능이 없습니다. |
| 11 | 평가용 질의응답은 개발용 코퍼스 기준 | `testset*.json`은 개발용 코퍼스 기준으로 만들었습니다. 코퍼스가 다르면 평가 결과가 달라지므로([추정]) 점수를 개발자 값과 직접 비교하지 마세요. |

**수집 후 점검 (권장)** — 법령 파일의 내부 법령명이 요청한 이름과 같은지 확인합니다. 아래를 `check_laws.py`로 저장해 `python check_laws.py ./data/01_raw_data/laws`로 실행하세요(파일명 앞의 `law_` 접두는 있어도 없어도 됩니다).

```python
import json, glob, sys, os, unicodedata

def find(o, keys):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in keys and isinstance(v, (str, int)):
                return v
        for v in o.values():
            r = find(v, keys)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find(v, keys)
            if r:
                return r

bad = 0
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*_raw.json"))):
    d = json.load(open(f, encoding="utf-8"))
    name = find(d, {"법령명_한글", "법령명한글", "법령명"})
    want = os.path.basename(f)[: -len("_raw.json")]
    if want.startswith("law_"):
        want = want[len("law_"):]
    # macOS 등은 파일명을 NFD 로 돌려주므로 NFC 로 맞춰 비교한다
    ok = unicodedata.normalize("NFC", str(name)) == unicodedata.normalize("NFC", want)
    bad += not ok
    print("OK  " if ok else "불일치", want, "->", name)
print("불일치 %d건" % bad)
```

그리고 수집 대상 목록(`TARGET_TAX_LAWS`)의 각 이름에 해당하는 파일이 있는지, 파일 수·가공 후 행 수·수집일·`--limit`을 기록해 두세요.

---

## 목차

1. [3분 안에 이해하기](#1-3분-안에-이해하기)
2. [빠른 시작](#2-빠른-시작)
3. [디렉터리 지도](#3-디렉터리-지도)
4. [계약 — 이 시스템의 심장](#4-계약--이-시스템의-심장)
5. [요청 하나가 지나는 길](#5-요청-하나가-지나는-길)
6. [검색 파이프라인](#6-검색-파이프라인)
7. [세션과 로그 격리](#7-세션과-로그-격리)
8. [자원 관리와 동시성](#8-자원-관리와-동시성)
9. [엔드포인트 전체](#9-엔드포인트-전체)
10. [요청 옵션 전체](#10-요청-옵션-전체)
11. [환경변수 전체](#11-환경변수-전체)
12. [관측·디버깅](#12-관측디버깅)
13. [평가 하네스](#13-평가-하네스)
14. [테스트](#14-테스트)
15. [확장하기](#15-확장하기)
16. [알려진 한계](#16-알려진-한계)
17. [처음 읽는 사람을 위한 순서](#17-처음-읽는-사람을-위한-순서)
18. [자주 막히는 곳](#18-자주-막히는-곳)
19. [라이선스와 데이터 출처](#19-라이선스와-데이터-출처)

---

## 1. 3분 안에 이해하기

### 프로세스 1개, 클라이언트 4종

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                                                                  │
│  index.html          app.py              test_api.py         cs_console.py       │
│  (웹 브라우저)        (Streamlit 모바일)   (CLI / REPL)        (CS 콘솔)           │
│       │                   │                   │                   │              │
│       └───────────────────┴───────────────────┴───────────────────┘              │
│                                           │  HTTP + SSE                          │
│                                           ▼                                      │
│              ┌─────────────────────────────┐                                     │
│              │  api_server.py     :8000    │                                     │
│              │  ─────────────────────────  │                                     │
│              │  세션 · 게이트 · 가드레일    │                                     │
│              │  검색 · 융합 · 리랭킹        │                                     │
│              │  생성 · 평가                 │                                     │
│              │         workers=1 (설계)     │                                     │
│              └───┬──────────┬──────────┬───┘                                     │
│                  │          │          │                                         │
│         psycopg2 │          │ bolt     │ HTTPS / 로컬                            │
│                  ▼          ▼          ▼                                         │
│      ┌───────────────┐ ┌─────────┐ ┌──────────────────┐                         │
│      │ PostgreSQL    │ │ Neo4j   │ │ OpenRouter       │                         │
│      │ + pgvector    │ │ (선택)  │ │ 또는 로컬 GGUF   │                         │
│      │ + pg_trgm     │ │         │ │ (llama-cpp)      │                         │
│      └───────────────┘ └─────────┘ └──────────────────┘                         │
│                                                                                  │
│  logs/sessions/{sid}.jsonl   세션 이벤트 시계열                                   │
│  logs/runs/{sid}/{run}.json  요청 1건 전체 덤프 (capture=full 이상)               │
│  logs/server.jsonl           세션 없는 로그                                       │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 2단계 흐름 — 접속과 질문이 분리돼 있다

```
  ① GET /session/connect          SSE. 끊길 때까지 열려 있다.
     ← event: session {"event":"connected", "session_id":"…"}
     ← 15초마다 ping (유휴 타이머 갱신)

  ② POST /ask   헤더에 X-Session-Id
     ← event: log     서버 진행 로그 (레벨 필터 적용)
     ← event: notice  무시된 옵션 안내
     ← event: system  대기열 안내, 컨텍스트 없음 안내
     ← event: prompt  LLM 프롬프트 트레이스 (trace_prompts)
     ← event: answer  guard_out 직후 최종 답변
     ← event: eval    DeepEval 결과
     ← event: done    {"status":"completed", "run_id":…, "timing_ms":…}
```

**왜 접속과 질문을 나눴는가:**
`/ask` 하나로 합치면 (a) 서버가 지금 몇 명 붙어 있는지 알 수 없고,
(b) 질문 없이 붙어만 있는 사용자를 정원에 셀 수 없으며,
(c) 세션 로그 파일을 어느 시점에 열지 정할 수 없다.
`/session/connect`가 **접속의 수명**을, `/ask`가 **작업의 수명**을 각각 소유한다.

---

## 2. 빠른 시작

> 이 절은 **Docker 없이 내 컴퓨터에 직접 설치해 실행하는 방법**입니다(Python·PostgreSQL을 직접 준비). Docker로 환경만 빠르게 띄우려면 위 3절 "빠른 시작 (환경만 설치)"을 따르세요.

### 2-1. 요구사항

- Python 3.11+
- PostgreSQL 14+ (`pgvector`, `pg_trgm` 확장) — 3절의 `deploy.sh`를 쓰면 Docker 컨테이너로 대신할 수 있습니다
- OpenRouter API 키 **또는** 로컬 GGUF + `llama-cpp-python`
- Neo4j (선택 — `use_graphdb=true` 일 때만)

### 2-2. 설치

> 이 저장소에는 데이터가 없습니다. 아래 `db_build/` 단계는 **코퍼스를 직접 구축하는 과정**이며, 인증키·이용 조건·개인정보에 관해서는 위 3절을 먼저 읽어 주세요.

```bash
pip install -r requirements.txt

# 1) PostgreSQL 확장(pgvector, pg_trgm)만 설치. 테이블/인덱스는 여기서 안 만든다.
psql -d <DB> -f database/init.sql

# 2) 마스터 테이블 생성 + 데이터 적재 (01~03), 마지막에 full_text pg_trgm GIN
#    인덱스까지 자동으로 건다 — database/postgres_repo.apply_search_support() 호출.
python db_build/01_collect_tax_data.py
python db_build/02_build_tax_db.py
python db_build/03_build_tax_amendments.py
```

인덱싱을 별도로 다시 걸어야 할 때(예: 위 스크립트 밖에서 직접 INSERT 했거나,
`--reset-db`로 테이블만 새로 만들고 적재는 나중에 한 경우)는 아래처럼 수동 실행한다:

```bash
python -c "
import psycopg2, os
from dotenv import load_dotenv; load_dotenv()
from database.postgres_repo import apply_search_support
conn = psycopg2.connect(os.getenv('TAX_DB_URI'))
apply_search_support(conn, include_indexes=True)
conn.close()
"
```

벡터(`embedding`) 컬럼에는 ivfflat/hnsw 같은 ANN 인덱스를 의도적으로 두지 않는다 —
이유는 `ADR.md`의 ADR-010 참고 (코퍼스 규모가 작아 전수 스캔이 더 정확해서 그렇다).

### 2-3. 반드시 있어야 하는 파일

부팅 전에 프로젝트 루트에 아래가 있어야 한다. **없으면 `[Fatal Error]`로 즉시 죽는다.**  
(설정 JSON 및 프롬프트 파일은 기본 파일이 저장소에 기본 포함되어 있으며, `.example.json` 템플릿도 함께 제공됩니다)

| 파일 | 없을 때 메시지 | 기본 제공 및 준비 방법 |
|---|---|---|
| `.env` | `DEFAULT_LLM 환경변수가 없습니다` | `.env.example` 복사 후 본인 API 키/DB 주소 기재 |
| `embedding_models.json` | `모델 목록 파일이 없습니다` | ✔ 저장소에 기본 제공 (필요 시 모델 추가) |
| `reranker_models.json` | 동일 | ✔ 저장소에 기본 제공 |
| `openrouter_models.json` | `OpenRouter 모델 목록 파일이 없습니다` | ✔ 저장소에 기본 카탈로그 제공 |
| `prompts_ko_세무.json` | `프롬프트 파일을 찾을 수 없습니다` | ✔ 저장소에 기본 제공 (필요 시 수정) |

`embedding_models.json` — **`indexed: true` 가 최소 1개 있어야 한다.**
이게 DB `embedding` 컬럼을 만든 모델이라는 표시다. 없으면 부팅이 실패한다.

```json
[
  {"id": "jhgan/ko-sroberta-multitask", "label": "ko-sroberta (기본, DB 인덱스)", "indexed": true},
  {"id": "BAAI/bge-m3",                 "label": "bge-m3 (비교용)"}
]
```

`prompts_ko_세무.json` — **`main_answer` 가 비면 부팅이 실패한다.**

```json
{
  "guardrail_in":    "…도메인 적합성 판정. PASS 또는 BLOCK 만 출력…",
  "guardrail_out":   "…답변이 컨텍스트에 근거하는지 판정. PASS 또는 BLOCK…",
  "query_expansion": "…질문을 검색에 유리하게 확장…",
  "vocab_mapping":   "…단어장에서 검색 키워드를 고른다…",
  "main_answer":     "…주어진 법령을 근거로 답한다…",
  "json_eval":       "…DeepEval 심판용 시스템 프롬프트…"
}
```

### 2-4. 실행

```bash
# 1) 백엔드 서버 기동
python api_server.py

# 옵션 부여 예시
python api_server.py \
  --prompt-file prompts_ko_세무.json \
  --capture full \
  --max-concurrent-user 4 \
  --max-waiting 5 \
  --max-connected 10 \
  --login-mode \
  --print-models
```

```bash
# 2) 클라이언트 실행 (택 1)

# (A) 웹 브라우저 UI
# index.html 을 브라우저로 직접 열거나 http://127.0.0.1:8000/ 접속

# (B) 모바일 / 브라우저 Streamlit 채팅 UI
streamlit run app.py

# (C) CLI 클라이언트 (대화형 REPL)
python test_api.py

# (D) CLI 단건 질의
python test_api.py --question "거주자 판정 기준 일수는?" --rerank --top-k 5
```

### 2-5. 잘 떴는지 확인

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "ok": true,
  "boot": {"started_at": "...", "prompt_file": "...", "capture_ceiling": "lean",
           "login_mode": false, "env_warnings": []},
  "contracts": {"ok": true, "problems": []},
  "models": {"llm_catalog": 42, "embed": {...}, "rerank": {...}}
}
```

⚠️ **`contracts.problems`를 반드시 확인할 것.**
현재 `ok`는 항상 `true`라서 여기 문제가 있어도 초록불이다. (→ [16절](#16-알려진-한계))

### 2-6. 첫 실행이 실패한다면

| 증상 | 원인 | 해결 |
|---|---|---|
| `[Fatal Error] DEFAULT_LLM 환경변수가 없습니다` | `.env` 없음/미로드 | `.env`를 프로젝트 루트에 |
| `❌ POSTGRES_PASSWORD 가 비어 있습니다` / `❌ NEO4J_PASSWORD 가 비어 있습니다` / `NEO4J_PASSWORD 는 8자 이상이어야 합니다` (compose가 직접 멈출 때는 `required variable … is missing a value`) | `.env`에 DB 비밀번호를 적지 않았음(저장소에는 비밀번호가 없음) | `.env`에 `POSTGRES_PASSWORD`, `NEO4J_PASSWORD`(8자 이상)를 직접 정해 적고 다시 실행 |
| `[Fatal Error] TAX_DB_URI 환경변수가 없습니다` | 동일 | `postgresql://user:pw@host/db` |
| `[Fatal Error] PostgreSQL 확장이 없습니다: ['vector']` | 확장 미설치 | `psql -f database/init.sql` |
| `[Fatal Error] 모델 목록 파일이 없습니다` | `embedding_models.json` 없음 | 2-3절 |
| `[Fatal Error] 임베딩 목록에 indexed=true 항목이 없습니다` | 플래그 누락 | 기본 모델에 `"indexed": true` |
| `[Fatal Error] 프롬프트 파일에 main_answer 가 비어 있습니다` | 프롬프트 JSON | 2-3절 |
| `[WARN] Neo4j 미연결` | 정상 | `use_graphdb=false`면 무시 |
| `아직 없는 검색 테이블: …` (경고) | 인제스트 전 | `python db_build/02_build_tax_db.py` 후 `python db_build/03_build_tax_amendments.py` |
| `pytest` → `ImportError` | `pythonpath` 미설정 | `pytest.ini` 추가 또는 `python -m pytest` |

---

## 3. 디렉터리 지도

```
taxlaw-rag/
│
├── api_server.py            🔴 진입점. FastAPI · SSE · lifespan
├── workflow_orchestrator.py 🔴 5노드 파이프라인
├── retriever.py             🔴 하이브리드 검색 + 리랭킹 + 그래프 보충
├── retriever_kiwi_patch.py     Kiwi 형태소 기반 법률 키워드 정밀 추출기
├── llm_handler.py           🔴 LLMController (openrouter / local 단일 인터페이스)
├── model_registry.py        🔴 임베딩·리랭커 슬롯 (refcount + TTL)
├── local_llm_gate.py        🔴 입장 세마포어 + 로컬 GGUF 락
├── session_manager.py       🔴 세션 · ContextVar 로그 라우팅
├── session_auth.py             로그인 모드 아이디 입구컷 + /admin
├── bootstrap.py                서버·하네스 공용 부팅
├── evaluator.py                DeepEval 어댑터
│
├── ── 계약 계층 (로컬 import 0개. 여기부터 읽을 것) ──
├── schema.py                   테이블 허용목록 · DDL · 식별자 규칙
├── contracts.py                요청 Pydantic 스키마
├── validation.py               값 범위 · 조합 제약 · 교습형 hint
├── error_envelope.py           에러 taxonomy (HTTP/exit/severity)
├── sse.py                      SSE 프레임 8종 직렬화
├── errors.py                   transient / overload 판정
├── rag_run.py                  AskRun · capture 3단계 · 덤프
├── rag_core.py                 가드레일 (verdict 정규화)
├── search_constants.py         키워드 추출 상수
├── settings.py                 환경변수 읽기
├── text_norm.py                문자열 정규화
│
├── database/
│   ├── postgres_repo.py        하이브리드 SQL (RRF / score_based)
│   ├── neo4j_repo.py           그래프 보충 컨텍스트
│   └── init.sql                확장(pgvector, pg_trgm)만 설치. 인덱스는 01~03이 적재 후 자동 생성
│
├── ── 클라이언트 ──
├── index.html                  웹 데모 UI
├── app.py                      모바일 / 브라우저 Streamlit 채팅 UI (.streamlit/)
├── test_api.py                 CLI REPL + 배치
├── cs_console.py               CS 콘솔 (HTTP만, 서버 import 안 함)
├── scripts/ask_client.py        최소 예제 클라이언트
│
├── ── 환경 구성 ──
├── deploy.sh / deploy.ps1      빈 DB 환경 기동 스크립트 (Docker 점검 → .env → DB 비밀번호 확인 → 포트 점검 → 기동 → /health)
├── docker-compose.yml          PostgreSQL(pgvector) + Neo4j + API
├── docker/Dockerfile.api       API 이미지
│
├── ── 주요 문서 ──
├── DB_SETUP_GUIDE.md           세법 DB 마스터 재구축 가이드 (SOP)
├── DEMO_DISCLAIMER.txt         데모 면책 및 이용 안내 원문
├── ADR.md                      아키텍처 결정 기록 (ADR-001 ~ 014)
├── STATE.md                    전역 상태 지도 (단일 프로세스 vs 워커 동시성)
│
├── ── 데이터 구축 (db_build/) : 저장소 루트에서 python db_build/01_....py ──
├── 01_collect_tax_data.py      원천 수집
├── 02_build_tax_db.py          법령·판례·심판원 파싱·임베딩·적재
├── 03_build_tax_amendments.py  개정세법 PDF 파싱·적재
├── 04_build_compare_embedding_tax.py  비교 임베딩 테이블
├── 05_verify_processed_chunks.py  가공 JSON 사전 무결성 대조 (3초)
├── 06_verify_db_migration.py   신·구 DB 스키마 및 전수 MD5 대조 검증
├── build_graph_db.py           Neo4j 그래프 구축 (루트)
├── check_graph_db.py           그래프 점검 (루트)
├── scripts/seed_dev_db.py      개발용 시드
│
├── ── 평가 ──
├── build_testset.py            골든셋 생성 (template / gemini 엔진)
├── evaluate_rag.py             채점 하네스 (검색 + 답변 분리 채점)
├── build_faithfulness_testset.py
├── evaluate_faithfulness.py
├── run_eval_batch.py           단건 스모크
│
├── tests/                      pytest (25파일, 104항목)
│   └── fixtures/               동결 픽스처 (채점기 회귀 방어)
│
└── logs/                       ⚠️ .gitignore — 세션/런 덤프
    ├── sessions/{sid}.jsonl
    ├── runs/{sid}/{run_id}.json
    └── server.jsonl
```

🔴 = 실행 경로. 처음 읽을 때는 계약 계층부터 볼 것. ([17절](#17-처음-읽는-사람을-위한-순서))

---

<a id="4-계약--이-시스템의-심장"></a>
## 4. 계약 — 이 시스템의 심장

### 4-1. "유일한 소유자" 규율

계약 계층 파일은 전부 이 형식의 docstring으로 시작한다.

```python
"""SSE 프레임 직렬화의 유일한 소유자.

kind 가 answer/log/system/prompt/notice 이면 data 는 {"text": ...} JSON.
kind 가 error/done 이면 data 는 구조화 객체 JSON 을 한 겹만 쓴다.
텍스트 스니핑으로 종류를 추측하지 않는다. FastAPI 를 import 하지 않는다.
"""
```

**세 가지를 동시에 선언한다:**
1. 무엇의 **유일한 소유자**인가
2. **어떤 규칙**을 지키는가
3. 무엇을 **import 하지 않는가** ← 레이어 침범 방지

| 개념 | 유일한 소유자 |
|---|---|
| 검색 테이블 이름 / DDL / 식별자 규칙 | `schema.py` |
| 요청 스키마 (타입) | `contracts.py` |
| 값 범위 · 조합 제약 · 사용자 안내 문구 | `validation.py` |
| 에러 코드 → HTTP/exit/severity | `error_envelope.py` |
| SSE 프레임 kind | `sse.py` |
| 일시적 오류 판정 키워드 | `errors.py` |
| capture 등급과 덤프 형식 | `rag_run.py` |
| 환경변수 읽기 | `settings.py` |
| 키워드 추출 상수 | `search_constants.py` |

> **상수를 두 곳에 복사한 순간부터 시스템은 썩기 시작한다.**
> 새 상수를 만들 때는 반드시 "이건 누구 소유인가"를 먼저 정할 것.

### 4-2. 에러 taxonomy — 4개 축

`error_envelope.py`는 하나의 코드에 4개 축을 붙인다.

| 코드 | HTTP | exit | severity | stream only |
|---|:---:|:---:|---|:---:|
| `INVALID_BODY` | 400 | 2 | reject | — |
| `INVALID_OPTION_VALUE` | 400 | 2 | reject | — |
| `UNKNOWN_MODEL` | 400 | 2 | reject | — |
| `INCOMPATIBLE_OPTIONS` | 400 | 2 | reject | — |
| `PROMPT_INVALID` | 400 | 2 | reject | — |
| `MISSING_SESSION` | 400 | 3 | reject | — |
| `AUTH_REQUIRED` | 401 | 3 | reject | — |
| `AUTH_FORBIDDEN` | 403 | 3 | reject | — |
| `SESSION_BUSY` | 409 | 3 | reject | — |
| `SESSION_CAPACITY` | 503 | 3 | retryable | — |
| `GATE_FULL` | — | 3 | retryable | ✓ |
| `RESOURCE_UNAVAILABLE` | — | 3 | retryable | ✓ |
| `UPSTREAM_OVERLOAD` | — | 4 | upstream | ✓ |
| `UPSTREAM_ERROR` | — | 4 | upstream | ✓ |
| `PIPELINE_ERROR` | — | 4 | upstream | ✓ |
| `CANCELLED` | — | 130 | cancelled | ✓ |
| `INTERNAL_ERROR` | 500 | 4 | upstream | — |
| `ADMIN_*` (5종) | 400~409 | 2~3 | reject | — |

**이 표는 `GET /errors/codes`로 그대로 나온다.** 클라이언트가 하드코딩할 필요가 없다.

계층 침범은 코드가 막는다:

```python
def http_error(code, message, ...):
    if code in STREAM_ONLY_CODES:
        raise AssertionError(f"{code} 는 SSE 전용 코드다")
```

> **왜 에러 문서를 코드에서 생성하는가:**
> 손으로 쓴 에러 코드 표는 반드시 썩는다. 코드는 바뀌고 문서는 안 바뀐다.
> `errors_codes_payload()`가 세 dict의 합집합에서 표를 만들면,
> 새 코드는 자동으로 나타나고 **빼먹은 코드는 `null`이 보여 즉시 티가 난다.**

### 4-3. 봉투(envelope) — 사전 거부와 스트림 오류가 같은 모양

```json
{
  "ok": false,
  "error": {
    "code": "INVALID_OPTION_VALUE",
    "message": "요청 옵션 1건이 허용되지 않는 값입니다.",
    "violations": [{
      "field": "rerank_settings.top_k",
      "given": 999,
      "reason": "허용 범위를 벗어났습니다.",
      "expected": {"type": "int", "min": 1, "max": 20},
      "default": 3,
      "hint": "리랭킹 후 최종적으로 LLM 에 넣을 문서 개수입니다. …"
    }],
    "ignored_fields": ["general_settings.unknown_opt"],
    "request_id": "…", "session_id": "…",
    "docs": "GET /options/schema",
    "exit_hint": 2,
    "severity": "reject",
    "terminal": true,
    "retry_after_sec": 30
  }
}
```

- HTTP 응답은 FastAPI가 `{"detail": {...}}`로 한 겹 감싼다 → `unwrap_envelope()`가 흡수
- 스트림은 `event: error` / `data: {봉투}` 로 그대로 나간다
- 클라이언트 3종이 전부 `error_envelope`를 import 해서 같은 함수로 해석한다

**`hint`가 이 시스템의 특징이다.** 타입이 아니라
"이 필드를 틀리면 무슨 일이 생기는지"를 쓴다.

### 4-4. 모르는 옵션은 막지 않고 알려준다

```python
outcome.ignored_fields   # extra="allow" 로 들어온 미지의 필드
   → logger.info("[VALIDATE] 모르는 옵션 %d개 무시")
   → session.write_event("options_ignored", fields=…)
   → SSE  event: notice  "무시된 옵션: general_settings.typo_opt"
   → done 프레임과 AskRun 요약에도 기록
```

**오타 옵션이 조용히 무시되는 것을 막는다.** 4곳에 남는다.

---

## 5. 요청 하나가 지나는 길

```
클라이언트                api_server              orchestrator            LLM/DB
    │                        │                        │                    │
    │ GET /session/connect   │                        │                    │
    ├───────────────────────▶│ create_session()       │                    │
    │◀── event: session ─────┤ (정원 10 초과 → 503)   │                    │
    │    connected           │                        │                    │
    │◀── ping (15s) ─────────┤ session.touch()        │                    │
    │                        │                        │                    │
    │ POST /ask              │                        │                    │
    │  X-Session-Id          │                        │                    │
    ├───────────────────────▶│ _prepare_ask()         │                    │
    │                        │  ├ _require_session    │                    │
    │                        │  ├ bind_session ← ContextVar                │
    │                        │  ├ gate() 로그인검사   │                    │
    │                        │  ├ QueryRequest 파싱   │                    │
    │                        │  ├ validate_request()  │                    │
    │◀── 400 봉투 ───────────┤  │   위반 → 즉시 거부  │                    │
    │                        │  ├ apply_embed_choice  │                    │
    │                        │  ├ merge_prompts       │                    │
    │                        │  └ try_begin_ask() ← 409 SESSION_BUSY       │
    │                        │                        │                    │
    │◀── event: notice ──────┤ 무시된 옵션 안내       │                    │
    │                        │ ensure_search_tables() ────────────────────▶│
    │◀── event: error ───────┤ 테이블 없음 → RESOURCE_UNAVAILABLE          │
    │                        │                        │                    │
    │                        │ acquire_ask_slot()     │                    │
    │◀── event: system ──────┤ "앞에 2명 대기 중…"    │                    │
    │◀── event: error ───────┤ 대기 초과 → GATE_FULL  │                    │
    │                        │                        │                    │
    │                        │ get_orchestrator(llm)  │                    │
    │◀── event: error ───────┤ 로드 실패 → UNKNOWN_MODEL                   │
    │                        │                        │                    │
    │                        │ acquire_local_llm_lock() ← local: 일 때만   │
    │                        │                        │                    │
    │                        │ run_pipeline() ───────▶│                    │
    │                        │                        │ NODE1 guard_in ───▶│
    │◀── event: log ─────────┤◀── out_queue ──────────┤   BLOCK → 종료      │
    │                        │                        │ NODE2 retrieve ───▶│
    │◀── event: log ─────────┤                        │   빈 결과 → 종료    │
    │                        │                        │ NODE3 generate ───▶│
    │◀── event: prompt ──────┤ trace_prompts          │                    │
    │                        │                        │ NODE4 guard_out ──▶│
    │                        │                        │   BLOCK → 1회 재생성│
    │◀── event: answer ──────┤ guard_out 직후 emit    │                    │
    │                        │                        │ NODE5 evaluate ───▶│
    │◀── event: eval ────────┤                        │                    │
    │                        │                        │                    │
    │                        │ finally:               │                    │
    │                        │  ├ rag_task 대기(30/600s)                   │
    │                        │  ├ release_local_llm_lock()                 │
    │                        │  ├ release_ask_slot()  │                    │
    │                        │  ├ persist_ask_run()  → logs/runs/…json     │
    │                        │  ├ slim_run_bodies()   │                    │
    │                        │  ├ session.end_ask()   │                    │
    │                        │  └ reset_session()     │                    │
    │◀── event: done ────────┤ {status, run_id, reason, timing_ms,         │
    │                        │  dropped_frames, partial}                   │
```

**중간에 클라이언트가 끊기면** `request.is_disconnected()`가 0.15초 간격으로
감지 → `session.request_cancel()` → `cancel_event.set()` +
**LLM HTTP 스트림 강제 종료** → 워커 스레드의 블로킹이 깨진다.
`finally`는 그대로 돌아 자원을 반납한다.

---

## 6. 검색 파이프라인

### 6-1. 전체 흐름

```
질문
 │
 ├─ 조합 검증 (여기서 막지 않으면 나중에 조용히 이상해진다)
 │    vocab_mapping + embed_variant → ValueError
 │    vocab_mapping + kiwi          → ValueError
 │
 ├─ query_mode
 │    none          원문 그대로
 │    expansion     LLM이 질문 확장 → 확장문에서 키워드 재추출
 │    vocab_mapping LLM이 DB 단어장(테이블별)에서 키워드 선택
 │
 ├─ keyword_mode
 │    trigram  정규식 \b\w{2,}\b, 불용어 제외, 최대 12개
 │    kiwi     형태소 분석 → "소득세법상" → "소득세법"
 │
 ├─ 임베딩 1회  (use_embed + infer_lock; embed_variant 시 use_embed_any)
 │     embed_variant 가 있으면 비교 모델로 질문 벡터 재계산
 │
 ├─ FOR table IN sources:                 (최대 4개)
 │      fusion_mode = rrf
 │        WITH src AS (…),
 │             keyword_search AS (ROW_NUMBER() OVER (ORDER BY similarity DESC)),
 │             vector_search  AS (ROW_NUMBER() OVER (ORDER BY embedding <=> q))
 │        SELECT …, 1.0/(60+k.rank) + 1.0/(60+v.rank) AS rrf_score
 │        FROM keyword_search k FULL OUTER JOIN vector_search v ON k.id = v.id
 │
 │      fusion_mode = score_based
 │        α · vector_sim + (1-α) · keyword_sim
 │
 ├─ 전체 후보 점수 내림차순 정렬  → candidates (덤프에 전량 보관)
 │
 ├─ use_reranker
 │    true   CrossEncoder(query, full_text) 재점수
 │             → min_score 필터
 │             → ⚠️ 전부 걸러지면 필터 무시 + WARN (fail-open)
 │             → top_k
 │    false  diversify_tagged(): top_k 가 한 테이블 독점이면
 │             마지막 자리를 다른 테이블 최고점과 교체
 │
 └─ use_graphdb  Neo4j 에서 시행령/판례 보충 → 컨텍스트 뒤에 병합
```

### 6-2. RRF — 왜 k=60 인가

```
 순위 →   1        2        3        5        10       20
 1/(60+r) 0.01639  0.01613  0.01587  0.01538  0.01429  0.01250
```

감쇠가 완만하다. 그래서:

```
 문서 A: dense 1위, sparse 미포함   →  0.01639          (단독 1위)
 문서 B: dense 5위, sparse 5위      →  0.01538 × 2 = 0.03077   ← 이긴다
```

**한쪽에서 1등인 것보다 양쪽에서 5등인 게 이긴다.**
이것이 RRF의 전부다 — **합의가 확신을 이긴다.**

k를 작게 하면(예: 10) 상위 순위 가중이 커져 단독 1위가 유리해지고,
k를 크게 하면 모든 순위가 평평해져 "많이 등장한 문서"가 유리해진다.
k=60은 원 논문(Cormack et al., 2009)의 값이며,
**현재 SQL에 하드코딩돼 있다** (`database/postgres_repo.py`의 RRF SQL, `1.0 / (60 + rank)` 부분).

### 6-3. `min_score` 주의

리랭커 `min_score`가 후보를 **전부** 걸러내면
필터를 무시하고 `top_k`를 쓴다. 로그에 남는다:

```
[RERANK] min_score=0.5000 가 후보를 전부 걸렀습니다. 필터 없이 top_k=3 를 씁니다.
```

**교육용 선택이다.** 빈 답변보다 낮은 품질 답변이 학습에 낫다는 판단.
**운영이라면 반대가 맞다** — 근거 없는 답변을 내느니 "찾지 못했습니다"가 정직하다.
→ `ADR-001`

리랭커 점수 범위도 매번 로그에 남으므로 `min_score`를 정할 때 참고할 것:

```
[RERANK] 점수 범위 min=-3.2451 max=6.8832 min_score=0.0000 통과=18/20
```

---

## 7. 세션과 로그 격리

### 7-1. 전역 로거 1개, 세션별 라우팅

```
                logging.getLogger("my_rag")     ← 전역 1개
                          │
                          ▼
                SessionLogRouter (Handler)
                          │
                   get_current_session()        ← ContextVar
                     ┌────┴─────┐
                세션 있음      세션 없음
                     │             │
        ┌────────────┴──┐          ▼
        ▼               ▼    logs/server.jsonl
 logs/sessions/     out_queue
   {sid}.jsonl    (asyncio.Queue)
   (레벨 무관 전량)     │
                        ▼  레벨 필터 (log_level 1/3/5)
                  SSE event: log → 그 사용자에게만
```

### 7-2. 스레드 경계를 넘는 법 ★

동기 코드(임베딩, SQL, llama.cpp)는 스레드풀에서 돈다.
**`run_in_executor`는 ContextVar를 자동 전파하지 않는다.**

```python
# session_manager.py
async def run_in_session_executor(func, /, *args, **kwargs):
    """워커 스레드에도 현재 세션 ContextVar를 복사한다."""
    ctx = contextvars.copy_context()                    # ① 스냅샷
    loop = asyncio.get_running_loop()
    def _call():
        return ctx.run(lambda: func(*args, **kwargs))   # ② 스레드 안에서 복원
    return await loop.run_in_executor(None, _call)
```

**오케스트레이터의 모든 블로킹 호출이 예외 없이 이 함수를 거친다.**
이걸 우회하면 그 안에서 찍은 로그가 `server.jsonl`로 새거나
**엉뚱한 사용자에게 간다.** 재현이 거의 불가능한 버그다.

> 🚨 **새 블로킹 호출을 추가할 때 반드시 `run_in_session_executor`를 쓸 것.**

### 7-3. 세션 수명

```
 GET /session/connect  →  create_session()      정원 초과 시 503
    │                     presence_open = True
    │
    ├─ POST /ask  →  try_begin_ask()            중복 시 409
    │                 processing = True
    │                 out_queue / ask_loop 생성
    │              …
    │                end_ask()  →  processing = False
    │                              maybe_destroy()
    │
    ├─ 클라 끊김  →  heartbeat finally
    │                 mark_presence_closed()
    │                 presence_open = False
    │                 request_cancel()
    │                 maybe_destroy()
    │
    └─ 60초마다  sweep_idle_sessions(SESSION_IDLE_TTL_SEC)
                   processing 중인 세션은 건드리지 않는다 ✔

 maybe_destroy():  presence_open == False  AND  processing == False  일 때만 제거
```

### 7-4. 로그 유실 계측

`emit_to_client()`가 큐에 못 넣으면 `dropped_frames`를 센다.
이 값은 `done` 프레임과 `AskRun` 요약에 들어간다.

```json
{"status": "completed", "dropped_frames": 3, ...}
```

**`dropped_frames > 0` 이면 "화면에서 본 게 전부가 아니다"를 뜻한다.**
유실을 세지 않으면 "로그가 왜 안 보이지?"에 영원히 답할 수 없다.

---

## 8. 자원 관리와 동시성

### 8-1. 임베딩 · 리랭커 슬롯

```
 use_embed("jhgan/ko-sroberta-multitask")    ← indexed 기본 임베딩
 use_embed_any("BAAI/bge-m3")               ← 비교(embed_variant) 임베딩
    ├ validate → 허용목록 아니면 ValueError
    ├ _acquire  refcount += 1  (double-checked locking)
    │  yield slot
    │     with slot.infer_lock:      ← 슬롯별 추론 직렬화
    │         model.get_text_embedding(...)
    └ finally: _release  refcount -= 1

 60초마다 sweep_idle_models()
    refcount == 0  AND  유휴 >= MODEL_IDLE_TTL_SEC(600)
       → 언로드 → gc.collect() → torch.cuda.empty_cache()
```

**새 모델을 쓸 때는 반드시 `use_embed` / `use_embed_any` / `use_rerank` 컨텍스트를 통과할 것.**
직접 `HuggingFaceEmbedding(...)`을 만들면 TTL·refcount·락이 전부 사라진다.
비교 임베딩(`embed_variant`)은 `retriever`가 `use_embed_any`로 슬롯을 쓴다.

### 8-2. 동시성 3단

```
 ① 입장 게이트   _ask_sem = Semaphore(--max-concurrent-user, 기본 4)
      대기 상한 --max-waiting (기본 5) 초과 → GATE_FULL 입구컷
      대기 중 "앞에 N명" SSE 안내
      대기 중 0.5초마다 is_disconnected() 확인 → 끊기면 락 안 잡음 ★

 ② 로컬 LLM 락   _local_sem = Semaphore(1)
      is_local_model(llm_model) OR is_local_model(eval_model) 일 때만
      llama_cpp.Llama 는 스레드 안전하지 않다

 ③ 컨트롤러 락   LLMController._infer_lock  (local 전용, 이중 방어)
```

②가 없으면 **GGUF 동시 추론 → 세그폴트 또는 출력 뒤섞임.**
`DEFAULT_LLM`이 `openrouter:`면 이 문제는 **잠복**한다.
`local:`로 바꾸는 순간 발현하고 원인 추적이 매우 어렵다.

### 8-3. 취소 3단

```
 사용자 Stop / 브라우저 닫기
    ├─① request.is_disconnected() 폴링 (0.15s 펌프 / 0.5s 게이트)
    ├─② session.cancel_event.set()  → raise_if_cancelled() 가 각 노드에서 확인
    └─③ session.request_cancel()    → _active_stream 강제 close
                                        → 워커 스레드의 for chunk 루프가 깨짐
```

**③이 핵심이다.** `Task.cancel()`은 스레드풀 안에서 블로킹 중인
동기 코드를 깨우지 못한다. 유일한 방법은 **기다리고 있는 자원을 닫는 것**이다.

### 8-4. `workers=1` 은 설계다

```
 프로세스 메모리에 있고 워커 간 공유 불가:
   _sessions        session_manager.py       ✗ 세션 끊김
   _bound           session_auth.py          ✗ 중복 로그인 차단 무력화
   _local_sem       local_llm_gate.py        ⚠ GGUF 동시추론 → 크래시
   _ask_sem         local_llm_gate.py        △ 상한이 N배로 느슨
   _slots           model_registry.py        △ 메모리 N배
   _LLM_CACHE       llm_handler.py           △ GGUF N배 (치명적)
```

**늘리려면:**
1. `X-Session-Id` sticky routing (nginx/traefik) → `_sessions`/`_bound` 해결
2. `DEFAULT_LLM=local:*` 이면 그래도 워커 1개 유지 (필수)
3. 그 전에 **먼저 측정할 것.** `logs/runs/*/*.json`의 `timing`을
   모으면 병목이 검색인지 리랭커인지 LLM인지 바로 나온다.

→ `ADR-005`

---

## 9. 엔드포인트 전체

| 메서드 | 경로 | 하는 일 |
|---|---|---|
| GET | `/session/connect` | **SSE.** 세션 생성 + 15초 하트비트. 정원 초과 시 503 |
| GET | `/session/status` | 접속 목록·처리 중 수 |
| POST | `/ask` | **SSE.** 질문 → 파이프라인 → 답변 |
| POST | `/ask/validate` | LLM·DB 없이 옵션만 점검 (무료) |
| GET | `/status` | 모니터링 종합 (connected/processing/waiting/local_llm/models) |
| GET | `/local-llm/status` | 로컬 GGUF 락 상태 |
| GET | `/health` | 부팅 상태·계약 점검·모델 카탈로그 수 |
| GET | `/options/schema` | **요청 옵션 전체 + 기본값 + hint** |
| GET | `/errors/codes` | **에러 코드 taxonomy 전체** |
| GET | `/llm/models` | 허용 LLM 목록 |
| GET | `/embed/models` | 허용 임베딩 목록 (+ `indexed`, `default`) |
| GET | `/rerank/models` | 허용 리랭커 목록 |
| GET | `/quota` | 현재 접속 클라이언트의 일일 질문 쿼터와 쿨다운 잔여 상태 |
| GET | `/` | 로컬 데모 UI(`index.html`) |
| GET | `/DEMO_DISCLAIMER.txt` | 이용 안내·면책 전문 |
| GET | `/admin/status` | 🔒 `CS_ALLOW_IP` 만 |
| GET | `/admin/users` | 🔒 로그인 모드 명단 조회 |
| POST | `/admin/users` | 🔒 명단 추가 |
| DELETE | `/admin/users/{user_id}` | 🔒 명단 삭제 |

### 대표 요청

```bash
# 1) 접속
curl -N http://127.0.0.1:8000/session/connect
# ← event: session
#   data: {"event":"connected","session_id":"a1b2c3…"}

# 2) 옵션만 점검 (LLM 안 씀, 무료)
curl -X POST http://127.0.0.1:8000/ask/validate \
  -H "X-Session-Id: a1b2c3…" -H "Content-Type: application/json" \
  -d '{"query":"거주자 판정 기준 일수는?",
       "general_settings":{"query_mode":"expansion","keyword_mode":"kiwi"}}'

# 3) 질문
curl -N -X POST http://127.0.0.1:8000/ask \
  -H "X-Session-Id: a1b2c3…" -H "Content-Type: application/json" \
  -d '{
    "query": "거주자 판정 기준 일수는?",
    "general_settings": {
      "llm_model": "openrouter:google/gemini-3.5-flash-lite",
      "query_mode": "expansion",
      "fusion_mode": "rrf",
      "sources": ["tax_laws_master", "tax_precedents_master"],
      "keyword_mode": "kiwi",
      "capture": "full",
      "log_level": 3
    },
    "rerank_settings": {"enabled": true, "top_k": 5,
                        "advanced": {"min_score": 0.0}},
    "eval_settings": {"enabled": false}
  }'
```

---

## 10. 요청 옵션 전체

**실시간 최신본은 `GET /options/schema`.** 아래는 사본이다.

### `general_settings`

| 필드 | 타입 | 허용/범위 | 기본 |
|---|---|---|---|
| `llm_model` | model_ref | `GET /llm/models` | `.env` `DEFAULT_LLM` |
| `embed_model` | model_ref | `GET /embed/models` | `RAG_EMBEDDING_MODEL_ID` |
| `embed_variant` | model_ref | 비교 실험용. 보통 비움 | `null` |
| `log_level` | enum | `1` 오류만 / `3` 일반 / `5` 디버그 | `3` |
| `query_mode` | enum | `none` / `expansion` / `vocab_mapping` | `none` |
| `fusion_mode` | enum | `rrf` / `score_based` | `rrf` |
| `alpha` | float | 0.0 ~ 1.0 (`score_based` 전용) | `0.5` |
| `sources` | enum_list | 허용 4테이블의 부분집합 | 전체 |
| `keyword_mode` | enum | `trigram` / `kiwi` | `trigram` |
| `capture` | enum | `lean` / `full` / `forensic` | 서버 상한 |
| `temperature` | float | 0.0 ~ 2.0 | `LLM_TEMPERATURE` |
| `max_tokens` | int | 1 ~ 32768 | `LLM_MAX_TOKENS` |
| `use_guardrails` | bool | | `true` |
| `trace_prompts` | bool | `forensic`이면 강제 `true` | `false` |
| `stream_logs` | bool | | `true` |
| `use_graphdb` | bool | ⚠️ `/options/schema`에 안 나옴 (규칙 누락) | `false` |
| `company_summary` | str | ⚠️ 규칙 없음. **빈 문자열 금지** (16절) | `.env` |

### `rerank_settings`

| 필드 | 타입 | 범위 | 기본 |
|---|---|---|---|
| `enabled` | bool | | `false` |
| `top_k` | int | 1 ~ 20 | `3` |
| `advanced.model` | model_ref | `GET /rerank/models` | `RAG_RERANKER_MODEL_ID` |
| `advanced.min_score` | float | 0.0 ~ 1.0 | `0.0` |

### `eval_settings`

| 필드 | 타입 | 범위 | 기본 |
|---|---|---|---|
| `enabled` | bool | | `false` |
| `eval_model` | model_ref | `GET /llm/models` | `DEFAULT_EVAL_LLM` |
| `threshold` | float | 0.0 ~ 1.0 | `0.5` |
| `strict_mode` | bool | | `false` |
| `advanced.metrics` | enum | `all` / `answer_relevancy` / `context_precision` | `all` |
| `advanced.async_mode` | bool | ⚠️ **읽지 않음.** 구 클라 호환용 | `true` |
| `advanced.use_cache` | bool | ⚠️ 동일 | `false` |

### 조합 제약 (`INCOMPATIBLE_OPTIONS`)

```
 query_mode=vocab_mapping  +  keyword_mode=kiwi       → 거부
   vocab_mapping 은 LLM 이 키워드를 만드는 경로라 형태소 분석이 개입할 자리가 없다.

 query_mode=vocab_mapping  +  embed_variant           → 거부
   단어장 매핑은 원본 테이블 검색 경로다.

 embed_model 과 embed_variant 가 서로 다른 비교 모델    → 거부
```

---

## 11. 환경변수 전체

**전체 목록과 "안 넣으면 무슨 일이 생기는지"는 `.env.example` 참조.**

| 변수 | 기본 | 없으면 |
|---|---|---|
| `DEFAULT_LLM` | — | 🔴 **부팅 실패** |
| `TAX_DB_URI` | — | 🔴 **부팅 실패** |
| `OPENROUTER_API_KEY` | — | `openrouter:` 모델 로드 실패 |
| `DEFAULT_EVAL_LLM` | `openrouter:google/gemini-3.5-flash-lite` | 기본값 |
| `DEFAULT_PROMPT_FILE` | `prompts_ko_세무.json` | 기본 파일 탐색 |
| `RAG_EMBEDDING_MODEL_ID` | `jhgan/ko-sroberta-multitask` | 카탈로그 첫 `indexed` |
| `RAG_RERANKER_MODEL_ID` | `Dongjin-kr/ko-reranker` | 카탈로그 첫 항목 |
| `RAG_CAPTURE` | `lean` | `lean` |
| `LLM_TEMPERATURE` | `0.1` | 기본값 + 경고 |
| `LLM_MAX_TOKENS` | `4096` | 기본값 + 경고 |
| `SESSION_IDLE_TTL_SEC` | `1800` | 기본값 (최소 60) |
| `MODEL_IDLE_TTL_SEC` | `600` | 기본값 (최소 30) |
| `COMPANY_SUMMARY` | 내장 문구 | 내장 문구 |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | `bolt://localhost:7687` / `neo4j` / — | 그래프 비활성 |
| `CS_ALLOW_IP` | `127.0.0.1` | localhost만 `/admin` |
| `RAG_SERVER` | `http://127.0.0.1:8000` | (cs_console 전용) |

> **잘못된 값은 부팅을 막지 않고 경고만 남긴 뒤 기본값을 쓴다.**
> 경고는 `GET /health` 의 `boot.env_warnings` 에 노출된다.
> **부팅 후 반드시 한 번 확인할 것.**

### 서버 CLI

```
 --prompt-file PATH          프롬프트 JSON (기본: DEFAULT_PROMPT_FILE)
 --capture {lean,full,forensic}  캡처 등급 = 서버 상한 (기본: RAG_CAPTURE)
 --max-concurrent-user N     동시 /ask 상한 (기본 4)
 --max-waiting N             대기열 상한, 초과 시 입구컷 (기본 5)
 --max-connected N           동시 접속 정원 (기본 10)
 --login-mode                X-User-Id 검사 활성화
 --print-models              기동 시 LLM 목록 출력
```

---

## 12. 관측·디버깅

### 12-1. 3중 기록

```
 ① logs/sessions/{sid}.jsonl     세션 이벤트 시계열 — 항상
      connected / ask_start / log / options_ignored / validate_rejected /
      gate_full / eval_report / answer / error / ask_run(요약) / ask_end

 ② logs/runs/{sid}/{run_id}.json 실행 전체 덤프 — capture=full 이상
      request / resolved / retrieval(hits + candidates) / generation /
      guardrails(raw) / eval / timing / infra
      forensic 이면 llm_calls[].messages 까지
      → tmp + os.replace 원자적 쓰기 (덤프가 깨지지 않는다)

 ③ logs/server.jsonl             세션 없는 로그 — 부팅·스윕·세션 밖 예외
```

### 12-2. `capture` 3단계

| 등급 | 세션 JSONL | JSON 덤프 | 검색 후보/문서 | LLM 프롬프트 |
|---|:---:|:---:|:---:|:---:|
| `lean` | ✓ | — | — | — |
| `full` | ✓ | ✓ | ✓ | — |
| `forensic` | ✓ | ✓ | ✓ | ✓ |

**클라이언트 요청과 서버 상한의 교집합이 적용된다.**
클라가 `forensic`을 요청해도 서버가 `--capture lean`이면 `lean`이다.
강등되면 로그에 남는다:

```
[CAPTURE] 요청=forensic 를 서버 상한=lean 로 내립니다.
```

`forensic`은 `trace_prompts`를 **강제로 켠다.**
"프롬프트를 다 기록하는 모드인데 트레이스는 꺼져 있는" 모순을 막는다.

### 12-3. 디버깅 경로

| 증상 | 볼 곳 |
|---|---|
| 답변이 안 나온다 | `logs/sessions/{sid}.jsonl` → `ask_run` 의 `stage` / `error_where` |
| 검색이 이상하다 | `capture=full` 재실행 → 덤프의 `retrieval.candidates` (점수 포함) |
| 프롬프트가 의심된다 | `capture=forensic` → 덤프의 `llm_calls[].messages` |
| 가드레일이 왜 막았나 | 덤프의 `guardrails_in.raw` / `guardrails_out.raw` |
| 로그 일부가 안 보인다 | `done` 프레임의 `dropped_frames` |
| 400/422가 난다 | 봉투의 `violations[].hint` — **무엇을 고칠지 이미 적혀 있다** |
| 옵션 이름을 모르겠다 | `GET /options/schema` |
| 에러 코드 의미 | `GET /errors/codes` |
| 동시성이 이상하다 | `GET /status` |
| 서버가 안 뜬다 | `[Fatal Error]` grep → `bootstrap` / `model_registry` / `settings` |

### 12-4. `AskRun` 요약 필드

`ask_run` 이벤트에 들어가는 것 (전체는 덤프에):

```
 run_id  status  stage  error_message  error_where
 timing{t_llm_init_ms, t_retrieve_ms, t_generate_ms, t_eval_ms, t_total_ms}
 dump_path  dump_failed
 context_chars  answer_chars  hit_count  candidate_count  llm_call_count
 guardrails_in / guardrails_out (+ raw 200자)
 capture  dropped_frames  ignored_fields  partial
 eval_enabled  eval_error  eval_metrics[]
 expanded_query  keywords[]
```

`partial: ["eval"]` — **평가는 실패했지만 답변은 나왔다**를 표현한다.
이분법(성공/실패)으로 뭉개지 않는다.

---

## 13. 평가 하네스

### 13-1. 실행

```bash
# 시험지 검증만 (LLM·DB 불필요)
python evaluate_rag.py --testset testset.json --validate-only

# 채점기 자체 검증 (LLM·DB 불필요)
python evaluate_rag.py --self-test

# 기본 — 검색과 답변을 분리 채점
python evaluate_rag.py --testset testset.json \
  --prompt-file prompts_ko_세무.json \
  --model openrouter:google/gemini-3.5-flash-lite

# 엔드투엔드 (가드레일·평가 포함) + 의미 유사도
python evaluate_rag.py --testset testset.json --mode pipeline --semantic

# 검색 파라미터 오버라이드
python evaluate_rag.py --testset testset.json \
  --top-k 5 --fusion rrf --query-mode expansion --use-reranker
```

### 13-2. 두 모드의 용도

```
 --mode nodes    (기본)  node_retrieve → node_generate
                          가드레일·DeepEval 생략
                          → 검색 실패와 생성 실패를 분리할 수 있다
                          → 성능을 '개선'하는 중이라면 이것

 --mode pipeline          run_pipeline (서버 /ask 와 동일)
                          가드레일·재시도·평가 포함
                          → 배포 직전 '회귀'를 확인한다면 이것
```

**왜 나눠 재는가:** RAG가 틀렸을 때 원인은 둘 중 하나다.
(a) 검색이 정답 문서를 못 가져왔다 → 임베딩/리랭커/청킹 문제
(b) 문서는 왔는데 LLM이 못 읽었다 → 프롬프트/모델 문제
**최종 답변만 보면 이 둘을 구분할 수 없어 엉뚱한 곳을 고치며 시간을 태운다.**

### 13-3. 지표

| 지표 | 무엇 | 조건 |
|---|---|---|
| `context_hit` | 정답 본문이 컨텍스트에 들어왔나 | 항상 |
| `hit@1/3/5/10` | gold_doc_id 를 순위 안에 끌어왔나 | `match_method=="id"` 만 |
| `MRR` | 정답을 얼마나 위에 올렸나 | 동일 |
| `pool_hit@k` | 후보 풀 기준 (top_k 컷 전) | 동일 |
| `cut_by_topk_rate` | 후보엔 있었는데 top_k 에서 밀린 비율 | 동일 |
| `kw_recall` | must_include 재현율 0~1 | 항상 |
| `hallucination` | must_not_include 등장 | 항상 |
| `absence_ok` | absence 문항에서 올바르게 기권했나 | absence 만 |
| `sem_sim` | 답변 vs 정답 코사인 | `--semantic` |
| `llm_judge` | LLM 심판 0~1 | `--judge` |

### 13-4. 숫자를 읽는 법

**hit@k 는 `--top-k` 와 분리돼 있다.**
고정 집합(1/3/5/10)으로 잰다. 이유:

> 과거에는 k = args.top_k 로 두어, top_k 를 바꾸면 채점 기준까지 함께 움직여
> **점수 변화의 원인(검색 개선 vs 채점 완화)을 구분할 수 없었다.**

그래서 이제:
- **hit@1이 오르면** → 순위 개선 (리랭커가 일을 했다)
- **hit@10만 오르면** → 그물 확대 (더 많이 가져왔을 뿐)

> 🎓 **자(ruler)와 대상을 분리하라.**
> 시스템 설정이 바뀔 때 측정 기준도 같이 바뀌면 **어떤 개선도 검증할 수 없다.**

**통계적 유의성.** `wilson_ci()`가 신뢰구간을 낸다.
n=50이면 ±12%p 수준이다. **50문항에서 0.68 → 0.74 는 의미 없다.**

### 13-5. 산출물

```
 logs/testruns/{run_id}/
   manifest.json     ← testset_sha256 + git rev + config 전체 + fail_ids
   summary.json      ← 집계
   results.csv       ← 문항별 행
   cases/{item}.json ← AskRun 전체 (통과/실패 모두)
```

**`manifest.json`이 재현성의 핵심이다.**
6개월 뒤 "0.68이 뭐였지?"에 답할 수 있다.

⚠️ **모델 A/B 비교 시 반드시 같은 testset 파일 + 같은 옵션으로 돌릴 것.**
`kw_threshold`, `require_context`, `id_check` 중 하나만 달라도 통과율이 크게 바뀐다.

---

## 14. 테스트

```bash
# 서버·DB 없이. 몇 초.
python -m pytest tests/ -q

# pytest.ini 가 있으면
pytest
```

`pytest.ini` 가 없으면 `ImportError`가 난다 (rootdir 문제). 반드시 추가할 것:

```ini
[pytest]
pythonpath = .
testpaths = tests
asyncio_mode = auto
```

### 지금 지켜지는 것

```
 계약      SSE 프레임 경계 · 이중인코딩 금지 · 봉투/exit코드 정합성
           Pydantic 스키마 · 옵션 규칙 · 테이블 허용목록
 순수함수  transient vs "generate" 오탐 방지 ★
           키워드 바인딩 (정규식 vs trigram 분리) ★
           diversify · capture 상한 강등 · 임베딩 선택 번역
 하네스    동결 픽스처 회귀 방어 ★ · 골든셋 검증 · self-test 통합
 세션      레벨별 스트리밍/파일 분리
```

### 지금 안 지켜지는 것 ⚠️

```
 ✗ 세션 생명주기 (create → ask → sweep → destroy)
 ✗ 게이트 동시성 (정원초과 / 대기열 / 반납 / 대기중 끊김)
 ✗ 취소 경로
 ✗ finally 자원 반납 (세마포어 누수)
 ✗ /ask 스트리밍 통합
```

`local_llm_gate`는 `tests/test_gate.py`로 DB·LLM 없이 검증한다.
임베딩 슬롯·비교 임베딩은 `tests/test_embed_slots.py` 등으로 커버한다.

---

## 15. 확장하기

### 15-1. 새 검색 테이블 — **2줄**

```python
# schema.py
ALLOWED_SOURCE_TABLES = (..., "tax_rulings_master")          # ①
SOURCE_LABEL_COLS = {..., "tax_rulings_master": ("title", "ruling_no")}   # ②
```

검증(`normalize_sources`), SQL(`_src_cte`), validation enum,
trigram 인덱스 DDL, `/options/schema` 가 **전부 자동으로 따라온다.**

### 15-2. 새 임베딩/리랭커 — **1줄**

```json
{"id": "BAAI/bge-m3", "label": "bge-m3 (비교용)"}
```

카탈로그·검증·`/embed/models`가 자동. 비교 테이블은 저장소 루트에서
`python db_build/04_build_compare_embedding_tax.py --model BAAI/bge-m3` 로 만든다.
한 테이블만 만들 때는 `--source-table` 을 붙인다.

### 15-3. 새 파이프라인 노드

```python
# workflow_orchestrator.py
async def node_my_step(self, state: RAGState) -> RAGState:
    run = state.run
    if run is not None:
        run.mark(stage="my_step")
    raise_if_cancelled()                              # ① 취소 확인
    result = await run_in_session_executor(...)       # ② 반드시 이걸로
    return state

# run_pipeline 순서에 삽입
state = await self.node_retrieve(state)
state = await self.node_my_step(state)                # ③
state = await self.node_generate(state)
```

**①②를 빠뜨리면 취소가 안 먹고 로그가 샌다.**

### 15-4. 새 에러 코드

```python
# error_envelope.py — 3개 dict 에 추가하면 /errors/codes 에 자동 반영
HTTP_BY_CODE     = {..., "MY_CODE": 400}       # 스트림 전용이면 생략
EXIT_BY_CODE     = {..., "MY_CODE": 2}
SEVERITY_BY_CODE = {..., "MY_CODE": "reject"}
STREAM_ONLY_CODES = frozenset({..., "MY_CODE"})   # 스트림 전용일 때만
```

빠뜨리면 `/errors/codes` 에 `null`이 보여 **즉시 티가 난다.**

### 15-5. 새 SSE kind

```python
# sse.py
SSE_KIND_MYKIND = "mykind"
_SSE_KINDS = frozenset({..., SSE_KIND_MYKIND})
_STRUCTURED_KINDS = frozenset({...})   # JSON 객체를 보낼 거면 여기도
```

목록에 없는 kind는 `log`로 강등되고 `logger.error`가 남는다.

### 15-6. 새 도메인 — **어렵다** ⚠️

`schema.py`가 세법 4테이블을 하드코딩하고,
`rag_run.LAW_GRAPH_TABLE = "tax_laws_master"` 가 고정이며,
전 파일 헤더 주석이 `DOMAIN: 세법`이다.
도메인 중립화가 필요하면 별도 설계가 선행돼야 한다.

---

## 16. 알려진 한계

정직하게 나열한다. **아는 한계는 모르는 한계보다 훨씬 낫다.**

### 16-1. ~~`company_summary=""` → NameError~~ ✅ 해결

`workflow_orchestrator`가 `settings.company_summary`를 fallback으로 쓰고,
`validation.FIELD_RULES`에 `general_settings.company_summary`(빈 문자열 거부)가 있다.

### 16-2. 🟠 LLM 캐시 축출이 제한적이다

`_LLM_CACHE`(`llm_handler.py`)와 `_ORCHESTRATOR_CACHE`
(`workflow_orchestrator.py`)는 **한 번 들어가면 프로세스 종료까지 유지**된다.

**완화:** `local:*` GGUF 교체 시 이전 로컬 LLM·Orchestrator 캐시를 비우는 축출이 들어가 있다.
`openrouter:`만 쓰면 여전히 가볍고, 서로 다른 `local:*.gguf`를 번갈아 쓰면 메모리 피크는 줄지만
**영구 상주 캐시 구조 자체는 그대로**다.

임베딩·리랭커·비교 임베딩은 `_slots` + `sweep_idle_models()`로 refcount + TTL 관리된다.

### 16-3. ~~`_variant_embed_models` 슬롯 밖~~ ✅ 해결

비교 임베딩은 `model_registry.use_embed_any()` → `_slots` (refcount, TTL, `infer_lock`).
`retriever`의 독립 dict 캐시는 제거했다.

### 16-4. 🟡 `/health` 가 실패할 수 없다

```python
_BOOT["contracts"] = {"ok": True, "problems": problems}   # ok 는 항상 True
return {..., "db": {"pool": "ready"}}                     # 하드코딩
```

**`contracts.problems` 를 직접 볼 것.**

### 16-5. 🟡 종료 시 in-flight 드레인이 없다

`lifespan` 종료에 신규 요청 차단·드레인·세션 종료 이벤트가 없다.
처리 중이던 `/ask` 가 중간에 끊기고 덤프가 미완성으로 남는다.
데모·학습용 단일 서버에서는 피해가 작지만 운영 전환 시 필수.

### 16-6. ~~검색 루프 중 취소를 못 본다~~ ✅ 해결

`retriever` 테이블 루프에 `raise_if_cancelled()`가 있다.

### 16-7. ~~`use_graphdb` 가 `/options/schema` 에 없다~~ ✅ 해결

`validation.FIELD_RULES`에 `general_settings.use_graphdb` 규칙이 있다.

### 16-8. 🟢 로그 로테이션이 없다

`capture=forensic` 으로 오래 돌리면 디스크가 찬다.
덤프 하나가 프롬프트 전문 + 문서 원문을 담아 수백 KB가 될 수 있다.
주기적으로 `logs/` 를 비우거나 `logrotate` 를 붙일 것.

### 16-9. 🟢 RRF k=60 이 SQL 에 하드코딩

`database/postgres_repo.py`의 RRF SQL(`1.0 / (60 + rank)`). RRF 실험을 하려면 SQL 을 고쳐야 한다.
`search_constants.RRF_K` 로 빼는 것이 이 프로젝트 원칙에 맞다.

### 16-10. 🟢 `Neo4jRepo.close()` 는 no-op

`PostgresRepo.close()` 는 풀 반납(필수), `Neo4jRepo.close()` 는 아무것도 안 한다.
`retriever.py` 가 둘을 같은 모양으로 써서 읽는 사람이 혼동한다.

### 16-11. 🟢 `eval_settings.advanced.async_mode` / `use_cache` 는 읽히지 않는다

구 클라이언트 호환용. `contracts.py` 주석에 정직하게 적혀 있다.

### 16-12. 🟢 성능을 아무도 측정한 적이 없다

`timing_ms` 는 나오지만 모아본 적이 없다.
**병목이 검색인지 리랭커인지 LLM인지 모르는 상태에서
동시성 파라미터를 조정하는 것은 추측이다.**

### 16-13. 🟠 인터넷에 직접 노출하지 말 것

이 서버는 로컬·사설망 데모용이다. 공개 인터넷에 그대로 열면 안 된다.

- CORS가 `allow_origins=["*"]` 로 열려 있다 (`api_server.py`).
- `--login-mode` 의 아이디 입구컷은 **명단에 있는 아이디인지만** 본다. 비밀번호 검증이 없다.
- `/admin/*` 은 IP 허용 목록(`CS_ALLOW_IP`)만 본다. 인증 토큰이 없다.
- 기본 바인드는 `127.0.0.1` 이다. `--host 0.0.0.0` 은 신뢰할 수 있는 네트워크에서만 쓴다.

외부에 열어야 한다면 앞단에 인증이 있는 리버스 프록시를 두고 CORS 를 좁힐 것.

---

## 17. 처음 읽는 사람을 위한 순서

```
 1단계 (30분) ── 계약부터. 여기가 시스템의 절반이다.
    schema.py            테이블 허용목록, 식별자 규칙
    sse.py               프레임 8종
    error_envelope.py    에러 taxonomy   ★ 가장 중요
    errors.py            transient / overload 판정
    contracts.py         요청 스키마
    validation.py        값 범위 + 조합 제약 + hint

 2단계 (30분) ── 관측. "무엇이 기록되는가"
    rag_run.py           AskRun / capture 3단계 / 원자적 덤프
    session_manager.py   ContextVar 로그 라우팅   ★ 이 시스템의 백미

 3단계 (60분) ── 파이프라인
    workflow_orchestrator.py   5노드
    retriever.py               검색 (mode 분기 주의)
    database/postgres_repo.py  하이브리드 SQL

 4단계 (45분) ── 자원 · 동시성
    model_registry.py    슬롯 refcount + TTL   ★ 모범 사례
    local_llm_gate.py    세마포어 2종 + 대기 상한
    llm_handler.py       LLMController, 취소 전파   ★

 5단계 (60분) ── 진입점
    api_server.py        lifespan → _prepare_ask → _ask_stream_events
                         ⚠️ _ask_stream_events 는 약 300줄짜리 긴 함수다.
                            함수 끝의 finally 블록을 먼저 읽으면 구조가 보인다.

 6단계 (45분) ── 평가
    evaluate_rag.py      docstring 0~6절만 읽어도 절반이 이해된다

 ─────────────────────────────────
 총 약 4시간 30분
```

**왜 계약부터인가:** 계약을 먼저 읽으면 나머지가 전부
**"그 계약을 지키는 방법"** 으로 읽힌다.
`api_server.py` 를 먼저 열면 약 300줄짜리 함수 안에서 길을 잃는다.

---

## 18. 자주 막히는 곳

| 증상 | 원인 | 해결 |
|---|---|---|
| `MISSING_SESSION` | `X-Session-Id` 헤더 없음 | `GET /session/connect` 먼저 |
| `SESSION_BUSY` (409) | 같은 세션이 이미 처리 중 | 이전 요청 완료 대기 |
| `SESSION_CAPACITY` (503) | 접속 정원 초과 | `--max-connected` 상향 |
| `GATE_FULL` | 대기열 초과 | `--max-waiting` 상향 또는 재시도 |
| `UNKNOWN_MODEL` | 모델 id 오타 | `GET /llm/models` — **fallback 안 한다** |
| `INCOMPATIBLE_OPTIONS` | 조합 제약 위반 | 봉투의 `violations[].reason` |
| `RESOURCE_UNAVAILABLE` | 검색/비교 테이블 없음 | 메시지에 만드는 명령이 들어 있다 |
| 답변이 "관련된 문서를 찾을 수 없습니다" | 검색 0건 | `capture=full` → `candidates` 확인 |
| 리랭커 켰는데 결과가 이상 | `min_score` 과다 | `[RERANK] 점수 범위` 로그 확인 |
| 로컬 GGUF 가 느리다 | `_local_sem` 동시 1건 | 설계다. `--max-waiting` 조정 |
| 로그가 안 보인다 | `log_level` 또는 `stream_logs` | `done` 의 `dropped_frames` 확인 |
| `pytest` ImportError | rootdir 문제 | `pytest.ini` 또는 `python -m pytest` |
| 세션이 자꾸 끊긴다 | 유휴 타임아웃 | `SESSION_IDLE_TTL_SEC` 상향 |
| 서버 재시작하면 모델 재로드 | 정상 | `_slots` 는 프로세스 메모리 |

---

## 19. 라이선스와 데이터 출처

- **코드**: 별도의 라이선스 파일이 없으며, 코드의 사용·복제·수정·재배포에 관한 권리는 저작권자에게 있습니다.
- **데이터**: 법령·판례·조세심판원 결정례는 법제처 국가법령정보 Open API로,
  개정세법 해설은 국세청(법규과) 발행 자료(PDF)에서 추출합니다. 이용 조건과 고지는 3절을 보세요.
- **저장소에 포함된 데이터**: `testset*.json`(평가용 질의응답)에는 법령·판례 원문 발췌가 들어 있습니다.
  법령·판례는 저작권법 제7조(보호받지 못하는 저작물)에 해당한다는 취지로 안내되지만, 이 저장소는 법적 판단을 보증하지 않으며 데이터의 이용 조건은 출처 기관의 안내가 우선합니다.
- **저장소에 포함하지 않는 데이터**: 법령·판례·조세심판원 재결례를 포함한 원본 수집 데이터(`data/`)와 개정세법 해설서 본문 추출물, DB 덤프는
  **제공하지 않으며 재배포하지 않습니다.** 원문 PDF를 직접 구해
  [`DB_SETUP_GUIDE.md`](DB_SETUP_GUIDE.md)의 절차(`db_build/03_build_tax_amendments.py`)로 재생성하세요.
  이 때문에 `tax_amendments_2026` 테이블 없이는 개정세법 관련 검색·평가가 비어 있을 수 있습니다.
- API 키 등 비밀값은 `.env`에만 두며 저장소에 포함되지 않습니다 (`.env.example` 참고).

---

## 부록 — 관련 문서

| 문서 | 내용 |
|---|---|
| [`ADR.md`](ADR.md) | 설명 없이는 오해받을 설계 결정 (ADR-001 ~ 014) |
| [`STATE.md`](STATE.md) | 전역 상태 지도. 워커를 늘리면 무엇이 깨지는가 |
| [`DB_SETUP_GUIDE.md`](DB_SETUP_GUIDE.md) | 세법 DB 마스터 재구축 순서 및 SOP (`db_build/`) |
| [`DEMO_DISCLAIMER.txt`](DEMO_DISCLAIMER.txt) | 데모 웹 UI 이용 안내 및 면책 조항 전문 |
| [`.env.example`](.env.example) | 환경변수별 "안 넣으면 무슨 일이 생기는지" |
