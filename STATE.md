# taxlaw-rag — 전역 상태 지도

> **왜 이 문서가 필요한가**
> `api_server.py` 마지막 줄이 `workers=1` 이다.
> 그 이유가 코드 한 줄에만 있으면, 6개월 뒤 누군가 그걸 늘렸다가
> **재현 불가능한 장애**를 만난다.
>
> 이 표는 "무엇이 어디에 있고, 워커를 늘리면 무엇이 깨지는가" 에 답한다.

---

## 1. 한눈에

```
                        프로세스 메모리                    디스크
                    ┌──────────────────────┐      ┌──────────────────────┐
 api_server :8000   │ _sessions       ✗공유 │      │ logs/sessions/*.jsonl│
                    │ _bound          ✗공유 │      │ logs/runs/{sid}/*.json│
                    │ _bound_by_session ✗   │      │ logs/server.jsonl    │
                    │ _slots          △중복 │      └──────────────────────┘
                    │ _LLM_CACHE      ⚠️위험 │              ✅ 공유 가능
                    │ _ORCHESTRATOR_CACHE ⚠️ │
                    │ _ask_sem        △워커별│      │ PostgreSQL 커넥션 풀  │
                    │ _local_sem      ⚠️위험 │      │ SimpleConnectionPool  │
                    │ _users          △동일 │      │ (1, 20)  프로세스별   │
                    │ GLOBAL_PROMPTS  △동일 │      │ Neo4j 드라이버        │
                    │ _BOOT           △동일 │      └──────────────────────┘
                    │ _server_capture △동일 │
                    └──────────────────────┘
```

```
 ✗공유  워커를 늘리면 기능이 깨진다
 ⚠️위험  워커를 늘리면 크래시 또는 OOM
 △중복  워커별로 복제되지만 기능은 정상 (메모리만 N배)
 △동일  워커별로 복제되지만 값이 같아 무해
 ✅      워커 간 공유 가능
```

---

## 2. 상태 목록

| 이름 | 위치 | 파일 | 수명 | 워커 간 | 없어지면 |
|---|---|---|---|:---:|---|
| `_sessions` | 메모리 dict | `session_manager.py` | 접속 동안 (유휴 30분) | **✗ 깨짐** | 세션 끊김 → `MISSING_SESSION` 400 |
| `_sessions_lock` | `threading.Lock` | `session_manager.py` | 프로세스 | — | — |
| `_current_session` | `ContextVar` | `session_manager.py` | 요청 스코프 | ✅ 무관 | 로그가 `server.jsonl` 로 샘 |
| `_max_connected` | int | `session_manager.py` | 프로세스 | △ 동일 | — |
| `_bound` | 메모리 dict | `session_auth.py` | 세션 수명 | **✗ 깨짐** | 중복 로그인 차단 무력화 |
| `_bound_by_session` | 메모리 dict | `session_auth.py` | 세션 수명 | **✗ 깨짐** | 동일 |
| `_users` | 메모리 dict | `session_auth.py` | 프로세스 | △ 동일 | 파일에서 다시 읽음 |
| `_login_mode` | bool | `session_auth.py` | 프로세스 | △ 동일 | — |
| `_slots` | 메모리 dict | `model_registry.py` | refcount=0 후 TTL | △ 중복 OK | 임베딩·리랭커·비교(`use_embed_any`) 재로드 |
| `_lock` | `threading.Lock` | `model_registry.py` | 프로세스 | — | — |
| `_embed_allow` / `_rerank_allow` | 메모리 dict | `model_registry.py` | 프로세스 | △ 동일 | JSON 에서 다시 읽음 |
| **`_LLM_CACHE`** | 메모리 dict | `llm_handler.py` | **영구 (로컬 교체 시 축출)** | **⚠️ 위험** | 컨트롤러 재생성 |
| **`_ORCHESTRATOR_CACHE`** | 메모리 dict | `workflow_orchestrator.py` | **영구 (로컬 교체 시 축출)** | **⚠️ 위험** | 오케스트레이터 재생성 |
| `_ask_sem` | `asyncio.Semaphore(4)` | `local_llm_gate.py` | 프로세스 | △ 워커별 | 동시성 상한이 N배로 느슨 |
| **`_local_sem`** | `asyncio.Semaphore(1)` | `local_llm_gate.py` | 프로세스 | **⚠️ 위험** | GGUF 동시 추론 → 크래시 |
| `_local_waiting` / `_ask_waiting` | int | `local_llm_gate.py` | 프로세스 | △ 워커별 | 대기열 카운트 부정확 |
| `GLOBAL_PROMPTS` | 모듈 전역 | `api_server.py` | 프로세스 | △ 동일 | 프롬프트 없음 → 부팅 실패 |
| `_BOOT` | 모듈 전역 dict | `bootstrap.py` | 프로세스 | △ 동일 | `/health` 정보 손실 |
| `_server_capture` | str | `rag_run.py` | 프로세스 | △ 동일 | `lean` 으로 복귀 |
| `_router_installed` | bool | `session_manager.py` | 프로세스 | △ 동일 | 핸들러 중복 등록 방지 |
| `PostgresRepo.connection_pool` | 클래스 변수 | `postgres_repo.py` | 프로세스 | △ 워커별 | 요청 시 `Exception` |
| `Neo4jRepo.driver` | 클래스 변수 | `neo4j_repo.py` | 프로세스 | △ 워커별 | 그래프 채널 비활성 |
| 세션 로그 | 디스크 | `logs/sessions/{sid}.jsonl` | 영구 | ✅ | 세션 이력 소실 |
| 실행 덤프 | 디스크 | `logs/runs/{sid}/{run_id}.json` | 영구 | ✅ | 사후 추적 불가 |

---

## 3. 위험 항목 상세

### 3-1. `_local_sem` — 워커를 늘리면 크래시한다

```python
# local_llm_gate.py
_local_sem = asyncio.Semaphore(1)   # init_gate() 에서 생성
```

`llama_cpp.Llama` 는 **스레드 안전하지 않다.**
`_local_sem` 이 동시 추론을 1건으로 제한한다.

워커를 N개 띄우면 **세마포어도 N개**가 되어 동시 추론 N건이 된다.
같은 GGUF 파일을 여러 프로세스가 각자 로드하므로 메모리도 N배다.
운이 좋으면 출력 뒤섞임, 나쁘면 세그폴트다.

> **`DEFAULT_LLM` 이 `openrouter:` 이면 이 문제는 잠복한다.**
> `local:` 로 바꾸는 순간 발현하고, **원인 추적이 매우 어렵다.**

`LLMController._infer_lock` (프로세스 내 이중 방어)도 워커 간에는 무력하다.

---

### 3-2. `_LLM_CACHE` / `_ORCHESTRATOR_CACHE` — 축출이 제한적이다

```python
_LLM_CACHE = {}                 # llm_handler.py
_ORCHESTRATOR_CACHE = {}        # workflow_orchestrator.py
```

프로세스가 죽을 때까지 캐시 dict 에 남는다. **완화:** `local:*` GGUF 를 바꿀 때
이전 로컬 항목을 비우는 축출이 `get_llm_controller` / `get_orchestrator` 에 있다.

```
 임베딩·리랭커·비교 임베딩  →  _slots + sweep_idle_models()  ✔
 LLM (수 GB)               →  영구 캐시, 로컬 교체 시만 축출  ⚠️
```

**단일 워커에서도** 서로 다른 `local:*.gguf` 를 동시에 상주시키면 OOM 이 난다.
`openrouter:` 만 쓰면 컨트롤러가 가벼워 잠복한다.

**추가 개선 여지:** OpenRouter 항목 축출, 유휴 TTL, 워커 간 sticky routing 전 전수 검증.

---

### 3-4. `_sessions` / `_bound` — sticky routing 으로 해결 가능

이 둘만 있으면 **코드 수정 없이** 해결된다.
`X-Session-Id` 를 consistent hashing 으로 워커에 고정하면 된다.
nginx / traefik 설정만으로 가능하다.

Redis 로 옮기면 무중단 재시작까지 되지만 코드 변경이 크다.
**학습 환경에는 과하다.** 실서비스 전환 시에만 검토할 것.

---

## 4. 워커를 늘리려면 — 순서를 지킬 것

```
 ① 먼저 측정한다                    ← 이걸 건너뛰면 나머지가 전부 추측이다
      logs/runs/*/*.json 의 timing 을 모은다
        t_retrieve_ms / t_generate_ms / t_eval_ms / t_total_ms
      병목이 t_generate_ms 라면(대부분 그렇다) 워커를 늘려도 소용없다.
      LLM 응답 대기는 I/O 라 단일 프로세스에서도 동시 처리된다.

 ② _LLM_CACHE 축출 강화 검토        ← 로컬 교체 축출은 구현됨 (3-2 참조)

 ③ ~~비교 임베딩 슬롯 통합~~ ✅ `use_embed_any` 로 `_slots` 에 편입 완료

 ④ DEFAULT_LLM 확인
      local:*  →  여기서 멈춘다. 워커 1개 유지가 필수다.
      openrouter:*  →  다음으로

 ⑤ sticky routing 설정
      nginx:   hash $http_x_session_id consistent;
      → _sessions / _bound 해결

 ⑥ workers 를 올린다
      api_server.py 마지막 줄

 ⑦ 검증
      동시 1 / 2 / 4 / 8 에서 p50·p95 재측정
      evaluate_rag 로 hit@k 회귀 확인 (모델 슬롯이 워커별로 갈리므로)
```

---

## 5. 디스크 상태

| 경로 | 내용 | 정리 정책 |
|---|---|---|
| `logs/sessions/{sid}.jsonl` | 세션 이벤트 시계열 | **없음** — 수동 |
| `logs/runs/{sid}/{run_id}.json` | 실행 덤프 (capture 등급별 상세도) | **없음** — 수동 |
| `logs/server.jsonl` | 세션 없는 로그 | **없음** — 수동 |
| `logs/testruns/{id}/` | 평가 산출물 + manifest | 영구 보관 (비교 기준점) |

> ⚠️ **로그 로테이션이 없다.**
> `capture=forensic` 으로 오래 돌리면 디스크가 찬다.
> 덤프 하나가 프롬프트 전문 + 문서 원문을 담아 **수백 KB** 가 될 수 있다.
> 수업 단위로 `logs/` 를 비우거나 `logrotate` / 크론을 붙일 것.

**세션 로그는 자기 치유 기능이 있다.** append 가 실패하면
`_rotate_log_file()` 이 이 세션만 새 파일로 갈아탄다.
RAG 는 중단하지 않는다.

```
logs/sessions/{sid}.jsonl              ← 원본
logs/sessions/{sid}_20260830T091203.jsonl  ← 실패 후 교체본
   첫 줄에 {"event": "log_file_rotated", "prev_file": ..., "reason": ...}
```

---

## 6. 종료 시 무슨 일이 일어나나

```
  SIGTERM / Ctrl-C
      │
      ▼  uvicorn 이 처리 (앱에는 시그널 핸들러가 없다)
  lifespan yield 이후
      ├─ ttl_task.cancel()          유휴 스윕 루프 중단
      ├─ await (CancelledError 흡수)
      ├─ PostgresRepo.close_pool()  커넥션 전부 반납
      └─ Neo4jRepo.close_driver()
      │
      ▼
  프로세스 종료
```

**현재 빠져 있는 것:**

| 빠진 것 | 결과 | 심각도 |
|---|---|:---:|
| 신규 요청 차단 플래그 | 종료 중에도 새 `/ask` 를 받는다 | 중 |
| in-flight 드레인 | 처리 중 `/ask` 가 중간에 끊김. 덤프 미완성 | 중 |
| 세션 종료 이벤트 기록 | JSONL 에 `shutdown` 이 안 남아 사후 분석 시 혼란 | 낮 |
| 모델 언로드 | GPU 메모리 반납 없이 종료 (OS 가 회수) | 낮 |

**보완안 (약 25줄):**

```python
_accepting = asyncio.Event(); _accepting.set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    ...
    yield
    # ① 신규 요청 차단  (_prepare_ask 에서 _accepting.is_set() 확인)
    _accepting.clear()

    # ② in-flight 드레인 (최대 30초)
    deadline = time.time() + 30
    while any(s.processing for s in list(_sessions.values())) and time.time() < deadline:
        await asyncio.sleep(0.5)

    # ③ 남은 세션에 종료 이벤트
    for s in list(_sessions.values()):
        s.write_event("server_shutdown")

    # ④ 백그라운드 태스크 (기존)
    ttl_task.cancel()
    try: await ttl_task
    except asyncio.CancelledError: pass

    # ⑤ 자원 반납 (기존)
    PostgresRepo.close_pool()
    Neo4jRepo.close_driver()
```

**급하지 않다.** 학습 환경에서 강제 종료의 피해는 작다.
다만 운영 전환 시에는 반드시 필요하다.

---

## 7. 요청 1건 동안의 상태 변화

```
 t0  POST /ask 도착
       _sessions[sid].processing        False → True
       _sessions[sid].request_id        None  → uuid4
       _sessions[sid].out_queue         None  → asyncio.Queue()
       _sessions[sid].ask_loop          None  → running loop
       _sessions[sid].cancel_event      clear()
       _sessions[sid].dropped_frames    0
       _current_session (ContextVar)    set(session)

 t1  게이트
       _ask_waiting                     +1  →  획득 후 −1
       _ask_sem                         value −1
       (local: 이면) _local_waiting     +1  →  −1
       (local: 이면) _local_sem         value −1

 t2  파이프라인
       _slots[("embed", id)].refcount   +1  →  검색 끝나면 −1
       _slots[("rerank", id)].refcount  +1  →  리랭킹 끝나면 −1
       _LLM_CACHE[llm_id]               (없으면 생성. 이후 영구)
       _ORCHESTRATOR_CACHE[llm_id]      (없으면 생성. 이후 영구)
       session.current_run              AskRun
       session.turns[run_id]            AskRun  (상한 200)

 t3  finally
       _local_sem / _ask_sem            release
       디스크                            logs/runs/{sid}/{run_id}.json  (capture≥full)
       AskRun 본문                       slim_run_bodies() 로 메모리에서 비움
       session.processing               True → False
       session.out_queue / ask_loop     → None
       session.current_run              → None
       _current_session                 reset(token)
       디스크                            logs/sessions/{sid}.jsonl  += ask_run, ask_end
```

**`slim_run_bodies()` 가 중요하다.** 덤프를 디스크에 쓴 뒤
`context_text`, `hits`, `candidates`, `answer`, `llm_calls` 를 메모리에서 비운다.
세션이 200턴까지 살아도 메모리가 선형 증가하지 않는다.

---

## 8. 요약 — 이 문서에서 기억할 세 가지

1. **`workers=1` 은 실수가 아니라 설계다.**
   세션(`_sessions`), 사용자 바인딩(`_bound`), 로컬 LLM 세마포어(`_local_sem`)가
   프로세스 메모리에 있기 때문이다. **늘리기 전에 4절을 읽을 것.**

2. **관리되는 캐시와 관리 밖 캐시가 섞여 있다.**
   임베딩·리랭커·비교 임베딩은 `_slots` + TTL 로 관리된다.
   `_LLM_CACHE` / `_ORCHESTRATOR_CACHE` 는 영구 캐시이지만 **로컬 GGUF 교체 시 축출**이 있다.
   워커를 늘리기 전에 4절을 읽을 것.

3. **PostgreSQL 커넥션 풀은 올바르다.**
   `SimpleConnectionPool(1, 20)` 이 프로세스 수명 동안 살아 있고,
   `PostgresRepo()` 는 빌리고 `close()` 는 반납한다.
   요청당 3~4회 빌리지만 순차라 동시 점유는 1개다.
   **여기는 손댈 필요 없다.**
