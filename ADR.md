# taxlaw-rag — 아키텍처 결정 기록 (ADR)

> **이 문서의 목적**
> 설명이 없으면 **오해받거나 "버그인 줄 알고 고쳐질" 결정**만 모은다.
> 자명한 결정은 적지 않는다.
>
> 각 항목은 **맥락 / 결정 / 이유 / 트레이드오프 / 뒤집어야 할 때** 순이다.
> 마지막 항목이 가장 중요하다. 어떤 결정도 영원하지 않지만,
> **언제 뒤집어야 하는지를 적어 두지 않으면 영원히 안 뒤집히거나 잘못 뒤집힌다.**

---

## 목록

| # | 결정 | 상태 |
|---|---|---|
| ADR-001 | 리랭커 `min_score` 는 fail-open | 교육용 한정 |
| ADR-002 | 가드레일은 fail-closed | 유지 |
| ADR-003 | `capture` 3단계와 서버 상한 | 유지 |
| ADR-004 | 로그는 전역 로거 1개 + ContextVar 라우팅 | 유지 |
| ADR-005 | `workers=1` | 유지 (조건부) |
| ADR-006 | 서버와 평가 하네스가 `boot_core()` 를 공유 | 유지 |
| ADR-007 | `is_transient` 는 `"rate"` 단독 매칭을 쓰지 않는다 | 유지 |
| ADR-008 | hit@k 를 `--top-k` 와 분리 | 유지 |
| ADR-009 | 테스트셋 생성은 별도 LLM SDK 를 쓴다 | 유지 |
| ADR-010 | 벡터 ANN 인덱스를 두지 않는다 | 측정 후 재검토 |
| ADR-011 | 모르는 옵션은 막지 않고 알려준다 | 유지 |
| ADR-012 | 모델 id 오타에 fallback 하지 않는다 | 유지 |
| ADR-013 | trigram 인덱스는 ingest 스크립트가 직접 건다 | 유지 |
| ADR-014 | 다층 공격 방어 및 비즈니스 쿼터 통제 체계 | 유지 |

---

## ADR-001 — 리랭커 `min_score` 는 fail-open

**맥락**
`rerank_settings.advanced.min_score` 는 CrossEncoder 점수 하한이다.
그런데 CrossEncoder 점수는 모델마다 스케일이 완전히 다르다.
`ko-reranker` 는 대략 −8 ~ +8 범위를 쓰는데, 사용자가 "0.5면 절반이겠지"
하고 넣으면 **후보가 전부 걸러진다.**

**결정**
필터가 후보를 전부 제거하면 **필터를 무시하고 `top_k` 를 그대로 쓴다.**
경고 로그를 남긴다.

```python
# retriever.py
if not filtered and scored_results:
    logger.warning("[RERANK] min_score=%.4f 가 후보를 전부 걸렀습니다. "
                   "필터 없이 top_k=%d 를 씁니다.", min_score, top_k)
    top_scored = scored_results[:top_k]
```

**이유**
빈 컨텍스트로 넘어가면 답변이 "관련된 문서를 찾을 수 없습니다"가 되고,
사용자는 **검색이 실패했다고 오해한다.** 실제로는 검색은 성공했고
자기가 넣은 숫자가 스케일에 안 맞았을 뿐이다.
로그에 점수 범위를 같이 찍어 스스로 알아채게 하는 것이 학습에 낫다.

```
[RERANK] 점수 범위 min=-3.2451 max=6.8832 min_score=0.5000 통과=0/20
[RERANK] min_score=0.5000 가 후보를 전부 걸렀습니다. 필터 없이 top_k=3 를 씁니다.
```

**트레이드오프**
- ✅ 옵션 실수가 침묵의 실패가 되지 않는다
- ❌ **`min_score` 가 실제로는 강제되지 않는다.** "품질 하한"으로 믿으면 안 된다
- ❌ 자동화된 배치에서 이 경고를 아무도 안 볼 수 있다

**뒤집어야 할 때 — 🔴 운영 전환 시 반드시 뒤집을 것**
운영 시스템에서는 근거 없는 답변을 내는 것보다 "찾지 못했습니다"가 정직하다.
`min_score` 를 강제하고 빈 결과를 그대로 반환해야 한다.
`--strict-min-score` 같은 플래그로 두 동작을 모두 지원하는 것도 방법이다.

---

## ADR-002 — 가드레일은 fail-closed

**맥락**
입·출구 가드레일은 LLM에게 `PASS` 또는 `BLOCK`을 출력하라고 시킨다.
그런데 LLM은 `"PASS입니다"`, `"NOT PASS"`, `"음, 이 질문은…"`,
빈 문자열 등 온갖 것을 반환한다.

**결정**
`_normalize_verdict()` 가 **판정이 애매하면 `BLOCK`** 을 반환한다.

```python
# rag_core.py
if not raw:                                        return "BLOCK"   # 빈 응답
if re.search(r"\bNOT\s*PASS\b|...|\bBLOCK\b", t):  return "BLOCK"   # 부정 우선
if first == "PASS" or t == "PASS":                 return "PASS"
if re.search(r"\bPASS\b", t):                      return "PASS"
return "BLOCK"                                                      # 모르면 차단
```

추가로 **LLM 호출 없이 규칙으로 먼저 거른다:**
```python
if (not context or not context.strip()) and len(generated_answer.strip()) > 40:
    return False    # 검색 문맥 없이 긴 답변 = 할루시네이션 의심
```

**이유**
`"NOT PASS"` 를 `PASS` 로 오독하면 **차단해야 할 것을 통과시킨다.**
반대 실수(통과할 것을 차단)는 사용자가 즉시 알아채고 재시도하지만,
전자는 조용히 잘못된 답변이 나간다. **비대칭한 위험에는 비대칭한 기본값을.**

**트레이드오프**
- ✅ 안전한 기본값. 오탐(false block)이 미탐(false pass)보다 낫다
- ❌ LLM이 장황하게 답하는 경향이 있으면 정상 질문도 차단될 수 있다
- ❌ 가드레일용 LLM 품질이 낮으면 사용성이 급격히 나빠진다

**뒤집어야 할 때**
가드레일 차단율이 비정상적으로 높다면(예: 20% 초과)
`_normalize_verdict` 를 고치기 전에 **먼저 `guardrails_in.raw` 를 볼 것.**
덤프에 원문이 남아 있다. 대개 프롬프트 문제이지 정규화 문제가 아니다.

---

## ADR-003 — `capture` 는 3단계이고, 클라 요청은 서버 상한을 넘지 못한다

**맥락**
디버깅에는 프롬프트 전문과 검색 후보 전량이 필요하다.
그런데 그걸 항상 남기면 (a) 디스크가 차고,
(b) 수업 환경에서는 **수강생이 강사 프롬프트를 덤프로 빼갈 수 있다.**

**결정**
```
 lean      세션 JSONL 만
 full      + JSON 덤프 (검색 후보/문서 포함)
 forensic  + LLM 프롬프트 전문
```

클라이언트가 `general_settings.capture` 로 요청하되,
**서버 `--capture` 가 상한**이다. 넘는 요청은 조용히 내리지 않고 로그를 남긴다.

```python
if _CAPTURE_RANK[requested] > _CAPTURE_RANK[ceiling]:
    logger.warning("[CAPTURE] 요청=%s 를 서버 상한=%s 로 내립니다.", requested, ceiling)
    return ceiling
```

그리고 **`forensic` 은 `trace_prompts` 를 강제로 켠다.**

**이유**
등급을 하나로 하면 "항상 켜서 디스크가 차거나" "항상 꺼서 디버깅을 못 하거나"
둘 중 하나가 된다. 상한 개념이 없으면 수강생이 `forensic` 을 넣어 프롬프트를 본다.
`forensic` 인데 `trace_prompts=false` 인 상태는 **의미상 모순**이라 강제한다.

**트레이드오프**
- ✅ 디스크·기밀·디버깅 편의의 균형
- ❌ 상한 때문에 "왜 덤프가 안 남지?"로 헤맬 수 있다 (→ 로그를 보면 나온다)
- ❌ 등급이 3개라 개념을 하나 더 배워야 한다

**뒤집어야 할 때**
수업이 아닌 내부 개발 전용이 되면 상한 개념은 불필요하다.
`--capture` 를 기본값으로만 쓰고 상한 로직은 남겨 둬도 무해하다.

---

## ADR-004 — 로그는 전역 로거 1개 + ContextVar 라우팅

**맥락**
서버 1개에 10명이 붙는다. 각자 **자기 로그만** SSE로 받아야 한다.
그런데 `logging.getLogger("my_rag")` 는 전역이다.
검색·임베딩·추론은 **스레드풀**에서 돈다.

**결정**
로거는 하나로 두고, `SessionLogRouter` 핸들러가 **ContextVar 를 조회**해
레코드를 세션 파일과 큐로 보낸다. 세션이 없으면 `server.jsonl`.

스레드 경계는 `run_in_session_executor()` 가 `copy_context()` + `ctx.run()` 으로 넘긴다.

**이유**
대안은 세 가지였다.

| 대안 | 문제 |
|---|---|
| 세션마다 Logger 인스턴스 | 모든 함수에 logger 를 인자로 넘겨야 함. 침습적 |
| 로그에 session_id 를 붙이고 나중에 필터 | SSE 실시간 전송이 불가능 |
| threading.local | asyncio 코루틴은 스레드가 아니라 동작 안 함 |

**ContextVar 만이 "코드를 안 건드리면서" + "실시간" + "asyncio 안전" 셋을 만족한다.**

**트레이드오프**
- ✅ 기존 `logger.info(...)` 호출을 하나도 안 고쳐도 라우팅된다
- ❌ **`run_in_session_executor` 를 우회하면 조용히 로그가 샌다.**
  이게 이 설계의 유일한 함정이고, 가장 찾기 어려운 종류의 버그다
- ❌ 핸들러 안에서 예외가 나면 로그 자체가 사라질 수 있다 (try 로 감쌈)

**뒤집어야 할 때 — 없다.** 대신 **규칙을 강제할 것.**
새 블로킹 호출을 추가할 때 `run_in_session_executor` 를 쓰는지
코드 리뷰에서 반드시 확인한다. AST 테스트로 자동화하는 것도 가능하다:
`workflow_orchestrator` 안의 `await` 중 executor 를 안 타는 것을 찾는 식.

---

## ADR-005 — `workers=1`

**맥락**
`uvicorn.run(..., workers=1)`. 늘리면 처리량이 오를 것 같다.

**결정**
**1로 고정한다.** 프로세스 메모리에 있는 상태가 워커 간에 공유되지 않기 때문이다.

```
 _sessions     session_manager.py:26   ✗ 워커 A 세션을 워커 B 가 모른다 → 400
 _bound        session_auth.py:33      ✗ 중복 로그인 차단 무력화
 _local_sem    local_llm_gate.py:24    ⚠ 세마포어가 N개 → GGUF 동시추론 N건 → 크래시
 _ask_sem      local_llm_gate.py:25    △ 상한이 N배로 느슨
 _slots        model_registry.py:32    △ 임베딩/리랭커 메모리 N배
 _LLM_CACHE    llm_handler.py:667      △ GGUF 메모리 N배 (치명적)
```

**이유**
`_local_sem` 이 특히 위험하다. `llama_cpp.Llama` 는 스레드 안전하지 않고,
프로세스가 N개면 세마포어도 N개가 되어 **동시 추론 N건**이 된다.
운이 좋으면 출력 뒤섞임, 나쁘면 세그폴트다.

**`DEFAULT_LLM` 이 `openrouter:` 이면 이 문제는 잠복한다.
`local:` 로 바꾸는 순간 발현하고 원인 추적이 매우 어렵다.**

**트레이드오프**
- ✅ 상태 동기화 코드가 0줄
- ❌ 단일 프로세스 처리량 한계. GIL 영향은 작다(대부분 I/O 대기)
- ❌ 무중단 재배포 불가

**뒤집어야 할 때 — 순서를 지킬 것**
```
 ① 먼저 측정한다.  logs/runs/*/*.json 의 timing 을 모아
                   병목이 검색인지 리랭커인지 LLM인지 확인.
                   대부분 LLM 대기이고, 그렇다면 워커를 늘려도 소용없다.
 ② X-Session-Id sticky routing (nginx/traefik consistent hashing)
                   → _sessions / _bound 가 코드 수정 없이 해결
 ③ DEFAULT_LLM=local:* 이면 그래도 워커 1개 유지 (필수)
 ④ _LLM_CACHE 축출을 먼저 구현 (안 하면 GGUF 가 N배)
 ⑤ 그 다음에 workers 를 올린다
```
Redis 로 세션을 옮기면 무중단 재시작까지 가능하지만 코드 변경이 크다.
**학습 환경에는 과하다.**

---

## ADR-006 — 서버와 평가 하네스가 `boot_core()` 를 공유한다

**맥락**
`evaluate_rag.py` 는 웹서버 없이 오케스트레이터를 직접 호출한다.
그러려면 DB 풀, 모델 레지스트리, OpenRouter 카탈로그, 프롬프트를
스스로 준비해야 한다.

**결정**
초기화를 `bootstrap.boot_core()` 하나로 모으고 서버 `lifespan` 과 하네스가
**둘 다 그것을 부른다.**

```python
def boot_core(*, prompt_cli=None, fail_fast_prompts=True) -> PromptsConfig:
    require_tax_db_uri()
    PostgresRepo.init_pool()
    init_model_registry()
    load_openrouter_catalog_file()
    prompts = load_prompts(prompt_cli, fail_fast=fail_fast_prompts)
    problems = verify_contracts(repo.conn)      # 확장·테이블·차원 점검
    ...
```

**이유**
진입점마다 초기화를 따로 하면 **`is_allowed_model_id()` 가
"파일 화이트리스트" 와 "네트워크 fallback" 으로 갈린다.**
그러면 서버에서는 거부되는 모델이 하네스에서는 통과하거나 그 반대가 되고,
평가 결과를 서버 동작의 근거로 쓸 수 없게 된다.

`bootstrap.py` 는 **`api_server` 를 import 하지 않는다.** 그게 이 분리의 전제다.

**트레이드오프**
- ✅ 서버와 하네스가 같은 카탈로그·같은 프롬프트·같은 계약 점검을 통과한다
- ❌ 하네스가 DB 없이는 못 돈다 (순수 채점 함수는 `--self-test` 로 분리)
- ❌ Neo4j·세션·게이트는 서버 전용이라 `lifespan` 에 남아 완전 통일은 아니다

**뒤집어야 할 때**
없다. `evaluate_rag.py` 와 `run_eval_batch.py` 가 `contracts` / `database.postgres_repo` / `database.neo4j_repo` / `local_llm_gate` 를 직접 import 하도록 정리 완료되어 하네스 → 서버 역방향 의존이 완전히 제거되었다.

---

## ADR-007 — `is_transient` 는 `"rate"` 단독 매칭을 쓰지 않는다

**맥락**
LLM 오류 메시지를 보고 "재시도하면 될 것"과 "모델이 그냥 실패한 것"을
갈라야 한다. 갈라지지 않으면 무한 재시도에 빠지거나,
할당량 오류를 모델 오답으로 기록한다.

**결정**
```python
TRANSIENT_KEYS = ("429", "quota", "rate limit", "resource exhausted",
                  "500", "502", "503", "504", "unavailable", "overloaded",
                  "high demand", "try again", "retry", "timeout", "deadline")
```

**`"rate"` 가 아니라 `"rate limit"` 이다.**

```python
"""'rate' 단독 매칭은 쓰지 않는다. 'generate' 안에 'rate' 가 들어 있어
평범한 생성 실패를 과부하로 오인하기 때문이다."""
```

그리고 `is_overload` 를 `is_transient` 보다 **좁게** 따로 뒀다.
(`timeout`, `500`, `retry` 는 transient 지만 overload 는 아니다)

**이유**
`"Failed to generate response"` ← 여기 `rate` 가 들어 있다.
`"gene**rate**"`. 이걸 놓치면 **모든 생성 실패가 과부하로 분류**된다.

두 개념을 분리한 이유는 **사용자 안내 문구가 달라야 하기 때문**이다.
과부하는 "잠시 후 다시" 가 맞고, 타임아웃은 "설정을 줄여 보세요" 가 맞다.

**트레이드오프**
- ✅ 재시도 폭주와 오분류를 막는다
- ❌ 키워드 목록은 본질적으로 취약하다. 새 프로바이더의 새 메시지는 놓친다
- ❌ 한국어 오류 메시지는 매칭 안 된다 (현재 업스트림이 전부 영어라 무해)

**뒤집어야 할 때**
프로바이더를 추가하면 **먼저 `tests/test_errors.py` 에 음성 케이스를 넣을 것.**
```python
assert not is_transient("Failed to generate response")   # ← 이걸 지우지 말 것
```
키워드를 넓힐 때마다 이 테스트가 방어선이다.

---

## ADR-008 — hit@k 를 `--top-k` 와 분리한다

**맥락**
`--top-k` 는 RAG가 실제로 LLM에 넣는 문서 개수(**동작 설정**)다.
`hit@k` 는 정답을 상위 k개 안에 끌어왔는지 재는 **채점 기준**이다.

초기 구현은 `k = args.top_k` 였다.

**결정**
채점은 **고정 집합 `HITK_LEVELS = (1, 3, 5, 10)`** 으로 동시 산출한다.
`--top-k` 를 바꿔도 채점 자는 안 움직인다.

**이유**
```
 top_k=3 일 때  hit@3 = 0.62
 top_k=5 로 바꿈 → hit@5 = 0.71

   → 검색이 좋아진 건가?  채점이 느슨해진 건가?  구분 불가.
```

자를 고정하면:
```
 hit@1  이 오르면  →  순위 개선 (리랭커가 일을 했다)
 hit@10 만 오르면  →  그물 확대 (더 많이 가져왔을 뿐)
 cut_by_topk_rate  →  후보엔 있었는데 top_k 에서 밀린 비율
```

**시스템 설정이 바뀔 때 측정 기준도 같이 바뀌면 어떤 개선도 검증할 수 없다.**

같은 원칙이 다른 곳에도 적용돼 있다:
- `summary["config"]` 에 판정 기준(`kw_threshold`, `require_context`, `id_check`)을 함께 저장
- `manifest.json` 에 `testset_sha256` 과 `git` 리비전 저장
- absence 문항을 검색 지표 분모에서 제외
- hit@k/MRR 을 `match_method == "id"` 인 행에서만 계산

**트레이드오프**
- ✅ 개선의 원인을 분리할 수 있다
- ❌ 지표가 많아져 리포트가 길다
- ❌ `top_k` 가 HITK_LEVELS 에 없으면 참고값 hit@N 이 하나 더 붙는다

**뒤집어야 할 때 — 없다.** 이건 실험 설계의 기본이다.

---

## ADR-009 — 테스트셋 생성은 별도 LLM SDK 를 쓴다

**맥락**
서버는 `LLMController` 로 OpenRouter / 로컬 GGUF 를 통일해 쓴다.
그런데 `build_testset.py --engine gemini` 는 `google.generativeai` 를 직접 부른다.

**얼핏 보면 일관성 위반이다.**

**결정**
**의도적으로 분리한다.** 출제자와 응시자는 다른 모델이어야 한다.

**이유**
답변을 생성하는 모델이 시험 문제도 출제하면,
**자기가 아는 것만 물어보는 시험지**가 나온다.
그 시험지로 잰 점수는 실제 성능이 아니라 자기 일관성이다.

`--engine template` (LLM 불필요) 도 함께 제공하는 이유가 같다 —
LLM 없이 만든 문항이 **가장 편향이 적은 기준선**이다.

지연 import 도 의도적이다:
```python
"""google-generativeai 는 무겁고, template 엔진만 쓰는 사용자에게는
아예 필요 없는 의존성이다. 모듈 최상단에서 import 하면 키도 없고 패키지도
없는 환경에서 스크립트가 시작조차 못 한다."""
```

**트레이드오프**
- ✅ 출제자 ≠ 응시자. 평가의 독립성
- ❌ 의존성이 하나 늘고, `LLMController` 의 취소·재시도 로직을 못 쓴다
- ❌ 일관성 위반처럼 보여 "통일해야 하나?" 하는 유혹이 생긴다 ← **이 문서의 존재 이유**

**뒤집어야 할 때**
`LLMController` 를 쓰더라도 **서버 `DEFAULT_LLM` 과 다른 모델을 강제**한다면
통일해도 된다. 단, "출제자 = 응시자" 가 되는 순간 점수는 무의미해진다.
통일하려면 `--judge-model` 처럼 **모델 분리를 강제하는 검증**을 먼저 넣을 것.

---

## ADR-010 — 벡터 ANN 인덱스를 두지 않는다

**맥락**
`schema.trigram_index_statements()` 는 `full_text` 에 GIN(trigram) 인덱스를 만든다.
그런데 `embedding` 컬럼에는 `ivfflat` / `hnsw` 인덱스가 없다.
`ORDER BY embedding <=> q` 는 **전수 스캔**이다.

**결정**
두지 않는다. 코드에 이유가 적혀 있다.

```python
"""벡터 ANN 인덱스(ivfflat/hnsw)는 두지 않는다. 코퍼스가 테이블마다 다르고
전수 스캔(<=>)이 작은 집합에서는 더 정확하다. ANN 이 필요해지면 측정 후에
이 함수 옆에 추가한다 — 지금은 누락이 아니라 의도다."""
```

**이유**
ANN 은 **근사**다. 정확도를 속도와 바꾼다.
수만 행 규모에서는 전수 스캔이 수십 ms 수준이라 바꿀 이유가 없다.
그리고 `ivfflat` 은 `lists` 파라미터를 코퍼스 크기에 맞춰야 하는데,
테이블 4개의 크기가 제각각이라 하나의 값으로 맞출 수 없다.

**"지금은 누락이 아니라 의도다"** 라는 문장이 이 ADR의 핵심이다.
없는 것을 보고 "빠뜨렸구나" 하고 추가하는 것을 막는다.

**트레이드오프**
- ✅ 정확도 100%. 튜닝 파라미터 0개
- ❌ 코퍼스가 수십만 행을 넘으면 선형으로 느려진다
- ❌ `search_hybrid_rrf` 는 `src` CTE 를 **두 번 스캔**한다 (키워드 1회 + 벡터 1회)

**뒤집어야 할 때 — 측정이 선행 조건**
```
 ① logs/runs/*/*.json 의 timing.t_retrieve_ms 를 모은다
 ② p95 가 500ms 를 넘고, 그 대부분이 SQL 이면
 ③ EXPLAIN ANALYZE 로 벡터 스캔 비중을 확인
 ④ 그때 hnsw 를 추가 (ivfflat 보다 lists 튜닝이 없어 안전)
 ⑤ 추가 후 반드시 evaluate_rag 로 hit@k 회귀를 확인 — ANN 은 정확도를 깎는다
```
**⑤를 빠뜨리면 "빨라졌는데 점수가 왜 떨어졌지?" 로 헤맨다.**

---

## ADR-011 — 모르는 옵션은 막지 않고 알려준다

**맥락**
`GeneralSettings` 등이 전부 `model_config = ConfigDict(extra="allow")` 다.
클라이언트가 `"typo_opt": 1` 을 보내도 Pydantic 이 안 막는다.

**결정**
막지 않는다. 대신 **수집해서 4곳에 알린다.**

```python
outcome.ignored_fields = collect_and_clear_extras(req)
   → logger.info("[VALIDATE] 모르는 옵션 %d개 무시: %s")
   → session.write_event("options_ignored", fields=…)
   → SSE  event: notice  "무시된 옵션: general_settings.typo_opt"
   → AskRun.resolved["ignored_fields"]  → 덤프와 done 프레임
```

**이유**
`extra="forbid"` 로 막으면 구 클라이언트가 새 서버에서 전부 깨진다.
`extra="ignore"` 로 조용히 버리면 **오타가 침묵의 실패**가 된다
("리랭커를 켰는데 왜 안 되지?" → 사실 `rerank_setting` 이라고 썼다).

**막지 않되 침묵하지 않는다** 가 이 프로젝트의 일관된 태도다.
같은 태도가 `dropped_frames`, `[CAPTURE] 강등 로그`, `[RERANK] 필터 무시 경고`
에도 나타난다.

**트레이드오프**
- ✅ 하위 호환 + 오타 발견
- ❌ 알림을 아무도 안 볼 수 있다 (그래서 4곳에 남긴다)
- ❌ 규칙이 없는 진짜 옵션은 `/options/schema` 에 안 나와 발견이 어렵다
  (과거 `use_graphdb` 누락 사례 — 지금은 규칙에 포함)

**뒤집어야 할 때**
없다. 다만 `contracts.py` 에 필드를 추가하면
**반드시 `validation.FIELD_RULES` 에도 규칙을 추가할 것.**
누락되면 (a) `/options/schema` 에 안 나오고 (b) 값 검증이 없어
이상한 값이 파이프라인 안까지 들어간다.
`use_graphdb`, `company_summary` 는 `FIELD_RULES` 에 포함됨 (2026-08).

---

## ADR-012 — 모델 id 오타에 fallback 하지 않는다

**맥락**
`openrouter:google/gemini-3.5-flash-lit` (오타) 를 받았을 때
"비슷한 모델로 대신 답할까?" 하는 유혹이 있다.

**결정**
**하지 않는다.** `ValueError` → `UNKNOWN_MODEL` 로 즉시 거부한다.

```python
class LLMController:
    """Local(GGUF) / OpenRouter 를 단일 인터페이스로 제어.
    오타 id 는 Fallback 하지 않는다."""
```

거부 메시지에 허용 목록 샘플과 규칙을 같이 넣는다:
```json
{"expected": {"type": "model_ref", "source": "GET /llm/models",
              "rule": "openrouter:<slug> 또는 local:<파일명.gguf>",
              "allowed_count": 42, "allowed_sample": ["…"]}}
```

**이유**
fallback 하면 **사용자가 A 모델을 평가했다고 믿는데 실제로는 B 모델의 결과**다.
평가 하네스에서 이게 일어나면 모든 비교 실험이 무의미해진다.
그리고 사용자는 오타를 영원히 모른다.

**트레이드오프**
- ✅ "내가 지정한 모델이 답했다" 가 보장된다
- ❌ 오타 하나에 요청 전체가 실패한다 (그래서 `/ask/validate` 를 무료로 제공)
- ❌ 모델이 프로바이더에서 사라지면 즉시 실패한다 (자동 대체 없음)

**뒤집어야 할 때**
운영에서 가용성이 정확성보다 중요하다면 **명시적 fallback 체인**을 도입한다.
단, 응답에 `"model_used"` 를 반드시 넣어 **무엇이 답했는지 알 수 있게** 할 것.
조용한 fallback 은 어떤 경우에도 안 된다.

---

## ADR-013 — trigram 인덱스는 ingest 스크립트가 직접 건다

**맥락**
`schema.trigram_index_statements()` 는 `full_text` 컬럼에 pg_trgm GIN 인덱스를 만드는
DDL을 갖고 있고, `database/postgres_repo.apply_search_support()` 가 그걸 실행한다.
그런데 실제로 이 함수를 호출하는 곳은 `scripts/seed_dev_db.py`(개발용 시드) 하나뿐이었다.
운영 DB를 만드는 진짜 경로인 `01_collect_tax_data.py → 02_build_tax_db.py →
03_build_tax_amendments.py` 는 이 함수를 몰랐다. `database/init.sql` 은 이름과 달리
확장(`CREATE EXTENSION`)만 깔고 인덱스는 만들지 않는데, README 에는 한동안
"확장 + 인덱스"라고 잘못 적혀 있었다. 그 결과 이 파이프라인으로 DB를 실제로
구축하면 `search_hybrid_rrf` / `search_hybrid_score_based` (postgres_repo.py) 가
매 쿼리마다 `full_text` 를 인덱스 없이 정규식·trigram 유사도로 전수 스캔하는
상태로 조용히 운영될 수 있었다 — 결과는 맞지만 테이블이 커질수록 계속 느려진다.

**결정**
`02_build_tax_db.py`, `03_build_tax_amendments.py` 모두 ingest 단계 맨 끝에서
`apply_search_support(conn, include_indexes=True)` 를 직접 호출한다.
`database.postgres_repo` 를 못 불러오면(예: 저장소 밖에서 이 스크립트만 떼어 실행)
조용히 건너뛰고 안내 메시지만 남기며, 인덱스 적용 자체가 실패해도 이미 커밋된
데이터 적재 결과는 롤백하지 않는다 — 인덱스는 언제든 재시도 가능한 부가 단계지,
적재 성공의 필요조건이 아니기 때문이다.

```python
# 02_build_tax_db.py / 03_build_tax_amendments.py 공통 패턴
if apply_search_support:
    try:
        apply_search_support(conn, include_indexes=True)
    except Exception as e:
        print(f"인덱스 적용 실패(데이터 적재는 정상 완료됨): {e}")
```

**이유**
인덱스 생성 로직은 이미 `schema.py`/`postgres_repo.py`에 정확히 있었다 — 없던 건
"어디서 호출하느냐"였다. 개발자가 "인덱스는 seed 스크립트가 알아서 하겠지" 라고
믿고 운영 파이프라인만 돌리면 인덱스 없는 DB가 만들어지는데, 쿼리 결과가 틀리지
않으니 몇 달간 아무도 눈치 못 챌 수 있는 조용한 성능 함정이었다.

**트레이드오프**
- ✅ `01~03` 만 순서대로 돌리면 검색까지 바로 쓸 수 있는 상태가 보장된다
- ✅ 인덱스 DDL의 유일한 소유자는 여전히 `schema.py` 다 (호출부만 늘었다)
- ❌ `--reset-db` 로 테이블만 새로 만들고 데이터는 다른 경로로 적재하면 이 자동 호출을
  못 타므로, 그런 경우엔 README 2-2절의 수동 실행 스니펫을 써야 한다

**뒤집어야 할 때**
인덱스 생성 시점을 "ingest 직후"가 아니라 "배치 적재 전부 끝난 뒤 1회"로 옮기고
싶다면(대용량 적재 중 인덱스 유지비용을 피하려는 경우), `apply_search_support` 호출을
02/03에서 빼고 별도의 `index` step(`--step index`)을 신설할 것. 단, 그 경우 README와
이 ADR을 반드시 같이 갱신할 것 — 이번 사고가 정확히 "코드는 있는데 아무도 안 부른다"
였다는 걸 잊지 말 것.

---

## ADR-014 — 다층 공격 방어(Cloudflare + Local Gate) 및 비즈니스 쿼터 통제 체계

**맥락**
- 공개 데모 서비스 특성상 악성 봇이나 매크로에 의한 공격성 호출, 과도한 LLM 토큰 소모를 줄여야 한다.
- 세션 단위 체크는 새로고침(F5) 시 새 세션이 발급되어 쿼터가 쉽게 무력화되며, 서버 재기동 시 휘발된다.
- 독립된 별도 모듈을 신설하여 한 줄 호출하는 구조는 책임 파편화 안티패턴이다.

**결정**
1. **다층 방어(Defense-in-Depth)**:
   - L3/L4/L7 대량 패킷 및 DDoS 봇 공격은 **Cloudflare WAF**가 앞단에서 걸러 낸다.
   - 비즈니스 룰("1일 5회, 15분 쿨다운")은 **`local_llm_gate.py` 내부의 게이트웨이**가 직접 통제한다.
2. **식별 기준**:
   - `CF-Connecting-IP` ➔ `X-Forwarded-For` ➔ `request.client.host` 순으로 실제 IP를 1차 기준으로 삼아 시크릿 창/로컬스토리지 삭제 우회를 막는다.
   - `X-Device-Id`는 헤더로 수신하여 클라이언트 추적 및 보완 식별자로 활용한다.
3. **선제 차감 및 우선순위 원칙**:
   - 질문(`/ask`) 진입 시 즉시 1회 선제 차감하며, 중간에 Stop을 누르거나 네트워크가 끊겨도 차감을 취소하지 않는다.
   - 상태 판정 시 `일일 5회 소진 여부`를 `15분 쿨다운`보다 먼저 검사하여, 5회 소진 후 "15분 대기"로 오인되는 UX 버그를 방지한다.
4. **원자적 디스크 영속 원장**:
   - `logs/quotas.json`에 KST 기준 일자별로 원자적 쓰기(`tmp` ➔ `replace`)를 수행하여 서버 재시작 시에도 쿼터가 보존된다.
5. **업계 표준(국룰) UX**:
   - 5회 소진 시 입력창 및 전송 버튼 `disabled`, placeholder 전환, 타임라인 시스템 공지 카드 노출.
   - 쿨다운 중 실시간 카운트다운 타이머 제공.

**이유**
- 세션 객체나 인메모리 딕셔너리에만 의존하면 F5 새로고침이나 서버 재시작으로 공격 차단이 즉시 무력화된다.
- 기존 입구컷/동시성 통제를 전담하던 `local_llm_gate.py`에 책임을 일체화하여 모듈 파편화 안티패턴을 배제하고 단일 진입점 방어를 달성한다.

**트레이드오프**
- ✅ 새로고침/시크릿 모드/서버 재시작에도 우회하기 어려운 쿼터 방어
- ✅ 모듈 난립 없는 깔끔한 응집도 유지
- ❌ 동일 공용 Wi-Fi나 NAT 환경의 복수 사용자가 동일 IP로 묶여 쿼터를 공유할 수 있음 (향후 개별 로그인 활성화 시 `user_id` 우선 바인딩으로 해소 가능)

**뒤집어야 할 때**
인증 로그인 모드(`--login-mode`)가 프로덕션 전면 기본값이 되어 회원별 계정 쿼터 체계로 전환될 때, IP 기반 쿼터 키를 `user_id` 1차 키로 확장할 것.

---

## 부록 — ADR 을 추가할 때

새 ADR 이 필요한지 판단하는 기준:

```
 ✅ 적을 것
    - 6개월 뒤 내가 "이거 버그 아닌가?" 할 만한 것
    - 명백한 대안이 있는데 그걸 안 고른 것
    - 교육용 / 운영용에서 답이 뒤집히는 것
    - 다른 곳과 일관성이 깨져 보이는데 의도적인 것

 ❌ 적지 않을 것
    - 자명한 것 ("FastAPI 를 썼다")
    - 코드를 읽으면 바로 아는 것
    - 취향 문제 (들여쓰기, 네이밍)
```

형식은 **맥락 / 결정 / 이유 / 트레이드오프 / 뒤집어야 할 때** 5단을 지킬 것.
특히 마지막 항목을 비우지 말 것 — **그게 이 문서의 존재 이유다.**
