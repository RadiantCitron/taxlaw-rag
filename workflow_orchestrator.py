# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
import asyncio
import logging
import threading
import time
import traceback

from error_envelope import PipelineFault
from errors import is_overload, is_transient
from evaluator import RAGEvaluator
from llm_handler import get_llm_controller, is_local_model
from retriever import get_hybrid_assembled_context
from rag_core import RAGGuardrails
from rag_run import AskRun, GuardrailResult, effective_trace_prompts
from session_manager import (
    SessionCancelled,
    get_current_session,
    raise_if_cancelled,
    run_in_session_executor,
)
from settings import company_summary as default_company_summary
from sse import SSE_KIND_ANSWER, SSE_KIND_EVAL, SSE_KIND_LOG
from search_constants import EMPTY_CONTEXT_MSG

logger = logging.getLogger("my_rag")

OUTPUT_DISCLAIMER = (
    "※ 이 답변은 AI가 생성한 참고 정보이며, 부정확할 수 있고 전문적인 조언을 대신하지 "
    "않습니다. 중요한 결정 전에는 관련 분야 전문가와 상담하시기 바랍니다."
)

_ORCHESTRATOR_CACHE = {}
_ORCHESTRATOR_LOCK = threading.Lock()


def _ms(start: float) -> float:
    return round((time.time() - start) * 1000, 1)


def _context_text(result) -> str:
    if isinstance(result, str):
        return result or ""
    return getattr(result, "context_text", None) or ""


def _publish_answer(
    session,
    answer: str,
    run: AskRun | None,
    *,
    trace_prompts: bool = False,
) -> None:
    if session is None:
        return
    if trace_prompts:
        session.emit_to_client("──────── ANSWER\n", kind=SSE_KIND_LOG)
    text = answer or ""
    session.write_event("answer", message=text)
    session.emit_to_client(text, kind=SSE_KIND_ANSWER)
    if run is not None:
        run.resolved["answer_sse_sent"] = True


def _eval_client_payload(req, run: AskRun | None) -> dict:
    """SSE 용 평가 요약. reason 은 200자로 자른다."""
    report = []
    for row in (run.eval.report if run is not None else None) or []:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason") or "")
        report.append({
            "metric": row.get("metric"),
            "score": row.get("score"),
            "pass": row.get("pass"),
            "reason": reason[:200],
        })
    return {
        "model": req.eval_settings.eval_model,
        "threshold": req.eval_settings.threshold,
        "error": run.eval.error if run is not None else None,
        "report": report,
    }


def _log_eval_report(report: list[dict]) -> None:
    if not report:
        return
    logger.info("\n" + "-" * 50)
    logger.info("[EVAL 결과 리포트]")
    for row in report:
        if not isinstance(row, dict):
            continue
        status = "PASS" if row.get("pass") else "FAIL"
        score = row.get("score")
        try:
            score_s = f"{float(score):.2f}"
        except (TypeError, ValueError):
            score_s = str(score)
        logger.info(f"▶ {row.get('metric') or '?'}: {score_s}점 ({status})")
        reason = (row.get("reason") or "").strip()
        if reason:
            logger.info(f"  ↳ 사유: {reason[:200]}")
    logger.info("-" * 50)


def _publish_eval_report(session, req, run: AskRun | None) -> None:
    if session is None or run is None:
        return
    payload = _eval_client_payload(req, run)
    session.write_event("eval_report", **payload)
    session.emit_to_client(payload, kind=SSE_KIND_EVAL)


def get_orchestrator(model_identifier: str) -> "RAGOrchestrator":
    """
    모델별로 Orchestrator를 한 번만 만들고 재사용한다.
    내부 LLM도 get_llm_controller로 캐시된다.
    """
    with _ORCHESTRATOR_LOCK:
        cached = _ORCHESTRATOR_CACHE.get(model_identifier)
        if cached is not None:
            logger.info(f"[CACHE] 캐시된 Orchestrator 재사용: {model_identifier}")
            return cached
    logger.info(f"[CACHE] 새 Orchestrator 생성: {model_identifier}")
    orch = RAGOrchestrator(model_identifier)
    with _ORCHESTRATOR_LOCK:
        existing = _ORCHESTRATOR_CACHE.get(model_identifier)
        if existing is not None:
            logger.info(f"[CACHE] 캐시된 Orchestrator 재사용: {model_identifier}")
            return existing
        if is_local_model(model_identifier):
            for key in list(_ORCHESTRATOR_CACHE):
                if key != model_identifier and is_local_model(key):
                    _ORCHESTRATOR_CACHE.pop(key, None)
                    logger.info("[CACHE] 로컬 Orchestrator 언로드: %s", key)
        _ORCHESTRATOR_CACHE[model_identifier] = orch
        return orch


class RAGState:
    """파이프라인 상태 객체"""
    def __init__(self, request_payload, active_prompts, run: AskRun | None = None):
        self.req = request_payload
        self.prompts = active_prompts
        self.query = request_payload.query
        self.trace_prompts = effective_trace_prompts(
            request_payload, capture=getattr(run, "capture", None)
        )
        self.run = run

        self.retrieved_context = ""
        self.generated_answer = ""
        self.raw_answer = ""
        self.eval_report = []

        self.retry_count = 0
        self.max_retries = 1
        self.error_message = ""
        self.status = "INIT"  # INIT -> RUNNING -> SUCCESS / FAIL


class RAGOrchestrator:
    """각 단계를 노드로 캡슐화하여 흐름을 제어"""

    def __init__(self, model_identifier: str):
        self.llm = get_llm_controller(model_identifier)
        self.optimizer = self.llm.get_optimizer()
        logger.info(f"[ORCHESTRATOR] 준비 완료 (Provider: {self.llm.provider})")

    async def node_input_guardrail(self, state: RAGState) -> RAGState:
        run = state.run
        if not state.req.general_settings.use_guardrails:
            logger.info("[NODE 1] 🛡️ 입구 가드레일: 옵션 비활성화로 생략합니다.")
            if run is not None:
                run.guardrails_in = GuardrailResult(enabled=False, verdict="skipped")
            return state

        logger.info("[NODE 1] 🛡️ 입구 가드레일 가동: 질문의 도메인 적합성 검사 중...")
        if run is not None:
            run.mark(stage="guard_in")
            run.guardrails_in = GuardrailResult(enabled=True)

        guardrails = RAGGuardrails(
            self.llm, state.prompts, trace_prompts=state.trace_prompts
        )
        raise_if_cancelled()
        is_safe = await run_in_session_executor(
            guardrails.check_input, state.query
        )

        if run is not None:
            run.guardrails_in.raw = guardrails.last_raw
        if not is_safe:
            state.status = "FAIL"
            state.error_message = "⚠️ [가드레일 차단] 질문이 도메인을 벗어났거나 부적절합니다."
            if run is not None:
                run.guardrails_in.verdict = "BLOCK"
                run.generation.answer = state.error_message
                run.mark(status="rejected", error=state.error_message, where="guard_in")
            return state

        if run is not None:
            run.guardrails_in.verdict = "PASS"
        return state

    async def node_retrieve(self, state: RAGState) -> RAGState:
        logger.info("[NODE 2] 🔍 자료 검색 및 리랭킹 가동")
        run = state.run
        if run is not None:
            run.mark(stage="retrieve")
        retrieval_start = time.time()
        raise_if_cancelled()
        result = await run_in_session_executor(
            lambda: get_hybrid_assembled_context(
                query_text=state.query,
                llm_optimizer=self.optimizer,
                active_prompts=state.prompts,
                query_mode=state.req.general_settings.query_mode,
                fusion_mode=state.req.general_settings.fusion_mode,
                alpha=state.req.general_settings.alpha,
                use_graphdb=state.req.general_settings.use_graphdb,
                embed_model_name=state.req.general_settings.embed_model,
                use_reranker=state.req.rerank_settings.enabled,
                rerank_model_name=state.req.rerank_settings.advanced.model,
                top_k=state.req.rerank_settings.top_k,
                min_score=state.req.rerank_settings.advanced.min_score,
                rerank_candidate_cap=state.req.rerank_settings.advanced.candidate_cap,
                trace_prompts=state.trace_prompts,
                embed_variant=getattr(state.req.general_settings, "embed_variant", None),
                sources=getattr(state.req.general_settings, "sources", None),
                keyword_mode=getattr(state.req.general_settings, "keyword_mode", "trigram"),
                base_limit=state.req.general_settings.base_limit,
            )
        )
        if run is not None:
            state.retrieved_context = run.apply_retrieval(result)
            run.timing.t_retrieve_ms = _ms(retrieval_start)
        else:
            state.retrieved_context = _context_text(result)
        return state

    async def node_generate(self, state: RAGState) -> RAGState:
        logger.info("[NODE 3] ✍️ LLM 답변 생성 가동")
        run = state.run
        if run is not None:
            run.mark(stage="generate")
        g = state.req.general_settings
        company_summary = getattr(
            g, "company_summary", None
        ) or default_company_summary()
        temperature = getattr(g, "temperature", None)
        max_tokens = getattr(g, "max_tokens", None)

        gen_start = time.time()
        raise_if_cancelled()
        answer = await run_in_session_executor(
            lambda: self.llm.ask(
                state.query,
                company_summary,
                state.retrieved_context,
                state.prompts.main_answer,
                temperature=temperature,
                max_tokens=max_tokens,
                trace_prompts=state.trace_prompts,
                stage="main_answer",
            ),
        )
        state.generated_answer = answer
        state.raw_answer = answer
        if run is not None:
            run.generation.answer = answer or ""
            run.timing.t_generate_ms = _ms(gen_start)
        return state

    async def node_evaluate(self, state: RAGState) -> RAGState:
        run = state.run
        if not state.req.eval_settings.enabled:
            if run is not None:
                run.eval.enabled = False
            return state

        logger.info("[NODE 5] ⚖️ DeepEval 평가 가동")
        if run is not None:
            run.mark(stage="eval")
            run.eval.enabled = True
        eval_start = time.time()

        evaluator = RAGEvaluator(
            state.req.eval_settings, state.prompts, trace_prompts=state.trace_prompts
        )
        try:
            raise_if_cancelled()
            report = await run_in_session_executor(
                lambda: evaluator.run_evaluation(
                    state.query,
                    state.raw_answer,
                    state.retrieved_context,
                ),
            )
        except SessionCancelled:
            raise
        except Exception as e:
            logger.exception("[WORKFLOW] eval 예외")
            if run is not None:
                run.eval.error = str(e)
                run.timing.t_eval_ms = _ms(eval_start)
                partial = list((run.resolved or {}).get("partial") or [])
                if "eval" not in partial:
                    partial.append("eval")
                    run.resolved["partial"] = partial
            sess = get_current_session()
            if sess is not None:
                _publish_eval_report(sess, state.req, run)
            return state

        state.eval_report = report
        if run is not None:
            run.eval.report = list(report or [])
            run.timing.t_eval_ms = _ms(eval_start)
        _log_eval_report(list(report or []))
        sess = get_current_session()
        if sess is not None:
            _publish_eval_report(sess, state.req, run)
        return state

    async def node_preprocess(self, state: RAGState) -> RAGState:
        """입구 가드레일 전에 실행되는 전처리 확장 지점. 현재는 로직 없음."""
        return state

    async def node_postprocess(self, state: RAGState) -> RAGState:
        """출구 가드레일 통과 후, 클라이언트로 보낼 답변에 고정 문구를 덧붙인다.

        가드레일(판정)과는 무관한 별도 단계로 분리해 둔다 — 판정 함수가 답변을
        변형하면 그 변형을 평가 쪽에서 다시 벗겨내야 하는 식의 책임 혼재를
        피하기 위함. DeepEval 평가(`node_evaluate`)는 이 문구가 붙기 전에 저장된
        `state.raw_answer`를 보므로, 이 단계보다 먼저 실행되든 나중에 실행되든
        평가 결과에 영향을 주지 않는다 — 응답 지연 없이 바로 퍼블리시할 수 있도록
        가드레일 뒤·평가 앞에 배치한다.
        """
        state.generated_answer = f"{state.generated_answer}\n\n{OUTPUT_DISCLAIMER}"
        return state

    async def node_output_guardrail(self, state: RAGState) -> RAGState:
        run = state.run
        if not state.req.general_settings.use_guardrails:
            logger.info("[NODE 4] 🛡️ 출구 가드레일: 옵션 비활성화로 생략합니다.")
            if run is not None:
                run.guardrails_out = GuardrailResult(enabled=False, verdict="skipped")
            return state

        logger.info("[NODE 4] 🛡️ 출구 가드레일 가동: 답변의 사실 기반(Fact-checking) 검사 중...")
        if run is not None:
            run.mark(stage="guard_out")
            run.guardrails_out = GuardrailResult(enabled=True)

        guardrails = RAGGuardrails(
            self.llm, state.prompts, trace_prompts=state.trace_prompts
        )
        raise_if_cancelled()
        is_safe = await run_in_session_executor(
            lambda: guardrails.check_output(
                state.query,
                state.generated_answer,
                state.retrieved_context
            )
        )

        if run is not None:
            run.guardrails_out.raw = guardrails.last_raw
        if not is_safe:
            logger.warning("   🚨 할루시네이션 감지됨!")
            if run is not None:
                run.guardrails_out.verdict = "BLOCK"
            if state.retry_count < state.max_retries:
                state.retry_count += 1
                logger.info(
                    f"   🔄 [재시도 {state.retry_count}/{state.max_retries}] "
                    f"답변을 다시 생성합니다..."
                )
                state = await self.node_generate(state)
                return await self.node_output_guardrail(state)
            state.status = "FAIL"
            state.error_message = (
                "⚠️ [가드레일 차단] 생성된 답변의 신뢰성을 보장할 수 없어 "
                "출력을 차단합니다."
            )
            if run is not None:
                run.generation.answer = state.error_message
                run.mark(status="rejected", error=state.error_message, where="guard_out")
            return state

        if run is not None:
            run.guardrails_out.verdict = "PASS"
        return state

    async def run_pipeline(self, req, active_prompts, run: AskRun | None = None) -> str:
        """파이프라인 라우팅 엔진"""
        total_start_time = time.time()
        if run is None:
            run = AskRun.from_request(req)
        state = RAGState(req, active_prompts, run=run)
        state.status = "RUNNING"
        run.status = "running"

        try:
            state = await self.node_preprocess(state)

            state = await self.node_input_guardrail(state)
            if state.status == "FAIL":
                return state.error_message

            state = await self.node_retrieve(state)
            if not state.retrieved_context:
                msg = EMPTY_CONTEXT_MSG
                run.generation.answer = msg
                run.mark(stage="retrieve", status="completed")
                return msg

            state = await self.node_generate(state)

            state = await self.node_output_guardrail(state)
            if state.status == "FAIL":
                return state.error_message

            state = await self.node_postprocess(state)

            sess = get_current_session()
            if sess is not None:
                _publish_answer(
                    sess,
                    state.generated_answer,
                    run,
                    trace_prompts=state.trace_prompts,
                )
            await asyncio.sleep(0)

            state = await self.node_evaluate(state)
            state.status = "SUCCESS"
            if run.status == "running":
                run.mark(stage="done", status="completed")

        except SessionCancelled:
            logger.info("[WORKFLOW] 세션 취소")
            run.mark(status="cancelled", where=run.stage)
            raise
        except ValueError as e:
            logger.warning(f"[WORKFLOW] {e}")
            msg = str(e)
            run.generation.answer = msg
            run.mark(status="rejected", error=msg, where=run.stage)
            raise PipelineFault("RESOURCE_UNAVAILABLE", msg) from e
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"\n\033[91m[WORKFLOW FATAL ERROR] 파이프라인 실행 중 치명적 오류 발생!\033[0m")
            print(f"\033[91m▶ 에러 메시지: {str(e)}\033[0m")
            print(f"\033[91m▶ 상세 추적:\n{error_trace}\033[0m")
            logger.error(f"[WORKFLOW FATAL ERROR] {str(e)}")
            run.mark(status="error", error=str(e), where=run.stage)
            if is_overload(str(e)):
                raise PipelineFault("UPSTREAM_OVERLOAD", str(e)) from e
            if is_transient(str(e)):
                raise PipelineFault("UPSTREAM_ERROR", str(e)) from e
            raise PipelineFault(
                "PIPELINE_ERROR",
                "시스템 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            ) from e

        finally:
            run.timing.t_total_ms = _ms(total_start_time)
            logger.info(
                f"\n[TIME] 🚀 총 워크플로우 소요 시간: "
                f"{time.time() - total_start_time:.2f}초"
            )

        return f"__FINAL_ANSWER__:{state.generated_answer}"
