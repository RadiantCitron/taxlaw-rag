# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
import logging
import re

logger = logging.getLogger("my_rag")


# =====================================================================
# Guardrails — 오케스트레이터 / 평가기가 공유하는 입·출구 가드레일
# (구 execute_rag_pipeline · DeepEval 어댑터는 evaluator.py 로 이전됨)
# =====================================================================
class RAGGuardrails:
    def __init__(self, llm_controller, prompts, *, trace_prompts: bool = False):
        self.llm = llm_controller
        self.prompts = prompts
        self.trace_prompts = bool(trace_prompts)
        self.last_raw: str | None = None

    def _normalize_verdict(self, raw: str) -> str:
        if not raw:
            return "BLOCK"
        text = raw.strip().upper()
        if re.search(r"\bNOT\s*PASS\b|\bNO\s*PASS\b|\bBLOCK\b|\bFAIL\b|\bREJECT\b", text):
            return "BLOCK"
        first = re.split(r"[\s,.\n:：]", text)[0]
        if first == "PASS" or text == "PASS":
            return "PASS"
        if re.search(r"\bPASS\b", text):
            return "PASS"
        return "BLOCK"

    def check_input(self, user_query: str) -> bool:
        logger.info("\n[GUARDRAIL-IN] 입구 가드레일: 질문의 도메인 적합성 검사 중...")
        if not user_query or not user_query.strip():
            logger.warning("[GUARDRAIL-IN] 차단: 빈 질문")
            self.last_raw = ""
            return False
        prompt = f"{self.prompts.guardrail_in}\n\nUser Query: {user_query}\nResult:"
        raw = self.llm.ask_raw(
            prompt, trace_prompts=self.trace_prompts, stage="guardrail_in"
        )
        self.last_raw = raw
        verdict = self._normalize_verdict(raw)
        if verdict == "PASS":
            logger.info("[GUARDRAIL-IN] 통과: 세무 도메인 관련 질문으로 확인되었습니다.")
            return True

        logger.warning(
            f"[GUARDRAIL-IN] 차단: 도메인 외 질문 또는 부적절한 입력 감지 "
            f"(raw={str(raw)[:80]!r}, verdict={verdict})"
        )
        return False

    def check_output(self, user_query: str, generated_answer: str, context: str) -> bool:
        logger.info("\n[GUARDRAIL-OUT] 출구 가드레일: 답변의 사실 기반(Fact-checking) 검사 중...")
        if not generated_answer or not generated_answer.strip():
            logger.warning("[GUARDRAIL-OUT] 차단: 빈 답변")
            self.last_raw = ""
            return False
        prompt = (
            f"{self.prompts.guardrail_out}\n\n"
            f"[Retrieved Context]\n{context}\n\n"
            f"[Generated Answer]\n{generated_answer}\n\nResult:"
        )
        raw = self.llm.ask_raw(
            prompt, trace_prompts=self.trace_prompts, stage="guardrail_out"
        )
        self.last_raw = raw
        verdict = self._normalize_verdict(raw)
        if verdict == "PASS":
            logger.info("[GUARDRAIL-OUT] 통과: 답변이 검색된 문서에 기반하고 있습니다.")
            return True

        logger.warning(
            f"[GUARDRAIL-OUT] 차단: 문서에 없는 허위 사실(할루시네이션) 감지 "
            f"(raw={str(raw)[:80]!r}, verdict={verdict})"
        )
        return False
