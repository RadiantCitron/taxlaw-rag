# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
import logging
import re

from llm_handler import get_llm_controller
from session_manager import SessionCancelled

logger = logging.getLogger("my_rag")

_DEEPEVAL_INSTALL_HINT = (
    "DeepEval이 설치되지 않았습니다. 프로젝트 루트에서 "
    "pip install -r requirements.txt 를 실행하거나 eval_settings.enabled=false 로 요청하세요."
)

# DeepEval 이 심사 LLM 에게 요구하는 JSON 형식 지시.
# 컨트롤러는 일반 텍스트만 반환하고, JSON 강제는 이 어댑터에서만 한다.
_DEEPEVAL_JSON_HINT = (
    "\n\n[CRITICAL INSTRUCTION]\n"
    "You are a strict JSON output machine. You MUST output ONLY a valid JSON object.\n"
    "DO NOT include any conversational text, greetings, or explanations.\n"
    "DO NOT wrap the output in markdown code blocks (e.g., ```json ... ```).\n"
    "Your output must start exactly with '{' and end exactly with '}'.\n"
)

_CustomEvaluator_cls = None


def _deepeval_modules():
    try:
        from deepeval import evaluate
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            FaithfulnessMetric,
        )
        from deepeval.models import DeepEvalBaseLLM
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise ImportError(_DEEPEVAL_INSTALL_HINT) from exc
    return (
        LLMTestCase,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        FaithfulnessMetric,
        evaluate,
        DeepEvalBaseLLM,
    )


def _custom_evaluator_class():
    global _CustomEvaluator_cls
    if _CustomEvaluator_cls is not None:
        return _CustomEvaluator_cls

    _, _, _, _, _, DeepEvalBaseLLM = _deepeval_modules()

    class CustomEvaluator(DeepEvalBaseLLM):
        """DeepEval ↔ LLMController 어댑터. OpenRouter / local 식별자를 그대로 쓴다."""

        def __init__(
            self,
            llm_controller,
            eval_prompt_template: str = "",
            *,
            trace_prompts: bool = False,
        ):
            self.llm_controller = llm_controller
            self.eval_prompt_template = eval_prompt_template or ""
            self.trace_prompts = bool(trace_prompts)

        def load_model(self):
            pass

        def generate(self, prompt: str, *args, schema=None, **kwargs):
            return self._generate_with_optional_schema(prompt, schema=schema)

        async def a_generate(self, prompt: str, *args, schema=None, **kwargs):
            return self._generate_with_optional_schema(prompt, schema=schema)

        def _generate_with_optional_schema(self, prompt: str, schema=None):
            system = (self.eval_prompt_template or "") + self._json_hint_for_schema(schema)
            raw = self.llm_controller.evaluate_ask(
                prompt, system, trace_prompts=self.trace_prompts, stage="eval"
            )
            json_str = self._extract_json(raw, schema=schema)
            if schema is None:
                return json_str
            try:
                import json

                data = json.loads(json_str)
                return schema.model_validate(data)
            except Exception:
                return schema.model_validate(self._empty_payload_for_schema(schema))

        @staticmethod
        def _json_hint_for_schema(schema) -> str:
            if schema is None:
                return _DEEPEVAL_JSON_HINT
            try:
                import json

                return (
                    _DEEPEVAL_JSON_HINT
                    + "\nRequired JSON shape (JSON Schema):\n"
                    + json.dumps(schema.model_json_schema(), ensure_ascii=False)
                    + "\n"
                )
            except Exception:
                return _DEEPEVAL_JSON_HINT

        @staticmethod
        def _empty_payload_for_schema(schema) -> dict:
            fields = getattr(schema, "model_fields", None) or {}
            if "verdicts" in fields:
                return {"verdicts": []}
            if "statements" in fields:
                return {"statements": []}
            if "claims" in fields:
                return {"claims": []}
            if "truths" in fields:
                return {"truths": []}
            if "reason" in fields:
                return {"reason": "[파싱 실패]"}
            return {}

        def get_model_name(self):
            return (
                f"Custom Evaluator ({self.llm_controller.provider}:"
                f"{self.llm_controller.model_name})"
            )

        @staticmethod
        def _extract_json(raw: str, schema=None) -> str:
            """응답에서 JSON 객체를 뽑는다. 실패 시 schema 에 맞는 최소 JSON."""
            text = (raw or "").strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                candidate = match.group(0)
                try:
                    import json

                    json.loads(candidate)
                    return candidate
                except Exception:
                    pass
            safe = text.replace('"', "'").replace("\n", " ")[:200]
            fallback = CustomEvaluator._empty_payload_for_schema(schema) if schema else {
                "statements": [],
                "score": 0.0,
                "reason": "[파싱 실패] " + safe,
            }
            if schema is not None and "reason" in fallback and len(fallback) == 1:
                fallback["reason"] = "[파싱 실패] " + safe
            import json

            return json.dumps(fallback, ensure_ascii=False)

    _CustomEvaluator_cls = CustomEvaluator
    return _CustomEvaluator_cls


class RAGEvaluator:
    """단건 DeepEval 평가. 채점 모델은 eval_settings.eval_model (openrouter:/local:) 로 지정."""

    def __init__(self, eval_settings, prompts, main_llm_controller=None, *, trace_prompts: bool = False):
        self.settings = eval_settings
        self.prompts = prompts
        # main_llm_controller 인자는 하위 호환용으로 받되, 더 이상 '메인 재사용'에 쓰지 않는다.
        _ = main_llm_controller

        eval_model_id = (self.settings.eval_model or "").strip()
        if not eval_model_id:
            raise ValueError(
                "eval_settings.eval_model 이 비어 있습니다. "
                "openrouter:... 또는 local:... 식별자를 지정하세요."
            )
        if eval_model_id == "local":
            raise ValueError(
                "eval_model='local' 은 더 이상 '메인 LLM 재사용'을 뜻하지 않습니다. "
                "채점용 모델 식별자를 명시하세요. 예: openrouter:google/gemini-2.0-flash"
            )

        logger.info(f"[EVAL] 채점관 모델: {eval_model_id}")
        eval_ctrl = get_llm_controller(eval_model_id)
        json_eval = getattr(prompts, "json_eval", "") or ""
        CustomEvaluator = _custom_evaluator_class()
        self.eval_target_model = CustomEvaluator(
            eval_ctrl, json_eval, trace_prompts=trace_prompts
        )

    def run_evaluation(
        self,
        query: str,
        answer: str,
        context: str | list[str],
        expected_answer: str | None = None,
    ):
        """DeepEval 단건 평가."""
        (
            LLMTestCase,
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            FaithfulnessMetric,
            evaluate,
            _DeepEvalBaseLLM,
        ) = _deepeval_modules()

        metrics_to_run = []
        selected_metric = self.settings.advanced.metrics
        thres = self.settings.threshold
        strict = self.settings.strict_mode

        if selected_metric in ["answer_relevancy", "all"]:
            metrics_to_run.append(
                AnswerRelevancyMetric(
                    threshold=thres,
                    model=self.eval_target_model,
                    strict_mode=strict,
                    include_reason=True,
                )
            )
        if selected_metric in ["context_precision", "all"]:
            metrics_to_run.append(
                ContextualPrecisionMetric(
                    threshold=thres,
                    model=self.eval_target_model,
                    strict_mode=strict,
                    include_reason=True,
                )
            )
        if selected_metric in ["faithfulness", "all"]:
            metrics_to_run.append(
                FaithfulnessMetric(
                    threshold=thres,
                    model=self.eval_target_model,
                    strict_mode=strict,
                    include_reason=True,
                )
            )

        if not metrics_to_run:
            raise ValueError(
                f"평가 지표가 비어 있습니다: {selected_metric!r}. "
                "allowed: all, answer_relevancy, context_precision, faithfulness"
            )

        if isinstance(context, str):
            chunks = [context] if context else []
        else:
            chunks = [ch for ch in context if ch]

        ref = expected_answer if expected_answer else query

        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            expected_output=ref,
            retrieval_context=chunks,
        )

        try:
            from deepeval.evaluate.configs import CacheConfig
            from deepeval.test_run import global_test_run_manager

            # DeepEval 전역 TestRunManager 는 프로세스 단위 싱글톤이다.
            # Windows 에서 디스크 잠금(portalocker)이 실패하면 test_run=None 이 남고,
            # 다음 evaluate() 가 save_to_disk=True 로 update_test_run 을 타면 크래시한다.
            global_test_run_manager.reset()
            global_test_run_manager.save_to_disk = False

            from deepeval.evaluate.configs import DisplayConfig

            results = evaluate(
                test_cases=[test_case],
                metrics=metrics_to_run,
                cache_config=CacheConfig(write_cache=False, use_cache=False),
                display_config=DisplayConfig(show_indicator=False, print_results=False),
            )
            target_result = results[0] if isinstance(results, list) else results

            if hasattr(target_result, "test_results") and target_result.test_results:
                metrics_list = target_result.test_results[0].metrics_data
            elif hasattr(target_result, "metrics"):
                metrics_list = target_result.metrics
            else:
                raise ValueError("평가 결과에서 지표를 읽지 못했습니다.")

            report = []
            for metric_data in metrics_list:
                score = getattr(metric_data, "score", 0.0)
                reason = getattr(metric_data, "reason", "사유 없음")
                is_successful = getattr(
                    metric_data, "success", getattr(metric_data, "is_successful", False)
                )
                metric_name = getattr(metric_data, "name", "Unknown Metric")
                report.append({
                    "metric": metric_name,
                    "score": float(score) if score else 0.0,
                    "pass": is_successful,
                    "reason": reason,
                })
            return report
        except SessionCancelled:
            logger.info("[EVAL] 세션 연결 종료로 평가 중단")
            return []
        except Exception:
            raise
