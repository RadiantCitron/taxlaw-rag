# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""HTTP /ask 요청·프롬프트·평가 결과 스키마의 유일한 소유자.

FastAPI 앱을 import 하지 않는다. api_server / evaluate_rag / test_api 가
이 모델을 공유한다. 값 범위 검증은 validation.py 가 한다. 여기서는 타입만 본다.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from schema import ALLOWED_SOURCE_TABLES
from settings import (
    company_summary,
    default_eval_llm,
    default_llm,
    embed_model_id,
    llm_max_tokens,
    llm_temperature,
    rerank_model_id,
)
from sse import SSE_KIND_DONE, SSE_KIND_ERROR, SSE_KIND_EVAL, SSE_KIND_NOTICE

__all__ = [
    "PromptsConfig",
    "GeneralSettings",
    "RerankAdvancedSettings",
    "RerankSettings",
    "EvalAdvancedSettings",
    "EvalSettings",
    "QueryRequest",
    "MetricResult",
    "SSE_KIND_DONE",
    "SSE_KIND_ERROR",
    "SSE_KIND_EVAL",
    "SSE_KIND_NOTICE",
]


class PromptsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    guardrail_in: str = ""
    guardrail_out: str = ""
    query_expansion: str = ""
    vocab_mapping: str = ""
    main_answer: str = ""
    json_eval: str = ""


class GeneralSettings(BaseModel):
    model_config = ConfigDict(extra="allow")
    llm_model: str = Field(default_factory=default_llm)
    embed_model: str = Field(default_factory=embed_model_id)
    log_level: int = Field(default=3)
    query_mode: str = Field(default="none")
    use_guardrails: bool = Field(default=True)
    trace_prompts: bool = Field(default=False)
    fusion_mode: str = Field(default="rrf")
    alpha: float = Field(default=0.5)
    use_graphdb: bool = Field(default=False)
    temperature: float = Field(default_factory=llm_temperature)
    max_tokens: int = Field(default_factory=llm_max_tokens)
    company_summary: str = Field(default_factory=company_summary)
    stream_logs: bool = Field(default=True)
    capture: str | None = Field(default=None)
    embed_variant: str | None = Field(default=None)
    sources: list[str] = Field(default_factory=lambda: list(ALLOWED_SOURCE_TABLES))
    keyword_mode: str = Field(default="trigram")
    base_limit: int = Field(default=20)

    @field_validator("sources", mode="before")
    @classmethod
    def coerce_sources(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v


class RerankAdvancedSettings(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = Field(default_factory=rerank_model_id)
    min_score: float = Field(default=0.0)
    candidate_cap: int = Field(default=50)


class RerankSettings(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = Field(default=False)
    top_k: int = Field(default=3)
    advanced: RerankAdvancedSettings = Field(default_factory=RerankAdvancedSettings)


class EvalAdvancedSettings(BaseModel):
    model_config = ConfigDict(extra="allow")
    metrics: str = Field(default="all")
    # evaluator 가 읽지 않는다. 옛 클라 페이로드 호환용.
    async_mode: bool = Field(default=True)
    use_cache: bool = Field(default=False)


class EvalSettings(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = Field(default=False)
    eval_model: str = Field(default_factory=default_eval_llm)
    threshold: float = Field(default=0.5)
    strict_mode: bool = Field(default=False)
    advanced: EvalAdvancedSettings = Field(default_factory=EvalAdvancedSettings)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    query: str = Field(min_length=1)
    general_settings: GeneralSettings = Field(default_factory=GeneralSettings)
    rerank_settings: RerankSettings = Field(default_factory=RerankSettings)
    eval_settings: EvalSettings = Field(default_factory=EvalSettings)
    custom_prompts: PromptsConfig | None = None


class MetricResult(BaseModel):
    """단건 DeepEval 한 과목. evaluator 는 같은 키의 dict 를 반환해도 된다."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    metric: str
    score: float = 0.0
    passed: bool = Field(default=False, alias="pass")
    reason: str = "사유 없음"
