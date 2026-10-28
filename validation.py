# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""요청 옵션의 허용값·기본값·설명의 유일한 소유자.

Pydantic 은 타입만 검사한다. 값 범위·열거형·조합 제약은 여기서 모아서 거부한다.
FastAPI / psycopg2 / llama_cpp 를 import 하지 않는다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from error_envelope import violation
from rag_run import effective_capture, server_capture
from schema import ALLOWED_SOURCE_TABLES
from settings import (
    company_summary as default_company_summary_setting,
    default_eval_llm,
    default_llm,
    embed_model_id,
    llm_max_tokens,
    llm_temperature,
    rerank_model_id,
)

# 테이블당 hybrid SQL LIMIT (retriever.get_hybrid_assembled_context.base_limit).
BASE_LIMIT_DEFAULT = 20
BASE_LIMIT_MIN = 1
BASE_LIMIT_MAX = 100
# rerank_settings.top_k = 최종 LLM 컨텍스트 문서 개수 상한.
TOP_K_MAX = 20
RERANK_CANDIDATE_CAP = 50
QUERY_MAX_LEN = 4000

logger = logging.getLogger("my_rag")

Checker = Callable[[str], tuple[bool, str]]


@dataclass(frozen=True)
class FieldRule:
    path: str
    kind: str
    allowed: tuple | None = None
    min: float | None = None
    max: float | None = None
    default: Any = None
    hint: str = ""
    resolver: str | None = None


@dataclass
class ValidationOutcome:
    violations: list[dict] = field(default_factory=list)
    ignored_fields: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)


def _default_of(rule: FieldRule) -> Any:
    val = rule.default
    return val() if callable(val) else val


def _rule_expected(rule: FieldRule, samples: dict | None = None) -> dict[str, Any]:
    if rule.kind == "enum":
        return {"type": "enum", "allowed": list(rule.allowed or ())}
    if rule.kind == "enum_list":
        return {"type": "enum", "allowed": list(rule.allowed or ())}
    if rule.kind == "float":
        return {"type": "float", "min": rule.min, "max": rule.max}
    if rule.kind == "int":
        return {
            "type": "int",
            "min": int(rule.min) if rule.min is not None else None,
            "max": int(rule.max) if rule.max is not None else None,
        }
    if rule.kind == "model_ref":
        extra = (samples or {}).get(rule.resolver or "", {})
        expected: dict[str, Any] = {
            "type": "model_ref",
            "source": extra.get("source") or "",
            "rule": extra.get("rule") or "openrouter:<slug> 또는 local:<파일명.gguf>",
        }
        allowed = extra.get("allowed")
        if allowed is not None and len(allowed) <= 10:
            expected["allowed"] = list(allowed)
        else:
            expected["allowed_count"] = extra.get("allowed_count")
            expected["allowed_sample"] = list(extra.get("allowed_sample") or [])[:10]
        return expected
    if rule.kind == "bool":
        return {"type": "bool"}
    if rule.kind == "str":
        return {"type": "str", "min": rule.min, "max": rule.max}
    return {"type": rule.kind}


FIELD_RULES: tuple[FieldRule, ...] = (
    FieldRule(
        "general_settings.llm_model", "model_ref", default=default_llm, resolver="llm",
        hint="답변 생성과 질문 확장에 쓰는 메인 LLM 입니다. GET /llm/models 목록의 id 를 씁니다. "
             "OpenRouter 모델은 openrouter: 접두사가 필요하고, 로컬 GGUF 는 local:파일명.gguf 형식입니다.",
    ),
    FieldRule(
        "general_settings.embed_model", "model_ref", default=embed_model_id, resolver="embed",
        hint="검색용 임베딩 모델입니다. 기본(indexed) 모델은 DB 의 원본 embedding 컬럼을 검색합니다. "
             "다른 카탈로그 모델을 고르면 그 모델 전용 비교 테이블을 검색합니다.",
    ),
    FieldRule(
        "general_settings.embed_variant", "model_ref", default=None, resolver="embed",
        hint="비교 실험용입니다. 보통은 비워 둡니다. embed_model 로 비교 모델을 고르면 서버가 자동으로 채웁니다.",
    ),
    FieldRule(
        "general_settings.log_level", "enum", allowed=(1, 3, 5), default=3,
        hint="서버가 보내주는 로그의 상세도입니다. 1=오류만, 3=일반, 5=디버그.",
    ),
    FieldRule(
        "general_settings.query_mode", "enum",
        allowed=("none", "expansion", "vocab_mapping"), default="none",
        hint="검색 전에 질문을 다듬는 방식입니다. none=원문 그대로, expansion=LLM 이 질문을 확장, "
             "vocab_mapping=LLM 이 DB 단어장에서 키워드를 고름.",
    ),
    FieldRule(
        "general_settings.fusion_mode", "enum", allowed=("rrf", "score_based"), default="rrf",
        hint="키워드 검색과 벡터 검색 결과를 합치는 방식입니다. rrf=순위 기반(안정적), "
             "score_based=점수 가중합(alpha 로 비중 조절).",
    ),
    FieldRule(
        "general_settings.alpha", "float", min=0.0, max=1.0, default=0.5,
        hint="fusion_mode=score_based 일 때만 쓰입니다. 벡터 검색에 주는 가중치이고, 나머지가 키워드 검색 몫입니다. "
             "0.5 는 동일 비중, 1.0 은 벡터만, 0.0 은 키워드만.",
    ),
    FieldRule(
        "general_settings.sources", "enum_list", allowed=ALLOWED_SOURCE_TABLES,
        default=lambda: list(ALLOWED_SOURCE_TABLES),
        hint="검색할 원본 테이블입니다. tax_laws_master=법령, tax_precedents_master=판례, "
             "tax_tribunal_master=심판, tax_amendments_2026=개정해설. 여러 개를 고를 수 있습니다.",
    ),
    FieldRule(
        "general_settings.keyword_mode", "enum", allowed=("trigram", "kiwi"), default="trigram",
        hint="키워드 추출 방식입니다. trigram=정규식+trigram(기본), kiwi=형태소 분석(kiwipiepy 설치 필요). "
             "vocab_mapping 과는 함께 쓸 수 없습니다.",
    ),
    FieldRule(
        "general_settings.base_limit", "int",
        min=BASE_LIMIT_MIN, max=BASE_LIMIT_MAX, default=BASE_LIMIT_DEFAULT,
        hint="테이블당 hybrid 검색 SQL LIMIT 입니다. 키워드·벡터를 합친 후보 풀 크기를 늘리거나 줄입니다. "
             "기본 20, 1~100.",
    ),
    FieldRule(
        "general_settings.capture", "enum", allowed=("lean", "full", "forensic"), default=None,
        hint="이 질문의 실행 기록을 얼마나 남길지입니다. lean=요약만, full=검색 문서까지 JSON 덤프, "
             "forensic=LLM 프롬프트까지. 서버 상한을 넘는 요청은 상한으로 낮춰 실행합니다.",
    ),
    FieldRule(
        "general_settings.temperature", "float", min=0.0, max=2.0, default=llm_temperature,
        hint="LLM 출력의 무작위성입니다. 0 에 가까울수록 같은 질문에 같은 답을 냅니다. "
             "법령 질의응답은 낮은 값이 적합합니다.",
    ),
    FieldRule(
        "general_settings.max_tokens", "int", min=1, max=32768, default=llm_max_tokens,
        hint="답변의 최대 길이(토큰)입니다. 너무 작으면 답변이 중간에 잘립니다.",
    ),
    FieldRule("general_settings.use_guardrails", "bool", default=True,
              hint="도메인 이탈·부적절 질문을 입구에서 거를지입니다."),
    FieldRule("general_settings.use_graphdb", "bool", default=False,
              hint="Neo4j 법령 그래프 컨텍스트를 검색에 포함합니다."),
    FieldRule(
        "general_settings.company_summary", "str", min=1, max=8000,
        default=default_company_summary_setting,
        hint="회사·학습 맥락 요약입니다. 빈 문자열은 허용하지 않습니다. 생략하면 서버 기본값을 씁니다.",
    ),
    FieldRule("general_settings.trace_prompts", "bool", default=False,
              hint="LLM 에 실제로 넘긴 프롬프트를 로그로 보낼지입니다."),
    FieldRule("general_settings.stream_logs", "bool", default=True,
              hint="서버 로그를 SSE 로 클라이언트가 받을지입니다."),
    FieldRule("rerank_settings.enabled", "bool", default=False,
              hint="CrossEncoder 로 검색 후보를 다시 줄지입니다."),
    FieldRule(
        "rerank_settings.top_k", "int", min=1, max=TOP_K_MAX, default=3,
        hint="최종적으로 LLM 에게 넘길 문서 개수입니다. 1~20.",
    ),
    FieldRule(
        "rerank_settings.advanced.model", "model_ref", default=rerank_model_id, resolver="rerank",
        hint="리랭커 모델입니다. GET /rerank/models 목록의 id 를 씁니다.",
    ),
    FieldRule(
        "rerank_settings.advanced.min_score", "float", min=0.0, max=1.0, default=0.0,
        hint="리랭킹 점수가 이 값 미만인 문서를 버립니다. 0.0 이면 버리지 않습니다. "
             "너무 높이면 문서가 하나도 안 남아 답변이 비어 있을 수 있습니다.",
    ),
    FieldRule(
        "rerank_settings.advanced.candidate_cap", "int",
        min=1, max=RERANK_CANDIDATE_CAP, default=RERANK_CANDIDATE_CAP,
        hint="CrossEncoder 에 넣을 최대 후보 수. 테이블 합산 상위부터.",
    ),
    FieldRule("eval_settings.enabled", "bool", default=False,
              hint="DeepEval 로 답변을 채점할지입니다. 문항마다 추가 LLM 호출이 납니다."),
    FieldRule(
        "eval_settings.eval_model", "model_ref", default=default_eval_llm, resolver="llm",
        hint="채점용 LLM 입니다. GET /llm/models 목록의 id 를 씁니다.",
    ),
    FieldRule(
        "eval_settings.threshold", "float", min=0.0, max=1.0, default=0.5,
        hint="평가 합격 커트라인입니다. 이 점수 이상이면 PASS.",
    ),
    FieldRule("eval_settings.strict_mode", "bool", default=False, hint="DeepEval 엄격 모드입니다."),
    FieldRule(
        "eval_settings.advanced.metrics", "enum",
        allowed=("all", "answer_relevancy", "context_precision"), default="all",
        hint="평가 항목입니다. answer_relevancy=답변이 질문에 맞는가, "
             "context_precision=검색이 정확했는가, all=둘 다.",
    ),
    FieldRule("eval_settings.advanced.async_mode", "bool", default=True,
              hint="미사용입니다. 옛 클라 호환용 필드이며 평가기는 읽지 않습니다."),
    FieldRule("eval_settings.advanced.use_cache", "bool", default=False,
              hint="미사용입니다. 옛 클라 호환용 필드이며 평가기는 읽지 않습니다."),
    FieldRule(
        "query", "str", min=1, max=QUERY_MAX_LEN, default=None,
        hint="질문 본문입니다. 1자 이상 4000자 이하여야 합니다.",
    ),
)

_RULES_BY_PATH = {r.path: r for r in FIELD_RULES}


def schema_payload() -> dict:
    fields = []
    for rule in FIELD_RULES:
        item: dict[str, Any] = {
            "path": rule.path,
            "type": "enum" if rule.kind == "enum_list" else rule.kind,
            "default": _default_of(rule),
            "hint": rule.hint,
        }
        expected = _rule_expected(rule)
        for key in ("allowed", "min", "max", "source", "rule", "allowed_count", "allowed_sample"):
            if key in expected:
                item[key] = expected[key]
        fields.append(item)
    return {"fields": fields}


def rule_for_path(path: str) -> FieldRule | None:
    return _RULES_BY_PATH.get(path)


def pydantic_violation(err: dict) -> dict:
    loc = [str(x) for x in (err.get("loc") or []) if x != "body"]
    path = ".".join(loc) or "body"
    rule = _RULES_BY_PATH.get(path)
    reason = "요청 본문 형식이 올바르지 않습니다."
    if rule is not None:
        return violation(
            path,
            err.get("input"),
            reason,
            _rule_expected(rule),
            _default_of(rule),
            rule.hint,
        )
    return violation(path, err.get("input"), reason, {"type": "unknown"}, None, "")


def _dig(raw: Any, path: str) -> tuple[bool, Any]:
    cur = raw
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def collect_and_clear_extras(obj: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    extra = getattr(obj, "__pydantic_extra__", None)
    if isinstance(extra, dict):
        for key in list(extra.keys()):
            paths.append(f"{prefix}.{key}" if prefix else str(key))
        extra.clear()
    fields = getattr(type(obj), "model_fields", None)
    if not fields:
        return paths
    for name in fields:
        child = getattr(obj, name, None)
        if child is None:
            continue
        if getattr(type(child), "model_fields", None) is not None:
            child_prefix = f"{prefix}.{name}" if prefix else name
            paths.extend(collect_and_clear_extras(child, child_prefix))
    return paths


def _as_number(value: Any):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return None
    return None


def _check_rule(
    rule: FieldRule,
    given: Any,
    *,
    checkers: dict[str, Checker] | None,
    samples: dict | None,
) -> dict | None:
    expected = _rule_expected(rule, samples)
    default = _default_of(rule)
    hint = rule.hint

    if rule.kind == "bool":
        return None

    if rule.kind == "enum":
        normalized = given
        if isinstance(given, str) and all(isinstance(a, str) for a in (rule.allowed or ())):
            normalized = given.strip().lower()
        if given is None or given == "":
            return None
        if normalized not in (rule.allowed or ()):
            return violation(
                rule.path, given, "허용 목록에 없는 값입니다.", expected, default, hint,
            )
        return None

    if rule.kind == "enum_list":
        items = given
        if isinstance(items, str):
            items = [p.strip() for p in items.split(",") if p.strip()]
        if not isinstance(items, (list, tuple)):
            return violation(
                rule.path, given, "테이블 목록은 배열이어야 합니다.", expected, default, hint,
            )
        if not items:
            return violation(
                rule.path, given, "검색 테이블이 비어 있습니다.", expected, default, hint,
            )
        bad = [x for x in items if x not in (rule.allowed or ())]
        if bad:
            return violation(
                rule.path, given,
                f"허용되지 않은 검색 테이블: {bad[0]}", expected, default, hint,
            )
        return None

    if rule.kind in ("float", "int"):
        num = _as_number(given)
        if num is None:
            return violation(
                rule.path, given, "허용 범위를 벗어났습니다.", expected, default, hint,
            )
        if rule.kind == "int" and int(num) != num:
            return violation(
                rule.path, given, "정수가 아닙니다.", expected, default, hint,
            )
        if rule.min is not None and num < rule.min:
            return violation(
                rule.path, given, "허용 범위를 벗어났습니다.", expected, default, hint,
            )
        if rule.max is not None and num > rule.max:
            return violation(
                rule.path, given, "허용 범위를 벗어났습니다.", expected, default, hint,
            )
        return None

    if rule.kind == "str":
        text = "" if given is None else str(given)
        if not text.strip():
            reason = "질문이 비어 있습니다."
            if rule.path.endswith("company_summary"):
                reason = (
                    "company_summary 는 빈 문자열일 수 없습니다. "
                    "필드를 생략하거나 내용을 넣으세요."
                )
            elif rule.path != "query":
                reason = "값이 비어 있습니다."
            return violation(
                rule.path, given, reason, expected, default, hint,
            )
        if rule.max is not None and len(text) > rule.max:
            return violation(
                rule.path, given, f"최대 {int(rule.max)}자까지입니다.", expected, default, hint,
            )
        return None

    if rule.kind == "model_ref":
        if given is None or given == "":
            return None
        checker = (checkers or {}).get(rule.resolver or "")
        if checker is None:
            return None
        ok, _err = checker(str(given).strip())
        if not ok:
            return violation(
                rule.path, given, "허용 목록에 없는 모델입니다.", expected, default, hint,
            )
        return None
    return None


def _combo_violations(raw: dict, req: Any) -> list[dict]:
    out: list[dict] = []
    g = getattr(req, "general_settings", None)
    query_mode = str(getattr(g, "query_mode", None) or "none").strip().lower()
    keyword_mode = str(getattr(g, "keyword_mode", None) or "trigram").strip().lower()
    _present, variant_val = _dig(raw, "general_settings.embed_variant")
    variant = (str(variant_val).strip() if variant_val else "") or None
    if not variant:
        raw_var = getattr(g, "embed_variant", None)
        variant = (str(raw_var).strip() if raw_var else "") or None

    q_rule = _RULES_BY_PATH["general_settings.query_mode"]
    k_rule = _RULES_BY_PATH["general_settings.keyword_mode"]
    v_rule = _RULES_BY_PATH["general_settings.embed_variant"]

    if query_mode == "vocab_mapping" and keyword_mode != "trigram":
        reason = (
            "query_mode=vocab_mapping 과 keyword_mode=kiwi 는 함께 쓸 수 없습니다. "
            "vocab_mapping 은 LLM 이 키워드를 만드는 경로라 형태소 분석이 개입할 자리가 없습니다. "
            "query_mode 를 none 또는 expansion 으로 바꾸거나, keyword_mode 를 trigram 으로 되돌리세요."
        )
        out.append(violation(
            q_rule.path, query_mode, reason, _rule_expected(q_rule),
            _default_of(q_rule), q_rule.hint,
        ))
        out.append(violation(
            k_rule.path, keyword_mode, reason, _rule_expected(k_rule),
            _default_of(k_rule), k_rule.hint,
        ))

    if query_mode == "vocab_mapping" and variant:
        reason = (
            "query_mode=vocab_mapping 과 embed_variant 는 함께 쓸 수 없습니다. "
            "단어장 매핑은 원본 테이블 검색 경로입니다. query_mode 를 바꾸거나 embed_variant 를 비우세요."
        )
        out.append(violation(
            q_rule.path, query_mode, reason, _rule_expected(q_rule),
            _default_of(q_rule), q_rule.hint,
        ))
        out.append(violation(
            v_rule.path, variant, reason, _rule_expected(v_rule),
            _default_of(v_rule), v_rule.hint,
        ))
    return out


def _apply_normalizations(req: Any, raw: dict) -> None:
    g = getattr(req, "general_settings", None)
    if g is None:
        return
    present, qmode = _dig(raw, "general_settings.query_mode")
    if present and isinstance(qmode, str):
        g.query_mode = qmode.strip().lower()
    present, fmode = _dig(raw, "general_settings.fusion_mode")
    if present and isinstance(fmode, str):
        g.fusion_mode = fmode.strip().lower()
    present, kmode = _dig(raw, "general_settings.keyword_mode")
    if present and isinstance(kmode, str):
        g.keyword_mode = kmode.strip().lower()
    present, cap = _dig(raw, "general_settings.capture")
    if present:
        if cap is None or (isinstance(cap, str) and not str(cap).strip()):
            g.capture = None
        elif isinstance(cap, str):
            g.capture = cap.strip().lower()
    present, metrics = _dig(raw, "eval_settings.advanced.metrics")
    if present and isinstance(metrics, str) and getattr(req, "eval_settings", None):
        req.eval_settings.advanced.metrics = metrics.strip().lower()


def validate_request(
    req: Any,
    raw_body: dict | None,
    *,
    model_checkers: dict[str, Checker] | None = None,
    model_samples: dict | None = None,
) -> ValidationOutcome:
    """위반을 전부 모아서 반환한다. 첫 위반에서 멈추지 않는다."""
    raw = raw_body if isinstance(raw_body, dict) else {}
    outcome = ValidationOutcome()
    outcome.ignored_fields = collect_and_clear_extras(req)

    for rule in FIELD_RULES:
        if rule.kind == "bool":
            continue
        present, given = _dig(raw, rule.path)
        if not present:
            if rule.path == "query":
                given = getattr(req, "query", None)
            else:
                continue
        item = _check_rule(rule, given, checkers=model_checkers, samples=model_samples)
        if item is not None:
            outcome.violations.append(item)

    outcome.violations.extend(_combo_violations(raw, req))

    if not outcome.violations:
        _apply_normalizations(req, raw)
        try:
            requested = getattr(getattr(req, "general_settings", None), "capture", None)
            used = effective_capture(req)
            if requested and used != requested:
                ceiling = server_capture()
                outcome.notices.append(
                    f"요청한 capture={requested} 은 서버 상한 {ceiling} 을 넘습니다. "
                    f"{used} 으로 낮춰 실행합니다."
                )
        except Exception as exc:
            logger.warning("[VALIDATE] capture notice 생성 실패: %s", exc)
        _fp, fusion_val = _dig(raw, "general_settings.fusion_mode")
        alpha_present, _ = _dig(raw, "general_settings.alpha")
        fusion = str(
            fusion_val
            or getattr(getattr(req, "general_settings", None), "fusion_mode", "")
            or "rrf"
        ).strip().lower()
        if fusion == "rrf" and alpha_present:
            outcome.notices.append(
                "alpha 는 fusion_mode=score_based 일 때만 사용됩니다. 이번 요청은 rrf 라 무시됩니다."
            )

    return outcome


def pick_error_code(violations: list[dict]) -> str:
    reasons = " ".join(str(v.get("reason") or "") for v in violations)
    if "함께 쓸 수 없습니다" in reasons:
        return "INCOMPATIBLE_OPTIONS"
    model_only = all(
        (v.get("reason") or "").startswith("허용 목록에 없는 모델") for v in violations
    ) if violations else False
    if model_only:
        return "UNKNOWN_MODEL"
    return "INVALID_OPTION_VALUE"


def summarize_violations(violations: list[dict]) -> str:
    parts = []
    for v in violations:
        field = v.get("field") or "?"
        given = v.get("given")
        shown = repr(given)
        if field == "query" and isinstance(given, str) and len(given) > 40:
            shown = repr(given[:40] + "…")
        parts.append(f"{field}={shown}")
    return ", ".join(parts)
