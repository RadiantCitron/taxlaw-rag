# -*- coding: utf-8 -*-
# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""스마트폰용 세법 RAG 채팅. 검색·생성 로직은 기존 파이프라인을 그대로 호출한다.

실행: streamlit run app.py
FastAPI(api_server.py)와 index.html 은 이 앱과 별개로 남는다.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

import streamlit as st
from dotenv import load_dotenv

from bootstrap import boot_core, merge_prompts, resolve_prompt_file_path
from contracts import QueryRequest
from database.neo4j_repo import Neo4jRepo
from error_envelope import PipelineFault
from llm_handler import is_allowed_model_id, models_catalog_payload
from model_registry import (
    apply_embed_choice,
    embedding_catalog_payload,
    reranker_catalog_payload,
    validate_embed_id_any,
    validate_rerank_model,
)
from rag_run import AskRun
from session_manager import SessionCancelled
from settings import DEFAULT_PROMPT_FILENAME
from validation import pick_error_code, validate_request
from workflow_orchestrator import get_orchestrator

logger = logging.getLogger("my_rag")

PREFER_LLM = "openrouter:google/gemini-3.5-flash-lite"
ANSWER_PREFIX = "__FINAL_ANSWER__:"
TERMS = """[면책 및 이용 안내]

본 사이트는 개인 학습 및 기술 포트폴리오용 데모입니다.
제공되는 모든 답변은 인공지능이 생성한 참고 정보일 뿐이며,
전문적인 조언(법률, 세무, 의료, 재무 등)을 대신하지 않습니다.

- AI 답변은 불완전하거나 실제와 다를 수 있습니다.
- 이 답변을 근거로 중요한 결정이나 행동을 하지 마십시오.
- 실제 문제는 반드시 해당 분야 전문가에게 상담받으시기 바랍니다.
- 질문 시 개인정보(실명, 주민번호, 연락처 등)를 입력하지 마십시오.

본 데모 이용으로 발생하는 결과에 대해 운영자는 책임을 지지 않습니다.
계속 이용하시면 위 내용에 동의한 것으로 간주됩니다.
"""

SOURCE_OPTIONS = (
    ("법령", "tax_laws_master"),
    ("판례", "tax_precedents_master"),
    ("심판", "tax_tribunal_master"),
    ("개정해설", "tax_amendments_2026"),
)
QUERY_MODES = (
    ("None (원문)", "none"),
    ("Expansion (확장)", "expansion"),
    ("Vocab Mapping (단어장)", "vocab_mapping"),
)
KEYWORD_MODES = (
    ("Trigram", "trigram"),
    ("Kiwi (형태소)", "kiwi"),
)
FUSION_MODES = (
    ("RRF (순위 기반)", "rrf"),
    ("Score (점수 기반)", "score_based"),
)
LOG_LEVELS = (
    ("Error", 1),
    ("Info", 3),
    ("Debug", 5),
)
QUERY_LABELS = {value: label for label, value in QUERY_MODES}
KEYWORD_LABELS = {value: label for label, value in KEYWORD_MODES}
FUSION_LABELS = {value: label for label, value in FUSION_MODES}
LOG_LABELS = {value: label for label, value in LOG_LEVELS}

MOBILE_CSS = """
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-left: 0.6rem;
        padding-right: 0.6rem;
        padding-bottom: 5rem;
    }
    header[data-testid="stHeader"] { display: none; }
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarCollapsedControl"] { display: none; }
    [data-testid="stToolbar"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
"""


def _load_env() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(root, ".env"))


def _prompt_cli_for_boot() -> str | None:
    """환경변수 경로가 없거나 깨졌으면 한국어 기본 프롬프트를 쓴다."""
    resolved = resolve_prompt_file_path(None)
    if resolved and os.path.exists(resolved):
        return None
    return DEFAULT_PROMPT_FILENAME


@st.cache_resource(show_spinner="세법 RAG를 준비하는 중...")
def boot_app():
    """DB 풀·모델 레지스트리·프롬프트. Neo4j 실패는 그래프 미사용으로 넘긴다."""
    _load_env()
    prompts = boot_core(prompt_cli=_prompt_cli_for_boot())
    try:
        Neo4jRepo.init_driver()
        if Neo4jRepo.driver is None:
            logger.warning("Neo4j 미연결. use_graphdb 는 빈 보충만 반환합니다.")
    except Exception as exc:
        logger.warning("Neo4j 초기화 실패(그래프 미사용이면 무시): %s", exc)
    return prompts


def _model_checkers() -> dict:
    return {
        "llm": is_allowed_model_id,
        "embed": validate_embed_id_any,
        "rerank": validate_rerank_model,
    }


def _ids(payload: dict) -> list[str]:
    return [m["id"] for m in (payload.get("models") or []) if m.get("id")]


def _pick_index(options: list[str], preferred: str) -> int:
    if preferred in options:
        return options.index(preferred)
    return 0


def iter_answer(text: str):
    """완료된 답을 짧게 잘라 st.write_stream 에 넘긴다."""
    body = text or ""
    step = 24
    for i in range(0, len(body), step):
        yield body[i:i + step]
        time.sleep(0.015)


def _format_violations(violations: list[dict]) -> str:
    lines = []
    for item in violations:
        field = item.get("field") or ""
        reason = item.get("reason") or "허용되지 않는 값입니다."
        lines.append(f"{field}: {reason}" if field else reason)
    return "\n".join(lines)


def _format_eval(run: AskRun) -> str:
    if getattr(run.eval, "error", None):
        return str(run.eval.error)
    lines = []
    for item in run.eval.report or []:
        if not isinstance(item, dict):
            continue
        metric = item.get("metric") or "metric"
        score = item.get("score")
        passed = item.get("pass", item.get("passed"))
        mark = "PASS" if passed else "FAIL"
        reason = item.get("reason") or ""
        lines.append(f"{metric}: {score} ({mark})\n{reason}".strip())
    return "\n\n".join(lines)


def _strip_answer(raw: str) -> str:
    text = raw or ""
    if text.startswith(ANSWER_PREFIX):
        return text[len(ANSWER_PREFIX):]
    return text


def run_ask(query: str, settings: dict, prompts) -> dict:
    """QueryRequest 검증 후 run_pipeline. 비즈니스 로직은 호출만 한다."""
    raw = {
        "query": query,
        "general_settings": {
            "llm_model": settings["llm_model"],
            "embed_model": settings["embed_model"],
            "log_level": settings["log_level"],
            "query_mode": settings["query_mode"],
            "keyword_mode": settings["keyword_mode"],
            "use_guardrails": settings["use_guardrails"],
            "trace_prompts": settings["trace_prompts"],
            "fusion_mode": settings["fusion_mode"],
            "alpha": settings["alpha"],
            "use_graphdb": settings["use_graphdb"],
            "sources": settings["sources"],
        },
        "rerank_settings": {
            "enabled": settings["rerank_enabled"],
            "top_k": settings["rerank_top_k"],
            "advanced": {
                "min_score": settings["rerank_min_score"],
            },
        },
        "eval_settings": {
            "enabled": settings["eval_enabled"],
            "eval_model": settings["eval_model"],
            "threshold": settings["eval_threshold"],
            "strict_mode": settings["eval_strict"],
        },
    }
    if settings.get("rerank_model"):
        raw["rerank_settings"]["advanced"]["model"] = settings["rerank_model"]

    try:
        req = QueryRequest.model_validate(raw)
    except Exception as exc:
        return {"error": f"요청 형식이 올바르지 않습니다.\n{exc}"}

    outcome = validate_request(
        req, raw, model_checkers=_model_checkers(),
    )
    if outcome.violations:
        code = pick_error_code(outcome.violations)
        return {"error": f"{code}\n{_format_violations(outcome.violations)}"}

    notices = list(outcome.notices or [])
    try:
        apply_embed_choice(req.general_settings)
        active_prompts = merge_prompts(prompts, req.custom_prompts)
        orchestrator = get_orchestrator(req.general_settings.llm_model)
        run = AskRun.from_request(req)
        result = asyncio.run(orchestrator.run_pipeline(req, active_prompts, run=run))
    except PipelineFault as exc:
        return {"error": exc.message, "notices": notices}
    except SessionCancelled:
        return {"error": "요청이 취소되었습니다.", "notices": notices}
    except Exception as exc:
        logger.exception("[APP] 파이프라인 실패")
        return {"error": f"시스템 오류가 발생했습니다.\n{exc}", "notices": notices}

    answer = _strip_answer(result if isinstance(result, str) else str(result or ""))
    sources = (run.retrieval.context_text or "").strip()
    eval_text = ""
    if req.eval_settings.enabled:
        eval_text = _format_eval(run).strip()
    return {
        "answer": answer,
        "sources": sources,
        "eval": eval_text,
        "notices": notices,
    }


def _render_assistant(msg: dict, *, stream: bool) -> None:
    for note in msg.get("notices") or []:
        st.caption(note)
    if msg.get("error"):
        st.error(msg["error"])
        return
    if stream:
        st.write_stream(iter_answer(msg.get("content") or ""))
    else:
        st.markdown(msg.get("content") or "")
    sources = (msg.get("sources") or "").strip()
    if sources:
        with st.expander("📄 출처 확인"):
            st.text(sources)
    eval_text = (msg.get("eval") or "").strip()
    if eval_text:
        with st.expander("평가 결과"):
            st.text(eval_text)


def _settings_form(llm_ids: list[str], embed_rows: list[dict], rerank_ids: list[str]) -> dict:
    embed_ids = [row["id"] for row in embed_rows if row.get("id")]
    embed_labels = {}
    embed_default = ""
    for row in embed_rows:
        mid = row.get("id")
        if not mid:
            continue
        label = row.get("label") or mid
        if row.get("default") or row.get("indexed"):
            embed_labels[mid] = f"{label} (기본)"
            if row.get("default"):
                embed_default = mid
        else:
            embed_labels[mid] = f"{label} (비교)"
    if not embed_default and embed_ids:
        embed_default = embed_ids[0]

    with st.expander("⚙️ 모델 및 검색 설정", expanded=False):
        st.caption("기본 설정")
        llm_model = st.selectbox(
            "LLM",
            llm_ids,
            index=_pick_index(llm_ids, PREFER_LLM),
            key="opt_llm",
        )
        embed_model = st.selectbox(
            "Embed",
            embed_ids,
            index=_pick_index(embed_ids, embed_default),
            format_func=lambda mid: embed_labels.get(mid, mid),
            key="opt_embed",
        )
        use_graphdb = st.checkbox("Graph DB", value=False, key="opt_graphdb")
        sources = []
        for label, table in SOURCE_OPTIONS:
            if st.checkbox(label, value=True, key=f"src_{table}"):
                sources.append(table)
        query_mode = st.selectbox(
            "Query Mode",
            [item[1] for item in QUERY_MODES],
            format_func=lambda value: QUERY_LABELS[value],
            key="opt_query_mode",
        )
        keyword_mode = st.selectbox(
            "Keyword",
            [item[1] for item in KEYWORD_MODES],
            format_func=lambda value: KEYWORD_LABELS[value],
            key="opt_keyword_mode",
        )
        log_level = st.selectbox(
            "Log",
            [item[1] for item in LOG_LEVELS],
            index=1,
            format_func=lambda value: LOG_LABELS[value],
            key="opt_log",
        )
        use_guardrails = st.checkbox("Guardrails", value=True, key="opt_guardrail")
        trace_prompts = st.checkbox("Trace prompts", value=False, key="opt_trace")
        fusion_mode = st.selectbox(
            "Fusion Mode",
            [item[1] for item in FUSION_MODES],
            format_func=lambda value: FUSION_LABELS[value],
            key="opt_fusion",
        )
        alpha = st.slider("Alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="opt_alpha")

        st.caption("리랭킹")
        rerank_enabled = st.checkbox("Enable", value=False, key="opt_rerank_enable")
        rerank_top_k = st.slider("Top-K", min_value=1, max_value=20, value=3, step=1, key="opt_rerank_topk")
        rerank_min_score = st.slider(
            "Min Score", min_value=0.0, max_value=1.0, value=0.0, step=0.1, key="opt_rerank_minscore",
        )
        rerank_model = ""
        if rerank_ids:
            rerank_model = st.selectbox("Model", rerank_ids, key="opt_rerank_model")
        else:
            st.caption("리랭크 모델 목록이 없습니다.")

        st.caption("평가 (DeepEval)")
        eval_enabled = st.checkbox("Enable", value=False, key="opt_eval_enable")
        eval_threshold = st.slider(
            "Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="opt_eval_thres",
        )
        eval_strict = st.checkbox("Strict", value=False, key="opt_eval_strict")
        eval_model = st.selectbox(
            "Judge",
            llm_ids,
            index=_pick_index(llm_ids, PREFER_LLM),
            key="opt_eval_model",
        )

    return {
        "llm_model": llm_model,
        "embed_model": embed_model,
        "use_graphdb": use_graphdb,
        "sources": sources,
        "query_mode": query_mode,
        "keyword_mode": keyword_mode,
        "log_level": int(log_level),
        "use_guardrails": use_guardrails,
        "trace_prompts": trace_prompts,
        "fusion_mode": fusion_mode,
        "alpha": round(float(alpha), 1),
        "rerank_enabled": rerank_enabled,
        "rerank_top_k": int(rerank_top_k),
        "rerank_min_score": round(float(rerank_min_score), 1),
        "rerank_model": rerank_model,
        "eval_enabled": eval_enabled,
        "eval_threshold": round(float(eval_threshold), 1),
        "eval_strict": eval_strict,
        "eval_model": eval_model,
    }


def main() -> None:
    st.set_page_config(
        page_title="세법 RAG",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agreed" not in st.session_state:
        st.session_state.agreed = False

    if not st.session_state.agreed:
        with st.expander("⚠️ 면책 및 이용 안내", expanded=True):
            st.text(TERMS)
            if st.checkbox("위 약관 및 면책조항을 충분히 인지하였고 동의합니다.", key="agree_check"):
                st.session_state.agreed = True
                st.rerun()
        st.caption("동의하면 질문을 입력할 수 있습니다.")
        return

    try:
        prompts = boot_app()
    except Exception as exc:
        st.error(f"시작에 실패했습니다.\n{exc}")
        return

    llm_ids = _ids(models_catalog_payload())
    embed_payload = embedding_catalog_payload()
    embed_rows = list(embed_payload.get("models") or [])
    rerank_ids = _ids(reranker_catalog_payload())
    if not llm_ids:
        st.error("LLM 목록이 없습니다. openrouter_models.json 과 서버 환경을 확인하세요.")
        return
    if not embed_rows:
        st.error("임베딩 모델이 없습니다.")
        return

    settings = _settings_form(llm_ids, embed_rows, rerank_ids)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_assistant(msg, stream=False)
            else:
                st.markdown(msg.get("content") or "")

    prompt = st.chat_input("세무 질문을 입력하세요")
    if not prompt:
        return
    question = prompt.strip()
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not settings["sources"]:
            result = {"error": "검색할 자료를 하나 이상 고르세요."}
        else:
            with st.spinner("검색하고 답변을 생성하는 중..."):
                result = run_ask(question, settings, prompts)
        answer = result.get("answer") or ""
        if not result.get("error") and not answer.strip():
            answer = "(답변이 비어 있습니다.)"
        assistant = {
            "role": "assistant",
            "content": answer,
            "sources": result.get("sources") or "",
            "eval": result.get("eval") or "",
            "notices": result.get("notices") or [],
            "error": result.get("error") or "",
        }
        _render_assistant(assistant, stream=not assistant["error"])
    st.session_state.messages.append(assistant)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
else:
    main()
