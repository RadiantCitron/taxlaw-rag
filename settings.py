# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""프로세스 환경변수 읽기의 유일한 소유자.

import 시점에 서버를 죽이지 않는다. 필수 값 검사는 require_server_env() 가
부팅 경로에서만 수행한다. FastAPI / psycopg2 를 import 하지 않는다.
"""
from __future__ import annotations

import os

DEFAULT_COMPANY_SUMMARY = (
    "우리 회사는 세무·회계 업무를 수행하며 관련 법령을 준수합니다."
)
DEFAULT_EMBED_MODEL = "jhgan/ko-sroberta-multitask"
DEFAULT_RERANK_MODEL = "Dongjin-kr/ko-reranker"
DEFAULT_EVAL_LLM = "openrouter:google/gemini-3.5-flash-lite"
DEFAULT_PROMPT_FILENAME = "prompts_ko_세무.json"
PROMPT_FILE_EN = "prompts_en_세무.json"
OPENROUTER_MODELS_FILE = "openrouter_models.json"


def _env(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip()


def default_llm() -> str:
    return _env("DEFAULT_LLM", "") or ""


def default_eval_llm() -> str:
    return _env("DEFAULT_EVAL_LLM", DEFAULT_EVAL_LLM) or DEFAULT_EVAL_LLM


def embed_model_id() -> str:
    return _env("RAG_EMBEDDING_MODEL_ID", DEFAULT_EMBED_MODEL) or DEFAULT_EMBED_MODEL


def rerank_model_id() -> str:
    return _env("RAG_RERANKER_MODEL_ID", DEFAULT_RERANK_MODEL) or DEFAULT_RERANK_MODEL


def prompt_file() -> str:
    return _env("DEFAULT_PROMPT_FILE", DEFAULT_PROMPT_FILENAME) or DEFAULT_PROMPT_FILENAME


def company_summary() -> str:
    return _env("COMPANY_SUMMARY", DEFAULT_COMPANY_SUMMARY) or DEFAULT_COMPANY_SUMMARY


def tax_db_uri() -> str:
    return _env("TAX_DB_URI", "") or ""


_ENV_WARNINGS: list[str] = []


def env_warnings() -> list[str]:
    return list(_ENV_WARNINGS)


def _warn_env(msg: str) -> None:
    if msg not in _ENV_WARNINGS:
        _ENV_WARNINGS.append(msg)


def llm_temperature() -> float:
    raw = os.getenv("LLM_TEMPERATURE", "0.1")
    try:
        return float(raw)
    except (TypeError, ValueError):
        _warn_env(f"LLM_TEMPERATURE={raw!r} 는 실수가 아닙니다. 기본값 0.1 을 씁니다.")
        return 0.1


def llm_max_tokens() -> int:
    raw = os.getenv("LLM_MAX_TOKENS", "4096")
    try:
        val = int(raw)
        if val > 0:
            return val
        _warn_env(f"LLM_MAX_TOKENS={raw!r} 는 양수가 아닙니다. 기본값 4096 을 씁니다.")
        return 4096
    except (TypeError, ValueError):
        _warn_env(f"LLM_MAX_TOKENS={raw!r} 는 정수가 아닙니다. 기본값 4096 을 씁니다.")
        return 4096


def rag_capture() -> str:
    text = (_env("RAG_CAPTURE", "lean") or "lean").lower()
    if text in ("lean", "full", "forensic"):
        return text
    _warn_env(f"RAG_CAPTURE={text!r} 는 lean|full|forensic 이 아닙니다. 기본값 lean 을 씁니다.")
    return "lean"


def session_idle_ttl_sec() -> int:
    raw = os.getenv("SESSION_IDLE_TTL_SEC", "1800")
    try:
        return max(60, int(raw))
    except (TypeError, ValueError):
        _warn_env(f"SESSION_IDLE_TTL_SEC={raw!r} 는 정수가 아닙니다. 기본값 1800 을 씁니다.")
        return 1800


def cs_allow_ip() -> str:
    return os.environ.get("CS_ALLOW_IP", "127.0.0.1")


def require_server_env() -> str:
    """웹 서버 기동에 필수인 DEFAULT_LLM. 없으면 RuntimeError."""
    llm = default_llm()
    if not llm:
        raise RuntimeError(
            "[Fatal Error] .env 파일에 DEFAULT_LLM 환경변수가 없습니다. "
            "서버를 구동할 수 없습니다."
        )
    return llm


def require_tax_db_uri() -> str:
    uri = tax_db_uri()
    if not uri:
        raise RuntimeError(
            "[Fatal Error] TAX_DB_URI 환경변수가 없습니다. "
            "PostgreSQL 연결 문자열을 .env 에 넣으세요."
        )
    return uri
