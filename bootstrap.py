# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""서버 lifespan 과 evaluate_rag 가 공유하는 핵심 부팅.

진입점마다 카탈로그/프롬프트/DB 를 다르게 준비하면 is_allowed_model_id() 가
파일 화이트리스트와 네트워크 fallback 으로 갈린다. 그 차이를 여기서 없앤다.

이 모듈은 api_server 를 import 하지 않는다.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from contracts import PromptsConfig
from settings import DEFAULT_PROMPT_FILENAME, OPENROUTER_MODELS_FILE, env_warnings, prompt_file

logger = logging.getLogger("my_rag")

_BOOT: dict[str, Any] = {
    "started_at": None,
    "prompt_file": None,
    "login_mode": False,
    "contracts": {"ok": True, "problems": []},
    "env_warnings": [],
}


def boot_status() -> dict[str, Any]:
    return {
        "started_at": _BOOT.get("started_at"),
        "prompt_file": _BOOT.get("prompt_file"),
        "login_mode": bool(_BOOT.get("login_mode")),
        "contracts": dict(_BOOT.get("contracts") or {"ok": True, "problems": []}),
        "env_warnings": list(_BOOT.get("env_warnings") or []),
    }


def set_boot_login_mode(enabled: bool) -> None:
    _BOOT["login_mode"] = bool(enabled)


def resolve_prompt_file_path(cli_path: str | None = None) -> str:
    """프롬프트 JSON 경로. CLI > .env > 기본 한국어 파일.

    상대 경로는 작업 디렉터리보다 이 파일(프로젝트 루트) 옆을 우선한다.
    """
    raw = (cli_path or "").strip() or (prompt_file() or "").strip()
    if not raw:
        raw = DEFAULT_PROMPT_FILENAME
    if os.path.isabs(raw):
        return raw
    here = os.path.dirname(os.path.abspath(__file__))
    beside_app = os.path.join(here, raw)
    if os.path.exists(beside_app):
        return beside_app
    cwd_candidate = os.path.abspath(raw)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return beside_app


def load_prompts_from_file(cli_path: str | None = None) -> PromptsConfig:
    """프롬프트 JSON을 읽고, 없거나 main_answer 가 비면 즉시 실패한다."""
    path = resolve_prompt_file_path(cli_path)
    if not path or not os.path.exists(path):
        raise RuntimeError(
            f"[Fatal Error] 프롬프트 파일이 지정되지 않았거나 찾을 수 없습니다. (경로: {path})"
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            file_data = json.load(f)
        prompts = PromptsConfig(**file_data)
    except Exception as e:
        raise RuntimeError(f"[Fatal Error] 프롬프트 파일 로드 실패: {e}") from e
    if not (prompts.main_answer or "").strip():
        raise RuntimeError(
            f"[Fatal Error] 프롬프트 파일에 main_answer 가 비어 있습니다. (경로: {path})"
        )
    return prompts


def merge_prompts(base: PromptsConfig | None, overlay: PromptsConfig | None) -> PromptsConfig:
    """요청 custom_prompts 의 비어 있지 않은 필드만 서버 프롬프트 위에 덮는다."""
    if overlay is None:
        if base is None:
            raise RuntimeError("[Fatal Error] 프롬프트가 로드되지 않았습니다.")
        return base
    if base is None:
        if not (overlay.main_answer or "").strip():
            raise RuntimeError("[Fatal Error] 프롬프트에 main_answer 가 없습니다.")
        return overlay
    data = base.model_dump()
    for key, val in overlay.model_dump().items():
        if isinstance(val, str) and val.strip():
            data[key] = val
    merged = PromptsConfig(**data)
    if not (merged.main_answer or "").strip():
        raise RuntimeError("[Fatal Error] 프롬프트에 main_answer 가 없습니다.")
    return merged


def load_prompts(cli_path: str | None = None, *, fail_fast: bool = True) -> PromptsConfig:
    if fail_fast:
        return load_prompts_from_file(cli_path)
    try:
        return load_prompts_from_file(cli_path)
    except Exception as e:
        logger.warning("[BOOT] 프롬프트 로드 실패(%s) → 빈 프롬프트", e)
        return PromptsConfig()


def load_openrouter_catalog_file() -> None:
    from llm_handler import load_openrouter_catalog_from_file

    catalog_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), OPENROUTER_MODELS_FILE
    )
    if not os.path.exists(catalog_path):
        raise RuntimeError(
            f"[Fatal Error] OpenRouter 모델 목록 파일이 없습니다: {catalog_path}"
        )
    load_openrouter_catalog_from_file(catalog_path)


def verify_contracts(conn: Any) -> list[str]:
    """기동 시 검색 전제를 한 번 확인한다. 확장 누락은 치명, 빈 테이블은 경고."""
    from schema import ALLOWED_SOURCE_TABLES, VECTOR_DIM

    problems: list[str] = []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm')"
        )
        found = {row[0] for row in cur.fetchall()}
        missing = {"vector", "pg_trgm"} - found
        if missing:
            raise RuntimeError(
                f"[Fatal Error] PostgreSQL 확장이 없습니다: {sorted(missing)}. "
                f"database/init.sql 을 실행하세요."
            )
        cur.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = current_schema()
              AND table_name = ANY(%s)
            """,
            (list(ALLOWED_SOURCE_TABLES),),
        )
        present = {row[0] for row in cur.fetchall()}
        absent = [t for t in ALLOWED_SOURCE_TABLES if t not in present]
        if absent:
            msg = f"아직 없는 검색 테이블: {', '.join(absent)}"
            logger.warning("[BOOT] %s", msg)
            problems.append(msg)
        for table in present:
            cur.execute(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = current_schema()
                  AND c.relname = %s
                  AND a.attname = 'embedding'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                """,
                (table,),
            )
            row = cur.fetchone()
            if not row:
                msg = f"{table}.embedding 컬럼이 없습니다"
                logger.warning("[BOOT] %s", msg)
                problems.append(msg)
                continue
            typ = (row[0] or "").lower()
            if "vector" not in typ:
                raise RuntimeError(
                    f"[Fatal Error] {table}.embedding 타입이 예상과 다릅니다: {row[0]}"
                )
            if f"({VECTOR_DIM})" not in typ and "vector" in typ:
                msg = (
                    f"{table}.embedding={row[0]} "
                    f"(원본 기대는 vector({VECTOR_DIM}); 비교 테이블은 다를 수 있음)"
                )
                logger.warning("[BOOT] %s", msg)
                problems.append(msg)
    return problems


def boot_core(*, prompt_cli: str | None = None, fail_fast_prompts: bool = True) -> PromptsConfig:
    """Postgres 풀 + 모델 레지스트리 + OpenRouter 파일 카탈로그 + 프롬프트.

    Neo4j / 세션 / 게이트는 서버 전용이라 lifespan 에 남긴다.
    """
    from database.postgres_repo import PostgresRepo
    from model_registry import init_model_registry
    from settings import require_tax_db_uri

    require_tax_db_uri()
    PostgresRepo.init_pool()
    init_model_registry()
    load_openrouter_catalog_file()
    prompts = load_prompts(prompt_cli, fail_fast=fail_fast_prompts)
    repo = PostgresRepo()
    try:
        problems = verify_contracts(repo.conn)
    finally:
        repo.close()
    warns = env_warnings()
    for msg in warns:
        logger.warning("[SETTINGS] %s", msg)
    _BOOT["started_at"] = datetime.now(timezone.utc).isoformat()
    _BOOT["prompt_file"] = resolve_prompt_file_path(prompt_cli)
    _BOOT["contracts"] = {"ok": True, "problems": problems}
    _BOOT["env_warnings"] = warns
    return prompts
