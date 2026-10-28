# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
import logging
import asyncio
import json
import argparse
import os
import sys
import time

# Windows 콘솔 기본 코드페이지(cp949)는 이모지/체크마크 등 일부 유니코드 문자를
# 인코딩하지 못한다. DeepEval(rich 콘솔 출력)이 평가 완료 후 그런 문자를 찍다가
# UnicodeEncodeError로 죽어 이미 계산된 평가 결과를 통째로 날리는 문제가 있어,
# 프로세스 시작 시점에 표준출력/에러를 UTF-8로 강제한다.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
try:
    import uvicorn
except ImportError:
    uvicorn = None
from workflow_orchestrator import get_orchestrator

from database.postgres_repo import PostgresRepo
from database.neo4j_repo import Neo4jRepo

from local_llm_gate import (
    init_gate,
    acquire_local_llm_lock,
    release_local_llm_lock,
    acquire_ask_slot,
    release_ask_slot,
    get_local_llm_status,
    get_gate_status,
    get_quota_status,
    check_and_consume_quota,
)
from llm_handler import (
    is_allowed_model_id,
    is_local_model,
    models_catalog_payload,
    format_allowed_models_help,
)
from model_registry import (
    apply_embed_choice,
    embedding_catalog_payload,
    indexed_embedding_id,
    loaded_models_status,
    reranker_catalog_payload,
    sweep_idle_models,
    unload_all_models,
    validate_embed_id_any,
    validate_rerank_model,
)
from bootstrap import (
    boot_core,
    boot_status,
    load_prompts_from_file,
    merge_prompts,
    resolve_prompt_file_path,
    set_boot_login_mode,
)
from contracts import (
    EvalAdvancedSettings,
    EvalSettings,
    GeneralSettings,
    PromptsConfig,
    QueryRequest,
    RerankAdvancedSettings,
    RerankSettings,
)
from error_envelope import (
    PipelineFault,
    build_envelope,
    errors_codes_payload,
    http_error,
    unwrap_envelope,
    violation,
)
from rag_run import AskRun, build_done_payload, persist_ask_run, server_capture, set_server_capture, slim_run_bodies
from session_manager import (
    SessionCancelled,
    SessionCapacityError,
    begin_shutdown,
    bind_session,
    create_session,
    drain_in_flight,
    get_session,
    init_sessions,
    install_log_router,
    is_shutting_down,
    mark_presence_closed,
    reset_session,
    run_in_session_executor,
    status_payload,
    sweep_idle_sessions,
)
from session_auth import (
    gate,
    init as init_session_auth,
    login_mode_enabled,
    router as admin_router,
)
from sse import as_sse_item, drain_nowait, format_done, format_error, format_sse
from validation import (
    pick_error_code,
    pydantic_violation,
    schema_payload,
    summarize_violations,
    validate_request,
)
from settings import (
    DEFAULT_PROMPT_FILENAME,
    PROMPT_FILE_EN,
    rag_capture,
    require_server_env,
    session_idle_ttl_sec,
)

load_dotenv()
ENV_DEFAULT_LLM = require_server_env()
PROMPT_FILE_KO = DEFAULT_PROMPT_FILENAME

GLOBAL_PROMPTS = None

# =====================================================================
# 2. 글로벌 로깅 설정
# =====================================================================
logger = logging.getLogger("my_rag")
logger.setLevel(logging.DEBUG)
logger.propagate = False

# =====================================================================
# 3. 서버 수명 주기 (Lifespan) 관리 - 여기서 딱 한 번만 실행됨
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global GLOBAL_PROMPTS
    install_log_router()
    
    parser = argparse.ArgumentParser(description="FastAPI RAG Server")
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help=(
            f"프롬프트 JSON 경로. 미지정 시 DEFAULT_PROMPT_FILE 또는 "
            f"{DEFAULT_PROMPT_FILENAME} (영어: {PROMPT_FILE_EN})"
        ),
    )
    parser.add_argument(
        "--print-models",
        action="store_true",
        help="기동 시 GET /llm/models 와 같은 LLM 목록을 콘솔에 출력",
    )
    parser.add_argument(
        "--max-concurrent-user",
        type=int,
        default=4,
        help="동시에 처리할 /ask 상한 (기본 4)",
    )
    parser.add_argument(
        "--max-waiting",
        type=int,
        default=5,
        help="상한 초과 시 대기 인원 상한. 넘으면 입구컷 (기본 5)",
    )
    parser.add_argument(
        "--max-connected",
        type=int,
        default=10,
        help="동시 접속(세션) 정원 (기본 10)",
    )
    parser.add_argument(
        "--capture",
        type=str,
        choices=["lean", "full", "forensic"],
        default=rag_capture(),
        help="턴 캡처 모드 (기본 lean, 환경변수 RAG_CAPTURE). 서버 상한. full 이상은 JSON 덤프, forensic 은 LLM 프롬프트까지",
    )
    parser.add_argument(
        "--login-mode",
        action="store_true",
        help="수업 로그인 모드. /ask 에 X-User-Id 가 필요하고 명단 아이디만 통과",
    )
    args, _unknown = parser.parse_known_args()

    try:
        GLOBAL_PROMPTS = boot_core(prompt_cli=args.prompt_file, fail_fast_prompts=True)
    except Exception as e:
        print(f"[Fatal Error] 핵심 부팅 실패: {e}", flush=True)
        raise
    loaded_path = resolve_prompt_file_path(args.prompt_file)
    logger.info(f"[PROMPT] 프롬프트 파일 로드 완료: {loaded_path}")
    print(f"[PROMPT] 프롬프트 파일 로드 완료: {loaded_path}", flush=True)

    try:
        Neo4jRepo.init_driver()
        if Neo4jRepo.driver is None:
            print("[WARN] Neo4j 미연결. use_graphdb 는 빈 보충만 반환합니다.", flush=True)
    except Exception as e:
        print(f"[WARN] Neo4j 초기화 실패(그래프 미사용이면 무시): {e}", flush=True)
    try:
        init_sessions(max_connected=args.max_connected)
        capture_mode = set_server_capture(args.capture)
        init_session_auth(login_mode=args.login_mode)
        set_boot_login_mode(args.login_mode)
        init_gate(
            max_concurrent_user=args.max_concurrent_user,
            max_waiting=args.max_waiting,
        )
    except Exception as e:
        print(f"[Fatal Error] 세션/게이트 초기화 실패: {e}", flush=True)
        raise

    print(
        f"[GATE] max_connected={args.max_connected} "
        f"max_concurrent_user={args.max_concurrent_user} "
        f"max_waiting={args.max_waiting} capture={capture_mode} "
        f"login_mode={str(args.login_mode).lower()}",
        flush=True,
    )

    if args.print_models:
        print("\n" + format_allowed_models_help(), flush=True)

    session_idle_ttl = session_idle_ttl_sec()

    async def _idle_sweep_loop():
        while True:
            await asyncio.sleep(60)
            try:
                sweep_idle_models()
            except Exception:
                logger.exception("[MODEL] TTL sweep 실패")
            try:
                n = sweep_idle_sessions(session_idle_ttl)
                if n:
                    logger.info("[SESSION] idle sweep closed %s session(s)", n)
            except Exception:
                logger.exception("[SESSION] idle sweep 실패")

    ttl_task = asyncio.create_task(_idle_sweep_loop())
    yield
    grace = float(os.getenv("SHUTDOWN_GRACE_SEC", "30"))
    logger.info("[SHUTDOWN] 시작 (grace=%ss)", grace)
    begin_shutdown()
    remaining = await drain_in_flight(grace)
    if remaining:
        logger.warning("[SHUTDOWN] %s in-flight 요청이 남은 채 종료", remaining)
    unload_all_models()
    ttl_task.cancel()
    try:
        await ttl_task
    except asyncio.CancelledError:
        pass
    PostgresRepo.close_pool()
    Neo4jRepo.close_driver()
    logger.info("[SHUTDOWN] 완료")

# FastAPI 앱 생성 시 lifespan 등록
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*", "X-Session-Id", "X-User-Id", "Content-Type"],
)
app.include_router(admin_router)

# QueryRequest 등 스키마는 contracts.py 가 소유. 여기서 re-export 한다.


def _system_sse_chunks(session, body: str) -> list[str]:
    session.record_client_text(body, event="system")
    return [format_sse("system", body)]


def _error_sse(session, env: dict) -> str:
    session.write_event(
        "error",
        code=(env.get("error") or {}).get("code"),
        message=(env.get("error") or {}).get("message"),
    )
    return format_error(env)


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _require_session(request: Request):
    session_id = (request.headers.get("X-Session-Id") or "").strip()
    session = get_session(session_id)
    if session is None:
        raise http_error("MISSING_SESSION", "세션이 없습니다. 먼저 GET /session/connect 로 접속하세요.")
    return session


def _model_checkers():
    return {
        "llm": is_allowed_model_id,
        "embed": validate_embed_id_any,
        "rerank": validate_rerank_model,
    }


def _model_samples():
    llm_ids = [m.get("id") for m in (models_catalog_payload().get("models") or []) if m.get("id")]
    embed_ids = [m.get("id") for m in (embedding_catalog_payload().get("models") or []) if m.get("id")]
    rerank_ids = [m.get("id") for m in (reranker_catalog_payload().get("models") or []) if m.get("id")]
    return {
        "llm": {
            "source": "GET /llm/models",
            "allowed_count": len(llm_ids),
            "allowed_sample": llm_ids[:10],
            "rule": "openrouter:<slug> 또는 local:<파일명.gguf>",
        },
        "embed": {
            "source": "GET /embed/models",
            "allowed": embed_ids,
            "rule": "GET /embed/models 의 id",
        },
        "rerank": {
            "source": "GET /rerank/models",
            "allowed": rerank_ids,
            "rule": "GET /rerank/models 의 id",
        },
    }


def _raise_option_errors(outcome, *, session=None, request_id=None):
    code = pick_error_code(outcome.violations)
    n = len(outcome.violations)
    message = f"요청 옵션 {n}건이 허용되지 않는 값입니다."
    if code == "UNKNOWN_MODEL":
        message = "허용 목록에 없는 모델입니다."
    elif code == "INCOMPATIBLE_OPTIONS":
        message = "함께 쓸 수 없는 옵션 조합입니다."
    sid = getattr(session, "session_id", None) if session is not None else None
    rid = request_id or getattr(session, "request_id", None)
    logger.warning(
        "[VALIDATE] 거부 code=%s session=%s violations=%s %s",
        code, sid, n, summarize_violations(outcome.violations),
    )
    if session is not None:
        session.write_event(
            "validate_rejected",
            code=code,
            fields=[v.get("field") for v in outcome.violations],
        )
    raise http_error(
        code, message,
        violations=outcome.violations,
        ignored=outcome.ignored_fields,
        request_id=rid,
        session_id=sid,
    )


async def _read_json_body(request: Request) -> dict:
    try:
        raw = await request.json()
    except Exception:
        raise http_error("INVALID_BODY", "요청 본문이 JSON 이 아닙니다.")
    if not isinstance(raw, dict):
        raise http_error("INVALID_BODY", "요청 본문이 객체가 아닙니다.")
    return raw


def _parse_query_request(raw: dict) -> QueryRequest:
    try:
        return QueryRequest.model_validate(raw)
    except Exception as exc:
        if isinstance(exc, ValidationError):
            violations = [pydantic_violation(e) for e in exc.errors()]
            raise http_error(
                "INVALID_BODY",
                "요청 본문 형식이 올바르지 않습니다.",
                violations=violations,
            ) from exc
        raise http_error("INVALID_BODY", f"요청 본문을 해석할 수 없습니다: {exc}") from exc


async def _prepare_ask(request: Request, *, occupy: bool):
    """세션 확인 → bind → gate → parse → validate → embed → prompts. occupy 면 try_begin_ask."""
    if is_shutting_down():
        raise http_error("SERVER_SHUTTING_DOWN", "서버 종료 중입니다.", status=503)
    raw = await _read_json_body(request)
    session = _require_session(request)
    token = bind_session(session)
    try:
        gate(request, session)
        req = _parse_query_request(raw)
        outcome = validate_request(
            req, raw, model_checkers=_model_checkers(), model_samples=_model_samples(),
        )
        if outcome.ignored_fields:
            logger.info(
                "[VALIDATE] 모르는 옵션 %d개 무시: %s",
                len(outcome.ignored_fields),
                ", ".join(outcome.ignored_fields),
            )
            session.write_event("options_ignored", fields=outcome.ignored_fields)
        if outcome.violations:
            _raise_option_errors(outcome, session=session)
        try:
            embed_used = apply_embed_choice(req.general_settings)
        except ValueError as exc:
            raise http_error("INCOMPATIBLE_OPTIONS", str(exc), session_id=session.session_id) from exc
        try:
            active_prompts = merge_prompts(GLOBAL_PROMPTS, req.custom_prompts)
        except RuntimeError as exc:
            raise http_error("PROMPT_INVALID", str(exc), session_id=session.session_id) from exc
        if occupy:
            if not session.try_begin_ask():
                raise http_error(
                    "SESSION_BUSY",
                    "이 세션은 이미 질문을 처리 중입니다.",
                    session_id=session.session_id,
                    status=409,
                )
            try:
                check_and_consume_quota(request, session=session)
            except Exception:
                session.end_ask("rejected")
                raise
        return req, session, active_prompts, embed_used, outcome, token
    except Exception:
        reset_session(token)
        raise


async def _pump_until(
    task: asyncio.Task,
    request: Request,
    queue: asyncio.Queue,
    state: dict,
):
    if queue is None:
        return
    while not task.done() or not queue.empty():
        if await request.is_disconnected():
            state["disconnected"] = True
            break
        try:
            msg = await asyncio.wait_for(queue.get(), timeout=0.15)
            kind, text = as_sse_item(msg)
            yield format_sse(kind, text)
        except asyncio.TimeoutError:
            continue


async def _acquire_gate_with_pump(
    task: asyncio.Task,
    request: Request,
    out_queue: asyncio.Queue,
    session,
    res: dict,
):
    """게이트/세마포어 태스크를 대기하며 SSE 큐를 펌핑하고 연결 해제 시 안전하게 정리한다."""
    pump_state: dict = {"disconnected": False}
    async for frame in _pump_until(task, request, out_queue, pump_state):
        yield frame
    if pump_state.get("disconnected"):
        res["disconnected"] = True
        session.request_cancel()
        res["ask_reason"] = "cancelled"
        if not task.done():
            task.cancel()
        try:
            res["acquired"] = bool(await task)
        except (asyncio.CancelledError, Exception):
            res["acquired"] = False
    else:
        try:
            res["acquired"] = bool(await task)
        except (asyncio.CancelledError, Exception):
            res["acquired"] = False


def _finalize_ask_run(
    run: AskRun,
    session,
    ask_reason: str,
    slot_acquired: bool,
    lock_acquired: bool,
) -> None:
    """AskRun 을 닫고 세션 로그에 남긴다. JSON 덤프는 capture=full 만.

    [2계층 상태 계약 - D1 / ADR-014]
    - 원장 계층(Audit): run.status 는 가드레일 거부 시 'rejected', 오류 시 'error', 취소 시 'cancelled'.
    - 스트림 계층(Transport): 클라이언트 SSE done.status 는 가드레일 안내가 정상 전송되었을 때
      'completed' 로 종료하여 웹 UI 중복 에러 배너 및 CLI 비정상 종료를 방지한다.
    """
    run.infra.slot_acquired = slot_acquired
    run.infra.local_lock = lock_acquired
    if run.status in ("pending", "running"):
        mapped = {
            "cancelled": "cancelled",
            "rejected": "rejected",
            "error": "error",
        }.get(ask_reason, "completed")
        run.mark(status=mapped)
    if run.status in ("pending", "running"):
        logger.error(
            "[ASK] finalize: run still running ask_reason=%s run_id=%s",
            ask_reason,
            run.run_id,
        )
        run.mark(status="error", error="pipeline incomplete", where="finalize")
    persist_ask_run(run)
    slim_run_bodies(run)
    session.write_event("ask_run", **run.summary_dict())


class _AskFinished(Exception):
    """스트림 본문을 끝낸다. 바깥 래퍼가 done 프레임을 보낸다."""


async def _ask_stream_events(
    req: QueryRequest,
    request: Request,
    session,
    active_prompts,
    embed_used: str,
    outcome,
    ctx: dict,
):
    level_mapping = {1: logging.ERROR, 3: logging.INFO, 5: logging.DEBUG}
    session.stream_min_level = level_mapping.get(req.general_settings.log_level, logging.INFO)
    session.stream_logs = bool(getattr(req.general_settings, "stream_logs", True))

    token = bind_session(session)
    ctx["token"] = token
    client_disconnected = False
    slot_acquired = False
    lock_acquired = False
    rag_task = None
    ask_reason = "error"
    done_reason = None
    out_queue = session.out_queue
    run = AskRun.from_request(
        req,
        run_id=session.request_id or "",
        session_id=session.session_id,
    )
    ctx["run"] = run
    session.current_run = run
    session.remember_turn(run)
    embed_requested = (req.general_settings.embed_model or "").strip()
    run.resolved = {
        "embed_requested": embed_requested,
        "embed_used": embed_used,
        "llm": req.general_settings.llm_model,
        "rerank": (
            req.rerank_settings.advanced.model if req.rerank_settings.enabled else None
        ),
        "ignored_fields": list(outcome.ignored_fields or []),
    }

    async def drain_queue():
        async for msg in drain_nowait(out_queue):
            yield format_sse(*as_sse_item(msg))

    try:
        if outcome.ignored_fields:
            yield format_sse("notice", "무시된 옵션: " + ", ".join(outcome.ignored_fields))
        for note in outcome.notices or []:
            yield format_sse("notice", note)

        logger.info(f"[EMBED] 검색 공간: {embed_used}")
        logger.info(f"[SEARCH] 테이블: {req.general_settings.sources}")
        try:
            def _check_tables():
                repo = PostgresRepo()
                try:
                    repo.ensure_search_tables(
                        req.general_settings.sources,
                        req.general_settings.embed_variant,
                    )
                finally:
                    repo.close()

            await run_in_session_executor(_check_tables)
        except ValueError as e:
            logger.warning(f"[EMBED] {e}")
            run.mark(status="rejected", error=str(e), where="embed_table")
            env = build_envelope(
                "RESOURCE_UNAVAILABLE",
                str(e),
                request_id=session.request_id,
                session_id=session.session_id,
                terminal=True,
            )
            yield _error_sse(session, env)
            ask_reason = "rejected"
            done_reason = "RESOURCE_UNAVAILABLE"
            raise _AskFinished()

        session.write_event(
            "ask_start",
            query=req.query[:200],
            llm=req.general_settings.llm_model,
            embed=embed_used,
            rerank=(
                req.rerank_settings.advanced.model if req.rerank_settings.enabled else None
            ),
            run_id=run.run_id,
            capture=run.capture,
            stream_logs=session.stream_logs,
        )

        gate_res: dict = {"acquired": False, "disconnected": False, "ask_reason": None}
        acquire_task = asyncio.create_task(acquire_ask_slot(request, out_queue))
        async for frame in _acquire_gate_with_pump(acquire_task, request, out_queue, session, gate_res):
            yield frame
        slot_acquired = gate_res["acquired"]
        if gate_res["disconnected"]:
            client_disconnected = True
            ask_reason = gate_res["ask_reason"]

        if not client_disconnected and not slot_acquired:
            async for item in drain_queue():
                yield item
            run.mark(status="rejected", error="ask slot full", where="gate")
            ask_reason = "rejected"
            done_reason = "GATE_FULL"
            raise _AskFinished()

        if not client_disconnected:
            try:
                orchestrator = await run_in_session_executor(
                    get_orchestrator, req.general_settings.llm_model
                )
            except SessionCancelled:
                raise
            except (ValueError, RuntimeError) as e:
                logger.error(f"[LLM] 모델 로드 실패: {e}")
                run.mark(status="rejected", error=str(e), where="llm_init")
                env = build_envelope(
                    "UNKNOWN_MODEL",
                    str(e),
                    request_id=session.request_id,
                    session_id=session.session_id,
                    terminal=True,
                )
                yield _error_sse(session, env)
                ask_reason = "rejected"
                done_reason = "UNKNOWN_MODEL"
                raise _AskFinished()

            needs_local = is_local_model(req.general_settings.llm_model)
            if req.eval_settings.enabled and is_local_model(req.eval_settings.eval_model):
                needs_local = True
            if needs_local:
                lock_res: dict = {"acquired": False, "disconnected": False, "ask_reason": None}
                lock_task = asyncio.create_task(acquire_local_llm_lock(request, out_queue))
                async for frame in _acquire_gate_with_pump(lock_task, request, out_queue, session, lock_res):
                    yield frame
                lock_acquired = lock_res["acquired"]
                if lock_res["disconnected"]:
                    client_disconnected = True
                    ask_reason = lock_res["ask_reason"]

                if not client_disconnected and not lock_acquired:
                    async for item in drain_queue():
                        yield item
                    run.mark(status="rejected", error="local llm lock full", where="gate")
                    ask_reason = "rejected"
                    done_reason = "GATE_FULL"
                    raise _AskFinished()

            async def consume_pipeline():
                try:
                    return await orchestrator.run_pipeline(req, active_prompts, run=run)
                except SessionCancelled:
                    logger.info("[ASK] 파이프라인 취소")
                    return None
                except PipelineFault:
                    raise
                except Exception as e:
                    logger.exception("[ASK] 파이프라인 예외")
                    raise PipelineFault(
                        "PIPELINE_ERROR",
                        "시스템 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
                    ) from e

            rag_task = asyncio.create_task(consume_pipeline())
            ctx["rag_task"] = rag_task
            pipe_state: dict = {"disconnected": False}
            async for frame in _pump_until(rag_task, request, out_queue, pipe_state):
                yield frame
            if pipe_state.get("disconnected"):
                client_disconnected = True
                session.request_cancel()
                ask_reason = "cancelled"
            else:
                for _ in range(10):
                    await asyncio.sleep(0.03)
                    async for item in drain_queue():
                        yield item
                final_answer = None
                if rag_task.done():
                    try:
                        final_answer = rag_task.result()
                    except (asyncio.CancelledError, SessionCancelled):
                        final_answer = None
                    except PipelineFault as fault:
                        env = build_envelope(
                            fault.code,
                            fault.message,
                            request_id=session.request_id,
                            session_id=session.session_id,
                            terminal=True,
                            retry_after_sec=fault.retry_after_sec,
                        )
                        yield _error_sse(session, env)
                        ask_reason = "error"
                        done_reason = fault.code
                        final_answer = None
                    except Exception as e:
                        logger.exception("[ASK] 파이프라인 예외: %s", e)
                        env = build_envelope(
                            "PIPELINE_ERROR",
                            "시스템 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
                            request_id=session.request_id,
                            session_id=session.session_id,
                            terminal=True,
                        )
                        yield _error_sse(session, env)
                        ask_reason = "error"
                        done_reason = "PIPELINE_ERROR"
                        final_answer = None
                if final_answer:
                    if isinstance(final_answer, str) and final_answer.startswith("__FINAL_ANSWER__:"):
                        if not run.resolved.get("answer_sse_sent"):
                            final_answer = final_answer.replace("__FINAL_ANSWER__:", "")
                            session.write_event("answer", message=final_answer)
                            if await request.is_disconnected():
                                client_disconnected = True
                                session.request_cancel()
                                ask_reason = "cancelled"
                            else:
                                yield format_sse("answer", final_answer)
                    else:
                        for chunk in _system_sse_chunks(session, str(final_answer)):
                            yield chunk
                if done_reason is None and ask_reason != "cancelled":
                    ask_reason = (
                        "cancelled" if session.cancel_event.is_set() else "completed"
                    )

    except _AskFinished:
        pass
    except asyncio.CancelledError:
        session.request_cancel()
        ask_reason = "cancelled"
        client_disconnected = True
        raise
    except SessionCancelled:
        ask_reason = "cancelled"
        client_disconnected = True
    finally:
        if rag_task is not None:
            session.request_cancel()
            try:
                wait_timeout = 600 if lock_acquired else 30
                await asyncio.wait_for(asyncio.shield(rag_task), timeout=wait_timeout)
            except asyncio.TimeoutError:
                if lock_acquired:
                    logger.warning("[ASK] rag_task 대기 타임아웃 lock_acquired")
            except Exception:
                logger.exception("[ASK] rag_task 예외")
                if run.status in ("pending", "running"):
                    run.mark(status="error", error="pipeline task failed", where="task")
                if ask_reason not in ("cancelled", "rejected", "error"):
                    ask_reason = "error"
                    done_reason = done_reason or "PIPELINE_ERROR"
            if rag_task.done() and not rag_task.cancelled():
                exc = rag_task.exception()
                if exc is not None and not isinstance(exc, (SessionCancelled, asyncio.CancelledError)):
                    if isinstance(exc, PipelineFault):
                        if ask_reason not in ("error", "rejected"):
                            ask_reason = "error"
                            done_reason = exc.code
                    else:
                        logger.exception("[ASK] rag_task 미처리 예외", exc_info=exc)
                        if run.status in ("pending", "running"):
                            run.mark(status="error", error=str(exc), where="task")
                        ask_reason = "error"
                        done_reason = done_reason or "PIPELINE_ERROR"
            if not rag_task.done():
                rag_task.cancel()
                try:
                    await rag_task
                except (asyncio.CancelledError, Exception):
                    pass
        if lock_acquired:
            release_local_llm_lock()
        if slot_acquired:
            release_ask_slot()
        run.resolved["dropped_frames"] = int(getattr(session, "dropped_frames", 0) or 0)
        _finalize_ask_run(run, session, ask_reason, slot_acquired, lock_acquired)
        session.end_ask(ask_reason)
        reset_session(token)
        ctx["ask_reason"] = ask_reason
        ctx["done_reason"] = done_reason
        ctx["dropped_frames"] = int(getattr(session, "dropped_frames", 0) or 0)


async def stream_generator(
    req: QueryRequest,
    request: Request,
    session,
    active_prompts,
    embed_used: str,
    outcome,
):
    ctx: dict = {}
    async for frame in _ask_stream_events(
        req, request, session, active_prompts, embed_used, outcome, ctx,
    ):
        yield frame
    run = ctx.get("run")
    if run is not None:
        payload = build_done_payload(
            run,
            ctx.get("ask_reason", "error"),
            ctx.get("done_reason"),
            dropped_frames=int(ctx.get("dropped_frames") or 0),
        )
        yield format_done(payload["status"], **{
            k: v for k, v in payload.items() if k != "status"
        })


def monitor_status_payload() -> dict:
    sess = status_payload()
    gate = get_gate_status()
    models = loaded_models_status()
    return {
        "ok": True,
        "connected": {
            "current": sess["connected"],
            "max": sess["max_connected"],
        },
        "processing": {
            "current": sess["processing"],
            "max": gate["max_concurrent_user"],
        },
        "waiting": {
            "current": gate["ask_waiting"],
            "max": gate["max_waiting"],
        },
        "local_llm": {
            "busy": gate["local_busy"],
            "waiting": gate["local_waiting"],
        },
        "embed": models["embed"],
        "rerank": models["rerank"],
    }


@app.get("/quota")
async def client_quota_status(request: Request):
    """현재 접속 클라이언트의 일일 질문 쿼터 및 15분 쿨다운 잔여 상태 조회"""
    return get_quota_status(request)


@app.get("/status")
async def server_status():
    """모니터링용 서버 상태. RAG 로그/답변 SSE 와 분리한다."""
    return monitor_status_payload()


@app.get("/local-llm/status")
async def local_llm_status():
    """로컬 GGUF 락 상태. 종합 조회는 GET /status."""
    return get_local_llm_status()


@app.get("/llm/models")
async def llm_models():
    """클라이언트가 고를 수 있는 LLM 목록과 로컬 파일 규칙."""
    return models_catalog_payload()


@app.get("/embed/models")
async def embed_models():
    return embedding_catalog_payload()


@app.get("/rerank/models")
async def rerank_models():
    return reranker_catalog_payload()


@app.get("/session/status")
async def session_status():
    return status_payload()


async def presence_heartbeat_loop(
    request: Request,
    session,
    *,
    poll_sec: float = 1.0,
    ping_sec: float = 15.0,
):
    """disconnect 는 poll_sec 마다 보고, ping/touch 는 ping_sec 주기로 보낸다.

    sleep(ping_sec) 한 방에 두면 종료 감지가 최대 ping_sec 만큼 지연된다.
    """
    last_ping_at = time.monotonic()
    while True:
        if await request.is_disconnected():
            return
        await asyncio.sleep(poll_sec)
        now = time.monotonic()
        if now - last_ping_at >= ping_sec:
            last_ping_at = now
            session.touch()
            ping = json.dumps({"event": "ping", "session_id": session.session_id})
            yield f"event: session\ndata: {ping}\n\n"


@app.get("/session/connect")
async def session_connect(request: Request):
    if is_shutting_down():
        raise http_error("SERVER_SHUTTING_DOWN", "서버 종료 중입니다.", status=503)
    host = request.client.host if request.client else ""
    try:
        session = create_session(host)
    except SessionCapacityError as e:
        raise http_error(
            "SESSION_CAPACITY",
            str(e),
            status=503,
        ) from e

    async def heartbeat():
        try:
            payload = json.dumps(
                {"event": "connected", "session_id": session.session_id},
                ensure_ascii=False,
            )
            yield f"event: session\ndata: {payload}\n\n"
            async for frame in presence_heartbeat_loop(request, session):
                yield frame
        finally:
            mark_presence_closed(session.session_id)

    return StreamingResponse(
        heartbeat(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.post("/ask")
async def ask(request: Request):
    req, session, active_prompts, embed_used, outcome, token = await _prepare_ask(
        request, occupy=True,
    )
    reset_session(token)
    return StreamingResponse(
        stream_generator(req, request, session, active_prompts, embed_used, outcome),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.post("/ask/validate")
async def ask_validate(request: Request):
    req, session, _prompts, embed_used, outcome, token = await _prepare_ask(
        request, occupy=False,
    )
    reset_session(token)
    return {
        "ok": True,
        "resolved": {
            "embed_used": embed_used,
            "query_mode": req.general_settings.query_mode,
            "keyword_mode": req.general_settings.keyword_mode,
            "sources": req.general_settings.sources,
        },
        "ignored_fields": outcome.ignored_fields,
        "notices": outcome.notices,
        "session_id": session.session_id,
    }


@app.get("/health")
async def health():
    boot = boot_status()
    models = loaded_models_status()
    llm_n = len((models_catalog_payload().get("models") or []))
    embed_n = len((embedding_catalog_payload().get("models") or []))
    rerank_n = len((reranker_catalog_payload().get("models") or []))
    contracts = boot.get("contracts") or {"ok": True, "problems": []}
    return {
        "ok": bool(contracts.get("ok")),
        "boot": {
            "started_at": boot.get("started_at"),
            "prompt_file": boot.get("prompt_file"),
            "capture_ceiling": server_capture(),
            "login_mode": login_mode_enabled(),
            "env_warnings": boot.get("env_warnings") or [],
        },
        "contracts": contracts,
        "db": {"pool": "ready"},
        "models": {"llm_catalog": llm_n, "embed": embed_n, "rerank": rerank_n, **models},
    }


@app.get("/")
async def index_page():
    """로컬 데모 UI. index.html 을 같은 포트에서 연다."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    return FileResponse(path, media_type="text/html; charset=utf-8")


@app.get("/DEMO_DISCLAIMER.txt")
async def demo_disclaimer():
    """데모 면책 및 이용 안내 원문."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DEMO_DISCLAIMER.txt")
    if not os.path.isfile(path):
        return JSONResponse({"ok": False, "message": "DEMO_DISCLAIMER.txt not found"}, status_code=404)
    return FileResponse(path, media_type="text/plain; charset=utf-8")


@app.get("/options/schema")
async def options_schema():
    return schema_payload()


@app.get("/errors/codes")
async def errors_codes():
    return errors_codes_payload()


@app.exception_handler(RequestValidationError)
async def _on_validation_error(request: Request, exc: RequestValidationError):
    violations = [pydantic_violation(e) for e in exc.errors()]
    env = build_envelope(
        "INVALID_BODY",
        "요청 본문 형식이 올바르지 않습니다.",
        violations=violations,
    )
    return JSONResponse(status_code=400, content={"detail": env})


@app.exception_handler(Exception)
async def _on_unhandled(request: Request, exc: Exception):
    if isinstance(exc, StarletteHTTPException):
        raise exc
    logger.exception("[HTTP] 미처리 예외 path=%s", request.url.path)
    env = build_envelope("INTERNAL_ERROR", "서버 내부 오류입니다.")
    return JSONResponse(status_code=500, content={"detail": env})


if __name__ == "__main__":
    grace = max(1, int(os.getenv("SHUTDOWN_GRACE_SEC", "30")))
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8000,
        workers=1,
        timeout_graceful_shutdown=grace,
    )
