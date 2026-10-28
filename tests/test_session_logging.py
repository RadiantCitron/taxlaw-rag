# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""세션 로깅 A1–A4 수용 기준."""
import asyncio
import json
import logging

from rag_run import AskRun, GuardrailResult, _clip_summary_text
from session_manager import (
    ClientSession,
    SessionLogRouter,
    bind_session,
    reset_session,
)
from sse import format_sse, SSE_KIND_EVAL


def _log_record(level: int, msg: str) -> logging.LogRecord:
    name = logging.getLevelName(level)
    return logging.LogRecord(
        name="my_rag",
        level=level,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_ac03_info_logged_not_streamed_at_error_level():
    async def _run():
        session = ClientSession("127.0.0.1")
        session.stream_min_level = logging.ERROR
        session.stream_logs = True
        session.try_begin_ask()
        token = bind_session(session)
        written = []
        emitted = []
        session.write_log_line = lambda level, msg: written.append((level, msg))
        session.emit_to_client = lambda msg, kind="log": emitted.append((kind, msg))
        router = SessionLogRouter()
        router.emit(_log_record(logging.INFO, "info line"))
        router.emit(_log_record(logging.ERROR, "err line"))
        assert written == [("INFO", "info line"), ("ERROR", "err line")]
        assert len(emitted) == 1
        assert emitted[0][0] == "log"
        reset_session(token)

    asyncio.run(_run())


def test_summary_guardrail_raw_clipped():
    run = AskRun(run_id="r1")
    run.guardrails_in = GuardrailResult(raw="x" * 250)
    summary = run.summary_dict()
    assert len(summary["guardrails_in_raw"]) == 201
    assert summary["guardrails_in_raw"].endswith("…")


def test_summary_eval_metrics_structured():
    run = AskRun(run_id="r1")
    run.eval.enabled = True
    run.eval.report = [
        {"metric": "answer_relevancy", "score": 0.9, "pass": True, "reason": "long"},
    ]
    metrics = run.summary_dict()["eval_metrics"]
    assert metrics == [{"metric": "answer_relevancy", "score": 0.9, "pass": True}]
    assert "reason" not in metrics[0]


def test_eval_sse_kind_structured():
    payload = {"report": [{"metric": "m", "score": 1, "pass": True}], "error": None}
    frame = format_sse(SSE_KIND_EVAL, payload)
    data = json.loads(frame.split("data: ", 1)[1].strip())
    assert data["report"][0]["metric"] == "m"
    assert "text" not in data


def test_clip_summary_text():
    assert _clip_summary_text("abc", limit=5) == "abc"
    assert _clip_summary_text("abcdef", limit=5) == "abcde…"
