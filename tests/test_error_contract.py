# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""오류 계약 EC 수용 기준."""
from error_envelope import (
    build_envelope,
    errors_codes_payload,
    exit_code_from_done,
    exit_code_from_envelope,
    STREAM_ONLY_CODES,
)
from rag_run import AskRun, build_done_payload


def test_build_envelope_exit_hint_and_severity():
    env = build_envelope("INVALID_BODY", "bad")
    err = env["error"]
    assert err["exit_hint"] == 2
    assert err["severity"] == "reject"


def test_exit_code_from_envelope_uses_exit_hint():
    env = build_envelope("SESSION_BUSY", "busy")
    assert exit_code_from_envelope(env) == 3


def test_stream_only_codes_marked_in_registry():
    rows = {r["code"]: r for r in errors_codes_payload()["codes"]}
    for code in STREAM_ONLY_CODES:
        assert rows[code]["stream_only"]


def test_internal_error_in_registry():
    rows = {r["code"]: r for r in errors_codes_payload()["codes"]}
    assert rows["INTERNAL_ERROR"]["http"] == 500
    assert rows["INTERNAL_ERROR"]["exit"] == 4


def test_done_completed_exit_zero():
    assert exit_code_from_done({"status": "completed"}) == 0


def test_finalize_running_maps_to_error_payload():
    run = AskRun(run_id="r1", status="running")
    payload = build_done_payload(run, "error", "PIPELINE_ERROR")
    assert payload["status"] == "error"
    assert payload["reason"] == "PIPELINE_ERROR"


def test_summary_dict_includes_eval_error():
    run = AskRun(run_id="r1")
    run.eval.error = "eval blew up"
    run.resolved["partial"] = ["eval"]
    summary = run.summary_dict()
    assert summary["eval_error"] == "eval blew up"
    assert summary["partial"] == ["eval"]
    assert summary["eval_metrics"] == []
