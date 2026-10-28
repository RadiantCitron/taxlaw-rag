# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""통신 계약 AC-4~AC-8 수용 기준."""
import pytest

from error_envelope import extract_envelope, http_error
from sse import format_sse


def test_ac04_error_newline_single_frame_boundary():
    frame = format_sse("error", "a\nb")
    assert frame.count("\n\n") == 1


def test_ac05_unknown_kind_demoted_to_log():
    frame = format_sse("anser", "x")
    assert frame.startswith("event: log\n")


def test_ac06_gate_full_not_http_error():
    with pytest.raises(AssertionError, match="SSE 전용"):
        http_error("GATE_FULL", "gate full")


def test_ac07_validation_envelope_extractable():
    from error_envelope import build_envelope

    env = build_envelope("INVALID_BODY", "bad")
    wrapped = {"detail": env}
    assert extract_envelope(wrapped) is not None


def test_ac08_done_rejected_exit_nonzero():
    from error_envelope import exit_code_from_done

    assert exit_code_from_done({"status": "rejected", "reason": "GATE_FULL"}) != 0
