# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
from unittest.mock import MagicMock

import pytest

import llm_handler
from llm_handler import LLMController, QueryOptimizer, is_local_model
from rag_run import AskRun, build_done_payload


def test_is_local_model():
    assert is_local_model("local:qwen.gguf") is True
    assert is_local_model("qwen.gguf") is True
    assert is_local_model("openrouter:google/gemini-3.5-flash-lite") is False
    assert is_local_model("") is False
    assert is_local_model("local") is False


def test_get_openrouter_catalog_loads_local_file_first(monkeypatch):
    """카탈로그 초기화 시 _OPENROUTER_CATALOG is None 이면 로컬 JSON 파일을 우선 로드한다."""
    monkeypatch.setattr(llm_handler, "_OPENROUTER_CATALOG", None)
    catalog = llm_handler.get_openrouter_catalog()
    assert isinstance(catalog, list)
    assert len(catalog) >= 7
    ids = [m["id"] for m in catalog]
    assert "openrouter:google/gemini-3.5-flash-lite" in ids


def test_query_optimizer_expand_query_fallback_on_empty():
    """질문 확장이 빈 문자열이나 공백을 반환하면 원본 user_query 로 안전 폴백한다."""
    mock_llm = MagicMock()
    mock_llm.ask_raw.return_value = ""
    opt = QueryOptimizer(mock_llm)

    res = opt.expand_query("부가가치세 신고 기한", "dummy_template")
    assert res == "부가가치세 신고 기한"

    mock_llm.ask_raw.return_value = "   "
    res2 = opt.expand_query("양도소득세 세율", "dummy_template")
    assert res2 == "양도소득세 세율"


def test_llm_controller_ask_raises_on_error():
    """LLMController.ask() 중 예외 발생 시 빈 문자열 대신 예외를 상위로 re-raise 한다."""
    ctrl = LLMController.__new__(LLMController)
    ctrl.provider = "openrouter"
    ctrl.model_name = "google/gemini-3.5-flash-lite"
    ctrl.client = MagicMock()
    ctrl._maybe_record_llm_call = MagicMock()
    ctrl.client.chat.completions.create.side_effect = RuntimeError("OpenRouter 503 error")

    with pytest.raises(RuntimeError, match="OpenRouter 503 error"):
        ctrl.ask("query", "company", "context", "prompt")


def test_two_layer_state_contract_audit_rejected_vs_stream_completed():
    """원장 계층(Audit: run.status='rejected')과 스트림 계층(Transport: done.status='completed') 2계층 계약 검증."""
    run = AskRun(run_id="run-test-1")
    run.mark(status="rejected", error="가드레일 거부", where="guard_in")
    assert run.status == "rejected"
    assert run.error_message == "가드레일 거부"

    # 사용자 안내 메시지가 스트리밍 완료되면 전송 계층은 completed 로 정상 마감
    payload_completed = build_done_payload(run, "completed", None)
    assert payload_completed["status"] == "completed"
    assert run.status == "rejected"  # 원장은 'rejected' 불변 유지

    # 전송 계층 에러/거부인 경우
    payload_rejected = build_done_payload(run, "rejected", "GUARD_REJECTED")
    assert payload_rejected["status"] == "rejected"
    assert payload_rejected["reason"] == "GUARD_REJECTED"
