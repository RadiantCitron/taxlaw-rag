# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""워크플로 answer-before-eval 스트리밍 계약."""
import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
# retriever 는 torch 등 무거운 ML 모듈을 import 하므로 단위 테스트용 Mock 제공.
# [T-1 버그 수정]: evaluator 는 sys.modules 에 주입하지 않아 test_presence_sse.py 오염을 방지한다!
sys.modules.setdefault("retriever", MagicMock())

from contracts import PromptsConfig, QueryRequest
from rag_run import AskRun
from session_manager import ClientSession, bind_session, reset_session
from sse import SSE_KIND_ANSWER, SSE_KIND_EVAL
from workflow_orchestrator import (
    OUTPUT_DISCLAIMER,
    RAGOrchestrator,
    RAGState,
    _publish_answer,
    _publish_eval_report,
)


@pytest.mark.asyncio
async def test_publish_answer_sets_sse_sent_flag():
    emitted = []
    session = ClientSession("127.0.0.1")
    session.try_begin_ask()
    token = bind_session(session)
    session.emit_to_client = lambda msg, kind="log": emitted.append((kind, msg))
    run = AskRun(run_id="r1")

    _publish_answer(session, "hello", run, trace_prompts=True)

    kinds = [k for k, _ in emitted]
    assert kinds[0] == "log"
    assert SSE_KIND_ANSWER in kinds
    assert run.resolved.get("answer_sse_sent") is True
    reset_session(token)


@pytest.mark.asyncio
async def test_run_pipeline_emits_answer_before_eval():
    emitted = []
    session = ClientSession("127.0.0.1")
    session.try_begin_ask()
    token = bind_session(session)
    session.emit_to_client = lambda msg, kind="log": emitted.append((kind, msg))

    req = QueryRequest(
        query="q",
        general_settings={"use_guardrails": False},
        eval_settings={"enabled": True, "eval_model": "openrouter:test"},
    )
    run = AskRun.from_request(req, session_id=session.session_id)
    prompts = PromptsConfig()

    orch = RAGOrchestrator.__new__(RAGOrchestrator)
    orch.llm = MagicMock()
    orch.optimizer = MagicMock()

    async def _guard_in(state: RAGState) -> RAGState:
        return state

    async def _retrieve(state: RAGState) -> RAGState:
        state.retrieved_context = "ctx"
        return state

    async def _generate(state: RAGState) -> RAGState:
        state.generated_answer = "answer text"
        if state.run is not None:
            state.run.generation.answer = "answer text"
        return state

    async def _guard_out(state: RAGState) -> RAGState:
        return state

    async def _evaluate(state: RAGState) -> RAGState:
        if state.run is not None:
            state.run.eval.report = [
                {"metric": "m", "score": 1.0, "pass": True, "reason": "ok"},
            ]
        sess = session
        _publish_eval_report(sess, state.req, state.run)
        return state

    orch.node_input_guardrail = _guard_in
    orch.node_retrieve = _retrieve
    orch.node_generate = _generate
    orch.node_output_guardrail = _guard_out
    orch.node_evaluate = _evaluate

    result = await orch.run_pipeline(req, prompts, run=run)

    kinds = [k for k, _ in emitted]
    assert SSE_KIND_ANSWER in kinds
    assert SSE_KIND_EVAL in kinds
    assert kinds.index(SSE_KIND_ANSWER) < kinds.index(SSE_KIND_EVAL)
    assert run.resolved.get("answer_sse_sent") is True
    assert result == f"__FINAL_ANSWER__:answer text\n\n{OUTPUT_DISCLAIMER}"
    reset_session(token)


@pytest.mark.asyncio
async def test_node_evaluate_success_no_unbound_local_error():
    emitted = []
    session = ClientSession("127.0.0.1")
    session.try_begin_ask()
    token = bind_session(session)
    session.emit_to_client = lambda msg, kind="log": emitted.append((kind, msg))

    req = QueryRequest(
        query="q",
        general_settings={"use_guardrails": False},
        eval_settings={"enabled": True, "eval_model": "openrouter:test"},
    )
    run = AskRun.from_request(req, session_id=session.session_id)
    state = RAGState(req, PromptsConfig(), run=run)
    state.generated_answer = "ans"
    state.retrieved_context = "ctx"

    mock_report = [{"metric": "m", "score": 0.9, "pass": True, "reason": "fine"}]
    orch = RAGOrchestrator.__new__(RAGOrchestrator)
    orch.llm = MagicMock()
    orch.optimizer = MagicMock()

    with patch("workflow_orchestrator.RAGEvaluator"), patch(
        "workflow_orchestrator.run_in_session_executor",
        new_callable=AsyncMock,
        return_value=mock_report,
    ):
        result = await orch.node_evaluate(state)

    assert result.eval_report == mock_report
    assert SSE_KIND_EVAL in [k for k, _ in emitted]
    assert run.eval.error is None
    reset_session(token)


@pytest.mark.asyncio
async def test_node_evaluate_failure_keeps_partial_and_publishes_eval():
    emitted = []
    session = ClientSession("127.0.0.1")
    session.try_begin_ask()
    token = bind_session(session)
    session.emit_to_client = lambda msg, kind="log": emitted.append((kind, msg))

    req = QueryRequest(
        query="q",
        general_settings={"use_guardrails": False},
        eval_settings={"enabled": True, "eval_model": "openrouter:test"},
    )
    run = AskRun.from_request(req, session_id=session.session_id)
    run.resolved["answer_sse_sent"] = True
    state = RAGState(req, PromptsConfig(), run=run)
    state.generated_answer = "ans"
    state.retrieved_context = "ctx"

    orch = RAGOrchestrator.__new__(RAGOrchestrator)
    orch.llm = MagicMock()
    orch.optimizer = MagicMock()

    with patch("workflow_orchestrator.RAGEvaluator"), patch(
        "workflow_orchestrator.run_in_session_executor",
        new_callable=AsyncMock,
        side_effect=RuntimeError("eval blew up"),
    ):
        result = await orch.node_evaluate(state)

    assert result is state
    assert run.eval.error == "eval blew up"
    assert run.resolved.get("partial") == ["eval"]
    assert SSE_KIND_EVAL in [k for k, _ in emitted]
    reset_session(token)
