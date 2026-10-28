# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
# 웹 서버 없이 오케스트레이터를 직접 호출하는 단건 DeepEval 스모크 스크립트.
# 대량 회귀는 evaluate_rag.py / evaluate_faithfulness.py 를 쓰라.

import asyncio
import csv
import os

from dotenv import load_dotenv

load_dotenv()

from contracts import PromptsConfig, QueryRequest
from workflow_orchestrator import RAGState, get_orchestrator

GOLDEN_SET = [
    {
        "query": "거주자 판정 기준 일수는?",
        "expected": "소득세법상 183일 이상 거주 등 요건을 본다.",
    },
]


async def run_batch_test():
    print("[BATCH TEST] 단건 DeepEval 스모크 시작...")

    from llm_handler import exit_if_invalid_model
    model = os.getenv("DEFAULT_LLM", "openrouter:google/gemini-3.5-flash-lite")
    eval_model = os.getenv("DEFAULT_EVAL_LLM", "openrouter:google/gemini-3.5-flash-lite")
    exit_if_invalid_model(model, label="생성 모델")
    exit_if_invalid_model(eval_model, label="채점 모델")
    try:
        orch = get_orchestrator(model)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    prompts = PromptsConfig()

    results = []
    for idx, item in enumerate(GOLDEN_SET, 1):
        print(f"\n--- [Test {idx}/{len(GOLDEN_SET)}] Query: {item['query']} ---")
        req = QueryRequest(query=item["query"])
        req.eval_settings.enabled = True
        req.eval_settings.eval_model = eval_model
        req.general_settings.use_guardrails = False

        state = RAGState(req, prompts)
        state = await orch.node_retrieve(state)
        state = await orch.node_generate(state)
        state = await orch.node_evaluate(state)

        score = state.eval_report[0]["score"] if state.eval_report else 0.0
        reason = state.eval_report[0]["reason"] if state.eval_report else "평가 실패"
        results.append({
            "query": item["query"],
            "generated_answer": state.generated_answer,
            "score": score,
            "reason": reason,
        })
        print(f"점수: {score} | 사유: {reason}")

    with open("eval_report.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["query", "generated_answer", "score", "reason"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n[BATCH TEST 완료] eval_report.csv 생성")


if __name__ == "__main__":
    asyncio.run(run_batch_test())
