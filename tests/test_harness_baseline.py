# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""동결 픽스처로 채점기 집계 기준선을 고정한다 (H4)."""
import json
from pathlib import Path

from evaluate_rag import aggregate, score_item

_FIX = Path(__file__).resolve().parent / "fixtures"


class _Args:
    mode = "nodes"
    model = "x"
    kw_threshold = 0.5
    require_context = True
    id_check = True
    top_k = 3
    judge = False


def test_scorer_baseline_is_stable():
    items = json.loads((_FIX / "mini_testset.json").read_text(encoding="utf-8"))
    answers = json.loads((_FIX / "mini_answers.json").read_text(encoding="utf-8"))
    expected = json.loads((_FIX / "mini_expected.json").read_text(encoding="utf-8"))
    args = _Args()
    rows = []
    for it in items:
        a = answers[it["id"]]
        rows.append(
            score_item(
                it,
                a["context"],
                a["answer"],
                a["ids"],
                args,
                None,
                None,
            )
        )
    got = aggregate(rows, args)
    assert got["overall_pass_rate"] == expected["overall_pass_rate"]
    assert got["retrieval"]["hit@3"] == expected["retrieval"]["hit@3"]
    assert got["retrieval"]["hit@10"] == expected["retrieval"]["hit@10"]
