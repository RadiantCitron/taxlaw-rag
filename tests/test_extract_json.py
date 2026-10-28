# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
import pytest

pytest.importorskip("deepeval")

from evaluator import _custom_evaluator_class


def test_extract_json_from_fenced_block():
    raw = '```json\n{"statements": [], "score": 0.5, "reason": "ok"}\n```'
    out = _custom_evaluator_class()._extract_json(raw)
    assert out.startswith("{")
    assert '"score": 0.5' in out


def test_extract_json_fallback_on_garbage():
    out = _custom_evaluator_class()._extract_json("not json at all")
    assert out.startswith("{")
    assert "파싱 실패" in out
