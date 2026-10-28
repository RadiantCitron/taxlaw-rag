# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""임베딩 슬롯 — 비교 모델(use_embed_any) 수용 기준."""
import pytest

pytest.importorskip("torch")

from model_registry import init_model_registry, use_embed_any


def setup_module():
    init_model_registry()


def test_use_embed_any_acquires_slot_with_infer_lock(monkeypatch):
    sentinel = object()
    monkeypatch.setattr("model_registry._load_model", lambda kind, mid: sentinel)

    with use_embed_any("BAAI/bge-m3") as slot:
        assert slot.model is sentinel
        assert slot.infer_lock is not None
        assert slot.model_id == "BAAI/bge-m3"


def test_use_embed_any_rejects_unknown(monkeypatch):
    monkeypatch.setattr("model_registry._load_model", lambda kind, mid: object())
    with pytest.raises(ValueError, match="허용 목록"):
        with use_embed_any("not-a-real-embed"):
            pass
