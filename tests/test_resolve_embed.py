# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
import pytest

pytest.importorskip("torch")

from model_registry import init_model_registry, resolve_embed_choice


def setup_module():
    init_model_registry()


def test_resolve_embed_indexed_default():
    encoder, variant = resolve_embed_choice("jhgan/ko-sroberta-multitask")
    assert encoder == "jhgan/ko-sroberta-multitask"
    assert variant is None


def test_resolve_embed_compare_model():
    encoder, variant = resolve_embed_choice("BAAI/bge-m3")
    assert encoder == "jhgan/ko-sroberta-multitask"
    assert variant == "BAAI/bge-m3"


def test_resolve_embed_unknown_raises():
    with pytest.raises(ValueError):
        resolve_embed_choice("not-a-real-model")
