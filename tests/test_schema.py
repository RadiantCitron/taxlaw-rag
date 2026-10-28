# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
import pytest

from schema import (
    ALLOWED_SOURCE_TABLES,
    VECTOR_DIM,
    cmp_table,
    normalize_sources,
    require_source,
)


def test_normalize_sources_default_is_all_four():
    assert normalize_sources(None) == list(ALLOWED_SOURCE_TABLES)
    assert len(ALLOWED_SOURCE_TABLES) == 4


def test_normalize_sources_rejects_unknown():
    with pytest.raises(ValueError, match="허용되지 않은"):
        normalize_sources(["labor_law"])


def test_cmp_table_producer_consumer_contract():
    name = cmp_table("tax_laws_master", "BAAI/bge-m3")
    assert name == "tax_laws_master_cmp_baai_bge_m3"
    assert cmp_table("tax_precedents_master", "intfloat/multilingual-e5-large").startswith(
        "tax_precedents_master_cmp_"
    )


def test_cmp_table_rejects_injection():
    with pytest.raises(ValueError):
        cmp_table("tax_laws_master; DROP TABLE tax_laws_master", "x")


def test_require_source_and_vector_dim():
    assert require_source("tax_amendments_2026") == "tax_amendments_2026"
    assert VECTOR_DIM == 768
