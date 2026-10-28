# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""골든셋 검증 validate_testset 수용 기준."""
from evaluate_rag import TESTSET_TYPES, validate_testset


def test_valid_item_passes():
    items = [
        {
            "id": "T1",
            "type": "exact",
            "query": "질문",
            "must_include": ["183일"],
            "gold_doc_id": "tax_laws_master:1",
            "gold_context": "183일",
        },
    ]
    assert validate_testset(items) == []


def test_empty_must_include_flagged():
    problems = validate_testset([
        {"id": "T2", "type": "exact", "query": "q", "must_include": [], "gold_doc_id": "tax_laws_master:1"},
    ])
    assert any("must_include" in p for p in problems)


def test_bad_type_flagged():
    problems = validate_testset([
        {"id": "T3", "type": "exsact", "query": "q", "must_include": ["a"], "gold_doc_id": "tax_laws_master:1"},
    ])
    assert any("type" in p for p in problems)


def test_absence_skips_gold_requirements():
    items = [{"id": "A1", "type": "absence", "query": "없는 조문?", "must_include": [], "gold_doc_id": ""}]
    assert validate_testset(items) == []


def test_testset_types_cover_build_testset():
    for t in ("exact", "criteria", "hybrid", "absence"):
        assert t in TESTSET_TYPES
