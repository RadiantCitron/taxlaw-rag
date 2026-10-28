# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
from rag_run import RetrievalHit, diversify_tagged, law_ids_from_hits


def _row(score: float):
    return ("id", "name", "art", "text", score)


def test_diversify_tagged_swaps_last_when_homogeneous():
    tagged = [
        ("tax_laws_master", _row(0.9)),
        ("tax_laws_master", _row(0.8)),
        ("tax_laws_master", _row(0.7)),
        ("tax_precedents_master", _row(0.4)),
    ]
    out = diversify_tagged(tagged, 3)
    assert [t for t, _ in out] == [
        "tax_laws_master",
        "tax_laws_master",
        "tax_precedents_master",
    ]


def test_diversify_tagged_keeps_mixed_head():
    tagged = [
        ("tax_laws_master", _row(0.9)),
        ("tax_precedents_master", _row(0.8)),
        ("tax_laws_master", _row(0.7)),
    ]
    out = diversify_tagged(tagged, 3)
    assert [t for t, _ in out] == [
        "tax_laws_master",
        "tax_precedents_master",
        "tax_laws_master",
    ]


def test_law_ids_from_hits_only_laws_master():
    hits = [
        RetrievalHit(doc_id="tax_laws_master:L1", source="법령", rank=1, parent_id="L1"),
        RetrievalHit(doc_id="tax_precedents_master:P1", source="판례", rank=2, parent_id="P1"),
        RetrievalHit(doc_id="tax_tribunal_master:T1", source="심판", rank=3, parent_id="T1"),
    ]
    assert law_ids_from_hits(hits) == ["L1"]
