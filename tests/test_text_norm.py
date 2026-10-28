# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
from text_norm import contains, norm, NUM_UNIT_RE


def test_norm_strips_spaces():
    assert norm("6억 원") == "6억원"


def test_contains_substring():
    assert contains("183일 규정", "183일")
    assert not contains("연차", "휴가")


def test_num_unit_pattern():
    assert NUM_UNIT_RE.search("183일 이상")
