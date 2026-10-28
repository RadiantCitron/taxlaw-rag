# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
from search_constants import SEARCH_KEYWORD_CAP, keyword_bind_values, keywords_from_text


def test_keywords_from_text_caps_and_skips():
    text = "원인 따른 적용 있는 대한 그리고 또는 후 소득세 공제 한도 가산 추가"
    kws = keywords_from_text(text, "fallback")
    assert "원인" not in kws
    assert "소득세" in kws
    assert len(kws) <= SEARCH_KEYWORD_CAP


def test_keywords_from_text_fallback():
    assert keywords_from_text("", "법령질문") == ["법령질문"]


def test_keyword_bind_values_splits_regex_and_similarity():
    pattern, text = keyword_bind_values(["소득세", "공제(한도)"])
    assert pattern == r"소득세|공제\(한도\)"
    assert text == "소득세 공제(한도)"
    assert "|" not in text


def test_keyword_bind_values_empty():
    assert keyword_bind_values([]) == ("", "")
    assert keyword_bind_values(["  ", ""]) == ("", "")

