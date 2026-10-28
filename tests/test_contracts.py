# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
from contracts import PromptsConfig, QueryRequest


def test_custom_prompts_optional():
    req = QueryRequest(query="근로소득세액공제는?")
    assert req.custom_prompts is None
    overlay = PromptsConfig(main_answer="답하라")
    req2 = QueryRequest(query="q", custom_prompts=overlay)
    assert req2.custom_prompts.main_answer == "답하라"


def test_sources_default_four_tables():
    req = QueryRequest(query="q")
    assert "tax_laws_master" in req.general_settings.sources
    assert len(req.general_settings.sources) == 4


def test_keyword_mode_default_and_valid():
    req = QueryRequest(query="q")
    assert req.general_settings.keyword_mode == "trigram"
    req2 = QueryRequest(query="q", general_settings={"keyword_mode": "kiwi"})
    assert req2.general_settings.keyword_mode == "kiwi"
