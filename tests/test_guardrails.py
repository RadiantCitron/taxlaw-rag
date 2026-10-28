# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
from rag_core import RAGGuardrails


def test_normalize_verdict_pass_and_block():
    g = RAGGuardrails(llm_controller=None, prompts=None)
    assert g._normalize_verdict("PASS") == "PASS"
    assert g._normalize_verdict("pass\nextra") == "PASS"
    assert g._normalize_verdict("BLOCK") == "BLOCK"
    assert g._normalize_verdict("NOT PASS") == "BLOCK"
    assert g._normalize_verdict("") == "BLOCK"
    assert g._normalize_verdict("maybe") == "BLOCK"
