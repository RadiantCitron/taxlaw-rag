# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
from types import SimpleNamespace

from rag_run import effective_capture, effective_trace_prompts, set_server_capture


def test_effective_capture_degrades_to_server_ceiling():
    set_server_capture("lean")
    req = SimpleNamespace(general_settings=SimpleNamespace(capture="forensic"))
    assert effective_capture(req) == "lean"


def test_effective_capture_allows_equal_or_lower():
    set_server_capture("full")
    req = SimpleNamespace(general_settings=SimpleNamespace(capture="lean"))
    assert effective_capture(req) == "lean"
    req2 = SimpleNamespace(general_settings=SimpleNamespace(capture="full"))
    assert effective_capture(req2) == "full"


def test_effective_trace_prompts_forensic_forces_true():
    set_server_capture("forensic")
    req = SimpleNamespace(general_settings=SimpleNamespace(capture=None, trace_prompts=False))
    assert effective_trace_prompts(req) is True
