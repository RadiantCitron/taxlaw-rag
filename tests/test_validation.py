# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""오류 계약 수용 기준 AC-02~AC-13."""
from contracts import QueryRequest
from error_envelope import build_envelope
from rag_run import effective_capture, set_server_capture
from validation import FIELD_RULES, pick_error_code, schema_payload, validate_request


def _run(raw: dict):
    req = QueryRequest.model_validate(raw)
    return req, validate_request(req, raw)


def test_schema_includes_use_graphdb_and_company_summary():
    schema = schema_payload()
    paths = {r["path"] for r in schema.get("fields") or []}
    assert "general_settings.use_graphdb" in paths
    assert "general_settings.company_summary" in paths


def test_company_summary_empty_rejected():
    _, outcome = _run({"query": "q", "general_settings": {"company_summary": ""}})
    assert len(outcome.violations) == 1
    assert outcome.violations[0]["field"] == "general_settings.company_summary"
    req, outcome = _run({"query": "q", "general_settings": {"alpha": 5.0}})
    assert req.general_settings.alpha == 5.0
    assert len(outcome.violations) == 1
    v = outcome.violations[0]
    assert v["field"] == "general_settings.alpha"
    assert v["given"] == 5.0
    assert v["expected"]["max"] == 1.0
    assert v["default"] == 0.5


def test_ac03_collects_all_violations():
    _, outcome = _run({
        "query": "q",
        "general_settings": {"alpha": 5.0},
        "rerank_settings": {"top_k": 0},
    })
    fields = {v["field"] for v in outcome.violations}
    assert len(outcome.violations) == 2
    assert "general_settings.alpha" in fields
    assert "rerank_settings.top_k" in fields


def test_ac04_query_mode_typo():
    _, outcome = _run({"query": "q", "general_settings": {"query_mode": "expension"}})
    assert len(outcome.violations) == 1
    allowed = outcome.violations[0]["expected"]["allowed"]
    assert set(allowed) == {"none", "expansion", "vocab_mapping"}


def test_ac05_query_mode_case_normalized():
    req, outcome = _run({"query": "q", "general_settings": {"query_mode": "EXPANSION"}})
    assert outcome.violations == []
    assert req.general_settings.query_mode == "expansion"


def test_ac06_capture_unknown():
    _, outcome = _run({"query": "q", "general_settings": {"capture": "verbose"}})
    assert len(outcome.violations) == 1
    assert outcome.violations[0]["field"] == "general_settings.capture"


def test_ac07_capture_ceiling_notice():
    prev = set_server_capture("lean")
    try:
        req, outcome = _run({"query": "q", "general_settings": {"capture": "forensic"}})
        assert outcome.violations == []
        assert len(outcome.notices) == 1
        assert "lean" in outcome.notices[0]
        assert effective_capture(req) == "lean"
    finally:
        set_server_capture(prev)


def test_ac08_unknown_field_ignored_not_rejected():
    _, outcome = _run({"query": "q", "general_settings": {"zzz": 1}})
    assert outcome.violations == []
    assert outcome.ignored_fields == ["general_settings.zzz"]


def test_ac09_vocab_mapping_kiwi_incompatible():
    _, outcome = _run({
        "query": "q",
        "general_settings": {
            "query_mode": "vocab_mapping",
            "keyword_mode": "kiwi",
        },
    })
    assert pick_error_code(outcome.violations) == "INCOMPATIBLE_OPTIONS"
    fields = {v["field"] for v in outcome.violations}
    assert "general_settings.query_mode" in fields
    assert "general_settings.keyword_mode" in fields


def test_ac10_top_k_negative():
    _, outcome = _run({"query": "q", "rerank_settings": {"top_k": -3}})
    assert len(outcome.violations) == 1
    v = outcome.violations[0]
    assert v["field"] == "rerank_settings.top_k"
    assert v["expected"]["min"] == 1


def test_ac11_metrics_typo():
    _, outcome = _run({
        "query": "q",
        "eval_settings": {"advanced": {"metrics": "answer_rel"}},
    })
    assert len(outcome.violations) == 1
    allowed = outcome.violations[0]["expected"]["allowed"]
    assert set(allowed) == {"all", "answer_relevancy", "context_precision"}


def test_ac12_envelope_keys():
    env = build_envelope(
        "INVALID_OPTION_VALUE",
        "msg",
        violations=[{"field": "general_settings.alpha"}],
        ignored=["general_settings.zzz"],
        request_id="r1",
    )
    err = env["error"]
    for key in ("code", "message", "violations", "ignored_fields", "request_id"):
        assert key in err
    assert env["ok"] is False
    from error_envelope import unwrap_envelope
    wrapped = {"detail": env}
    assert unwrap_envelope(wrapped) is env
    assert unwrap_envelope(env) is env


def test_ac13_schema_has_31_fields():
    payload = schema_payload()
    assert len(FIELD_RULES) == 31
    assert len(payload["fields"]) == 31
    paths = {item["path"] for item in payload["fields"]}
    assert "general_settings.sources" in paths
    assert "general_settings.base_limit" in paths
    assert "query" in paths


def test_base_limit_out_of_range():
    _, outcome = _run({"query": "q", "general_settings": {"base_limit": 101}})
    assert len(outcome.violations) == 1
    assert outcome.violations[0]["field"] == "general_settings.base_limit"
