# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
from errors import is_overload, is_transient, retry_after_seconds


def test_is_transient_rate_limit_not_generate():
    assert is_transient("429 rate limit exceeded")
    assert is_transient("503 high demand")
    assert not is_transient("Failed to generate response")
    assert not is_transient("could not generate")
    assert not is_transient("정답은 제12조입니다")


def test_is_overload_narrower_than_transient():
    assert is_overload("429 rate limit")
    assert is_transient("timeout waiting for upstream")
    assert not is_overload("timeout waiting for upstream")


def test_retry_after_seconds():
    assert retry_after_seconds("retry in 56s") == 56.0
    assert retry_after_seconds("please retry after 41s") == 41.0
    assert retry_after_seconds("no hint") is None
