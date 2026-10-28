# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""하네스 --self-test 를 pytest 로 통합한다 (H1)."""
import importlib
import pytest


@pytest.mark.parametrize(
    "mod",
    [
        "evaluate_rag",
        "evaluate_faithfulness",
        "build_faithfulness_testset",
    ],
)
def test_harness_selftest_passes(mod: str):
    m = importlib.import_module(mod)
    m._self_test()


def test_build_testset_selftest():
    pytest.importorskip("psycopg2")
    import build_testset
    build_testset._self_test()
