# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""SSE 직렬화·큐 비우기 수용 기준 AC-01, AC-14, AC-15."""
import asyncio
import json

from sse import drain_nowait, format_sse


def test_ac01_drain_nowait_yields_all():
    async def _run():
        queue = asyncio.Queue()
        for i in range(3):
            queue.put_nowait(("log", f"m{i}"))
        items = [item async for item in drain_nowait(queue)]
        assert items == [("log", "m0"), ("log", "m1"), ("log", "m2")]
        assert queue.empty()

    asyncio.run(_run())


def test_ac14_error_not_double_wrapped():
    envelope = {"ok": False, "error": {"code": "INVALID_OPTION_VALUE", "message": "x"}}
    frame = format_sse("error", envelope)
    assert frame.startswith("event: error\n")
    data_line = frame.split("data: ", 1)[1].strip()
    payload = json.loads(data_line)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_OPTION_VALUE"
    assert "text" not in payload


def test_ac15_answer_wraps_text():
    frame = format_sse("answer", "a\nb")
    assert frame.startswith("event: answer\n")
    data_line = frame.split("data: ", 1)[1].strip()
    assert json.loads(data_line) == {"text": "a\nb"}


def test_structured_error_newline_wrapped():
    frame = format_sse("error", "line1\nline2")
    assert frame.count("\n\n") == 1
    data_line = frame.split("data: ", 1)[1].strip()
    payload = json.loads(data_line)
    assert "text" in payload
