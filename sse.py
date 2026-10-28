# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""SSE 프레임 직렬화의 유일한 소유자.

kind 가 answer/log/system/prompt/notice 이면 data 는 {"text": ...} JSON.
kind 가 error/done 이면 data 는 구조화 객체 JSON 을 한 겹만 쓴다.
텍스트 스니핑으로 종류를 추측하지 않는다. FastAPI 를 import 하지 않는다.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

logger = logging.getLogger("my_rag")

SSE_KIND_LOG = "log"
SSE_KIND_ANSWER = "answer"
SSE_KIND_SYSTEM = "system"
SSE_KIND_PROMPT = "prompt"
SSE_KIND_NOTICE = "notice"
SSE_KIND_ERROR = "error"
SSE_KIND_DONE = "done"
SSE_KIND_EVAL = "eval"

_SSE_KINDS = frozenset({
    SSE_KIND_LOG,
    SSE_KIND_ANSWER,
    SSE_KIND_SYSTEM,
    SSE_KIND_PROMPT,
    SSE_KIND_NOTICE,
    SSE_KIND_ERROR,
    SSE_KIND_DONE,
    SSE_KIND_EVAL,
})
_STRUCTURED_KINDS = frozenset({
    SSE_KIND_ERROR,
    SSE_KIND_DONE,
    SSE_KIND_EVAL,
})


def as_sse_item(item: Any) -> tuple[str, Any]:
    if isinstance(item, tuple) and len(item) == 2:
        return str(item[0]), item[1]
    return SSE_KIND_LOG, item


def _serialize_data(kind: str, payload: Any) -> str:
    if kind in _STRUCTURED_KINDS:
        if isinstance(payload, (dict, list)):
            data = json.dumps(payload, ensure_ascii=False)
        elif isinstance(payload, str):
            data = payload
        else:
            data = json.dumps(payload, ensure_ascii=False) if payload is not None else "{}"
        if "\n" in data or "\r" in data:
            logger.error("[SSE] kind=%s data 에 개행이 있어 감쌌다", kind)
            data = json.dumps({"text": data}, ensure_ascii=False)
        return data
    text = "" if payload is None else str(payload)
    return json.dumps({"text": text}, ensure_ascii=False)


def format_sse(kind: str, payload: Any) -> str:
    """SSE 한 프레임. error/done 은 이중 인코딩하지 않는다."""
    name = kind or SSE_KIND_LOG
    if name not in _SSE_KINDS:
        logger.error("[SSE] 알 수 없는 kind=%r → log 로 강등", name)
        name = SSE_KIND_LOG
    data = _serialize_data(name, payload)
    return f"event: {name}\ndata: {data}\n\n"


def format_error(envelope: dict) -> str:
    return format_sse(SSE_KIND_ERROR, envelope)


def format_done(status: str, **fields: Any) -> str:
    body = {"status": status}
    body.update(fields)
    return format_sse(SSE_KIND_DONE, body)


async def drain_nowait(queue: asyncio.Queue | None) -> AsyncIterator[Any]:
    """큐를 비동기로 비운다. get 한 항목을 버리지 않는다."""
    while queue is not None and not queue.empty():
        try:
            msg = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        yield msg
