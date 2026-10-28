# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
# --- START OF FILE model_registry.py ---
"""임베딩/리랭커 허용 목록(JSON) + 모델 id별 1개 캐시 + 유휴 TTL 언로드.

taxlaw-rag(세무) 검색 공간:
    indexed=true  원본 tax_laws_master.embedding 컬럼(기본).
    그 외 카탈로그 id 는 build_compare_embedding_tax.py 비교 테이블.
클라이언트는 embed_model 하나만 고르면 되고, :func:`resolve_embed_choice` 가 두 갈래로 번역한다.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any

try:
    import torch
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from sentence_transformers import CrossEncoder
except ImportError:
    torch = None
    HuggingFaceEmbedding = None
    CrossEncoder = None

logger = logging.getLogger("my_rag")

EMBED_MODELS_FILE = "embedding_models.json"
RERANK_MODELS_FILE = "reranker_models.json"
DEFAULT_IDLE_TTL_SEC = 600

_lock = threading.Lock()
_embed_allow: dict[str, dict] = {}
_rerank_allow: dict[str, dict] = {}
_slots: dict[tuple[str, str], "_ModelSlot"] = {}
_idle_ttl_sec = DEFAULT_IDLE_TTL_SEC


class _ModelSlot:
    def __init__(self, kind: str, model_id: str, model: Any):
        self.kind = kind
        self.model_id = model_id
        self.model = model
        self.refcount = 0
        self.last_used = time.monotonic()
        self.infer_lock = threading.Lock()


def _here(*parts: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def _load_json_list(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise RuntimeError(f"[Fatal Error] 모델 목록 파일이 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"[Fatal Error] 모델 목록이 비어 있거나 배열이 아닙니다: {path}")
    out: list[dict] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        mid = str(raw.get("id") or "").strip()
        if not mid or mid in seen:
            continue
        seen.add(mid)
        item = {
            "id": mid,
            "label": str(raw.get("label") or mid),
        }
        if "indexed" in raw:
            item["indexed"] = bool(raw.get("indexed"))
        out.append(item)
    if not out:
        raise RuntimeError(f"[Fatal Error] 유효한 id 항목이 없습니다: {path}")
    return out


def init_model_registry(
    embed_path: str | None = None,
    rerank_path: str | None = None,
    idle_ttl_sec: int | None = None,
) -> None:
    global _embed_allow, _rerank_allow, _idle_ttl_sec
    embed_path = embed_path or _here(EMBED_MODELS_FILE)
    rerank_path = rerank_path or _here(RERANK_MODELS_FILE)
    embeds = _load_json_list(embed_path)
    reranks = _load_json_list(rerank_path)
    _embed_allow = {row["id"]: row for row in embeds}
    _rerank_allow = {row["id"]: row for row in reranks}
    if idle_ttl_sec is not None:
        _idle_ttl_sec = max(30, int(idle_ttl_sec))
    else:
        _idle_ttl_sec = max(30, int(os.getenv("MODEL_IDLE_TTL_SEC", str(DEFAULT_IDLE_TTL_SEC))))
    indexed = [row for row in embeds if row.get("indexed")]
    if not indexed:
        raise RuntimeError(
            f"[Fatal Error] 임베딩 목록에 indexed=true 항목이 없습니다: {embed_path}"
        )
    logger.info(
        f"[MODEL] 임베딩 {len(_embed_allow)}개 (indexed {len(indexed)}), "
        f"리랭커 {len(_rerank_allow)}개, ttl={_idle_ttl_sec}s"
    )


def embedding_catalog_payload() -> dict:
    default_id = indexed_embedding_id()
    models = []
    for row in _embed_allow.values():
        mid = row["id"]
        is_default = mid == default_id
        models.append(
            {
                "id": mid,
                "label": row.get("label") or mid,
                "indexed": bool(row.get("indexed")),
                "default": is_default,
            }
        )
    return {"models": models, "default_id": default_id}


def reranker_catalog_payload() -> dict:
    models = [
        {"id": row["id"], "label": row.get("label") or row["id"]}
        for row in _rerank_allow.values()
    ]
    return {"models": models}


def indexed_embedding_ids() -> set[str]:
    return {mid for mid, row in _embed_allow.items() if row.get("indexed")}


def indexed_embedding_id() -> str:
    """원본 tax_laws_master.embedding 을 만든 기본 임베딩. indexed=true 가 그 표시다."""
    indexed = [mid for mid, row in _embed_allow.items() if row.get("indexed")]
    if not indexed:
        raise RuntimeError("indexed=true 임베딩이 없습니다.")
    env = (os.getenv("RAG_EMBEDDING_MODEL_ID") or "").strip()
    if env in indexed:
        return env
    return indexed[0]


def _embed_not_allowed(model_id: str) -> str:
    allowed = ", ".join(_embed_allow.keys()) or "(없음)"
    return f"허용 목록에 없는 임베딩 모델입니다: {model_id}\n허용: {allowed}"


def resolve_embed_choice(
    embed_model: str | None,
    embed_variant: str | None = None,
) -> tuple[str, str | None]:
    """클라이언트가 고른 임베딩을 원본 컬럼 vs 비교 테이블로 번역한다.

    Returns:
        ``(encoder_id, variant_or_none)``. ``encoder_id`` 는 항상 indexed 기본
        모델(:func:`use_embed` 용). ``variant_or_none`` 이 있으면
        ``tax_laws_master_cmp_*`` 비교 테이블을 검색한다.

    Raises:
        ValueError: 목록에 없거나, embed_model 과 embed_variant 가 서로 다른
            비교 모델을 가리킬 때.
    """
    indexed_id = indexed_embedding_id()
    requested = (embed_model or "").strip() or indexed_id
    variant = (embed_variant or "").strip() or None

    if requested not in _embed_allow:
        raise ValueError(_embed_not_allowed(requested))

    if variant:
        if variant not in _embed_allow:
            raise ValueError(_embed_not_allowed(variant))
        if variant == indexed_id:
            if requested != indexed_id:
                raise ValueError(
                    "임베딩 선택이 충돌합니다. "
                    f"embed_model={requested} 는 비교 테이블인데 "
                    f"embed_variant={embed_variant} 는 기본(원본 컬럼)입니다."
                )
            return indexed_id, None
        if requested != indexed_id and requested != variant:
            raise ValueError(
                "임베딩 선택이 충돌합니다. "
                f"embed_model={requested}, embed_variant={embed_variant}. "
                "기본 모델과 비교 테이블을 같이 쓰거나, 같은 비교 모델만 지정하세요."
            )
        return indexed_id, variant

    if requested == indexed_id:
        return indexed_id, None
    return indexed_id, requested


def apply_embed_choice(general_settings: Any) -> str:
    """general_settings 의 embed_model / embed_variant 를 번역해 쓴다.

    Returns:
        실제로 검색하는 벡터 공간 id (비교 모델이 있으면 그 id, 없으면 기본).
    """
    encoder, variant = resolve_embed_choice(
        getattr(general_settings, "embed_model", None),
        getattr(general_settings, "embed_variant", None),
    )
    general_settings.embed_model = encoder
    if hasattr(general_settings, "embed_variant"):
        general_settings.embed_variant = variant
    return variant or encoder


def default_rerank_id() -> str:
    env = (os.getenv("RAG_RERANKER_MODEL_ID") or "").strip()
    if env and env in _rerank_allow:
        return env
    if _rerank_allow:
        return next(iter(_rerank_allow))
    return ""


def loaded_models_status() -> dict:
    indexed = indexed_embedding_id()
    embed_loaded = False
    loaded_rerank = None
    with _lock:
        for (kind, mid), slot in _slots.items():
            if slot.model is None:
                continue
            if kind == "embed" and mid == indexed:
                embed_loaded = True
            if kind == "rerank":
                loaded_rerank = mid
    rerank_id = loaded_rerank or default_rerank_id()
    return {
        "embed": {"id": indexed, "loaded": embed_loaded},
        "rerank": {"id": rerank_id, "loaded": loaded_rerank is not None},
    }


def validate_embed_id_any(model_id: str) -> tuple[bool, str]:
    """카탈로그에 있는 임베딩인지. indexed 여부는 보지 않는다(비교 모델용)."""
    text = (model_id or "").strip()
    if not text:
        return False, "임베딩 모델이 비어 있습니다."
    if text not in _embed_allow:
        allowed = ", ".join(_embed_allow.keys()) or "(없음)"
        return False, f"허용 목록에 없는 임베딩 모델입니다: {text}\n허용: {allowed}"
    return True, ""


def validate_embed_model(model_id: str) -> tuple[bool, str]:
    text = (model_id or "").strip()
    if not text:
        return False, "임베딩 모델이 비어 있습니다."
    row = _embed_allow.get(text)
    if row is None:
        allowed = ", ".join(_embed_allow.keys()) or "(없음)"
        return False, f"허용 목록에 없는 임베딩 모델입니다: {text}\n허용: {allowed}"
    if not row.get("indexed"):
        indexed = ", ".join(sorted(indexed_embedding_ids())) or "(없음)"
        return False, (
            f"이 임베딩은 DB 인덱스와 다릅니다: {text}\n"
            f"벡터 검색에는 indexed=true 모델만 사용할 수 있습니다: {indexed}"
        )
    return True, ""


def validate_rerank_model(model_id: str) -> tuple[bool, str]:
    text = (model_id or "").strip()
    if not text:
        return False, "리랭커 모델이 비어 있습니다."
    if text not in _rerank_allow:
        allowed = ", ".join(_rerank_allow.keys()) or "(없음)"
        return False, f"허용 목록에 없는 리랭커입니다: {text}\n허용: {allowed}"
    return True, ""


def _load_model(kind: str, model_id: str) -> Any:
    logger.info(f"[MODEL] {kind} 로드 시작: {model_id}")
    if kind == "embed":
        if HuggingFaceEmbedding is None:
            raise ImportError("임베딩 모델을 로드하려면 torch 및 llama-index 가 필요합니다.")
        model = HuggingFaceEmbedding(model_name=model_id)
    elif kind == "rerank":
        if CrossEncoder is None:
            raise ImportError("리랭커 모델을 로드하려면 torch 및 sentence-transformers 가 필요합니다.")
        model = CrossEncoder(model_id)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    logger.info(f"[MODEL] {kind} 로드 완료: {model_id}")
    return model


def _unload_slot(slot: _ModelSlot) -> None:
    logger.info(f"[MODEL] {slot.kind} 언로드: {slot.model_id}")
    slot.model = None
    try:
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _acquire(kind: str, model_id: str) -> _ModelSlot:
    key = (kind, model_id)
    with _lock:
        slot = _slots.get(key)
        if slot is not None and slot.model is not None:
            slot.refcount += 1
            slot.last_used = time.monotonic()
            return slot
    model = _load_model(kind, model_id)
    with _lock:
        slot = _slots.get(key)
        if slot is not None and slot.model is not None:
            slot.refcount += 1
            slot.last_used = time.monotonic()
            return slot
        slot = _ModelSlot(kind, model_id, model)
        slot.refcount = 1
        slot.last_used = time.monotonic()
        _slots[key] = slot
        return slot


def _release(kind: str, model_id: str) -> None:
    key = (kind, model_id)
    with _lock:
        slot = _slots.get(key)
        if slot is None:
            return
        slot.refcount = max(0, slot.refcount - 1)
        slot.last_used = time.monotonic()


def sweep_idle_models() -> None:
    now = time.monotonic()
    with _lock:
        stale = []
        for key, slot in _slots.items():
            if slot.refcount > 0:
                continue
            if now - slot.last_used >= _idle_ttl_sec:
                stale.append(key)
        for key in stale:
            slot = _slots.pop(key, None)
            if slot is not None:
                _unload_slot(slot)


def unload_all_models() -> None:
    """종료 시 refcount 와 무관하게 캐시된 모델을 모두 비운다."""
    with _lock:
        slots = list(_slots.values())
        _slots.clear()
    for slot in slots:
        _unload_slot(slot)


@contextmanager
def use_embed(model_id: str):
    ok, err = validate_embed_model(model_id)
    if not ok:
        raise ValueError(err)
    slot = _acquire("embed", model_id)
    try:
        yield slot
    finally:
        _release("embed", model_id)


@contextmanager
def use_embed_any(model_id: str):
    """indexed 여부와 무관한 임베딩 슬롯. 비교 테이블용 embed_variant 에 쓴다."""
    ok, err = validate_embed_id_any(model_id)
    if not ok:
        raise ValueError(err)
    slot = _acquire("embed", model_id)
    try:
        yield slot
    finally:
        _release("embed", model_id)


@contextmanager
def use_rerank(model_id: str):
    ok, err = validate_rerank_model(model_id)
    if not ok:
        raise ValueError(err)
    slot = _acquire("rerank", model_id)
    try:
        yield slot
    finally:
        _release("rerank", model_id)
