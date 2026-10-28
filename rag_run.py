# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
# --- START OF FILE rag_run.py ---
"""질문 1회(QaTurn / AskRun)의 실행 컨텍스트.

세션은 컨베이어다. 질문이 순서대로 들어오면 턴 객체가 하나씩 쌓인다.
lean: 질문·답·상태.
full: 설정 스냅샷 + DB 후보/최종 hit. JSON 덤프.
forensic: full + LLM messages + 가드레일 raw. JSON 덤프.
서버 --capture 는 기본값이자 상한. 클라 요청은 상한을 넘지 못한다.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

logger = logging.getLogger("my_rag")

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "runs")

CAPTURE_MODES = ("lean", "full", "forensic")
_CAPTURE_RANK = {"lean": 0, "full": 1, "forensic": 2}
_server_capture = "lean"


def normalize_capture(mode: str | None) -> str | None:
    text = (mode or "").strip().lower()
    return text if text in _CAPTURE_RANK else None


def set_server_capture(mode: str | None) -> str:
    global _server_capture
    _server_capture = normalize_capture(mode) or "lean"
    return _server_capture


def server_capture() -> str:
    return _server_capture


def dumps_run_json(capture: str) -> bool:
    return capture in ("full", "forensic")


def dumps_retrieval_docs(capture: str) -> bool:
    return capture in ("full", "forensic")


def dumps_llm_calls(capture: str) -> bool:
    return capture == "forensic"


def effective_capture(req: Any = None) -> str:
    """클라 요청과 서버 상한의 교집합. 상한을 넘는 요청은 내린다."""
    ceiling = _server_capture
    g = getattr(req, "general_settings", None)
    raw = getattr(g, "capture", None) if g is not None else None
    requested = normalize_capture(raw)
    if requested is None:
        return ceiling
    if _CAPTURE_RANK[requested] > _CAPTURE_RANK[ceiling]:
        logger.warning(
            "[CAPTURE] 요청=%s 를 서버 상한=%s 로 내립니다.",
            requested,
            ceiling,
        )
        return ceiling
    return requested


def effective_trace_prompts(req: Any = None, capture: str | None = None) -> bool:
    """forensic 은 프롬프트 트레이스를 강제한다. 그 외는 클라 플래그."""
    mode = capture if capture in _CAPTURE_RANK else effective_capture(req)
    if mode == "forensic":
        return True
    g = getattr(req, "general_settings", None)
    return bool(getattr(g, "trace_prompts", False)) if g is not None else False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_request(req: Any) -> dict[str, Any]:
    """QueryRequest(또는 유사 객체)를 JSON-safe dict 로 만든다."""
    if req is None:
        return {}
    for name in ("model_dump", "dict"):
        fn = getattr(req, name, None)
        if not callable(fn):
            continue
        try:
            data = fn(mode="json") if name == "model_dump" else fn()
        except TypeError:
            try:
                data = fn()
            except Exception as exc:
                logger.debug("[AskRun] dump_request %s 실패", name, exc_info=exc)
                continue
        except Exception as exc:
            logger.debug("[AskRun] dump_request %s 실패", name, exc_info=exc)
            continue
        if isinstance(data, dict):
            return data
    return {"query": getattr(req, "query", "") or ""}


@dataclass
class RetrievalHit:
    doc_id: str
    source: str
    rank: int
    score: float | None = None
    content: str = ""
    level: str = ""
    parent_id: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Neo4j Law 노드는 법령 원본 id 만 갖는다. 판례/심판/개정 id 를 넣으면 공회전한다.
LAW_GRAPH_TABLE = "tax_laws_master"


def diversify_tagged(tagged: list[tuple[str, tuple]], top_k: int) -> list[tuple[str, tuple]]:
    """점수순 top_k 를 유지하되, 전부 한 테이블이면 마지막 자리를 다른 테이블 최고점과 교체."""
    if top_k <= 1 or len(tagged) <= top_k:
        return tagged[:top_k]
    head = tagged[:top_k]
    tables = {t for t, _ in head}
    if len(tables) > 1:
        return head
    only = next(iter(tables))
    for item in tagged[top_k:]:
        if item[0] != only:
            return head[:-1] + [item]
    return head


def law_ids_from_hits(hits: list[RetrievalHit]) -> list:
    prefix = f"{LAW_GRAPH_TABLE}:"
    return [h.parent_id for h in hits if h.parent_id and (h.doc_id or "").startswith(prefix)]


@dataclass
class RetrievalResult:
    context_text: str = ""
    hits: list[RetrievalHit] = field(default_factory=list)
    candidates: list[RetrievalHit] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    expanded_query: str = ""
    graph_context: str = ""

    def id_rows(self) -> list[tuple[str, str, str]]:
        """평가기 호환: (id, law_name, article_number)."""
        return [(h.doc_id, h.source, h.level) for h in self.hits]


def format_context(hits: list[RetrievalHit]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"\n--- [관련 법령 {i}] ---\n[{h.level}] {h.content}\n")
    return "".join(parts)


@dataclass
class GuardrailResult:
    enabled: bool = False
    verdict: str | None = None
    raw: str | None = None


@dataclass
class RetrievalSnapshot:
    context_text: str = ""
    latency_ms: float | None = None
    error: str | None = None
    expanded_query: str | None = None
    keywords: list[str] = field(default_factory=list)
    candidate_count: int | None = None
    hit_count: int | None = None
    hits: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    graph_context: str = ""


@dataclass
class GenerationSnapshot:
    answer: str = ""
    latency_ms: float | None = None
    token_usage: dict[str, Any] | None = None


@dataclass
class EvalSnapshot:
    enabled: bool = False
    report: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float | None = None
    error: str | None = None


@dataclass
class TimingSnapshot:
    t_llm_init_ms: float | None = None
    t_retrieve_ms: float | None = None
    t_generate_ms: float | None = None
    t_eval_ms: float | None = None
    t_total_ms: float | None = None


@dataclass
class InfraSnapshot:
    slot_acquired: bool | None = None
    local_lock: bool | None = None


@dataclass
class AskRun:
    """컨베이어 위의 상자. 질문 1회 + (어떤 형태든) 답 + 그때의 컨텍스트."""

    run_id: str
    session_id: str = ""
    started_at: str = field(default_factory=_utc_now)
    ended_at: str | None = None
    status: str = "pending"
    stage: str = "init"
    capture: str = "lean"
    error_message: str | None = None
    error_where: str | None = None

    query: str = ""
    request: dict[str, Any] = field(default_factory=dict)
    resolved: dict[str, Any] = field(default_factory=dict)

    retrieval: RetrievalSnapshot = field(default_factory=RetrievalSnapshot)
    generation: GenerationSnapshot = field(default_factory=GenerationSnapshot)
    guardrails_in: GuardrailResult = field(default_factory=GuardrailResult)
    guardrails_out: GuardrailResult = field(default_factory=GuardrailResult)
    eval: EvalSnapshot = field(default_factory=EvalSnapshot)
    infra: InfraSnapshot = field(default_factory=InfraSnapshot)
    timing: TimingSnapshot = field(default_factory=TimingSnapshot)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)

    dump_path: str | None = None
    dump_failed: bool = False

    @classmethod
    def from_request(
        cls,
        req: Any,
        *,
        run_id: str | None = None,
        session_id: str = "",
    ) -> "AskRun":
        return cls(
            run_id=run_id or str(uuid4()),
            session_id=session_id or "",
            query=getattr(req, "query", "") or "",
            request=dump_request(req),
            capture=effective_capture(req),
            status="running",
            stage="validate",
        )

    def apply_retrieval(self, result: RetrievalResult) -> str:
        """검색 결과를 턴에 반영하고, LLM 에 넣을 컨텍스트 문자열을 돌려준다."""
        if isinstance(result, str):
            self.retrieval.context_text = result or ""
            return self.retrieval.context_text
        context = result.context_text or ""
        self.retrieval.context_text = context
        self.retrieval.expanded_query = result.expanded_query or None
        self.retrieval.keywords = list(result.keywords or [])
        self.retrieval.candidate_count = len(result.candidates)
        self.retrieval.hit_count = len(result.hits)
        if dumps_retrieval_docs(self.capture):
            self.retrieval.hits = [h.to_dict() for h in result.hits]
            self.retrieval.candidates = [h.to_dict() for h in result.candidates]
            self.retrieval.graph_context = result.graph_context or ""
        else:
            self.retrieval.hits = []
            self.retrieval.candidates = []
            self.retrieval.graph_context = ""
        return context

    def record_llm_call(
        self,
        stage: str,
        model: str,
        messages: list,
        output: str = "",
    ) -> None:
        if not dumps_llm_calls(self.capture):
            return
        copied = []
        for msg in messages or []:
            if isinstance(msg, dict):
                copied.append({
                    "role": str(msg.get("role") or "message"),
                    "content": str(msg.get("content") or ""),
                })
            else:
                copied.append({"role": "message", "content": str(msg)})
        self.llm_calls.append({
            "stage": (stage or "llm").strip() or "llm",
            "model": model,
            "messages": copied,
            "output": output or "",
        })

    def mark(
        self,
        *,
        stage: str | None = None,
        status: str | None = None,
        error: str | None = None,
        where: str | None = None,
    ) -> None:
        if stage is not None:
            self.stage = stage
        if status is not None:
            self.status = status
        if error is not None:
            self.error_message = error
            self.error_where = where or self.stage

    def finish(self, status: str | None = None) -> None:
        if status is not None:
            self.status = status
        if self.ended_at is None:
            self.ended_at = _utc_now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    def summary_dict(self) -> dict[str, Any]:
        """세션 JSONL 용 요약. 본문 전체는 dump 파일에만 둔다."""
        eval_metrics = [
            {
                "metric": r.get("metric"),
                "score": r.get("score"),
                "pass": r.get("pass"),
            }
            for r in (self.eval.report or [])
            if r.get("metric")
        ]
        return {
            "run_id": self.run_id,
            "status": self.status,
            "stage": self.stage,
            "error_message": self.error_message,
            "error_where": self.error_where,
            "timing": asdict(self.timing),
            "dump_path": self.dump_path,
            "context_chars": len(self.retrieval.context_text or ""),
            "answer_chars": len(self.generation.answer or ""),
            "guardrails_in": self.guardrails_in.verdict,
            "guardrails_out": self.guardrails_out.verdict,
            "guardrails_in_raw": _clip_summary_text(self.guardrails_in.raw),
            "guardrails_out_raw": _clip_summary_text(self.guardrails_out.raw),
            "capture": self.capture,
            "hit_count": self.retrieval.hit_count,
            "candidate_count": self.retrieval.candidate_count,
            "llm_call_count": len(self.llm_calls),
            "dump_failed": self.dump_failed,
            "dropped_frames": int((self.resolved or {}).get("dropped_frames") or 0),
            "ignored_fields": list((self.resolved or {}).get("ignored_fields") or []),
            "eval_enabled": self.eval.enabled,
            "eval_error": self.eval.error,
            "eval_metrics": eval_metrics,
            "expanded_query": self.retrieval.expanded_query,
            "keywords": list(self.retrieval.keywords or []),
            "partial": list((self.resolved or {}).get("partial") or []),
        }


def _clip_summary_text(text: str | None, *, limit: int = 200) -> str:
    raw = text or ""
    if len(raw) > limit:
        return raw[:limit] + "…"
    return raw


def build_done_payload(
    run: AskRun,
    ask_reason: str,
    done_reason: str | None,
    *,
    dropped_frames: int = 0,
) -> dict[str, Any]:
    """ask_reason → done 프레임 필드. 매핑은 이 함수 한 곳만.

    [2계층 상태 계약 - D1 / ADR-014]
    - ask_reason 매핑 결과(status)는 클라이언트로 전달되는 전송 계층 상태다.
    - 가드레일 거부 안내 텍스트가 정상 스트리밍된 경우 ask_reason='completed' 로 매핑되어
      done.status='completed' 가 된다 (웹 UI 에러 배너 오발생 방지).
    - 실제 내부 감사 원장(AskRun.status)은 'rejected' 로 온전히 보존된다.
    """
    status_map = {
        "cancelled": "cancelled",
        "rejected": "rejected",
        "error": "error",
    }
    status = status_map.get(ask_reason, "completed")
    reason = done_reason
    if status == "cancelled" and not reason:
        reason = "CANCELLED"
    payload: dict[str, Any] = {
        "status": status,
        "run_id": run.run_id,
        "reason": reason,
        "timing_ms": {"total": getattr(run.timing, "t_total_ms", None)},
        "dropped_frames": dropped_frames,
    }
    partial = list((run.resolved or {}).get("partial") or [])
    if partial:
        payload["partial"] = partial
    return payload


def persist_ask_run(run: AskRun) -> str | None:
    """full/forensic 일 때만 JSON 파일로 남긴다. lean 은 메모리 턴만 유지."""
    run.finish()
    if not dumps_run_json(run.capture):
        return None
    session_key = run.session_id or "_nosession"
    directory = os.path.join(RUNS_DIR, session_key)
    path = os.path.join(directory, f"{run.run_id}.json")
    tmp = path + ".tmp"
    try:
        os.makedirs(directory, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(run.to_json())
        os.replace(tmp, path)
        run.dump_path = path
        return path
    except Exception as exc:
        logger.exception("[AskRun] 덤프 실패 run_id=%s", run.run_id)
        run.dump_failed = True
        note = f"AskRun 덤프 실패: {exc}"
        if run.error_message:
            run.error_message = f"{run.error_message}; {note}"
        else:
            run.error_message = note
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return None


def slim_run_bodies(run: AskRun | None) -> None:
    """세션 메모리에서 본문을 비운다. persist 이후에 호출한다."""
    if run is None:
        return
    retr = getattr(run, "retrieval", None)
    if retr is not None:
        retr.context_text = ""
        retr.hits = []
        retr.candidates = []
        retr.graph_context = ""
    gen = getattr(run, "generation", None)
    if gen is not None:
        gen.answer = ""
    run.llm_calls = []

