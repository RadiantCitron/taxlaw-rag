#!/usr/bin/env python
# -*- coding: utf-8 -*-
# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""세무 RAG 평가 하네스.

================================================================================
0. 이 파일을 읽는 법 (학습 가이드)
================================================================================
build_testset.py 가 만든 테스트셋(JSON)을 로드해, 네 실제 RAG 파이프라인을
문항별로 돌리고 '검색'과 '답변'을 동시에 채점한다. 웹서버 없이 터미널에서 실행.

코드는 아래 7개 섹션으로 나뉘며, 데이터가 그 순서대로 흐른다.

    0) 테스트셋 로드/정규화 : 신·구 스키마를 하나의 내부 형식으로 통일
    1) 채점 헬퍼           : 문자열 비교의 규칙(공백 무시, 조문번호 변형 등)
    2) RAG 실행            : 요청 객체 조립 → 검색 → 생성
    3) 선택 채점기         : 의미 유사도 / LLM 심판
    4) 문항 채점           : 위 신호들을 모아 pass/fail 판정
    5) 집계 리포트         : 전체·유형별·검색 지표 산출과 출력
    6) 메인                : 초기화 → 루프 → 리포트 → 파일 저장

================================================================================
1. 무엇을 재는가
================================================================================
  [검색(Retrieval)]
    - context_hit   : 정답 문서(full_text 조각)가 검색 컨텍스트에 들어왔는가 (항상 측정)
    - hit@k / MRR   : gold_doc_id 를 실제로 끌어왔는가 (선택: 아래 '검색기 패치' 시)
  [답변(Answer)]
    - kw_recall     : must_include(정답 핵심 토큰) 재현율 0~1
    - hallucination : must_not_include(오답/오해) 등장 여부
    - absence_ok    : absence 문항에서 '없다/확인 불가'로 올바르게 답했는가
    - sem_sim       : 답변 vs 정답 코사인 유사도 (선택 --semantic, ko-sroberta 재사용)
    - llm_judge     : LLM 심판 0~1 (선택 --judge, 네 LLMController.evaluate_ask 재사용)

Why 검색과 답변을 나눠 재는가(이 도구의 존재 이유):
    RAG 가 틀렸을 때 원인은 둘 중 하나다.
      (a) 검색이 정답 문서를 못 가져왔다  → 임베딩/리랭커/청킹 문제
      (b) 문서는 왔는데 LLM 이 못 읽었다  → 프롬프트/모델 문제
    최종 답변만 보면 이 둘을 구분할 수 없어, 엉뚱한 곳을 고치며 시간을 태우게 된다.
    그래서 이 하네스는 **검색 컨텍스트를 직접 확보**해 두 단계를 따로 채점한다.
    (nodes 모드가 기본인 이유이기도 하다.)

================================================================================
2. 실행 모드
================================================================================
    --mode nodes     (기본) RAGState 에 AskRun 을 넣고 node_retrieve→node_generate.
                     가드레일/DeepEval 생략. 검색+답변을 분리 채점.
    --mode pipeline  같은 AskRun 을 run_pipeline 에 주입(서버 /ask 와 동일).
                     가드레일·재시도·평가 포함. 검색 스냅샷도 AskRun 으로 회수.

    두 모드의 용도 구분:
      * 성능을 '개선'하는 중이라면 nodes — 원인 분리가 되니까.
      * 배포 직전 '회귀'를 확인한다면 pipeline — 사용자가 실제로 받는 것과 동일하니까.

    산출물: logs/testruns/{실행id}/ 아래에 manifest.json, CSV/JSON 요약,
    cases/{문항id}.json (AskRun 전체). 저장은 통과/실패 모두.
    --print-failures 가 켜져 있으면 실패 문항만 진단 블록을 화면에 길게 출력.

================================================================================
3. 전제(네 코드 시그니처에 맞춤)
================================================================================
    import api_server                         # 그 안의 PostgresRepo/Neo4jRepo/init_gate/
                                              #   PromptsConfig/QueryRequest 를 재사용
    from workflow_orchestrator import get_orchestrator, RAGState
    from rag_run import AskRun, set_server_capture
    get_orchestrator(model_id: str) -> RAGOrchestrator      # 캐시 재사용
    RAGState(req, prompts, run=AskRun); orch.node_retrieve/node_generate/run_pipeline

================================================================================
4. 자체 초기화(웹서버 없이 도는 이유)
================================================================================
이 하네스는 서버의 lifespan 이 하던 일을 대신 한다:
  - PostgresRepo.init_pool() / Neo4jRepo.init_driver() 호출(DB 풀·드라이버)
  - 프롬프트 JSON 로드(--prompt-file 또는 .env의 DEFAULT_PROMPT_FILE)
  - set_server_capture("full") — 검색 후보·점수를 AskRun 에 남긴다
→ 그래서 api_server 를 띄우지 않아도 검색·생성이 실제와 동일하게 돈다.
  (프롬프트를 안 주면 main_answer 가 비어 답변 품질이 급락하니 반드시 지정 권장)

Why 서버를 안 띄우는가:
    HTTP 계층을 끼면 (a) 타임아웃/스트리밍 때문에 배치가 불안정하고,
    (b) 서버 로그와 평가 로그가 섞이며, (c) 배치 산출물을 실행 단위 폴더에
    모으기 어렵다. 평가는 같은 오케스트레이터 함수를 직접 부르되, 서버가
    하던 초기화는 bootstrap_environment 가 떠안는다.

================================================================================
5. 사용 예
================================================================================
    # 기본(검색+답변) — 프롬프트 파일 지정. 실패 문항은 화면에 진단 블록.
    python evaluate_rag.py --testset testset.json --prompt-file prompts.json \
        --model openrouter:google/gemini-2.0-flash

    # 엔드투엔드(가드레일 포함) + 의미유사도까지
    python evaluate_rag.py --testset testset.json --mode pipeline --semantic

    # 검색 파라미터 오버라이드
    python evaluate_rag.py --testset testset.json --top-k 5 --fusion rrf --query-mode expansion
    # → 최종 컨텍스트 기준 hit@k 와, 후보 풀 기준 pool_hit@k 를 함께 낸다.

================================================================================
6. 전체 주의사항
================================================================================
  * 이 하네스는 **채점만** 한다. 시험지(testset.json)가 나쁘면 점수도 무의미하다.
  * 모델 A/B 비교 시에는 반드시 같은 testset 파일 + 같은 옵션으로 돌려야 한다.
    kw_threshold, require_context, id_check 중 하나만 달라도 통과율이 크게 바뀐다.
    그래서 summary["config"] 에 판정 조건을 함께 저장한다(결과만 보고 조건을
    잊어버리는 사고 방지).
  * 무료 등급 API 로 돌릴 때는 --sleep 4 정도를 주자. 429 재시도로 시간이
    늘어나느니 처음부터 간격을 두는 편이 빠르다.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from types import ModuleType
from typing import TYPE_CHECKING, Any

from search_constants import EMPTY_CONTEXT_MSG

if TYPE_CHECKING:
    # 런타임 import 는 amain() 안에서 지연 수행한다(무겁고, DB/모델 로딩을 유발).
    # 시그니처에 실제 타입 이름을 쓰기 위한 '검사 전용' import.
    from contracts import PromptsConfig, QueryRequest as QueryRequestModel
    from rag_run import AskRun
    from workflow_orchestrator import RAGOrchestrator, RAGState as RAGStateType

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# hit@k 의 채점 기준 k 집합. --top-k(RAG 동작 설정)와 독립적으로 고정한다.
# 여러 k 를 동시에 재면 "순위가 좋아졌나(hit@1↑)"와 "그물만 넓혔나(hit@10만↑)"를
# 분리해 해석할 수 있다. 검색 결과 개수가 이보다 적으면 해당 k 는 자연히 상한이 된다.
HITK_LEVELS = (1, 3, 5, 10)
EMPTY_DOC_MSG = EMPTY_CONTEXT_MSG
_FINAL_PREFIX = "__FINAL_ANSWER__:"
_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _git_rev(root: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if proc.returncode == 0:
            return (proc.stdout or "").strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def _safe_filename(item_id: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", item_id or "item")[:120]


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    for name in ("model_dump", "dict"):
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                data = fn(mode="json") if name == "model_dump" else fn()
            except TypeError:
                try:
                    data = fn()
                except Exception:
                    continue
            except Exception:
                continue
            if isinstance(data, dict):
                return data
    if isinstance(obj, dict):
        return obj
    return {}


def _hit_doc_id(row: Any) -> str:
    if isinstance(row, dict):
        return str(row.get("doc_id") or "")
    if isinstance(row, (list, tuple)) and row:
        return str(row[0])
    return str(row or "")


def _gold_ids_match(hit_id: str, gold: str) -> bool:
    """접두어 있는 히트 id 와 예전 원본 gold id 를 같게 본다."""
    if not hit_id or not gold:
        return False
    if hit_id == gold:
        return True
    from schema import ALLOWED_SOURCE_TABLES
    for table in ALLOWED_SOURCE_TABLES:
        prefix = f"{table}:"
        if hit_id.startswith(prefix) and hit_id[len(prefix):] == gold:
            return True
        if gold.startswith(prefix) and gold[len(prefix):] == hit_id:
            return True
    return False


def rank_and_score(rows: list | None, gold_doc_id: str) -> tuple[int, float | None]:
    """목록에서 gold 의 1-base 순위와 점수를 찾는다. 없으면 (0, None)."""
    if not rows or not gold_doc_id:
        return 0, None
    gold = str(gold_doc_id)
    for i, row in enumerate(rows, 1):
        if not _gold_ids_match(_hit_doc_id(row), gold):
            continue
        score = None
        if isinstance(row, dict) and row.get("score") is not None:
            try:
                score = float(row["score"])
            except (TypeError, ValueError):
                score = None
        return i, score
    return 0, None


def ids_from_hits(hits: list | None) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for row in hits or []:
        if isinstance(row, dict):
            out.append((str(row.get("doc_id") or ""), str(row.get("source") or ""),
                        str(row.get("level") or "")))
        elif isinstance(row, (list, tuple)):
            out.append(tuple(row[:3]) if len(row) >= 3 else (str(row[0]), "", ""))  # type: ignore[arg-type]
        else:
            out.append((str(row), "", ""))
    return out


def strip_final_prefix(text: str | None) -> str:
    raw = text or ""
    if raw.startswith(_FINAL_PREFIX):
        return raw[len(_FINAL_PREFIX):]
    return raw


def answer_from_run(run: Any, fallback: str = "") -> str:
    """채점용 답. 가드레일 차단이면 사용자에게 보이는 메시지."""
    if run is None:
        return strip_final_prefix(fallback)
    status = getattr(run, "status", "") or ""
    err = getattr(run, "error_message", None) or ""
    gen = ""
    gen_snap = getattr(run, "generation", None)
    if gen_snap is not None:
        gen = getattr(gen_snap, "answer", "") or ""
    if status == "rejected":
        return err or gen or strip_final_prefix(fallback)
    return gen or err or strip_final_prefix(fallback)


def context_from_run(run: Any) -> str:
    if run is None:
        return ""
    snap = getattr(run, "retrieval", None)
    return (getattr(snap, "context_text", None) or "") if snap is not None else ""


def diagnose_row(item: dict, run: Any, passed: bool) -> str:
    """실패 원인을 한 단어로. 검색 풀 미적중 / top_k 밀림 / 생성 / 가드레일."""
    status = getattr(run, "status", "") or "" if run is not None else ""
    where = getattr(run, "error_where", "") or "" if run is not None else ""
    if status == "rejected":
        if where == "guard_in":
            return "guardrail_in"
        if where == "guard_out":
            return "guardrail_out"
        return "guardrail"
    if status == "error":
        return "error"
    if item.get("type") == "absence":
        return "ok" if passed else "absence_fail"
    gold = item.get("gold_doc_id") or ""
    hits = getattr(getattr(run, "retrieval", None), "hits", None) or []
    cands = getattr(getattr(run, "retrieval", None), "candidates", None) or []
    hit_rank, _ = rank_and_score(hits, gold)
    cand_rank, _ = rank_and_score(cands, gold)
    if hit_rank >= 1:
        return "ok" if passed else "generation"
    if cand_rank >= 1:
        return "cut_by_topk"
    return "leakage" if passed else "pool_miss"


def make_ask_run(req: Any, *, batch_id: str, item_id: str) -> "AskRun":
    from rag_run import AskRun
    return AskRun.from_request(req, run_id=item_id, session_id=batch_id)


def dump_case(batch_dir: str, item: dict, row: dict, run: Any, prompts: Any) -> str | None:
    """문항 1개 = JSON 1파일. AskRun 과 채점·진단을 같은 봉투에 담는다."""
    case_dir = os.path.join(batch_dir, "cases")
    path = os.path.join(case_dir, f"{_safe_filename(item.get('id') or row.get('id') or 'item')}.json")
    payload = {
        "source": "regression",
        "item": {
            "id": item.get("id"),
            "type": item.get("type"),
            "issue": item.get("issue"),
            "query": item.get("query"),
            "expected_answer": item.get("expected_answer"),
            "must_include": item.get("must_include"),
            "must_not_include": item.get("must_not_include"),
            "gold_doc_id": item.get("gold_doc_id"),
            "gold_law_name": item.get("gold_law_name"),
            "gold_article_number": item.get("gold_article_number"),
        },
        "score": row,
        "run": run.to_dict() if run is not None and hasattr(run, "to_dict") else None,
        "prompts": _as_dict(prompts),
    }
    tmp = path + ".tmp"
    try:
        os.makedirs(case_dir, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
        if run is not None:
            run.dump_path = path
        return path
    except OSError as exc:
        print(f"[WARN] 문항 덤프 실패({item.get('id')}): {exc}", file=sys.stderr)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return None


def write_manifest(batch_dir: str, args: argparse.Namespace, summary: dict,
                   testset_path: str, n_items: int, rows: list[dict]) -> str:
    manifest = {
        "source": "regression",
        "started_at": summary.get("started_at"),
        "ended_at": _utc_now(),
        "elapsed_sec": summary.get("elapsed_sec"),
        "mode": args.mode,
        "model": args.model,
        "testset": os.path.abspath(testset_path),
        "testset_sha256": _file_sha256(testset_path),
        "git": _git_rev(_HARNESS_DIR),
        "n_items": n_items,
        "config": summary.get("config"),
        "retrieval": summary.get("retrieval"),
        "overall_pass_rate": summary.get("overall_pass_rate"),
        "fail_ids": [r["id"] for r in rows if not r.get("pass")],
    }
    path = os.path.join(batch_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, default=str)
    return path


def snapshot_harness_config(args: argparse.Namespace, req: Any = None,
                            prompt_path: str | None = None) -> dict[str, Any]:
    """실행 재현용 설정 스냅샷. CLI 미지정(None)과 실제로 적용된 값을 같이 남긴다."""
    cfg: dict[str, Any] = {
        "mode": getattr(args, "mode", None),
        "model": getattr(args, "model", None),
        "kw_threshold": getattr(args, "kw_threshold", None),
        "require_context": getattr(args, "require_context", None),
        "id_check": getattr(args, "id_check", None),
        "capture": "full",
        "trace_prompts": bool(getattr(args, "trace_prompts", True)),
        "semantic": bool(getattr(args, "semantic", False)),
        "judge": bool(getattr(args, "judge", False)),
        "judge_model": getattr(args, "judge_model", None),
        "sleep": getattr(args, "sleep", 0.0),
        "max_retries": getattr(args, "max_retries", 4),
        "limit": getattr(args, "limit", 0) or None,
        "prompt_file": prompt_path or getattr(args, "prompt_file", None),
        "max_tokens": getattr(args, "max_tokens", None),
        "temperature": getattr(args, "temperature", None),
        "cli_overrides": {
            "top_k": getattr(args, "top_k", None),
            "query_mode": getattr(args, "query_mode", None),
            "expansion": bool(getattr(args, "expansion", False)),
            "fusion": getattr(args, "fusion", None),
            "alpha": getattr(args, "alpha", None),
            "embed_model": getattr(args, "embed_model", None),
            "embed_variant": getattr(args, "embed_variant", None),
            "sources": getattr(args, "sources", None),
            "rerank_model": getattr(args, "rerank_model", None),
            "min_score": getattr(args, "min_score", None),
            "use_reranker": getattr(args, "use_reranker", None),
            "use_guardrails": getattr(args, "use_guardrails", None),
            "use_graphdb": getattr(args, "use_graphdb", None),
        },
    }
    if req is None:
        return cfg
    dumped = _as_dict(req)
    dumped.pop("query", None)
    dumped.pop("custom_prompts", None)
    cfg["resolved"] = dumped
    g = getattr(req, "general_settings", None)
    r = getattr(req, "rerank_settings", None)
    ev = getattr(req, "eval_settings", None)
    adv = getattr(r, "advanced", None) if r is not None else None
    cfg["resolved_search"] = {
        "llm_model": getattr(g, "llm_model", None) if g is not None else None,
        "embed_model": getattr(g, "embed_model", None) if g is not None else None,
        "embed_variant": getattr(g, "embed_variant", None) if g is not None else None,
        "sources": getattr(g, "sources", None) if g is not None else None,
        "query_mode": getattr(g, "query_mode", None) if g is not None else None,
        "fusion_mode": getattr(g, "fusion_mode", None) if g is not None else None,
        "alpha": getattr(g, "alpha", None) if g is not None else None,
        "use_graphdb": getattr(g, "use_graphdb", None) if g is not None else None,
        "use_guardrails": getattr(g, "use_guardrails", None) if g is not None else None,
        "temperature": getattr(g, "temperature", None) if g is not None else None,
        "max_tokens": getattr(g, "max_tokens", None) if g is not None else None,
        "capture": getattr(g, "capture", None) if g is not None else None,
        "trace_prompts": getattr(g, "trace_prompts", None) if g is not None else None,
        "reranker_enabled": getattr(r, "enabled", None) if r is not None else None,
        "top_k": getattr(r, "top_k", None) if r is not None else None,
        "rerank_model": getattr(adv, "model", None) if adv is not None else None,
        "min_score": getattr(adv, "min_score", None) if adv is not None else None,
        "eval_enabled": getattr(ev, "enabled", None) if ev is not None else None,
        "eval_model": getattr(ev, "eval_model", None) if ev is not None else None,
    }
    return cfg


def print_failure_detail(item: dict, row: dict, run: Any) -> None:
    """실패한 문항만 검색 진단에 필요한 맥락을 길게 출력한다."""
    snap = getattr(run, "retrieval", None) if run is not None else None
    hits = getattr(snap, "hits", None) or []
    expanded = getattr(snap, "expanded_query", None) or ""
    keywords = getattr(snap, "keywords", None) or []
    print(f"\n----- FAIL {row.get('id')}  diag={row.get('diag')}  status={row.get('run_status')} -----")
    print(f"  Q: {item.get('query')}")
    if expanded:
        print(f"  expanded: {expanded}")
    if keywords:
        print(f"  keywords: {keywords}")
    print(f"  gold: {item.get('gold_doc_id')}  hit_rank={row.get('gold_hit_rank')}  "
          f"hit_score={row.get('gold_hit_score')}  "
          f"cand_rank={row.get('gold_candidate_rank')}  cand_score={row.get('gold_candidate_score')}")
    print(f"  note: {row.get('note')}  kw={row.get('kw_recall')}  halluc={row.get('hallucination')}")
    if hits:
        print("  top hits:")
        for h in hits[:10]:
            if isinstance(h, dict):
                print(f"    #{h.get('rank')} id={h.get('doc_id')} score={h.get('score')} "
                      f"[{h.get('level')}] {(h.get('content') or '')[:80]}")
            else:
                print(f"    {h}")
    else:
        print("  top hits: (없음)")
    ans = (row.get('answer') or "")[:400]
    if ans:
        print(f"  answer: {ans}")
    print("-----")


# =============================================================================
# 0) 테스트셋 로드/정규화 (신 스키마 + 구 스키마 alias 모두 수용)
# =============================================================================
def load_testset(path: str) -> list[dict]:
    """테스트셋 JSON 을 읽어 내부 표준 형식으로 정규화한다.

    Args:
        path: 테스트셋 JSON 경로.

    Returns:
        정규화된 문항 dict 목록. query 가 빈 문항은 제외된다.

    Raises:
        FileNotFoundError: 파일이 없을 때.
        json.JSONDecodeError: JSON 문법 오류.

    Note:
        ★ 이 함수의 정체는 '부패 방지 계층(anti-corruption layer)'이다(Why).
          테스트셋은 시간이 지나며 스키마가 바뀐다(구: input/expected_output/
          source_id, 신: query/expected_answer/gold_doc_id). 이 변환을 한곳에
          가두지 않으면, 채점 코드 곳곳에 `it.get("query") or it.get("input")`
          같은 방어 코드가 번져 나가 유지보수가 불가능해진다.
          **입구에서 한 번 정규화하고, 이후 코드는 신 스키마만 안다.**

        허용하는 입력 형태:
          - 최상위가 리스트인 경우(기본)
          - 최상위가 dict 이고 items/data 키에 리스트가 있는 경우
            (다른 도구가 메타데이터를 감싸 저장한 산출물 호환)

        빈 query 문항을 버리는 이유: 질문이 없으면 실행 자체가 불가능한데,
        분모에는 포함되어 통과율만 떨어뜨린다. 조용히 점수를 왜곡하느니 제외한다.

        id 자동 부여(`Q{i:03d}`): id 없는 문항이 섞여도 리포트에서 행을 식별할
        수 있어야 하기 때문이다.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):  # {"items":[...]} 형태도 허용
        data = data.get("items") or data.get("data") or []
    norm = []
    for i, it in enumerate(data, 1):
        # 신 스키마를 우선하고, 없으면 구 스키마 alias 로 폴백한다.
        q = it.get("query") or it.get("input") or ""
        exp = it.get("expected_answer") or it.get("expected_output") or ""
        gold_id = it.get("gold_doc_id") or it.get("source_id") or ""
        norm.append({
            "id": it.get("id") or f"Q{i:03d}",
            "type": (it.get("type") or "exact").lower(),
            "issue": it.get("issue") or "",
            "query": q,
            "expected_answer": exp,
            "must_include": _as_list(it.get("must_include")),
            "must_not_include": _as_list(it.get("must_not_include")),
            "gold_doc_id": gold_id,
            "gold_law_name": it.get("gold_law_name") or it.get("law_name") or "",
            "gold_article_number": str(it.get("gold_article_number") or it.get("article_number") or ""),
            "gold_context": it.get("gold_context") or it.get("evidence_text") or "",
        })
    skipped = sum(1 for x in norm if not x["query"])
    if skipped:
        print(f"[WARN] query 비어 {skipped}문항 제외", file=sys.stderr)
    return [x for x in norm if x["query"]]


TESTSET_TYPES = frozenset({"exact", "criteria", "hybrid", "absence"})


def validate_testset(items: list[dict]) -> list[str]:
    """골든셋 결함 목록. 비어 있으면 정상."""
    problems: list[str] = []
    seen: set[str] = set()
    for it in items:
        i = str(it.get("id") or "?")
        if i in seen:
            problems.append(f"{i}: id 중복")
        seen.add(i)
        q = (it.get("query") or "").strip()
        if not q:
            problems.append(f"{i}: query 비어 있음")
        typ = (it.get("type") or "exact").lower()
        if typ not in TESTSET_TYPES:
            problems.append(f"{i}: 알 수 없는 type={typ!r}")
        if typ != "absence":
            must = it.get("must_include") or []
            if not must:
                problems.append(f"{i}: must_include 가 비어 무조건 통과됨")
            gid = str(it.get("gold_doc_id") or "").strip()
            gctx = (it.get("gold_context") or "").strip()
            if not gid and not gctx:
                problems.append(f"{i}: gold_doc_id·gold_context 모두 비어 검색 채점 불가")
            if gid and re.fullmatch(r"\d+", gid):
                problems.append(
                    f"{i}: gold_doc_id='{gid}' 접두어 없음. "
                    f"tax_laws_master:{gid} 형식 권장"
                )
    return problems


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항 비율의 Wilson 신뢰구간. n 이 작을 때 정규근사보다 안정적이다."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _format_pass_rate_line(label: str, passed: int, total: int) -> str:
    if total <= 0:
        return f"{label}: (문항 없음)"
    rate = passed / total
    lo, hi = wilson_ci(passed, total)
    return (
        f"{label}: {rate * 100:.1f}%  ({passed}/{total}, "
        f"95% CI {lo * 100:.1f}~{hi * 100:.1f}%)"
    )


def _as_list(v: Any) -> list[str]:
    """어떤 형태로 저장된 토큰 목록이든 list[str] 로 정규화한다.

    Args:
        v: None / list / str / 기타 스칼라.

    Returns:
        문자열 리스트. 입력이 None 이거나 빈 문자열이면 빈 리스트.

    Note:
        Why 이렇게 관대한가:
          테스트셋은 JSON 으로도 CSV 로도 저장된다. CSV 를 거치면 리스트가
          `'["183일", "거주자"]'` 같은 **문자열**이 되어 돌아온다(emit_csv 가
          json.dumps 로 저장하므로). 그래서 '[' 로 시작하면 JSON 파싱을 한 번 더
          시도한다.

        파싱 실패 시 예외를 던지지 않고 '문자열 하나짜리 리스트'로 폴백하는 이유:
          토큰 목록이 조금 이상하다고 평가 전체를 중단시킬 이유가 없다.
          최악의 경우 그 문항의 채점이 느슨해질 뿐이다.
    """
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                return [str(x) for x in json.loads(s)]
            except Exception:
                pass
        return [s] if s else []
    return [str(v)]


# =============================================================================
# 1) 채점 헬퍼
#
# 이 섹션의 철학:
#   자동 채점은 결국 '문자열 비교'다. 그런데 LLM 답변은 띄어쓰기·표기가
#   제각각이라 순진한 `in` 비교는 정답을 오답으로 만든다. 그래서 비교 규칙을
#   여기 한곳에 모아, 채점의 관대함/엄격함을 통제 가능하게 만든다.
# =============================================================================
def _norm(s: str) -> str:
    """비교용 정규화: 공백 제거 + 소문자화.

    Args:
        s: 원본 문자열(None 허용).

    Returns:
        정규화된 문자열.

    Note:
        공백 제거는 '6억 원' vs '6억원' 같은 표기 흔들림을 흡수하기 위한 것이고,
        소문자화는 영문 약어(VAT/vat)를 위한 것이다. 한글에는 대소문자가 없어
        lower() 는 사실상 영문에만 작용한다.
    """
    return re.sub(r"\s+", "", (s or "")).lower()


# 조문번호형 토큰인지 판별하는 패턴('1', '1의4' 형태).
_ART_RE = re.compile(r"^\d+(?:의\d+)?$")


def contains_token(text: str, token: str) -> bool:
    """답변에 토큰이 들어있는지 판정한다(공백 무시, 조문번호는 변형도 시도).

    Args:
        text: 검사 대상(모델 답변).
        token: 찾을 토큰.

    Returns:
        포함하면 True. token 이 비어 있으면 True(빈 조건은 항상 만족).

    Note:
        ★ 조문번호 변형 처리가 이 함수의 핵심이다(Why).
          테스트셋에는 조문번호가 DB 원형('1의4')으로 저장되지만, LLM 은
          '제1조의4' 라고 쓰고, 인용 표기는 '[1의4]' 나 '(1의4)' 로 나온다.
          같은 대상을 가리키는데 표기가 4가지인 셈이다. 원형만 비교하면
          맞는 답을 계속 틀렸다고 채점하게 된다.
          그래서 토큰이 '숫자(의숫자)' 꼴이면 표시형·인용형까지 함께 확인한다.

        ★ 주의 — 이 관대함의 대가:
          '183' 같은 순수 숫자 토큰도 _ART_RE 에 매칭되어 '제183조' 변형까지
          찾는다. 즉 '183일'을 물었는데 답변에 '제183조'가 있으면 통과할 수 있다.
          실무상 이런 충돌은 드물고, 반대(정답을 오답 처리)의 피해가 훨씬 크기에
          관대한 쪽을 택했다.

        ★ 빈 토큰이 True 인 이유: must_include 에 빈 문자열이 섞여도 그 문항을
          통째로 실패시키지 않기 위해서다(시험지 결함이 모델 점수를 깎지 않도록).
    """
    if not token:
        return True
    nt, ntok = _norm(text), _norm(token)
    if ntok and ntok in nt:
        return True
    tk = token.strip()
    if _ART_RE.match(tk):
        # '1' → 제1조,  '1의4' → 제1조의4  (+ [1의4]/(1의4) 표지 형태)
        variants = [f"[{tk}]", f"({tk})"]
        if "의" in tk:
            base, sub = tk.split("의", 1)
            variants.append(f"제{base}조의{sub}")
        else:
            variants.append(f"제{tk}조")
        for v in variants:
            if _norm(v) in nt:
                return True
    return False


def kw_recall(answer: str, must_include: list[str]) -> tuple[float, list[str]]:
    """must_include 토큰의 재현율과 누락 목록을 계산한다.

    Args:
        answer: 모델 답변.
        must_include: 반드시 포함되어야 할 토큰 목록.

    Returns:
        (재현율 0~1, 누락 토큰 목록). must_include 가 비면 (1.0, []).

    Note:
        Why 재현율(비율)인가, 전부 포함(all)이 아니라:
            토큰 3개 중 2개를 맞힌 답과 0개를 맞힌 답은 품질이 명백히 다르다.
            all/nothing 으로 채점하면 그 차이가 사라져, 개선이 진행 중인지
            알 수 없게 된다. 비율로 두면 --kw-threshold 로 엄격도를 **사후에**
            조절할 수도 있다(같은 실행 결과를 다른 기준으로 재해석 가능).

        누락 목록을 함께 돌려주는 이유: 리포트에서 "무엇이 빠져서 틀렸는지"를
        바로 보여 주기 위해서다. 점수만 있으면 다음 행동을 정할 수 없다.
    """
    if not must_include:
        return 1.0, []
    missing = [t for t in must_include if not contains_token(answer, t)]
    hit = len(must_include) - len(missing)
    return hit / len(must_include), missing


def has_hallucination(answer: str, must_not_include: list[str]) -> tuple[bool, list[str]]:
    """금지 토큰이 답변에 등장했는지 확인한다.

    Args:
        answer: 모델 답변.
        must_not_include: 등장하면 안 되는 토큰 목록.

    Returns:
        (하나라도 등장했는가, 등장한 토큰 목록).

    Note:
        여기서 '할루시네이션'은 넓은 의미다. 사실 위조뿐 아니라 '흔한 오해'
        (예: 거주자 판정에서 '1년')를 말한 경우도 포함한다. 시험지가 지정한
        '말하면 안 되는 것'을 말했다는 뜻이다.

        주의: must_not_include 가 비면 항상 False 다. template 엔진으로 만든
        문항은 이 목록이 비어 있으므로, 그 시험지에서 할루시네이션율이 0% 로
        나오는 것은 '깨끗하다'가 아니라 '측정하지 않았다'는 뜻이다.
    """
    bad = [t for t in must_not_include if t and contains_token(answer, t)]
    return (len(bad) > 0), bad


# 기권(자료에 없음) 표현 마커. 어간 수준으로 짧게 잡아 표현 변주를 폭넓게 잡는다.
_ABSENCE_MARKERS = ["없", "확인할 수 없", "찾을 수 없", "제공된 자료", "근거가 없", "존재하지 않"]


def is_absence_answer(answer: str) -> bool:
    """답변이 '자료에 없다/확인 불가' 취지인지 판정한다.

    Args:
        answer: 모델 답변.

    Returns:
        기권 표현이 하나라도 있으면 True.

    Note:
        Why 이렇게 느슨한가:
            기권 표현은 무한히 다양하다('확인되지 않습니다', '규정이 없습니다',
            '자료에서 찾을 수 없습니다'…). 재현율을 우선해 어간('없')까지 넣었다.

        ★ 주의 — 정밀도가 낮다:
            "제외되는 경우는 **없**습니다" 처럼 다른 맥락의 '없'도 통과시킨다.
            즉 이 판정은 '기권했을 가능성'에 가깝다. 더 엄격한 판정이 필요하면
            evaluate_faithfulness.py 의 ABSTENTION_MARKERS(완성형 표현 목록)를
            참고할 것. 그쪽은 충실성 측정이 주 목적이라 정밀도를 더 높였다.

            '제공된 자료' 가 마커에 들어 있는 것도 같은 이유의 트레이드오프다.
            근거를 인용하며 정상 답변할 때도 이 문구를 쓸 수 있다.
    """
    a = answer or ""
    return any(m in a for m in _ABSENCE_MARKERS)


def context_hit(retrieved_context: str, item: dict) -> bool:
    """정답 컨텍스트가 검색 결과에 들어왔는지 부분일치로 판정한다.

    Args:
        retrieved_context: 검색기가 조립한 컨텍스트 전체 문자열.
        item: 정규화된 문항(gold_context / gold_article_number 사용).

    Returns:
        정답 문서 조각이나 조문번호 표지가 검색 결과에 있으면 True.

    Note:
        ★ 이 함수가 필요한 이유(Why) — id 없이도 검색을 채점하기 위해서다.
          이상적으로는 검색 결과의 **문서 id** 와 gold_doc_id 를 대조해야 정확하다.
          그런데 기존 retriever 는 컨텍스트를 하나의 긴 문자열로 조립해 돌려줄 뿐,
          어떤 문서에서 왔는지 id 를 주지 않는다. 검색기를 고치지 않고도 지금
          당장 측정하려면, **본문이 겹치는가**로 근사할 수밖에 없다.
          (id 를 주도록 패치하면 retrieval_rank 로 정확히 잰다. §모듈 docstring 참조)

        ★ 왜 '여러 지점의 40자 창'인가:
          - 앞부분만 비교하면, 청킹 때문에 조문 앞머리가 잘려 나간 경우를 놓친다.
          - 그래서 시작/1/3 지점/중간 세 곳에서 40자씩 뽑아 하나라도 걸리면 hit.
          - 40자는 '우연히 일치하기엔 충분히 길고, 청크 경계에 잘리기엔 충분히 짧은'
            경험적 절충값이다.

        ★ 조문번호 표지(`[1의4]`)는 '약한 신호'다.
          컨텍스트 조립 시 출처를 이 형식으로 붙이는 관례가 있어 보조로 확인하지만,
          다른 조문이 이 번호를 인용만 해도 걸릴 수 있다.

        주의: gold 본문이 20자 미만이면 창 비교를 건너뛴다. 너무 짧은 문자열은
        우연 일치 확률이 높아 신호가 아니라 잡음이 된다.
    """
    if not retrieved_context:
        return False
    nrc = _norm(retrieved_context)
    gold = item.get("gold_context") or ""
    ng = _norm(gold)
    # gold 본문에서 여러 지점의 40자 창을 뽑아 하나라도 매칭되면 hit
    if len(ng) >= 20:
        windows = []
        for start in (0, len(ng) // 3, len(ng) // 2):
            windows.append(ng[start:start + 40])
        if any(w and w in nrc for w in windows):
            return True
    # 조문번호 표지([1의4] 등) 약한 신호
    art = item.get("gold_article_number") or ""
    if art and _norm(f"[{art}]") in nrc:
        return True
    return False


def retrieval_rank(ids: list | None, gold_doc_id: str) -> int:
    """검색 결과 랭킹에서 정답 문서의 순위를 찾는다.

    Args:
        ids: 검색기가 돌려준 순위 목록. 각 원소는 id 스칼라이거나
            (id, law, art) 튜플/리스트일 수 있다.
        gold_doc_id: 정답 문서 id.

    Returns:
        1-base 순위. 목록에 없거나 입력이 비면 0.

    Note:
        Why 0 을 '없음'으로 쓰는가: 순위는 1부터 시작하므로 0 은 자연스러운
        sentinel 이다. 호출부는 `rank >= 1` 로 발견 여부를 판정하고,
        MRR 계산에서도 `rank >= 1` 인 것만 1/rank 를 더한다(0 나눗셈 방지).

        원소 형태를 두 가지로 받는 이유: 검색기 구현이 (id 만) 주는 버전과
        (id, 법령명, 조문번호) 튜플을 주는 버전이 혼재해 왔다. 어느 쪽이든
        첫 요소를 id 로 보고 문자열 비교한다(타입 차이 흡수를 위해 str() 캐스팅).
    """
    if not ids or not gold_doc_id:
        return 0
    for i, row in enumerate(ids, 1):
        rid = row[0] if isinstance(row, (list, tuple)) else row
        if _gold_ids_match(str(rid), str(gold_doc_id)):
            return i
    return 0


# =============================================================================
# 2) RAG 실행 (검색 컨텍스트 + 답변 확보)
# =============================================================================
def build_request(QueryRequest: type["QueryRequestModel"], query: str,
                  args: argparse.Namespace) -> "QueryRequestModel":
    """실제 서비스와 동일한 요청 객체를 만들되, CLI 옵션만 덮어쓴다.

    Args:
        QueryRequest: api_server 의 QueryRequest **클래스 자체**(인스턴스 아님).
        query: 질문 문자열.
        args: CLI 인자. None 인 항목은 '오버라이드 안 함'을 뜻한다.

    Returns:
        설정이 반영된 QueryRequest 인스턴스.

    Note:
        ★ 클래스를 인자로 받는 이유(Why):
          api_server 는 무거워서 모듈 최상단에서 import 하지 않는다(amain 에서 지연
          import). 그러면 이 함수가 QueryRequest 를 직접 참조할 수 없으므로,
          호출부가 클래스를 넘겨주는 **의존성 주입** 형태를 취했다.
          덕분에 이 함수는 단위 테스트 시 가짜 클래스로 대체하기도 쉽다.

        ★ 'None = 오버라이드 안 함' 규약:
          argparse 기본값을 None 으로 두고, None 이 아닐 때만 대입한다.
          그래야 **서비스 기본값(QueryRequest 의 Field default)을 그대로 쓰는 것**과
          '사용자가 명시적으로 그 값을 줬다'를 구분할 수 있다.
          예: --alpha 0.0 은 유효한 값이므로 `if args.alpha:` 로 검사하면 안 되고
          `if args.alpha is not None:` 이어야 한다. (문자열 옵션은 빈 문자열이
          의미 없어 truthiness 검사를 쓴다.)

        ★ eval_settings.enabled = False:
          파이프라인 내부의 자동 평가(deepeval 등)를 끈다. 채점은 이 하네스가
          하므로 중복이고, 문항마다 추가 LLM 호출이 발생해 비용·시간이 배가된다.
          try/except 로 감싼 것은 구버전 QueryRequest 에 이 필드가 없을 수 있어서다.

        ★ capture=full / trace_prompts:
          하네스는 검색 후보·점수를 남겨야 하므로 capture 를 항상 full 로 둔다.
          프롬프트 추적은 PoC 기본 on (--no-trace-prompts 로 끔).

        ★ getattr(..., None) 방어:
          general_settings/rerank_settings/advanced 구조가 버전마다 다를 수 있다.
          없으면 조용히 건너뛰어, 스키마가 달라도 평가가 죽지 않게 한다.
    """
    req = QueryRequest(query=query)
    # 평가 모듈은 하네스가 대신 채점하므로 기본 OFF(속도/비용)
    try:
        req.eval_settings.enabled = False
    except Exception:
        pass
    g = getattr(req, "general_settings", None)
    if g is not None:
        if hasattr(g, "capture"):
            try:
                from rag_run import set_server_capture
                set_server_capture("full")
            except Exception:
                pass
            g.capture = "full"
        if hasattr(g, "trace_prompts"):
            g.trace_prompts = bool(getattr(args, "trace_prompts", True))
        if args.model:
            g.llm_model = args.model
        if args.use_guardrails is not None:
            g.use_guardrails = args.use_guardrails
        if getattr(args, "expansion", False):
            g.query_mode = "expansion"
        elif args.query_mode:
            g.query_mode = args.query_mode
        if getattr(args, "keyword_mode", None) and hasattr(g, "keyword_mode"):
            g.keyword_mode = args.keyword_mode
        if args.fusion:
            g.fusion_mode = args.fusion
        if args.alpha is not None:
            g.alpha = args.alpha
        if args.embed_model:
            g.embed_model = args.embed_model
        if getattr(args, "embed_variant", None) and hasattr(g, "embed_variant"):
            g.embed_variant = args.embed_variant
        if args.use_graphdb is not None:
            g.use_graphdb = args.use_graphdb
        if getattr(args, "temperature", None) is not None and hasattr(g, "temperature"):
            g.temperature = args.temperature
        if getattr(args, "max_tokens", None) is not None and hasattr(g, "max_tokens"):
            g.max_tokens = args.max_tokens
        from model_registry import apply_embed_choice
        apply_embed_choice(g)
        if getattr(args, "sources", None) and hasattr(g, "sources"):
            from database.postgres_repo import normalize_sources
            g.sources = normalize_sources(args.sources)
    r = getattr(req, "rerank_settings", None)
    if r is not None:
        if args.use_reranker is not None:
            r.enabled = args.use_reranker
        if args.top_k is not None:
            r.top_k = args.top_k
        adv = getattr(r, "advanced", None)
        if adv is not None:
            if args.rerank_model:
                adv.model = args.rerank_model
            if args.min_score is not None:
                adv.min_score = args.min_score
    return req


def _assert_eval_request(req) -> None:
    from validation import pick_error_code, summarize_violations, validate_request

    raw = req.model_dump(mode="json") if hasattr(req, "model_dump") else {"query": getattr(req, "query", "")}
    outcome = validate_request(req, raw)
    if not outcome.violations:
        return
    code = pick_error_code(outcome.violations)
    print(
        f"[VALIDATE] 거부 code={code} {summarize_violations(outcome.violations)}",
        file=sys.stderr,
    )
    raise SystemExit(2)


async def run_nodes(orch: "RAGOrchestrator", RAGState: type["RAGStateType"],
                    prompts: "PromptsConfig",
                    req: "QueryRequestModel",
                    run: "AskRun") -> "AskRun":
    """검색+생성 노드만 실행한다. AskRun 에 검색 스냅샷이 채워진다.

    컨텍스트가 비면 생성을 건너뛴다. 배경지식으로 검색 실패를 덮지 않기 위함이다.
    """
    state = RAGState(req, prompts, run=run)
    state = await orch.node_retrieve(state)
    context = state.retrieved_context or ""
    if not context:
        if not (run.generation.answer or ""):
            run.generation.answer = EMPTY_DOC_MSG
        if run.status == "running":
            run.mark(stage="retrieve", status="completed")
        run.finish()
        return run
    state = await orch.node_generate(state)
    if run.status == "running":
        run.mark(stage="done", status="completed")
    run.finish()
    return run


async def run_pipeline(orch: "RAGOrchestrator", prompts: "PromptsConfig",
                       req: "QueryRequestModel",
                       run: "AskRun") -> "AskRun":
    """서버와 같은 run_pipeline 경로. AskRun 을 주입하고 회수한다."""
    out = await orch.run_pipeline(req, prompts, run=run)
    if not (run.generation.answer or "") and isinstance(out, str):
        run.generation.answer = strip_final_prefix(out)
    if run.status == "running":
        run.mark(stage="done", status="completed")
    run.finish()
    return run


# =============================================================================
# 3) 선택 채점기: 의미 유사도 / LLM 심판
#
# Why '선택'인가:
#   둘 다 비싸다(임베딩 모델 로딩 / 문항당 LLM 호출 1회). 기본 지표(kw_recall,
#   context_hit)만으로도 대부분의 회귀는 잡힌다. 정밀 분석이 필요할 때만 켠다.
# =============================================================================
class _Semantic:
    """답변과 정답의 의미 유사도를 계산하는 지연 로딩 래퍼.

    Attributes:
        model: 임베딩 모델 인스턴스. 최초 sim() 호출 전까지 None.
        model_name: HuggingFace 모델 이름.

    Note:
        ★ 지연 로딩(lazy loading)을 쓰는 이유(Why):
          임베딩 모델은 수백 MB 를 메모리에 올리고 로딩에 수십 초가 걸린다.
          --semantic 을 켰더라도, 실제로 유사도를 계산할 문항이 하나도 없을 수
          있다(예: 전부 absence 유형). 생성자에서 로딩하면 그 비용을 무조건
          치르게 되므로, **처음 필요한 순간까지 미룬다.**

        ★ 기본 indexed 임베딩을 쓰는 것이 중요하다:
          --semantic 은 검색에 쓰는 비교 모델과 별개로, 원본 컬럼을 만든
          indexed 모델로 답-정답을 잰다. 비교 테이블 실험 중에도 채점 축이
          흔들리지 않게 하기 위해서다.

        ★ 한계: 코사인 유사도는 '비슷하게 들리는가'만 본다. 숫자가 틀려도
          문장 구조가 같으면 높게 나온다(183일 vs 300일). 그래서 이 값은
          **보조 지표**이고, pass 판정에는 쓰지 않는다.
    """

    def __init__(self, model_name: str) -> None:
        """모델 이름만 기억하고, 실제 로딩은 미룬다.

        Args:
            model_name: HuggingFace 임베딩 모델 이름.
        """
        self.model = None
        self.model_name = model_name

    def _lazy(self) -> None:
        """최초 사용 시점에 임베딩 모델을 로드한다(멱등).

        Note:
            llama_index 의 HuggingFaceEmbedding 을 쓰는 이유는 검색 파이프라인과
            같은 래퍼를 재사용해 임베딩 결과가 완전히 동일하도록 하기 위해서다.
            import 도 이 안에서 한다(모듈 로드 비용 회피).
        """
        if self.model is None:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            print(f"[SEM] 임베딩 모델 로드: {self.model_name}", file=sys.stderr)
            self.model = HuggingFaceEmbedding(model_name=self.model_name)

    def sim(self, a: str, b: str) -> float:
        """두 문자열의 코사인 유사도를 계산한다.

        Args:
            a: 비교 문자열 1(모델 답변).
            b: 비교 문자열 2(정답).

        Returns:
            -1.0~1.0 범위의 코사인 유사도. 한쪽이 비면 0.0.

        Note:
            분모에 `or 1e-9` 를 둔 것은 영벡터로 인한 0 나눗셈 방지다.
            (빈 문자열은 위에서 걸러지지만, 임베딩이 영벡터가 되는 극단 상황 대비.)

            float32 로 캐스팅하는 이유는 메모리와 연산 속도이며, 유사도 계산
            정밀도에는 영향이 없다.
        """
        if not a or not b:
            return 0.0
        self._lazy()
        import numpy as np
        va = np.array(self.model.get_text_embedding(a), dtype="float32")
        vb = np.array(self.model.get_text_embedding(b), dtype="float32")
        denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-9
        return float(np.dot(va, vb) / denom)


# 심판 LLM 의 시스템 프롬프트.
_JUDGE_TEMPLATE = (
    "너는 세무 RAG 채점관이다. "
    '반드시 JSON 하나만 출력한다. 형식: {"score": 0.0~1.0, "reason": "한 줄"}'
)


def llm_judge(orch: "RAGOrchestrator", query: str, answer: str,
              expected: str, context: str,
              judge_llm: Any | None = None) -> float | None:
    """LLM 심판에게 답변 품질을 0.0~1.0 으로 채점받는다.

    judge_llm 이 있으면 그 컨트롤러를 쓰고, 없으면 orch.llm 을 쓴다.
    pass 판정에는 쓰지 않고 참고 지표만 남긴다.
    """
    prompt = f"""[질문]
{query}

[정답(모범답)]
{expected}

[모델 답변]
{answer}

[검색 근거(있으면)]
{(context or '')[:1500]}

지시: 모델 답변이 '정답'과 '검색 근거'에 부합하는 정도를 채점하라.
사실 오류/근거 없는 단정이 있으면 크게 감점. JSON만.
"""
    try:
        ctrl = judge_llm if judge_llm is not None else orch.llm
        raw = ctrl.evaluate_ask(prompt, _JUDGE_TEMPLATE, temperature=0.0, max_tokens=400)
        text = str(raw or "").strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                val = float(obj.get("score"))
            except Exception:
                val = None
        else:
            val = None
        if val is None:
            m2 = re.search(r"(\d+(?:\.\d+)?)", text)
            if not m2:
                return None
            val = float(m2.group(1))
        if val > 1.0:
            val = val / 100.0
        return max(0.0, min(1.0, val))
    except Exception as e:
        print(f"[JUDGE] 실패: {e}", file=sys.stderr)
        return None


# =============================================================================
# 4) 문항 채점
# =============================================================================
def score_item(item: dict, context: str, answer: str, ids: list | None,
               args: argparse.Namespace, semantic: "_Semantic | None",
               orch: "RAGOrchestrator",
               judge_llm: Any | None = None,
               run: Any = None) -> dict:
    """한 문항의 실행 결과를 채점해 리포트 행(dict)을 만든다.

    Args:
        item: 정규화된 문항.
        context: 검색 컨텍스트(pipeline 모드면 빈 문자열).
        answer: 모델 답변.
        ids: 검색 결과 id 목록(없으면 None).
        args: CLI 인자(판정 기준 포함).
        semantic: 의미 유사도 계산기(미사용 시 None).
        orch: 오케스트레이터(LLM 심판용).

    Returns:
        CSV/JSON 으로 그대로 나갈 수 있는 평면 dict 한 행.

    Note:
        ★ 반환 dict 의 모든 키를 **미리 빈 값으로 채워 두는** 이유(Why):
          이후 조건부로만 채워지는 필드(sem_sim, retrieval_rank 등)가 있는데,
          키가 없는 행이 섞이면 csv.DictWriter 가 깨지고 JSON 스키마도 들쭉날쭉해진다.
          "모든 행은 같은 모양"을 보장하는 것이 표 형태 산출물의 기본이다.
          빈 문자열은 '측정 안 함', False/숫자는 '측정 결과'라는 의미 구분도 된다.

        ★ 검색 정답 판정 방법 선택(3가지 경로):
            기본(id_check=True) : 조문 id 로 대조 → 정확. hit@k/MRR 계산 가능.
            --no-id-check       : id 무시, 본문 겹침(context_hit)으로만 판정.
            (id_check=True 라도 검색기가 id 를 안 주면 자동으로 본문 판정으로 폴백)
          어느 경로였는지 `match_method` 에 기록한다. 이걸 남기지 않으면 나중에
          "이 hit@k 는 믿을 수 있는 수치인가?"에 답할 수 없다.
          폴백이 일어나면 note 에도 흔적을 남긴다(id요청_but검색기미반환→본문판정).

        ★ context_hit 는 **항상** 계산한다(nodes 모드 한정).
          id 대조를 쓰더라도 본문 겹침은 별개의 정보다. 예컨대 id 는 못 맞혔는데
          본문은 겹친다면, 같은 내용의 다른 청크를 가져왔다는 뜻이라 청킹 전략을
          의심할 단서가 된다.

        ★ pass 판정 규칙:
            absence : 기권 표현이 있고 + 할루시네이션이 없어야 통과.
                      (검색 성공 여부를 보지 않는다. 정답 문서가 없으니까.)
            그 외    : kw_recall >= 임계 AND 할루시네이션 없음 AND (선택)검색 성공.
          세 조건을 **AND** 로 묶은 이유: RAG 답변은 '내용이 맞고, 거짓이 없고,
          근거에서 왔을 때' 비로소 쓸 수 있기 때문이다. 하나라도 빠지면 실무에서
          쓸 수 없는 답이므로 부분 점수를 주지 않는다.

        ★ note 필드에 실패 사유를 누적(`+=`)한다. 여러 이유로 동시에 실패할 수
          있으므로 덮어쓰지 않고 이어 붙인다. 리포트에서 이 문자열만 봐도
          무엇을 고쳐야 할지 알 수 있게 하는 것이 목적이다.

        ★ 답변은 500자로 잘라 저장한다. CSV 가 과도하게 커지는 것을 막기 위해서다.
          전체 답변이 필요하면 --mode 를 바꾸지 말고 별도로 로깅해야 한다.
    """
    typ = item["type"]
    rec, missing = kw_recall(answer, item["must_include"])
    halluc, bad = has_hallucination(answer, item["must_not_include"])

    # 모든 문항이 같은 컬럼 구조를 갖도록 뼈대를 먼저 만든다(위 Note 참조).
    row = {
        "id": item["id"], "type": typ, "issue": item["issue"],
        "query": item["query"],
        "answer": (answer or "").replace("\n", " ")[:500],
        "kw_recall": round(rec, 3),
        "missing_terms": "|".join(missing),
        "hallucination": halluc,
        "bad_terms": "|".join(bad),
        "context_hit": "",
        "retrieval_rank": "",
        "retrieval_found": "",
        "match_method": "",
        "gold_hit_rank": "",
        "gold_hit_score": "",
        "gold_candidate_rank": "",
        "gold_candidate_score": "",
        "expanded_query": "",
        "diag": "",
        "run_status": getattr(run, "status", "") if run is not None else "",
        "sem_sim": "",
        "llm_judge": "",
        "gold_doc_id": item["gold_doc_id"],
        "pass": False,
        "note": "",
    }

    # 검색 채점 (nodes 모드에서만 의미)
    # --- 정답 자료 판정 방법 선택 ---
    #   기본(id_check=True) : 조문 id 로 대조 → 정확. hit@k/MRR 계산 가능.
    #   --no-id-check       : id 무시, 본문 겹침(context_hit)으로만 판정.
    #   (id_check=True 라도 검색기가 id 를 안 주면 자동으로 본문 판정으로 폴백)
    snap = getattr(run, "retrieval", None) if run is not None else None
    hits = getattr(snap, "hits", None) if snap is not None else None
    cands = getattr(snap, "candidates", None) if snap is not None else None
    if snap is not None:
        row["expanded_query"] = getattr(snap, "expanded_query", None) or ""
    if hits:
        ids = ids_from_hits(hits)
    elif ids is None and hits is not None:
        ids = []

    ctx = context or context_from_run(run)
    gold = item["gold_doc_id"]
    have_retrieval = bool(ctx) or (hits is not None) or (ids is not None) or args.mode == "nodes"

    c_hit = None
    if have_retrieval:
        c_hit = context_hit(ctx, item)
        row["context_hit"] = c_hit
        use_id = bool(args.id_check) and (ids is not None)
        if use_id:
            rank = retrieval_rank(ids, gold)
            row["retrieval_rank"] = rank
            row["retrieval_found"] = rank >= 1
            row["match_method"] = "id"
            row["gold_hit_rank"] = rank
            hit_rank, hit_score = rank_and_score(hits, gold) if hits is not None else (rank, None)
            if hit_rank:
                row["gold_hit_rank"] = hit_rank
            if hit_score is not None:
                row["gold_hit_score"] = round(hit_score, 4)
            cand_rank, cand_score = rank_and_score(cands, gold)
            row["gold_candidate_rank"] = cand_rank
            if cand_score is not None:
                row["gold_candidate_score"] = round(cand_score, 4)
        else:
            row["retrieval_found"] = bool(c_hit)
            row["match_method"] = "text"
            if args.id_check and ids is None:
                row["note"] += "id요청_but검색기미반환→본문판정;"

    # 의미 유사도 (absence 는 '없다'가 정답이라 유사도 비교가 무의미)
    if semantic is not None and typ != "absence":
        try:
            row["sem_sim"] = round(semantic.sim(answer, item["expected_answer"]), 3)
        except Exception as e:
            row["note"] += f"sem_err:{e};"

    # LLM 심판 (역시 absence 제외)
    if args.judge and typ != "absence":
        j = llm_judge(orch, item["query"], answer, item["expected_answer"], ctx,
                      judge_llm=judge_llm)
        if j is not None:
            row["llm_judge"] = round(j, 3)

    # ---- pass 판정 ----
    status = row["run_status"]
    if status == "rejected":
        row["pass"] = False
        row["note"] += "가드레일차단;"
        row["diag"] = diagnose_row(item, run, False)
        return row
    if status == "error" or (answer or "").startswith("[실행오류]"):
        row["pass"] = False
        row["note"] += "실행오류;"
        row["diag"] = "error"
        return row

    if typ == "absence":
        ok = is_absence_answer(answer) and not halluc
        row["pass"] = ok
        if not ok:
            row["note"] += "absence_미표현;" if not is_absence_answer(answer) else "할루시;"
    else:
        cond_kw = rec >= args.kw_threshold
        cond_no_halluc = not halluc
        cond_ctx = True
        if args.require_context and have_retrieval:
            cond_ctx = bool(row.get("retrieval_found"))
        row["pass"] = cond_kw and cond_no_halluc and cond_ctx
        if not cond_kw:
            row["note"] += f"kw<{args.kw_threshold};"
        if not cond_no_halluc:
            row["note"] += "할루시;"
        if not cond_ctx:
            if isinstance(row.get("gold_candidate_rank"), int) and row["gold_candidate_rank"] >= 1:
                row["note"] += "topk밀림;"
            else:
                row["note"] += "검색실패;"

    row["diag"] = diagnose_row(item, run, bool(row["pass"]))
    return row


# =============================================================================
# 5) 집계 리포트
# =============================================================================
def aggregate(rows: list[dict], args: argparse.Namespace) -> dict:
    """행 목록을 전체/유형별/검색 지표로 집계한다.

    Args:
        rows: score_item 이 만든 행 목록.
        args: CLI 인자(설정을 요약에 함께 기록).

    Returns:
        요약 dict. JSON 으로 저장되며 리포트 출력의 입력이 된다.

    Note:
        ★ summary["config"] 에 판정 기준을 함께 저장하는 이유(Why):
          같은 시험지라도 kw_threshold 를 0.5 로 뒀는지 0.8 로 뒀는지에 따라
          통과율이 크게 달라진다. 결과 파일만 남고 실행 명령이 사라지면 그 숫자는
          해석 불가능해진다. **결과와 조건은 항상 붙어 다녀야 한다.**

        ★ 지표별 산출 조건이 다르다(중요):
          - context_hit_rate    : context_hit 이 계산된 행만(비-absence).
          - retrieval_found_rate: 선택된 판정 방법 기준 '정답 자료 발견'률.
          - hit@k / MRR         : **match_method == "id" 인 행만.**
            본문 겹침 판정에는 '순위' 개념이 없으므로 MRR 을 계산할 수 없다.
            id 대조가 실제로 적용된 문항에서만 계산해야 수치가 정직하다.
          absence 를 항상 제외하는 이유: 정답 문서가 없는 문항을 분모에 넣으면
          검색 성능이 실제보다 나쁘게 보인다.

        ★ hit@k 는 고정 집합 HITK_LEVELS(1/3/5/10)로 동시 산출한다.
          --top-k(RAG 가 실제로 가져오는 개수, 동작 설정)와 채점 기준 k 는
          별개다. 과거에는 k = args.top_k 로 두어, top_k 를 바꾸면 채점 기준까지
          함께 움직여 점수 변화의 원인(검색 개선 vs 채점 완화)을 구분할 수 없었다.
          이제 top_k 를 바꿔도 hit@1/3/5/10 은 같은 자로 재므로, hit@1 이 오르면
          순위 개선, hit@10 만 오르면 그물 확대로 해석할 수 있다.
          hit@N(=top_k) 표기는 현재 동작 설정 경계에서 몇 %가 살아남는지
          보여주는 참고값이다(HITK_LEVELS 에 없는 top_k 일 때만 추가).

        ★ MRR(Mean Reciprocal Rank)의 의미:
          정답이 1위면 1.0, 2위면 0.5, 3위면 0.33… 을 평균한 값이다.
          "정답을 얼마나 위에 올렸는가"를 보므로, 리랭커 개선 효과가
          hit@k 보다 민감하게 드러난다. 못 찾은 문항(rank=0)은 0점으로 들어간다
          (분모에는 포함, 분자에는 미포함). MRR 자체는 원래 top_k 와 무관하다.

        ★ sem_sim/llm_judge 는 값이 하나라도 있을 때만 요약에 넣는다.
          측정하지 않은 지표를 0.0 으로 표시하면 '성능이 0' 으로 오독된다.

        내부 헬퍼 rate/mean 은 빈 목록에서 0.0 을 돌려 0 나눗셈을 피한다.
    """
    n = len(rows)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    def rate(sub: list[dict], key: str, val: Any = True) -> float:
        """sub 중 row[key] == val 인 비율(빈 목록이면 0.0)."""
        vals = [r for r in sub if r[key] == val]
        return len(vals) / len(sub) if sub else 0.0

    def mean(sub: list[dict], key: str) -> float:
        """sub 의 숫자형 row[key] 평균(빈 문자열은 자동 제외)."""
        xs = [r[key] for r in sub if isinstance(r[key], (int, float))]
        return sum(xs) / len(xs) if xs else 0.0

    summary = {
        "total": n,
        "overall_pass_rate": round(rate(rows, "pass"), 4),
        "hallucination_rate": round(rate(rows, "hallucination"), 4),
        "mean_kw_recall": round(mean(rows, "kw_recall"), 4),
        "by_type": {},
        "retrieval": {},
        "config": {
            "mode": args.mode, "model": args.model,
            "kw_threshold": args.kw_threshold,
            "require_context": args.require_context,
            "id_check": args.id_check,
            "capture": "full",
            "trace_prompts": bool(getattr(args, "trace_prompts", True)),
            "top_k": args.top_k,
        },
    }
    for t, sub in by_type.items():
        summary["by_type"][t] = {
            "count": len(sub),
            "pass_rate": round(rate(sub, "pass"), 4),
            "mean_kw_recall": round(mean(sub, "kw_recall"), 4),
        }

    measured = [r for r in rows if r.get("retrieval_found") != ""]
    if measured:
        summary["retrieval"]["match_method"] = "id" if args.id_check else "text"
        ctx_rows = [r for r in rows if r["type"] != "absence" and r["context_hit"] != ""]
        if ctx_rows:
            summary["retrieval"]["context_hit_rate"] = round(rate(ctx_rows, "context_hit"), 4)
        found_rows = [r for r in rows if r["type"] != "absence" and r.get("retrieval_found") != ""]
        if found_rows:
            summary["retrieval"]["retrieval_found_rate"] = round(rate(found_rows, "retrieval_found"), 4)
        rank_rows = [r for r in rows if r.get("match_method") == "id"
                     and isinstance(r["retrieval_rank"], int) and r["retrieval_rank"] >= 0
                     and r["type"] != "absence"]
        if rank_rows:
            n_rank = len(rank_rows)
            for kk in HITK_LEVELS:
                n_hit = sum(1 for r in rank_rows if 1 <= r["retrieval_rank"] <= kk)
                summary["retrieval"][f"hit@{kk}"] = round(n_hit / n_rank, 4)
            mrr = sum((1.0 / r["retrieval_rank"]) for r in rank_rows if r["retrieval_rank"] >= 1)
            summary["retrieval"]["mrr"] = round(mrr / n_rank, 4)
            eff_k = args.top_k if args.top_k else None
            if eff_k and eff_k not in HITK_LEVELS:
                n_hit = sum(1 for r in rank_rows if 1 <= r["retrieval_rank"] <= eff_k)
                summary["retrieval"][f"hit@{eff_k}(=top_k)"] = round(n_hit / n_rank, 4)
        pool_rows = [r for r in rows if r.get("match_method") == "id"
                     and isinstance(r.get("gold_candidate_rank"), int)
                     and r["gold_candidate_rank"] >= 0
                     and r["type"] != "absence"]
        if pool_rows:
            n_pool = len(pool_rows)
            for kk in HITK_LEVELS:
                n_hit = sum(1 for r in pool_rows if 1 <= r["gold_candidate_rank"] <= kk)
                summary["retrieval"][f"pool_hit@{kk}"] = round(n_hit / n_pool, 4)
            n_cut = sum(1 for r in pool_rows
                        if r["gold_candidate_rank"] >= 1 and not r.get("retrieval_found"))
            summary["retrieval"]["cut_by_topk_rate"] = round(n_cut / n_pool, 4)
    diag_counts = Counter(r.get("diag") or "" for r in rows if r.get("diag"))
    if diag_counts:
        summary["diagnosis"] = dict(diag_counts)

    if any(isinstance(r["sem_sim"], (int, float)) for r in rows):
        summary["mean_sem_sim"] = round(mean(rows, "sem_sim"), 4)
    if any(isinstance(r["llm_judge"], (int, float)) for r in rows):
        summary["mean_llm_judge"] = round(mean(rows, "llm_judge"), 4)
    return summary


def print_report(summary: dict, rows: list[dict]) -> None:
    """사람이 읽는 요약 리포트를 화면에 출력한다.

    Args:
        summary: aggregate() 결과.
        rows: 개별 행(실패 미리보기에 사용).

    Note:
        구성 의도: 맨 위에 '어떤 조건으로 쟀는지'를 한 줄로 보여 준 뒤 지표를
        내보낸다. 조건 없이 숫자만 보면 반드시 오해가 생기기 때문이다.

        실패 문항 미리보기를 10개로 제한한 이유: 터미널에서 즉시 원인 감을 잡는
        것이 목적이며, 전체 목록은 CSV 에 있다. 화면을 스크롤로 채우면 오히려
        요약을 못 보게 된다.

        표시 형식 분기: 키에 'rate' 나 'hit@' 가 들어가면 백분율로, 문자열이면
        그대로, 나머지(mrr 등)는 소수 3자리로 찍는다.
    """
    print("\n" + "=" * 68)
    print("📊 [RAG 평가 리포트]")
    print("=" * 68)
    c = summary["config"]
    if c.get("mode") == "nodes":
        print("[주의] nodes 모드는 검색 컨텍스트가 비면 LLM 을 호출하지 않습니다.")
        print("       실제 서비스는 호출하므로, 이 통과율은 서비스 통과율의 하한입니다.")
    print(f"모드={c['mode']} | 모델={c['model']} | kw임계={c['kw_threshold']} | 검색요구={c['require_context']} | id대조={c.get('id_check', True)}")
    print(f"총 문항        : {summary['total']}")
    passed_n = sum(1 for r in rows if r.get("pass"))
    print(_format_pass_rate_line("전체 통과율", passed_n, summary["total"]))
    if summary["total"] < 40:
        print("  ↳ 표본이 40건 미만이라 ±10%p 수준의 차이는 잡음과 구분되지 않을 수 있습니다.")
    print(f"할루시네이션율 : {summary['hallucination_rate']*100:.1f}%")
    print(f"평균 핵심어재현: {summary['mean_kw_recall']*100:.1f}%")
    if "mean_sem_sim" in summary:
        print(f"평균 의미유사도: {summary['mean_sem_sim']:.3f}")
    if "mean_llm_judge" in summary:
        print(f"평균 LLM심판   : {summary['mean_llm_judge']:.3f}")

    if summary["retrieval"]:
        print("\n[검색]")
        for k, v in summary["retrieval"].items():
            if isinstance(v, str):
                disp = v
            elif ("rate" in k or "hit@" in k):
                disp = f"{v*100:.1f}%"
            else:
                disp = f"{v:.3f}"
            print(f"  - {k:22s}: {disp}")

    if summary.get("diagnosis"):
        print("\n[진단]")
        for k, v in summary["diagnosis"].items():
            print(f"  - {k:22s}: {v}")

    print("\n[유형별]")
    print(f"  {'type':10s} {'n':>3s} {'pass':>7s} {'kw_recall':>10s}")
    for t, s in summary["by_type"].items():
        sub_pass = sum(1 for r in rows if r.get("type") == t and r.get("pass"))
        print(
            f"  {t:10s} {s['count']:>3d} "
            f"{s['pass_rate']*100:>6.1f}% ({sub_pass}/{s['count']}) "
            f"{s['mean_kw_recall']*100:>9.1f}%"
        )

    fails = [r for r in rows if not r["pass"]]
    if fails:
        print(f"\n[실패 문항 {len(fails)}개 미리보기]")
        for r in fails[:10]:
            print(f"  · {r['id']}({r['type']}) diag={r.get('diag')} note={r['note']} kw={r['kw_recall']} "
                  f"hit={r.get('gold_hit_rank')} cand={r.get('gold_candidate_rank')} Q={r['query'][:34]}")
    print("=" * 68)


def write_outputs(rows: list[dict], summary: dict, csv_path: str, json_path: str) -> None:
    """상세 결과(CSV)와 요약+원본(JSON)을 파일로 저장한다.

    Args:
        rows: 개별 채점 행.
        summary: 집계 결과.
        csv_path: CSV 출력 경로.
        json_path: JSON 출력 경로.

    Note:
        Why 두 형식을 모두 쓰는가:
          - CSV : 사람이 Excel 로 정렬·필터해 실패 원인을 훑기 좋다.
          - JSON: summary 와 rows 를 함께 담아 **기계가 다시 읽어** 실행 간
                  비교(어제 vs 오늘)를 자동화하기 좋다.
          용도가 다르므로 하나로 합치지 않는다.

        cols 목록을 명시적으로 고정한 이유: 컬럼 순서를 안정시켜 여러 실행의
        CSV 를 diff 할 수 있게 하고, row 에 나중에 추가된 임시 키가 새어 나가지
        않게 한다. `r.get(k, "")` 로 누락 키도 안전하게 처리한다.

        utf-8-sig(BOM) + newline='' 는 한국어 Windows Excel 호환의 표준 조합이다.
    """
    cols = ["id", "type", "issue", "query", "answer", "kw_recall", "missing_terms",
            "hallucination", "bad_terms", "context_hit", "retrieval_rank",
            "retrieval_found", "match_method",
            "gold_hit_rank", "gold_hit_score", "gold_candidate_rank", "gold_candidate_score",
            "expanded_query", "diag", "run_status",
            "sem_sim", "llm_judge", "gold_doc_id", "pass", "note"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n[WRITE] {csv_path}\n[WRITE] {json_path}", file=sys.stderr)


def _is_transient(msg: str) -> bool:
    """429/503 등 '잠깐 기다리면 풀리는' 일시 오류인가(이런 것만 재시도).

    Args:
        msg: 오류 메시지 또는 답변 문자열.

    Returns:
        일시 오류로 보이면 True.

    Note:
        ★ 이 구분이 평가 신뢰성의 핵심이다(Why):
          재시도해야 할 것은 **인프라 문제**뿐이다. 모델이 내놓은 '틀린 답'을
          재시도하면, 여러 번 굴려 운 좋게 맞은 답을 채점하게 되어 점수가
          부풀려진다. 그건 평가가 아니라 자기기만이다.
          그래서 "일시 오류처럼 보이는 문자열"에만 재시도를 허용한다.

        ★ 한계 — 문자열 휴리스틱이다:
          예외 타입이 아니라 메시지 문자열을 보므로, 모델이 답변 본문에
          '429' 같은 숫자를 우연히 쓰면 오탐할 수 있다. 그래서 호출부
          (run_item_with_retry)는 '[실행오류] 로 시작' 또는 '연동 오류 포함'
          같은 **추가 조건**과 AND 로 묶어 오탐을 줄인다.
    """
    from errors import is_transient as _shared_is_transient
    return _shared_is_transient(msg)


def _retry_seconds(msg: str) -> float | None:
    """오류 메시지에서 'retry in 56s' 같은 대기 힌트를 초 단위로 읽는다.

    Args:
        msg: 오류 메시지.

    Returns:
        초 단위 대기 시간. 힌트가 없거나 파싱 실패면 None.

    Note:
        서버가 알려 준 대기 시간을 쓰는 것이 지수 백오프보다 항상 낫다(Why):
        서버는 할당량 창이 언제 리셋되는지 정확히 알고 있다. 임의로 5초를 쉬면
        너무 이르면 또 막히고, 너무 늦으면 시간을 낭비한다.
        힌트가 없을 때만 백오프로 폴백한다.
    """
    from errors import retry_after_seconds
    return retry_after_seconds(msg)


async def run_item_with_retry(orch: "RAGOrchestrator", RAGState: type["RAGStateType"],
                              prompts: "PromptsConfig", req: "QueryRequestModel",
                              args: argparse.Namespace,
                              mode: str,
                              *,
                              batch_id: str,
                              item_id: str) -> "AskRun":
    """한 문항을 실행하되, 일시 오류(429/503)면 잠깐 쉬고 재시도한다.

    매 시도마다 새 AskRun 을 만들어 이전 시도의 검색 스냅샷과 섞이지 않게 한다.
    """
    max_retries = getattr(args, "max_retries", 4)
    last: Any = None
    for attempt in range(1, max_retries + 1):
        run_id = item_id if attempt == 1 else f"{item_id}-try{attempt}"
        run = make_ask_run(req, batch_id=batch_id, item_id=run_id)
        try:
            if mode == "pipeline":
                run = await run_pipeline(orch, prompts, req, run)
            else:
                run = await run_nodes(orch, RAGState, prompts, req, run)
        except Exception as e:
            run.generation.answer = f"[실행오류] {e}"
            run.mark(status="error", error=str(e), where="harness")
            run.finish()

        answer = answer_from_run(run)
        transient = answer.startswith("[실행오류]") and _is_transient(answer)
        transient = transient or (_is_transient(answer) and "연동 오류" in answer)
        last = run
        if not transient or attempt >= max_retries:
            return run

        wait = _retry_seconds(answer) or (5.0 * attempt)
        wait = min(wait + 1.0, 65.0)
        print(f"      ↻ 일시 오류 감지 → {wait:.0f}초 대기 후 재시도 ({attempt}/{max_retries-1})",
              file=sys.stderr)
        await asyncio.sleep(wait)
    return last


def bootstrap_environment(args: argparse.Namespace) -> "PromptsConfig":
    """웹서버 lifespan 과 같은 boot_core() 로 평가 환경을 연다.

    Postgres/카탈로그/프롬프트는 서버와 동일해야 is_allowed_model_id() 가
    네트워크 fallback 으로 붕괴하지 않는다. Neo4j·게이트는 평가 전용으로 여기 남긴다.
    """
    from bootstrap import boot_core
    from database.neo4j_repo import Neo4jRepo
    from local_llm_gate import init_gate

    try:
        prompts = boot_core(prompt_cli=args.prompt_file, fail_fast_prompts=False)
        print("[BOOT] Postgres 풀·모델 카탈로그·프롬프트 초기화 완료", file=sys.stderr)
    except Exception as e:
        print(f"[BOOT][FATAL] 핵심 부팅 실패: {e}", file=sys.stderr)
        raise

    try:
        Neo4jRepo.init_driver()
        print("[BOOT] Neo4j 드라이버 초기화 완료", file=sys.stderr)
    except Exception as e:
        print(f"[BOOT][WARN] Neo4j 초기화 실패(그래프 미사용이면 무시): {e}", file=sys.stderr)

    try:
        init_gate()
    except Exception:
        pass

    return prompts


# =============================================================================
# 6) 메인
# =============================================================================
async def amain(args: argparse.Namespace) -> None:
    """평가 배치의 비동기 진입점.

    Args:
        args: CLI 인자.

    Raises:
        SystemExit: 테스트셋이 비었을 때.

    Note:
        ★ RAG 모듈을 여기서 import 하는 이유(Why):
          api_server / workflow_orchestrator 는 import 만으로도 무거운 초기화를
          유발한다(모델 로딩, 설정 파싱 등). 모듈 최상단에서 import 하면
          `--help` 조차 수 초가 걸리고, 의존성이 없는 환경에서는 도움말도 못 본다.
          **실제로 평가를 시작하는 시점**에 import 하는 것이 맞다.

        ★ 진행 로그를 문항마다 한 줄씩 stderr 로 찍는다:
          긴 배치(수십 분)에서 '지금 어디까지 갔는지'를 알 수 없으면, 멈춘 건지
          느린 건지 판단할 수 없다. kw/found/halluc 을 함께 보여 줘 중간에도
          경향을 파악할 수 있게 했다.

        ★ finally 에서 DB 자원을 정리한다:
          평가가 실패하든 성공하든 커넥션 풀과 드라이버는 닫아야 한다.
          서버의 lifespan 종료부에 대응하는 처리이며, 각각 try/except 로 감싼 것은
          한쪽 정리 실패가 다른 쪽 정리를 막지 않게 하기 위해서다.

        ★ --sleep 은 '다음 호출 전' 대기이며 마지막 문항 뒤에는 쉬지 않는다
          (`idx < len(items)`). 배치 끝에서 불필요하게 기다리지 않기 위함이다.
    """
    # 실제 RAG 모듈(실행 환경에 존재).
    # [ADR-006] api_server 역방향 의존을 제거하고 순수 계약/저장소 모듈을 직접 import한다.
    from contracts import QueryRequest
    from database.neo4j_repo import Neo4jRepo
    from database.postgres_repo import PostgresRepo
    from rag_run import set_server_capture
    from workflow_orchestrator import get_orchestrator, RAGState

    set_server_capture("full")
    prompts = bootstrap_environment(args)

    items = load_testset(args.testset)
    if not items:
        print("[ERROR] 테스트셋이 비었습니다.", file=sys.stderr)
        sys.exit(1)
    problems = validate_testset(items)
    if problems:
        for msg in problems:
            print(f"[TESTSET] {msg}", file=sys.stderr)
        if not getattr(args, "allow_dirty_testset", False):
            print(
                "[ERROR] 골든셋 검증 실패. --allow-dirty-testset 로 강행하거나 수정하세요.",
                file=sys.stderr,
            )
            sys.exit(2)
    if args.limit:
        items = items[: args.limit]

    from llm_handler import exit_if_invalid_model, get_llm_controller
    from model_registry import indexed_embedding_id, resolve_embed_choice
    exit_if_invalid_model(args.model, label="생성 모델")
    if args.judge:
        exit_if_invalid_model(args.judge_model or args.model, label="심판 모델")
    try:
        enc, var = resolve_embed_choice(
            args.embed_model or os.getenv("RAG_EMBEDDING_MODEL_ID"),
            args.embed_variant,
        )
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    if var:
        print(f"[BOOT] 검색 임베딩(비교 테이블): {var}", file=sys.stderr)
    else:
        print(f"[BOOT] 검색 임베딩(기본/원본 컬럼): {enc}", file=sys.stderr)
    from database.postgres_repo import normalize_sources
    try:
        src = normalize_sources(getattr(args, "sources", None))
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[BOOT] 검색 테이블: {', '.join(src)}", file=sys.stderr)
    try:
        with PostgresRepo() as repo:
            repo.ensure_search_tables(src, var)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 검색 테이블 확인 실패: {e}", file=sys.stderr)
        sys.exit(1)

    # 오케스트레이터는 모델별로 캐시되므로 루프 밖에서 한 번만 얻는다.
    try:
        orch = get_orchestrator(args.model)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    semantic = _Semantic(indexed_embedding_id()) if args.semantic else None
    judge_llm = None
    if args.judge:
        jmid = args.judge_model or args.model
        if jmid == args.model:
            print("[WARN] 심판 모델이 생성 모델과 같습니다. --judge-model 로 분리하세요.",
                  file=sys.stderr)
            judge_llm = orch.llm
        else:
            from llm_handler import get_llm_controller
            judge_llm = get_llm_controller(jmid)
            print(f"[JUDGE] 심판 모델: {jmid}", file=sys.stderr)

    batch_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    dump_root = args.dump_dir or os.path.join(_HARNESS_DIR, "logs", "testruns")
    batch_dir = os.path.join(dump_root, batch_id)
    os.makedirs(os.path.join(batch_dir, "cases"), exist_ok=True)
    csv_path = args.out_csv if os.path.isabs(args.out_csv) or os.path.dirname(args.out_csv) \
        else os.path.join(batch_dir, args.out_csv)
    json_path = args.out_json if os.path.isabs(args.out_json) or os.path.dirname(args.out_json) \
        else os.path.join(batch_dir, args.out_json)
    print(f"테스트셋 {len(items)}문항 | 모드={args.mode} | 모델={args.model} | dump={batch_dir}",
          file=sys.stderr)

    rows = []
    t0 = time.time()
    started_at = _utc_now()
    try:
        for idx, item in enumerate(items, 1):
            req = build_request(QueryRequest, item["query"], args)
            _assert_eval_request(req)
            run = await run_item_with_retry(
                orch, RAGState, prompts, req, args, args.mode,
                batch_id=batch_id, item_id=str(item["id"]))
            context = context_from_run(run)
            answer = answer_from_run(run)
            ids = ids_from_hits(getattr(getattr(run, "retrieval", None), "hits", None))

            row = score_item(item, context, answer, ids, args, semantic, orch,
                             judge_llm=judge_llm, run=run)
            dump_case(batch_dir, item, row, run, prompts)
            rows.append(row)
            flag = "✅" if row["pass"] else "❌"
            print(f"  [{idx}/{len(items)}] {flag} {item['id']} "
                  f"kw={row['kw_recall']} found={row.get('retrieval_found')}({row.get('match_method')}) "
                  f"diag={row.get('diag')} halluc={row['hallucination']}",
                  file=sys.stderr)
            if (not row["pass"]) and getattr(args, "print_failures", True):
                print_failure_detail(item, row, run)
            if getattr(args, "sleep", 0) and idx < len(items):
                await asyncio.sleep(args.sleep)

        summary = aggregate(rows, args)
        summary["elapsed_sec"] = round(time.time() - t0, 1)
        summary["started_at"] = started_at
        summary["batch_id"] = batch_id
        summary["dump_dir"] = batch_dir
        prompt_used = args.prompt_file or os.getenv("DEFAULT_PROMPT_FILE")
        probe = build_request(QueryRequest, "(config-snapshot)", args)
        summary["config"] = snapshot_harness_config(args, probe, prompt_used)
        print_report(summary, rows)
        write_outputs(rows, summary, csv_path, json_path)
        write_manifest(batch_dir, args, summary, args.testset, len(items), rows)
    finally:
        # 프로세스 종료 전 DB 자원 정리(서버 lifespan 종료부 대응)
        try:
            PostgresRepo.close_pool()
        except Exception:
            pass
        try:
            Neo4jRepo.close_driver()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    """명령행 인자를 정의하고 파싱한다.

    Returns:
        파싱된 인자.

    Note:
        옵션 그룹의 설계 의도:

        [실행 제어] --testset/--mode/--model/--limit/--prompt-file/--sleep/--max-retries
          --limit 은 스모크 테스트용이다. 40문항 배치를 돌리기 전에 3문항으로
          배관이 살아 있는지 확인하는 습관이 시간을 아낀다.

        [채점 기준] --kw-threshold / --require-context(--no-require-context)
          기본 0.5 는 "핵심 토큰의 절반 이상"이라는 뜻이다. 엄격하게 보려면 0.8 로
          올린다. 단, **비교하려는 실행들끼리는 같은 값**이어야 한다.
          --require-context 기본 on 은 "근거에서 나온 답만 정답으로 친다"는 선언이다.
          RAG 를 평가하는 것이지 LLM 의 암기력을 평가하는 게 아니기 때문이다.

        [검색 판정] --no-id-check
          기본 : 검색 결과의 조문 id 를 정답 id 와 대조(정확). hit@k/MRR 산출.
          지정 : id 를 무시하고 본문 겹침으로만 판정(검색기가 id 를 못 줘도 동작).

        [선택 채점기] --semantic / --judge
          비용이 있으므로 기본 off. 필요할 때만 켠다.

        [RAG 파라미터] --top-k/--fusion/--alpha/...
          전부 기본 None = '서비스 기본값 사용'. 검색 튜닝 실험을 이 하네스로
          바로 돌릴 수 있게 하려는 것이다(코드 수정 없이 A/B).

        `store_true`/`store_false` 쌍(--use-reranker/--no-reranker)을 둔 이유:
          기본값을 None 으로 유지한 채 '켜기/끄기'를 모두 명시할 수 있어야
          3-상태(미지정/켬/끔)를 표현할 수 있기 때문이다.
    """
    p = argparse.ArgumentParser(description="세무 RAG 평가 하네스")
    p.add_argument("--testset", default="", help="build_testset.py 산출 JSON")
    p.add_argument("--mode", choices=["nodes", "pipeline"], default="nodes")
    p.add_argument("--model",
                   default=os.getenv("RAG_EVAL_MODEL") or os.getenv("DEFAULT_LLM", "local:qwen2.5-3b-instruct-q4_k_m.gguf"),
                   help="생성 LLM (local:... 또는 openrouter:...). 기본=DEFAULT_LLM")
    p.add_argument("--limit", type=int, default=0, help="상위 N문항만(스모크 테스트)")
    p.add_argument("--prompt-file", dest="prompt_file", default=None,
                   help="RAG 프롬프트 JSON 경로(main_answer 등). 미지정 시 .env의 DEFAULT_PROMPT_FILE 사용")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="문항 사이 대기(초). 무료 등급 분당 한도(429) 예방용. 예: --sleep 4")
    p.add_argument("--max-retries", type=int, default=4, dest="max_retries",
                   help="429/503 등 '일시 오류' 시 최대 재시도 횟수(기본 4). 모델의 실제 오답은 재시도 안 함")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=None,
                   help="답변 최대 토큰")
    p.add_argument("--temperature", type=float, default=None,
                   help="생성 temperature")

    # 채점 기준
    p.add_argument("--kw-threshold", type=float, default=0.5, dest="kw_threshold",
                   help="핵심어 재현율 통과 기준(0~1)")
    p.add_argument("--require-context", dest="require_context", action="store_true", default=True,
                   help="검색성공을 pass 조건에 포함(기본 on). 컨텍스트에 정답이 있어야 통과")
    p.add_argument("--no-require-context", dest="require_context", action="store_false")

    # 검색 정답 판정 방법: 조문 id 대조(기본) vs 본문 겹침
    #   기본        : 검색 결과의 조문 id 를 정답 id 와 대조(정확). hit@k/MRR 산출.
    #   --no-id-check : id 를 무시하고 본문 겹침으로만 판정(검색기가 id 를 못 줘도 동작).
    p.add_argument("--no-id-check", dest="id_check", action="store_false", default=True,
                   help="조문 id 대조를 끄고 본문 겹침으로만 검색 정답 판정(기본: id 대조 ON)")

    # 선택 채점기
    p.add_argument("--semantic", action="store_true",
                   help="답변-정답 의미유사도 측정. 기본 indexed 임베딩을 쓰며 검색 모델과 별개")
    p.add_argument("--judge", action="store_true", help="LLM 심판 채점(evaluate_ask 재사용)")
    p.add_argument("--judge-model", dest="judge_model", default=None,
                   help="심판 모델 (미지정 시 --model 과 동일)")

    # RAG 파라미터 오버라이드 (미지정 시 QueryRequest 기본값 사용)
    p.add_argument("--top-k", type=int, default=None, dest="top_k")
    p.add_argument("--expansion", action="store_true",
                   help="질문 확장을 켠다 (--query-mode expansion). 기본 OFF")
    p.add_argument("--query-mode", default=None, dest="query_mode",
                   choices=["none", "expansion", "vocab_mapping"])
    p.add_argument("--keyword-mode", default=None, dest="keyword_mode",
                   choices=["trigram", "kiwi"],
                   help="키워드 추출. 기본 trigram. kiwi 는 형태소 분석(kiwipiepy).")
    p.add_argument("--fusion", default=None, choices=[None, "rrf", "score_based"])
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--embed-model", default=None, dest="embed_model",
                   help="검색 임베딩. 기본(indexed)이면 각 원본 embedding 컬럼. "
                        "그 외 카탈로그 모델이면 고른 테이블마다 비교 테이블을 쓴다.")
    p.add_argument("--embed-variant", dest="embed_variant", default=None,
                   help="[호환] 비교 테이블 모델 id. --embed-model 에 비교 모델을 "
                        "줘도 같다. 기본 모델과 다른 비교 모델을 동시에 주면 에러.")
    p.add_argument("--sources", default=None,
                   help="쉼표로 구분한 검색 테이블. 기본은 법령/판례/심판/개정세법 전부. "
                        "예: tax_laws_master,tax_amendments_2026")
    p.add_argument("--rerank-model", default=None, dest="rerank_model")
    p.add_argument("--min-score", type=float, default=None, dest="min_score")
    p.add_argument("--rerank", dest="use_reranker", action="store_true", default=None,
                   help="리랭킹을 켠다 (--use-reranker 와 동일). 기본 OFF")
    p.add_argument("--use-reranker", dest="use_reranker", action="store_true", default=None)
    p.add_argument("--no-reranker", dest="use_reranker", action="store_false")
    p.add_argument("--use-guardrails", dest="use_guardrails", action="store_true", default=None)
    p.add_argument("--no-guardrails", dest="use_guardrails", action="store_false")
    p.add_argument("--use-graphdb", dest="use_graphdb", action="store_true", default=None)
    p.add_argument("--no-graphdb", dest="use_graphdb", action="store_false")

    p.add_argument("--trace-prompts", dest="trace_prompts", action="store_true", default=True,
                   help="프롬프트 전문을 로그에 남긴다(하네스 기본 on)")
    p.add_argument("--no-trace-prompts", dest="trace_prompts", action="store_false")
    p.add_argument("--print-failures", dest="print_failures", action="store_true", default=True,
                   help="실패한 문항의 검색 진단을 화면에 길게 출력(기본 on)")
    p.add_argument("--no-print-failures", dest="print_failures", action="store_false")
    p.add_argument("--dump-dir", dest="dump_dir", default=None,
                   help="실행 산출물 루트(기본: <프로젝트>/logs/testruns)")

    # 출력 경로 (상대 파일명은 실행 폴더 아래)
    p.add_argument("--out-csv", default="eval_report.csv", dest="out_csv")
    p.add_argument("--out-json", default="eval_summary.json", dest="out_json")
    p.add_argument("--self-test", action="store_true", help="채점 헬퍼 단위검증 후 종료")
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="골든셋 검증만 수행하고 종료 (DB·LLM 불필요)",
    )
    p.add_argument(
        "--allow-dirty-testset",
        action="store_true",
        help="골든셋 검증 경고가 있어도 평가를 계속한다",
    )
    return p.parse_args()


def _self_test() -> None:
    """DB·LLM 없이 순위/집계 헬퍼만 검증한다."""
    rnk, sc = rank_and_score(
        [{"doc_id": "a", "score": 0.9}, {"doc_id": "b", "score": 0.31}],
        "b",
    )
    assert rnk == 2 and sc == 0.31, (rnk, sc)
    rnk0, sc0 = rank_and_score([{"doc_id": "a", "score": 0.9}], "z")
    assert rnk0 == 0 and sc0 is None

    class _A:
        mode = "nodes"; model = "x"; kw_threshold = 0.5
        require_context = True; id_check = True; top_k = 3
    rows = [
        {"type": "exact", "pass": True, "hallucination": False, "kw_recall": 1.0,
         "context_hit": True, "retrieval_found": True, "match_method": "id",
         "retrieval_rank": 2, "gold_candidate_rank": 2, "sem_sim": "", "llm_judge": "",
         "diag": "ok"},
        {"type": "exact", "pass": False, "hallucination": False, "kw_recall": 0.0,
         "context_hit": False, "retrieval_found": False, "match_method": "id",
         "retrieval_rank": 0, "gold_candidate_rank": 7, "sem_sim": "", "llm_judge": "",
         "diag": "cut_by_topk"},
    ]
    ret = aggregate(rows, _A())["retrieval"]
    assert ret["hit@3"] == 0.5 and ret["hit@10"] == 0.5, ret
    assert ret["pool_hit@3"] == 0.5 and ret["pool_hit@10"] == 1.0, ret
    assert ret["cut_by_topk_rate"] == 0.5, ret
    snap = snapshot_harness_config(_A(), None, "prompts.json")
    assert snap["mode"] == "nodes" and snap["capture"] == "full"
    assert snap["cli_overrides"]["top_k"] == 3
    assert "resolved" not in snap

    # 음성 통제: 떨어뜨려야 할 것을 떨어뜨리는가
    rec, _ = kw_recall("연차는 15일입니다.", ["30일", "평균임금"])
    assert rec == 0.0, rec
    hallu, hit = has_hallucination("제999조에 따라 지급한다.", ["제999조"])
    assert hallu and hit == ["제999조"]
    assert not context_hit(
        "전혀 무관한 문장",
        {"gold_context": "휴가를 주어야 한다", "gold_article_number": "59"},
    )
    assert retrieval_rank([("tax_laws_master:1", "법령", "조")], "tax_laws_master:99") == 0

    probs = validate_testset([
        {
            "id": "T1", "type": "exact", "query": "q",
            "must_include": ["183일"], "gold_doc_id": "tax_laws_master:1",
            "gold_context": "183일",
        },
    ])
    assert probs == []
    bad = validate_testset([
        {"id": "T2", "type": "exsact", "query": "q", "must_include": [], "gold_doc_id": "1"},
    ])
    assert any("type" in p for p in bad)
    assert any("must_include" in p for p in bad)

    lo, hi = wilson_ci(24, 40)
    assert 0.0 <= lo < hi <= 1.0

    print("[self-test] evaluate_rag scoring OK")


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        _self_test()
        sys.exit(0)
    if args.validate_only:
        if not args.testset:
            print("[ERROR] --validate-only 에는 --testset 이 필요합니다.", file=sys.stderr)
            sys.exit(1)
        items = load_testset(args.testset)
        problems = validate_testset(items)
        if problems:
            for msg in problems:
                print(f"[TESTSET] {msg}", file=sys.stderr)
            sys.exit(1)
        print(f"[OK] 골든셋 {len(items)}문항 검증 통과", file=sys.stderr)
        sys.exit(0)
    if not args.testset:
        print("[ERROR] --testset 이 필요합니다.", file=sys.stderr)
        sys.exit(1)
    asyncio.run(amain(args))
