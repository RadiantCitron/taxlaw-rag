# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
import requests
import json
import argparse
import sys
import os
import shlex
import threading
import time
import queue as queue_mod
from dotenv import load_dotenv

# 터미널에서 다중 줄 입력 및 Ctrl+Enter 감지를 위한 라이브러리 (선택적 의존성)
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
except ImportError:
    PromptSession = None
    KeyBindings = None

# =====================================================================
# [학습용 명세서] RAG 파이프라인 CLI 제어 파라미터 총정리
# =====================================================================
"""
세무 RAG CLI. FastAPI 서버(api_server.py)로 JSON을 보내 파이프라인을 제어한다.
이 프로젝트는 Postgres(법령/판례/심판/개정세법 해설) + 옵션 Neo4j 그래프다.
검색 테이블은 --sources 로 고른다. 비우면 네 테이블 전부. 그래프는 --use-graph.

실행 시 `python test_api.py --help` 또는 질문창에 `--help` 를 치면 옵션 설명을 본다.
`--question` 을 주면 대화창 없이 질문을 보내고 답변을 출력한 뒤 종료한다. 여러 번 지정하면 순서대로 보낸다.
질문창에서 줄이 `--` 로 시작하면 질문이 아니라 설정 변경이다. 잘못된 토큰만 무시하고 나머지를 적용한다.

[1. General Options (기본 설정)]
  --llm             : 답변 생성 및 질문 확장에 사용할 메인 LLM.
                      서버 GET /llm/models 목록의 id. OpenRouter 는 openrouter: 접두사 필수.
  --embed-model     : 검색 임베딩. 기본(indexed)은 각 원본 embedding 컬럼.
                      다른 카탈로그 모델이면 고른 테이블마다 비교 테이블(cmp)을 검색한다.
  --sources         : 쉼표로 구분한 검색 테이블. 비우면 네 테이블 전부.
                      tax_laws_master,tax_precedents_master,tax_tribunal_master,tax_amendments_2026
  --log-level       : 백엔드 로그 상세도 (1: Error, 3: Info, 5: Debug). 세션 JSONL 로거 하한.
  --query-mode      : 질문 최적화 ('none': 원문 검색(기본), 'expansion': 질문 확장, 'vocab_mapping': 단어장 매핑).
  --keyword-mode    : 키워드 추출 ('trigram' 기본, 'kiwi' 형태소). vocab_mapping 에서는 kiwi 가 무시된다.
  --expansion       : 질문 확장을 켠다 (--query-mode expansion 과 동일). 기본 OFF.
  --base-limit      : 테이블당 hybrid 검색 SQL LIMIT. 기본 20, 1~100.
  --use-graph       : Neo4j 그래프 보충 컨텍스트를 켠다. 기본 OFF. 페이로드 키는 use_graphdb.
  --no-use-graph    : 그래프 보충을 끈다.
  --stream-logs     : 서버 로그 SSE 를 클라로 보낸다 (기본 ON).
  --no-stream-logs  : 로그 SSE 만 끈다. 세션 파일·답변 스트림은 유지.
  --capture         : 턴 캡처 lean|full|forensic. 생략 시 서버 기본. 서버 상한을 넘지 못함.
  --question        : 질문을 보내고 답변을 출력한 뒤 종료. 여러 번 지정 가능. 없으면 대화형.

[2. Rerank Options (리랭킹 설정)]
  --rerank          : CrossEncoder 리랭킹을 켠다. 기본 OFF.
  --no-rerank       : 리랭킹을 끈다 (기본과 동일).
  --top-k           : 리랭킹 후 최종적으로 LLM 컨텍스트에 주입할 상위 문서의 개수.
  --rerank-model    : 리랭킹에 사용할 허깅페이스 모델 경로.
  --rerank-candidates : CrossEncoder 에 넣을 최대 후보 수 (1~50, 기본 20).
  --min-score       : 이 점수 미만의 문서는 노이즈로 간주하여 원천 배제함 (0.0 ~ 1.0).

[3. Eval Options (DeepEval 평가 설정)]
  --eval            : 답변 생성 후 품질 평가를 켠다. 기본 OFF.
  --no-eval         : 품질 평가를 끈다 (기본과 동일).
  --eval-model      : 채점 LLM. GET /llm/models 의 id 또는 local:<파일명>. 'local' 별칭은 불가.
  --eval-thres      : 합격 커트라인 점수 (0.0 ~ 1.0).
  --eval-strict     : 사소한 오류나 환각도 가차 없이 0점 처리하는 엄격한 채점 모드 켜기.
  --no-eval-strict  : 엄격한 채점 모드 끄기 (기본값).
  --eval-metrics    : 평가 과목 ('answer_relevancy': 답변 관련성, 'context_precision': 검색 정밀도, 'all': 모두).
"""
# argparse choices 는 UX 편의일 뿐 서버 검증을 대체하지 않는다.
# =====================================================================
from error_envelope import (
    envelope_code,
    exit_code_for,
    exit_code_from_done,
    exit_code_from_envelope,
    unwrap_envelope,
)
from validation import BASE_LIMIT_DEFAULT, BASE_LIMIT_MAX, BASE_LIMIT_MIN


def _base_limit_invalid(args) -> str | None:
    bl = int(getattr(args, "base_limit", BASE_LIMIT_DEFAULT) or BASE_LIMIT_DEFAULT)
    if bl < BASE_LIMIT_MIN or bl > BASE_LIMIT_MAX:
        return f"--base-limit 는 {BASE_LIMIT_MIN}~{BASE_LIMIT_MAX} 사이여야 합니다."
    return None


class EnvelopeError(Exception):
    def __init__(self, env: dict, status: int | None = None):
        self.env = env
        self.status = status
        super().__init__(envelope_code(env) or "error")


def render_error_envelope(env: dict) -> None:
    err = (env or {}).get("error") or {}
    print("\n" + "=" * 60)
    print(f"[요청 거부] {err.get('code') or '?'}")
    print(err.get("message") or "")
    print("-" * 60)
    for i, v in enumerate(err.get("violations") or [], 1):
        print(f"{i}) {v.get('field')}")
        print(f"   보낸 값 : {v.get('given')}")
        print(f"   사유    : {v.get('reason')}")
        expected = v.get("expected") or {}
        kind = expected.get("type")
        if kind == "enum" and expected.get("allowed") is not None:
            print(f"   허용    : {', '.join(str(x) for x in expected.get('allowed') or [])}")
        elif kind in ("float", "int"):
            print(f"   허용 범위: {expected.get('min')} 이상 {expected.get('max')} 이하 ({kind})")
        elif kind == "model_ref":
            if expected.get("allowed"):
                print(f"   허용    : {', '.join(str(x) for x in expected.get('allowed') or [])}")
            else:
                print(
                    f"   허용    : {expected.get('source') or ''} 에서 조회 "
                    f"({expected.get('allowed_count') or '?'}개)"
                )
                sample = expected.get("allowed_sample") or []
                if sample:
                    print(f"             예: {', '.join(str(x) for x in sample)}")
            if expected.get("rule"):
                print(f"   규칙    : {expected.get('rule')}")
        if "default" in v:
            print(f"   기본값  : {v.get('default')}")
        if v.get("hint"):
            print(f"   설명    : {v.get('hint')}")
        print()
    ignored = err.get("ignored_fields") or []
    if ignored:
        print("무시된 옵션: " + ", ".join(str(x) for x in ignored))
    print(f"전체 옵션 명세: {err.get('docs') or 'GET /options/schema'}")
    print("=" * 60)


def render_eval_report(obj: dict) -> None:
    """`--no-stream-logs` 여도 평가 결과는 화면에 남긴다."""
    if not isinstance(obj, dict):
        print(f"\n[EVAL] {obj}", flush=True)
        return
    if obj.get("error"):
        print(f"\n[EVAL 오류] {obj['error']}", flush=True)
    rows = obj.get("report") or []
    if not rows and not obj.get("error"):
        print("\n[EVAL] 결과 없음", flush=True)
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = "PASS" if row.get("pass") else "FAIL"
        metric = row.get("metric") or "?"
        score = row.get("score")
        try:
            score_s = f"{float(score):.2f}"
        except (TypeError, ValueError):
            score_s = str(score)
        print(f"\n[EVAL] {metric}: {score_s}점 ({status})", flush=True)
        reason = (row.get("reason") or "").strip()
        if reason:
            print(f"  ↳ {reason}", flush=True)
    if rows or obj.get("error"):
        print(flush=True)


def decode_sse_data(raw: str) -> str:
    """SSE data 줄을 본문 텍스트로. JSON {text} 또는 구형 <br> 평문."""
    text = (raw or "").strip()
    if not text:
        return ""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "text" in obj:
            return str(obj.get("text") or "")
    except json.JSONDecodeError:
        pass
    return text.replace("<br>", "\n")

def spinner_animation():
    """서버 응답을 기다리는 동안 회전하는 바를 보여주는 함수"""
    spinner_chars = ['|', '/', '-', '\\']
    idx = 0
    while is_waiting:
        # \r을 사용하여 줄바꿈 없이 제자리에서 글자를 덮어씁니다.
        # flush=True를 주어 버퍼에 담지 않고 즉시 화면에 쏘도록 강제합니다.
        sys.stdout.write(f"\r[System] 서버 응답 대기 중... {spinner_chars[idx % 4]}")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)
    
    # 대기가 끝나면 스피너 글자를 공백으로 덮어써서 깔끔하게 지웁니다.
    sys.stdout.write("\r" + " " * 40 + "\r")
    sys.stdout.flush()


def iter_sse_lines(url, payload, headers=None):
    """POST /ask 를 별도 스레드에서 읽고, 메인에서 Ctrl+C 로 소켓만 끊을 수 있게 한다."""
    session = requests.Session()
    out = queue_mod.Queue()
    stop = threading.Event()

    def worker():
        try:
            resp = session.post(url, json=payload, stream=True, headers=headers)
            if not resp.ok:
                try:
                    body = resp.json()
                    env = unwrap_envelope(body) or (
                        body if isinstance(body, dict) and "error" in body else None
                    )
                    if env:
                        out.put(("envelope", (env, resp.status_code)))
                        out.put(("done", None))
                        return
                except ValueError:
                    pass
                resp.raise_for_status()
            for line in resp.iter_lines(chunk_size=1, decode_unicode=True):
                if stop.is_set():
                    break
                if line:
                    out.put(("line", line))
            out.put(("done", None))
        except Exception as e:
            if stop.is_set():
                out.put(("done", None))
            else:
                out.put(("err", e))
        finally:
            try:
                session.close()
            except Exception:
                pass

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    try:
        while True:
            try:
                kind, val = out.get(timeout=0.1)
            except queue_mod.Empty:
                continue
            if kind == "line":
                yield val
            elif kind == "envelope":
                env, status = val
                raise EnvelopeError(env, status)
            elif kind == "err":
                raise val
            else:
                return
    except KeyboardInterrupt:
        raise
    finally:
        stop.set()
        try:
            session.close()
        except Exception:
            pass


def start_presence(session_url: str):
    """질문 없이도 서버에 붙어 있는 세션을 연다."""
    holder = {"id": None, "error": None}
    stop = threading.Event()
    http = requests.Session()
    resp_holder: dict = {}

    def _close_response() -> None:
        resp = resp_holder.get("resp")
        if resp is None:
            return
        try:
            resp.close()
        except Exception:
            pass
        resp_holder.pop("resp", None)

    def worker():
        try:
            resp = http.get(session_url, stream=True, timeout=(5, None))
            resp_holder["resp"] = resp
            http._close_presence_stream = _close_response
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if stop.is_set():
                    break
                if not line or not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if data.get("event") == "connected" and data.get("session_id"):
                    holder["id"] = data["session_id"]
        except Exception as e:
            holder["error"] = e
        finally:
            _close_response()
            try:
                http.close()
            except Exception:
                pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    for _ in range(50):
        if holder["id"] or holder["error"]:
            break
        time.sleep(0.1)
    if not holder["id"]:
        stop.set()
        _close_response()
        thread.join(timeout=2)
        err = holder["error"] or "세션 연결 시간 초과"
        raise RuntimeError(f"세션 접속 실패: {err}")
    return holder["id"], stop, http, thread


def make_repl_parser(launch_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """질문창용. 기본값을 넣지 않아, 이번 줄에 나온 옵션만 덮는다."""
    repl = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    skip_dest = {"questions"}
    for action in launch_parser._actions:
        if isinstance(action, argparse._HelpAction) or not action.option_strings:
            continue
        if action.dest in skip_dest:
            continue
        common = dict(dest=action.dest, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        if isinstance(action, argparse._StoreTrueAction):
            repl.add_argument(*action.option_strings, action="store_true", **common)
        elif isinstance(action, argparse._StoreFalseAction):
            repl.add_argument(*action.option_strings, action="store_false", **common)
        else:
            kwargs = dict(common)
            if action.type is not None:
                kwargs["type"] = action.type
            if action.choices is not None:
                kwargs["choices"] = action.choices
            repl.add_argument(*action.option_strings, **kwargs)
    return repl


def _option_value_count(action) -> int:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return 0
    if action.nargs == 0:
        return 0
    if isinstance(action.nargs, int):
        return action.nargs
    return 1


def apply_inline_options(text: str, args, launch_parser, repl_parser):
    """`--` 설정 줄을 토큰 단위로 적용. 잘못된 토큰만 건너뛴다."""
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError as exc:
        return False, [f"따옴표 오류: {exc}"], set()

    opt_map = {}
    for action in launch_parser._actions:
        for opt in action.option_strings:
            opt_map[opt] = action

    show_help = False
    errors = []
    applied = set()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("-h", "--help"):
            show_help = True
            i += 1
            continue
        if not tok.startswith("--"):
            errors.append(f"옵션이 아님: {tok}")
            i += 1
            continue
        if "=" in tok:
            flag, eq_val = tok.split("=", 1)
            action = opt_map.get(flag)
            if action is None:
                errors.append(f"알 수 없는 옵션: {tok}")
                i += 1
                continue
            chunk = [flag, eq_val]
            i += 1
        else:
            action = opt_map.get(tok)
            if action is None:
                errors.append(f"알 수 없는 옵션: {tok}")
                i += 1
                continue
            need = _option_value_count(action)
            if need > 0:
                if i + 1 >= len(tokens) or tokens[i + 1].startswith("-"):
                    errors.append(f"{tok} 값이 없습니다")
                    i += 1
                    continue
                chunk = [tok, tokens[i + 1]]
                i += 2
            else:
                chunk = [tok]
                i += 1
        try:
            ns = repl_parser.parse_args(chunk)
        except (argparse.ArgumentError, SystemExit) as exc:
            errors.append(f"{' '.join(chunk)} → {exc}")
            continue
        except Exception as exc:
            errors.append(f"{' '.join(chunk)} → {exc}")
            continue
        for key, val in vars(ns).items():
            setattr(args, key, val)
            applied.add(key)
            if key == "expansion" and val:
                args.query_mode = "expansion"
    return show_help, errors, applied


def main():
    load_dotenv()
        
    env_default_llm = os.getenv("DEFAULT_LLM")
    env_default_eval = os.getenv("DEFAULT_EVAL_LLM")
    env_default_embed = os.getenv("RAG_EMBEDDING_MODEL_ID", "jhgan/ko-sroberta-multitask")
    env_default_reranker = os.getenv("RAG_RERANKER_MODEL_ID", "Dongjin-kr/ko-reranker")

    # 🌟 argparse의 help 메시지를 주석과 완벽하게 동일한 수준으로 상세하게 작성
    parser = argparse.ArgumentParser(
        description="세무 RAG CLI. Postgres + 옵션 Neo4j(--use-graph). --db-mode 없음.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # [1. General Options]
    parser.add_argument("--llm", type=str, default=env_default_llm, 
                        help="메인 LLM. 서버 /llm/models 의 id (openrouter:... 또는 local:<gguf>)")
    parser.add_argument("--embed-model", type=str, default=env_default_embed, 
                        help="검색 임베딩. 기본(indexed)이면 원본 컬럼, 그 외 카탈로그 모델이면 비교 테이블")
    parser.add_argument("--sources", type=str, default="",
                        help="쉼표 구분 검색 테이블. 비우면 서버 기본(법령/판례/심판/개정 전부)")
    parser.add_argument("--log-level", type=int, choices=[1, 3, 5], default=3, 
                        help="백엔드에서 쏴주는 로그의 상세도 (1: Error, 3: Info, 5: Debug)")
    parser.add_argument("--query-mode", type=str, choices=["none", "expansion", "vocab_mapping"], default="none", 
                        help="질문 최적화 ('none': 원문(기본), 'expansion': 질문 확장, 'vocab_mapping': 단어장 매핑)")
    parser.add_argument("--keyword-mode", type=str, choices=["trigram", "kiwi"], default="trigram",
                        help="키워드 추출 ('trigram' 기본, 'kiwi' 형태소). vocab_mapping 에서는 kiwi 무시")
    parser.add_argument("--expansion", action="store_true",
                        help="질문 확장을 켠다 (--query-mode expansion). 기본 OFF")
    parser.add_argument("--base-limit", type=int, default=BASE_LIMIT_DEFAULT,
                        help=f"테이블당 hybrid 검색 LIMIT ({BASE_LIMIT_MIN}~{BASE_LIMIT_MAX}, 기본 {BASE_LIMIT_DEFAULT})")
    parser.add_argument("--guardrail", action="store_true", default=True,
                        help="가드레일(의미 통제)을 명시적으로 활성화함 (기본값)")
    parser.add_argument("--no-guardrail", action="store_false", dest="guardrail",
                        help="가드레일(의미 통제)을 비활성화함")
    parser.add_argument("--trace-prompts", action="store_true", default=False,
                        help="LLM에 실제로 넘긴 프롬프트를 클라이언트에 출력한다. 기본 OFF")
    parser.add_argument("--no-trace-prompts", action="store_false", dest="trace_prompts",
                        help="프롬프트 트레이스를 끈다")
    parser.add_argument("--fusion-mode", type=str, choices=["rrf", "score_based"], default="rrf", 
                        help="하이브리드 검색 융합 방식 ('rrf': 순위 기반, 'score_based': 점수 기반)")
    parser.add_argument("--alpha", type=float, default=0.5, 
                        help="Score-based 융합 시 벡터 검색 가중치 (0.0 ~ 1.0). 이상한 값이면 0.5 강제 적용")
    parser.add_argument("--use-graph", action="store_true",
                        help="이 플래그를 넣으면 Neo4j GraphDB를 연동하여 보충 컨텍스트를 가져옵니다.")
    parser.add_argument("--no-use-graph", action="store_false", dest="use_graph",
                        help="Neo4j 그래프 보충을 끈다 (기본)")
    parser.add_argument("--stream-logs", action="store_true", default=True,
                        help="서버 로그 SSE를 클라로 보낸다 (기본 ON)")
    parser.add_argument("--no-stream-logs", action="store_false", dest="stream_logs",
                        help="서버 로그 SSE를 끈다. 세션 파일·답변 스트림은 유지")
    parser.add_argument("--capture", type=str, choices=["lean", "full", "forensic"], default=None,
                        help="턴 캡처. 생략 시 서버 기본. 서버 상한을 넘지 못함. forensic 은 프롬프트+hit JSON")
    parser.add_argument("--id", type=str, default=None,
                        help="수업 로그인 아이디. /ask 헤더 X-User-Id")
    parser.add_argument("--question", action="append", dest="questions", default=None,
                        metavar="TEXT",
                        help="질문을 보내고 답변을 출력한 뒤 종료한다. 여러 번 지정하면 순서대로 보낸다. 없으면 대화형")
    
    # [2. Rerank Options]
    parser.add_argument("--rerank", action="store_true", default=False,
                        help="CrossEncoder 리랭킹을 켠다. 기본 OFF")
    parser.add_argument("--no-rerank", action="store_false", dest="rerank",
                        help="CrossEncoder 리랭킹을 끈다 (기본)")
    parser.add_argument("--top-k", type=int, default=3, 
                        help="리랭킹 후 최종적으로 LLM 컨텍스트에 주입할 상위 문서의 개수")
    parser.add_argument("--rerank-model", type=str, default=env_default_reranker, 
                        help="리랭킹에 사용할 허깅페이스 모델 경로")
    parser.add_argument("--rerank-candidates", type=int, default=20,
                        help="CrossEncoder 리랭킹 후보 상한 (1~50, 기본 20)")
    parser.add_argument("--min-score", type=float, default=0.0, 
                        help="이 점수 미만의 문서는 노이즈로 간주하여 원천 배제함 (0.0 ~ 1.0)")

    # [3. Eval Options]
    parser.add_argument("--eval", action="store_true", default=False,
                        help="답변 생성 후 DeepEval 품질 평가를 켠다. 기본 OFF")
    parser.add_argument("--no-eval", action="store_false", dest="eval",
                        help="품질 평가를 끈다 (기본)")
    parser.add_argument("--eval-model", type=str, default=env_default_eval, 
                        help="채점 LLM. /llm/models 의 id (openrouter:... 또는 local:<파일명>)")
    parser.add_argument("--eval-thres", type=float, default=0.5, 
                        help="합격 커트라인 점수 (0.0 ~ 1.0)")
    parser.add_argument("--eval-strict", action="store_true", default=False, 
                        help="사소한 오류나 환각도 가차 없이 0점 처리하는 엄격한 채점 모드 켜기")
    parser.add_argument("--no-eval-strict", action="store_false", dest="eval_strict", 
                        help="엄격한 채점 모드 끄기 (기본값)")
    parser.add_argument("--eval-metrics", type=str, choices=["answer_relevancy", "context_precision", "all"], default="all", 
                        help="평가 과목 ('answer_relevancy', 'context_precision', 'all')")
    parser.add_argument("--prompt-file", type=str, default=None, 
                        help="클라이언트 단에서 덮어쓸 커스텀 프롬프트 JSON 파일 경로")

    args = parser.parse_args()
    bad_bl = _base_limit_invalid(args)
    if bad_bl:
        print(f"\n[Error] {bad_bl}")
        sys.exit(1)
    if args.rerank and (args.rerank_candidates < 1 or args.rerank_candidates > 50):
        print("\n[Error] --rerank-candidates 는 1~50 사이여야 합니다.")
        sys.exit(1)
    if args.expansion:
        args.query_mode = "expansion"
    if args.questions is not None:
        args.questions = [q.strip() for q in args.questions if (q or "").strip()]
        if not args.questions:
            print("\n[Error] --question 값이 비어 있습니다.")
            sys.exit(1)
    repl_parser = make_repl_parser(parser)

    custom_prompts_payload = None

    def load_prompt_file(path: str | None) -> bool:
        nonlocal custom_prompts_payload
        if not path:
            custom_prompts_payload = None
            return True
        try:
            with open(path, "r", encoding="utf-8") as f:
                custom_prompts_payload = json.load(f)
            print(f"[System] 클라이언트 커스텀 프롬프트 로드 완료: {path}")
            return True
        except Exception as e:
            print(f"[Error] 프롬프트 파일 로드 실패: {e}")
            return False

    def build_payload(query_text: str | None) -> dict:
        general = {
            "llm_model": args.llm,
            "embed_model": args.embed_model,
            "log_level": args.log_level,
            "query_mode": args.query_mode,
            "keyword_mode": args.keyword_mode,
            "use_guardrails": args.guardrail,
            "trace_prompts": args.trace_prompts,
            "fusion_mode": args.fusion_mode,
            "alpha": args.alpha,
            "use_graphdb": args.use_graph,
            "stream_logs": args.stream_logs,
            "capture": args.capture,
            "base_limit": args.base_limit,
        }
        sources = [s.strip() for s in (args.sources or "").split(",") if s.strip()]
        if sources:
            general["sources"] = sources
        payload = {
            "query": query_text,
            "general_settings": general,
            "rerank_settings": {
                "enabled": args.rerank,
                "top_k": args.top_k,
                "advanced": {
                    "model": args.rerank_model,
                    "min_score": args.min_score,
                    "candidate_cap": args.rerank_candidates,
                },
            },
            "eval_settings": {
                "enabled": args.eval,
                "threshold": args.eval_thres,
                "strict_mode": args.eval_strict,
                "advanced": {
                    "metrics": args.eval_metrics,
                },
            },
        }
        if args.eval:
            payload["eval_settings"]["eval_model"] = args.eval_model
        if custom_prompts_payload:
            payload["custom_prompts"] = custom_prompts_payload
        return payload

    def print_settings(title: str) -> None:
        print("\n" + "=" * 60)
        print(f"[System] {title}")
        print(json.dumps(build_payload(None), indent=2, ensure_ascii=False))
        print("=" * 60 + "\n")

    if args.prompt_file and not load_prompt_file(args.prompt_file):
        sys.exit(1)

    url = "http://127.0.0.1:8000/ask"
    models_url = "http://127.0.0.1:8000/llm/models"
    session_url = "http://127.0.0.1:8000/session/connect"

    try:
        catalog_resp = requests.get(models_url, timeout=5)
        catalog_resp.raise_for_status()
        catalog = catalog_resp.json()
    except requests.exceptions.ConnectionError:
        print("\n[Error] 서버에 연결할 수 없습니다. api_server.py가 실행 중인지 확인하세요.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] 서버에서 LLM 목록을 받지 못했습니다: {e}")
        sys.exit(1)

    try:
        health = requests.get("http://127.0.0.1:8000/health", timeout=5).json()
        contracts = health.get("contracts") or {}
        problems = contracts.get("problems") or []
        if contracts.get("ok") is False or problems:
            print("[경고] 서버가 DB 계약 문제를 보고했습니다: " + "; ".join(problems or ["ok=false"]))
        for w in (health.get("boot") or {}).get("env_warnings") or []:
            print("[경고] " + w)
    except requests.exceptions.ConnectionError:
        print("\n[Error] 서버에 연결할 수 없습니다. api_server.py가 실행 중인지 확인하세요.")
        sys.exit(1)
    except Exception:
        pass

    models = catalog.get("models") or []
    local_rule = catalog.get("local_rule") or "local:<gguf파일명>"
    allowed_ids = [m.get("id") for m in models if m.get("id")]

    def _print_allowed() -> None:
        print("사용 가능한 LLM:")
        if allowed_ids:
            for item in models:
                mid = item.get("id", "")
                label = item.get("label", "")
                extra = f"  ({label})" if label else ""
                print(f"  - {mid}{extra}")
        else:
            print("  (서버 목록이 비어 있습니다)")
        print(f"로컬 규칙: {local_rule}")

    def _is_ok_on_client(name: str | None) -> bool:
        text = (name or "").strip()
        if not text or text == "local":
            return False
        if text in allowed_ids:
            return True
        if text.startswith("local:") and text != "local:":
            return True
        return False

    if not args.llm:
        print("\n[Error] --llm 이 비어 있습니다. 서버 목록에서 고르세요.")
        _print_allowed()
        sys.exit(1)
    if not _is_ok_on_client(args.llm):
        print(f"\n[Error] 지원하지 않는 LLM입니다: {args.llm}")
        _print_allowed()
        sys.exit(1)

    if args.eval:
        if not args.eval_model:
            args.eval_model = allowed_ids[0] if allowed_ids else None
        if not _is_ok_on_client(args.eval_model):
            print(f"\n[Error] 지원하지 않는 채점 LLM입니다: {args.eval_model}")
            _print_allowed()
            sys.exit(1)

    print("="*60)
    if args.questions:
        print("RAG 배치 질문 클라이언트 시작")
    else:
        print("RAG 대화형 콘솔 클라이언트 시작")
    print(f"▶ 대상 서버: {url}")
    print(f"▶ LLM: {args.llm}")
    print(f"▶ query_mode: {args.query_mode}  keyword_mode: {args.keyword_mode}  sources: {args.sources or '(전부)'}  base_limit: {args.base_limit}  use_graphdb: {args.use_graph}  rerank: {args.rerank}  eval: {args.eval}")
    if args.rerank:
        print(f"▶ rerank candidates: {args.rerank_candidates}")
    if args.eval:
        print(f"▶ 채점 LLM: {args.eval_model}")
    try:
        session_id, presence_stop, presence_http, presence_thread = start_presence(session_url)
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)
    print(f"▶ 세션: {session_id}")
    if args.id:
        print(f"▶ 로그인 아이디: {args.id}")
    if args.questions:
        print(f"▶ 질문 수: {len(args.questions)}")
    print("="*60)

    def run_one_ask(query_text: str) -> int:
        """한 질문을 보내고 SSE를 끝까지 출력한다. 종료 코드 0/1/2/3/4/130."""
        bad = _base_limit_invalid(args)
        if bad:
            print(f"\n[Error] {bad}")
            return 1
        if args.rerank and (args.rerank_candidates < 1 or args.rerank_candidates > 50):
            print("\n[Error] --rerank-candidates 는 1~50 사이여야 합니다.")
            return 1
        payload = build_payload(query_text)

        print("\n" + "=" * 60)
        print("[System] 아래의 설정값으로 질문을 전송합니다.")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("=" * 60 + "\n")

        global is_waiting
        is_waiting = True

        def spinner_animation():
            spinner_chars = ["|", "/", "-", "\\"]
            idx = 0
            while is_waiting:
                sys.stdout.write(f"\r[System] 서버 응답 대기 중... {spinner_chars[idx % 4]}")
                sys.stdout.flush()
                idx += 1
                time.sleep(0.1)
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()

        spinner_thread = threading.Thread(target=spinner_animation, daemon=True)
        spinner_thread.start()

        code = 0
        try:
            ask_headers = {"X-Session-Id": session_id}
            if args.id:
                ask_headers["X-User-Id"] = str(args.id).strip()
            first_chunk_received = False
            event_kind = "message"
            stop_after = None
            for line in iter_sse_lines(url, payload, headers=ask_headers):
                if stop_after is not None and time.monotonic() >= stop_after:
                    break
                if not first_chunk_received:
                    is_waiting = False
                    spinner_thread.join(timeout=1)
                    first_chunk_received = True
                if line.startswith("event:"):
                    event_kind = line[6:].strip() or "message"
                    continue
                if not line.startswith("data:"):
                    continue
                raw = line[5:].lstrip()
                if event_kind == "error":
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        obj = {}
                    env = unwrap_envelope(obj) or obj
                    render_error_envelope(env)
                    code = exit_code_from_envelope(env)
                    if ((env.get("error") or {}).get("terminal") is True
                            or (env.get("error") or {}).get("terminal") is None):
                        # terminal 기본은 스트림 종료. 최대 2초 동안 done 을 기다린다.
                        stop_after = time.monotonic() + 2.0
                    event_kind = "message"
                    continue
                if event_kind == "eval":
                    try:
                        eval_obj = json.loads(raw)
                    except json.JSONDecodeError:
                        eval_obj = {}
                    render_eval_report(eval_obj)
                    event_kind = "message"
                    continue
                if event_kind == "notice":
                    print("[안내] " + decode_sse_data(raw), flush=True)
                    event_kind = "message"
                    continue
                if event_kind == "done":
                    try:
                        done_obj = json.loads(raw)
                    except json.JSONDecodeError:
                        done_obj = {}
                    code = exit_code_from_done(done_obj)
                    partial = done_obj.get("partial") or []
                    if partial:
                        print(f"[경고] 부분 실패: {', '.join(str(x) for x in partial)}", flush=True)
                    dropped = int(done_obj.get("dropped_frames") or 0)
                    if dropped > 0:
                        print(f"[경고] dropped_frames={dropped}", flush=True)
                    if stop_after is not None:
                        break
                    event_kind = "message"
                    continue
                content = decode_sse_data(raw)
                if event_kind == "answer":
                    sys.stdout.write(content)
                elif content:
                    sys.stdout.write(content)
                sys.stdout.flush()
                event_kind = "message"
            print("\n" + "-" * 60)
        except EnvelopeError as e:
            render_error_envelope(e.env)
            code = exit_code_from_envelope(e.env)
        except KeyboardInterrupt:
            print("\n[System] 이 질문을 취소했습니다.")
            code = 130
        except requests.exceptions.ConnectionError:
            print("\n[Error] 서버에 연결할 수 없습니다. api_server.py가 실행 중인지 확인하세요.")
            code = 1
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            body = None
            if e.response is not None:
                try:
                    body = e.response.json()
                except ValueError:
                    body = None
            env = unwrap_envelope(body) if body else None
            if env:
                render_error_envelope(env)
                code = exit_code_from_envelope(env)
            else:
                print(f"\n[Error] HTTP {status}: {e}")
                if e.response is not None and e.response.text:
                    print(e.response.text)
                code = exit_code_for(None, http_status=status)
        except Exception as e:
            print(f"\n[Error] 통신 중 오류 발생: {str(e)}")
            code = 1
        finally:
            is_waiting = False
            if spinner_thread.is_alive():
                spinner_thread.join(timeout=1)
        return code

    try:
        if args.questions:
            worst = 0
            total = len(args.questions)
            for i, query in enumerate(args.questions, 1):
                print(f"\n[System] 질문 {i}/{total}")
                outcome = run_one_ask(query)
                if outcome == 130:
                    sys.exit(130)
                if outcome == 1:
                    sys.exit(1)
                if outcome == 2:
                    sys.exit(2)
                if outcome == 3:
                    sys.exit(3)
                if outcome > worst:
                    worst = outcome
            sys.exit(worst)

        if PromptSession is None or KeyBindings is None:
            print("[Error] 대화형 REPL 모드를 사용하려면 'prompt_toolkit' 패키지가 필요합니다: pip install prompt_toolkit")
            sys.exit(1)

        bindings = KeyBindings()

        @bindings.add('escape', 'enter') # Alt+Enter (윈도우에서 가장 확실한 다중 줄 전송 단축키)
        @bindings.add('c-j')        # Ctrl+Enter (일부 터미널용)
        def _(event):
            event.current_buffer.validate_and_handle()

        session = PromptSession(key_bindings=bindings)

        # 4. 무한 루프 대화창 (REPL)
        while True:
            try:
                print("\n[User] 질문 또는 설정(-- 로 시작). 줄바꿈: Enter, 전송: Alt+Enter 또는 Ctrl+Enter, 도움말: --help, 종료: q")
                query = session.prompt("❯ ", multiline=True).strip()
            except KeyboardInterrupt:
                print("\n\n[System] 클라이언트를 종료합니다.")
                break

            if query.lower() in ['q', 'quit', 'exit']:
                print("\n[System] 클라이언트를 종료합니다.")
                break
            if not query:
                continue

            if query.startswith("--"):
                show_help, errors, applied = apply_inline_options(
                    query, args, parser, repl_parser
                )
                if show_help:
                    print(parser.format_help())
                for msg in errors:
                    print(f"[Warn] 무시: {msg}")
                if "prompt_file" in applied and not load_prompt_file(args.prompt_file):
                    print("[Warn] 프롬프트 파일은 적용되지 않았습니다.")
                title = "설정을 갱신했습니다. 질문을 입력하세요."
                if errors and not applied and not show_help:
                    title = "적용된 옵션이 없습니다. 현재 설정입니다."
                print_settings(title)
                continue

            run_one_ask(query)
    finally:
        presence_stop.set()
        closer = getattr(presence_http, "_close_presence_stream", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
        try:
            presence_http.close()
        except Exception:
            pass
        presence_thread.join(timeout=2)


if __name__ == "__main__":
    main()