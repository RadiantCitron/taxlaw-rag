# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
import gc
import os
import re
import json
import logging
import sys
import threading
import time
import urllib.error
import urllib.request
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None
from openai import OpenAI
from session_manager import (
    SessionCancelled,
    get_current_session,
    raise_if_cancelled,
)
from errors import is_transient, retry_after_seconds
from settings import (
    OPENROUTER_MODELS_FILE,
    llm_max_tokens as get_default_max_tokens,
    llm_temperature as get_default_temperature,
)

logger = logging.getLogger("my_rag")

# =====================================================================
# OpenRouter 권장 헤더
# =====================================================================
OPENROUTER_HEADERS = {
    "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
    "X-Title": os.getenv("OPENROUTER_APP_NAME", "taxlaw-rag"),
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Gemini 3.5 등 thinking 모델은 reasoning 을 끌 수 없다. 본문 한도와 분리한다.
OPENROUTER_REASONING_MAX_TOKENS = 512

# OpenRouter 카탈로그 조회 실패 시에만 쓰는 기본값.
# google/gemini-flash-lite-latest 는 OpenRouter 슬러그가 아니다.
OPENROUTER_FALLBACK_MODEL = {
    "id": "openrouter:google/gemini-3.5-flash-lite",
    "label": "Gemini 3.5 Flash Lite (OpenRouter fallback)",
    "provider": "openrouter",
}

_OPENROUTER_CATALOG: list[dict] | None = None

_HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_SEARCH_DIRS = (
    _HERE,
    os.path.join(_HERE, "llm"),
    os.path.join(_HERE, "data"),
    os.path.abspath("."),
    os.path.abspath("llm"),
    os.path.abspath("data"),
)
LOCAL_RULE = (
    "local:<gguf파일명>  (파일이 프로젝트 폴더, llm/, data/ 또는 현재 폴더에 있을 때). "
    "gemini: / claude: / groq: 는 지원하지 않습니다."
)


def _split_model_identifier(identifier: str) -> tuple[str, str]:
    text = (identifier or "").strip()
    if not text:
        return "", ""
    if ":" not in text:
        return "local", text
    provider, name = text.split(":", 1)
    return provider.strip().lower(), name.strip()


def is_local_model(identifier: str) -> bool:
    """로컬 GGUF 인지. 판정은 이 함수 하나만 쓴다.

    별칭 'local'(메인 LLM 재사용)은 파일이 아니므로 False.
    """
    text = (identifier or "").strip()
    if not text or text == "local":
        return False
    provider, name = _split_model_identifier(text)
    return provider == "local" and bool(name)


def find_local_model_path(filename: str) -> str | None:
    """로컬 GGUF 경로를 찾는다. 없으면 None."""
    name = (filename or "").strip()
    if not name:
        return None
    candidates = [os.path.abspath(name)]
    for folder in LOCAL_MODEL_SEARCH_DIRS:
        candidates.append(os.path.abspath(os.path.join(folder, name)))
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path
    return None


def scan_local_gguf_models() -> list[dict]:
    """llm/, data/, 현재 폴더의 .gguf 파일을 목록에 넣는다."""
    found: dict[str, dict] = {}
    for folder in LOCAL_MODEL_SEARCH_DIRS:
        root = os.path.abspath(folder)
        if not os.path.isdir(root):
            continue
        try:
            names = os.listdir(root)
        except OSError:
            continue
        for name in names:
            if not name.lower().endswith(".gguf"):
                continue
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            mid = f"local:{name}"
            if mid not in found:
                found[mid] = {
                    "id": mid,
                    "label": f"{name} (local GGUF)",
                    "provider": "local",
                }
    return list(found.values())


def _is_chat_text_model(item: dict) -> bool:
    slug = str(item.get("id") or "")
    if not slug or ":batch" in slug:
        return False
    arch = item.get("architecture") or {}
    outputs = arch.get("output_modalities") or arch.get("output_modality") or []
    if isinstance(outputs, str):
        outputs = [outputs]
    if outputs and "text" not in [str(x).lower() for x in outputs]:
        return False
    return True


def _normalize_openrouter_item(item: dict) -> dict:
    slug = str(item.get("id") or "").strip()
    name = str(item.get("name") or slug)
    return {
        "id": f"openrouter:{slug}",
        "label": name,
        "provider": "openrouter",
    }


def fetch_openrouter_chat_models(timeout_sec: float = 15.0) -> list[dict]:
    """OpenRouter GET /models 에서 채팅(text 출력) 모델만 가져온다."""
    url = f"{OPENROUTER_BASE_URL}/models"
    req = urllib.request.Request(url, headers={"User-Agent": "taxlaw-rag/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("OpenRouter /models 응답이 목록이 아닙니다.")
    out: list[dict] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict) or not _is_chat_text_model(raw):
            continue
        item = _normalize_openrouter_item(raw)
        if not item["id"].startswith("openrouter:") or item["id"] == "openrouter:":
            continue
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        out.append(item)
    out.sort(key=lambda x: x["id"])
    return out


def refresh_openrouter_catalog() -> list[dict]:
    """카탈로그를 다시 받아 캐시한다. 실패 시 fallback 1개."""
    global _OPENROUTER_CATALOG
    try:
        models = fetch_openrouter_chat_models()
        if not models:
            raise ValueError("채팅용 모델이 0개입니다.")
        _OPENROUTER_CATALOG = models
        logger.info(f"[LLM] OpenRouter 채팅 모델 {len(models)}개 로드")
        return list(models)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError) as e:
        logger.warning(f"[LLM] OpenRouter 카탈로그 조회 실패: {e}. fallback 사용.")
        _OPENROUTER_CATALOG = [dict(OPENROUTER_FALLBACK_MODEL)]
        return list(_OPENROUTER_CATALOG)


def load_openrouter_catalog_from_file(path: str) -> list[dict]:
    """로컬 JSON 카탈로그를 읽어 캐시에 넣는다. OpenRouter 네트워크 요청은 하지 않는다."""
    global _OPENROUTER_CATALOG
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"OpenRouter 모델 목록 파일이 비어 있거나 배열이 아닙니다: {path}")
    models = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        mid = str(raw.get("id") or "").strip()
        if not mid.startswith("openrouter:") or mid == "openrouter:":
            continue
        models.append({
            "id": mid,
            "label": str(raw.get("label") or mid),
            "provider": str(raw.get("provider") or "openrouter"),
        })
    if not models:
        raise ValueError(f"유효한 openrouter: 항목이 없습니다: {path}")
    _OPENROUTER_CATALOG = models
    logger.info(f"[LLM] OpenRouter 모델 목록 파일 로드: {path} ({len(models)}개)")
    return list(models)


def get_openrouter_catalog() -> list[dict]:
    global _OPENROUTER_CATALOG
    if _OPENROUTER_CATALOG is None:
        catalog_path = os.path.join(_HERE, OPENROUTER_MODELS_FILE)
        if os.path.exists(catalog_path):
            try:
                load_openrouter_catalog_from_file(catalog_path)
            except Exception as e:
                logger.warning(f"[LLM] 로컬 카탈로그 로드 실패({e}) ➔ 원격 갱신 시도")
                refresh_openrouter_catalog()
        else:
            refresh_openrouter_catalog()
    return [dict(x) for x in (_OPENROUTER_CATALOG or [])]


def list_available_models() -> list[dict]:
    """OpenRouter 채팅 카탈로그 + 발견된 로컬 GGUF."""
    catalog = get_openrouter_catalog()
    seen = {item["id"] for item in catalog}
    for item in scan_local_gguf_models():
        if item["id"] not in seen:
            catalog.append(item)
            seen.add(item["id"])
    return catalog


def format_allowed_models_help() -> str:
    lines = [f"  - {item['id']}  ({item['label']})" for item in list_available_models()]
    body = "\n".join(lines) if lines else "  (등록된 모델 없음)"
    return (
        "사용 가능한 LLM:\n"
        f"{body}\n"
        f"로컬 규칙: {LOCAL_RULE}"
    )


def is_allowed_model_id(identifier: str) -> tuple[bool, str]:
    """허용 목록 id 또는 파일이 있는 local 모델인지 검사한다.

    Returns:
        (ok, error_message). ok 이면 error_message 는 빈 문자열.
    """
    text = (identifier or "").strip()
    if not text:
        return False, "모델 식별자가 비어 있습니다.\n" + format_allowed_models_help()
    if text == "local":
        return False, (
            "eval_model='local' 은 더 이상 메인 LLM 재사용을 뜻하지 않습니다. "
            "openrouter:... 또는 local:<파일명> 을 지정하세요.\n"
            + format_allowed_models_help()
        )

    allowed_ids = {item["id"] for item in list_available_models()}
    if text in allowed_ids:
        return True, ""

    provider, name = _split_model_identifier(text)
    if provider == "local":
        if find_local_model_path(name):
            return True, ""
        return False, (
            f"로컬 모델 파일을 찾을 수 없습니다: {name}\n"
            + format_allowed_models_help()
        )

    return False, (
        f"지원하지 않는 모델입니다: {text}\n"
        + format_allowed_models_help()
    )


def validate_model_identifier(identifier: str) -> str:
    """허용되지 않으면 ValueError. 허용되면 strip 된 id."""
    text = (identifier or "").strip()
    ok, err = is_allowed_model_id(text)
    if not ok:
        raise ValueError(err)
    return text


def exit_if_invalid_model(identifier: str, *, label: str = "모델") -> str:
    """CLI 기동 가드. 불법이면 stderr 에 목록을 찍고 종료한다."""
    text = (identifier or "").strip()
    ok, err = is_allowed_model_id(text)
    if not ok:
        print(f"[ERROR] {label} 식별자가 올바르지 않습니다.\n{err}", file=sys.stderr)
        raise SystemExit(1)
    return text


def models_catalog_payload() -> dict:
    """GET /llm/models 응답."""
    from settings import default_llm
    available = list_available_models()
    def_id = default_llm()
    if not any(m.get("id") == def_id for m in available) and available:
        def_id = available[0]["id"]
    return {"models": available, "default_id": def_id, "local_rule": LOCAL_RULE}


def format_prompt_trace(model_id: str, stage: str, messages: list) -> str:
    """클라이언트에 보낼 LLM 입력 트레이스 텍스트."""
    label = (stage or "llm").strip() or "llm"
    lines = ["", f"──────── LLM TRACE  [{label}]  {model_id}"]
    for msg in messages or []:
        if isinstance(msg, dict):
            role = str(msg.get("role") or "message")
            content = str(msg.get("content") or "")
        else:
            role = "message"
            content = str(msg)
        lines.append(f"role: {role}")
        lines.append(content)
        lines.append("")
    lines.append("────────")
    return "\n".join(lines)


# =====================================================================
# QueryOptimizer
# =====================================================================
class QueryOptimizer:
    def __init__(self, llm_controller):
        self.llm_controller = llm_controller

    def expand_query(self, user_query: str, prompt_template: str, *, trace_prompts: bool = False) -> str:
        logger.info("[LLM-OPT] Query Expansion (질문 확장) 프로세스 시작...")
        prompt = f"{prompt_template}\n\n[Actual Execution]\nUser Query: {user_query}\nExpanded Query:"
        expanded_query = self.llm_controller.ask_raw(
            prompt, trace_prompts=trace_prompts, stage="query_expansion"
        )
        if not expanded_query or not str(expanded_query).strip():
            logger.warning("[LLM-OPT] LLM 질문 확장 실패(빈 응답). 원본 질문으로 대체합니다.")
            return user_query

        expanded_query = str(expanded_query).strip()
        logger.info(f"[LLM-OPT] 확장된 질문: {expanded_query}")
        return expanded_query

    def optimize_to_sql_keywords(
        self, user_query: str, db_vocabulary: str, prompt_template: str, *, trace_prompts: bool = False
    ) -> list:
        logger.info("[LLM-OPT] Vocab Mapping (DB 단어장 매핑) 프로세스 시작...")

        prompt = f"{prompt_template}\n\n[DB Vocabulary]\n{db_vocabulary}\n\nUser Query: {user_query}\nKeywords:"
        raw_output = self.llm_controller.ask_raw(
            prompt, trace_prompts=trace_prompts, stage="vocab_mapping"
        )

        if not raw_output:
            logger.warning("[LLM-OPT] LLM이 키워드를 생성하지 못했습니다. 원본 질문을 키워드로 대체합니다.")
            return [word for word in user_query.split() if len(word) >= 2]

        keywords = [re.sub(r'[^\w\s]', '', k.strip()) for k in raw_output.split(',') if k.strip()]
        logger.info(f"[LLM-OPT] 매핑된 RDB 검색용 키워드: {keywords}")
        return keywords


# =====================================================================
# LLMController — local / openrouter 만 지원
# =====================================================================
class LLMController:
    """Local(GGUF) / OpenRouter 를 단일 인터페이스로 제어. 오타 id 는 Fallback 하지 않는다."""

    def __init__(self, model_identifier: str):
        logger.info(f"[INIT] LLMController 초기화 시작 (요청된 식별자: {model_identifier})")

        self.local_llm = None
        self.client = None
        self.provider = None
        self.model_name = None
        self.fallback_warning = ""

        ident = validate_model_identifier(model_identifier)
        err = self._try_load_model(ident)
        if err:
            logger.error(f"[INIT 오류] {err}")
            raise ValueError(err)

        self._infer_lock = threading.Lock() if self.provider == "local" else None
        self.optimizer = QueryOptimizer(self)
        logger.info(f"[INIT] LLMController 초기화 완료 (현재 Provider: {self.provider})")

    def _try_load_model(self, identifier: str) -> str:
        """성공 시 빈 문자열, 실패 시 이유."""
        try:
            self.provider, self.model_name = _split_model_identifier(identifier)

            if self.provider == "local":
                if Llama is None:
                    return "llama-cpp-python이 설치되어 있지 않습니다."
                model_path = find_local_model_path(self.model_name)
                if not model_path:
                    return f"로컬 모델 파일을 찾을 수 없습니다: {self.model_name}"
                logger.info(f"[INIT] Local Llama 로드 중: {model_path}")
                self.local_llm = Llama(
                    model_path=model_path,
                    n_ctx=4096,
                    n_gpu_layers=-1,
                    verbose=False,
                )
                logger.info("[INIT] Local Llama 로드 완료")
                return ""

            if self.provider == "openrouter":
                api_key = os.environ.get("OPENROUTER_API_KEY")
                if not api_key:
                    return "OPENROUTER_API_KEY가 없습니다."
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=OPENROUTER_BASE_URL,
                    default_headers=OPENROUTER_HEADERS,
                )
                return ""

            return (
                f"지원하지 않는 Provider입니다: {self.provider}.\n"
                + format_allowed_models_help()
            )
        except Exception as e:
            return f"{identifier} 로드 중 에러: {e}"

    def get_optimizer(self):
        return self.optimizer

    # -----------------------------------------------------------------
    # 통합 생성 경로 — ask / ask_raw / evaluate_ask 가 모두 여기를 탄다
    # -----------------------------------------------------------------
    def _generate(
        self,
        messages: list,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace_prompts: bool = False,
        stage: str = "",
    ) -> str:
        output = ""
        try:
            if trace_prompts:
                logger.info(
                    "%s",
                    format_prompt_trace(f"{self.provider}:{self.model_name}", stage, messages),
                    extra={"prompt_trace": True},
                )
            temp = get_default_temperature() if temperature is None else temperature
            tokens = get_default_max_tokens() if max_tokens is None else max_tokens
            session = get_current_session()

            if self.provider == "local":
                raise_if_cancelled()
                lock = self._infer_lock
                if lock is not None:
                    lock.acquire()
                try:
                    if hasattr(self.local_llm, "reset"):
                        self.local_llm.reset()
                    response = self.local_llm.create_chat_completion(
                        messages=messages,
                        max_tokens=tokens,
                        temperature=temp,
                    )
                    raise_if_cancelled()
                    output = (response["choices"][0]["message"]["content"] or "").strip()
                    return output
                finally:
                    if lock is not None:
                        lock.release()

            if self.provider == "openrouter":
                raise_if_cancelled()
                last_exc: Exception | None = None
                for attempt in range(2):
                    try:
                        output = self._openrouter_complete(
                            messages, temp=temp, tokens=tokens, session=session,
                        )
                        return output
                    except SessionCancelled:
                        raise
                    except Exception as exc:
                        last_exc = exc
                        if session is not None and session.cancel_event.is_set():
                            raise SessionCancelled() from None
                        if attempt == 0 and is_transient(str(exc)):
                            wait = retry_after_seconds(str(exc)) or 2.0
                            wait = min(max(wait, 0.5), 15.0)
                            logger.warning(
                                "[LLM] OpenRouter 일시 오류, %.1fs 후 1회 재시도: %s",
                                wait, exc,
                            )
                            time.sleep(wait)
                            raise_if_cancelled()
                            continue
                        raise
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError("OpenRouter 생성 실패")

            raise ValueError(f"지원하지 않는 provider: {self.provider}")
        finally:
            self._maybe_record_llm_call(stage, messages, output)

    def _openrouter_complete(self, messages, *, temp: float, tokens: int, session) -> str:
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temp,
            max_tokens=tokens,
            stream=True,
            extra_body={"reasoning": {"max_tokens": OPENROUTER_REASONING_MAX_TOKENS}},
        )
        if session is not None:
            session.attach_stream(stream)
        parts = []
        try:
            for chunk in stream:
                if session is not None and session.cancel_event.is_set():
                    raise SessionCancelled()
                if not getattr(chunk, "choices", None):
                    continue
                piece = getattr(chunk.choices[0].delta, "content", None)
                if piece:
                    parts.append(piece)
            return "".join(parts).strip()
        except SessionCancelled:
            raise
        except Exception:
            if session is not None and session.cancel_event.is_set():
                raise SessionCancelled() from None
            raise
        finally:
            if session is not None:
                session.detach_stream(stream)
            else:
                closer = getattr(stream, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception:
                        pass

    def _maybe_record_llm_call(self, stage: str, messages: list, output: str) -> None:
        session = get_current_session()
        run = getattr(session, "current_run", None) if session is not None else None
        record = getattr(run, "record_llm_call", None) if run is not None else None
        if not callable(record):
            return
        record(stage, f"{self.provider}:{self.model_name}", messages, output)

    def ask_raw(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace_prompts: bool = False,
        stage: str = "",
    ) -> str:
        """가드레일 / 옵티마이저용 순수 텍스트 질의."""
        try:
            return self._generate(
                [{"role": "user", "content": prompt}],
                temperature=temperature if temperature is not None else 0.1,
                max_tokens=max_tokens if max_tokens is not None else 2048,
                trace_prompts=trace_prompts,
                stage=stage,
            )
        except SessionCancelled:
            raise
        except Exception as e:
            logger.error(f"[LLM RAW 오류] {str(e)}")
            return ""

    def ask(
        self,
        query: str,
        company_summary: str,
        law_context: str,
        prompt_template: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace_prompts: bool = False,
        stage: str = "main_answer",
    ) -> str:
        """RAG 메인 답변 생성."""
        logger.info(f"[LLM] {self.provider} ({self.model_name}) 모델에 답변 생성 요청 중...")
        system_prompt = (
            f"{prompt_template}\n\n"
            f"[Company Summary]\n{company_summary}\n\n"
            f"[Retrieved Context]\n{law_context}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        try:
            answer = self._generate(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                trace_prompts=trace_prompts,
                stage=stage,
            )
            return answer
        except SessionCancelled:
            raise
        except Exception as e:
            error_msg = f"답변 생성 실패 ({self.provider} 연동 오류: {str(e)})"
            logger.error(f"[LLM 오류] {error_msg}")
            raise

    def evaluate_ask(
        self,
        prompt: str,
        eval_prompt_template: str = "",
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace_prompts: bool = False,
        stage: str = "eval",
    ) -> str:
        """채점/심사용 일반 텍스트 호출. JSON 강제는 호출부(평가 어댑터) 책임."""
        logger.info(f"[LLM] {self.provider} ({self.model_name}) 모델에 채점(Evaluation) 요청 중...")
        system = (eval_prompt_template or "").strip() or "너는 채점관이다. 지시에 따라 답하라."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            return self._generate(
                messages,
                temperature=temperature if temperature is not None else 0.1,
                max_tokens=max_tokens if max_tokens is not None else 1000,
                trace_prompts=trace_prompts,
                stage=stage,
            )
        except SessionCancelled:
            raise
        except Exception as e:
            logger.error(f"[LLM 채점 오류] {self.provider} 연동 실패: {str(e)}")
            return ""


# =========================================================
# LLM 캐시
# =========================================================
_LLM_CACHE = {}
_LLM_CACHE_LOCK = threading.Lock()


def get_llm_controller(model_identifier: str) -> LLMController:
    """모델을 한 번만 로드하고 재사용한다. 불법 id 는 ValueError."""
    ident = validate_model_identifier(model_identifier)
    with _LLM_CACHE_LOCK:
        cached = _LLM_CACHE.get(ident)
        if cached is not None:
            logger.info(f"[CACHE] 캐시된 LLM 재사용: {ident}")
            return cached
    logger.info(f"[CACHE] 새 LLM 로드: {ident}")
    ctrl = LLMController(ident)
    with _LLM_CACHE_LOCK:
        existing = _LLM_CACHE.get(ident)
        if existing is not None:
            logger.info(f"[CACHE] 캐시된 LLM 재사용: {ident}")
            return existing
        if is_local_model(ident):
            for key in list(_LLM_CACHE):
                if key != ident and is_local_model(key):
                    old = _LLM_CACHE.pop(key)
                    try:
                        old.local_llm = None
                    except Exception:
                        pass
                    logger.info("[CACHE] 로컬 LLM 언로드: %s", key)
            gc.collect()
        _LLM_CACHE[ident] = ctrl
        return ctrl
