# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""간단한 /ask end-to-end 확인용 SSE 클라이언트 (개발/데모 전용).

GET /session/connect 로 세션을 열고(백그라운드 유지), 그 세션으로 POST /ask 하여
스트리밍 응답을 출력한다.
"""
from __future__ import annotations

import json
import sys
import threading
import time

import requests

BASE = "http://127.0.0.1:8000"


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "근로소득세액공제는 어떻게 계산하나요?"
    llm = sys.argv[2] if len(sys.argv) > 2 else "local:qwen2.5-0.5b-instruct-q4_k_m.gguf"

    sess_holder: dict[str, str] = {}
    stop = threading.Event()

    def presence():
        with requests.get(f"{BASE}/session/connect", stream=True, timeout=120) as r:
            for line in r.iter_lines(decode_unicode=True):
                if stop.is_set():
                    break
                if line and line.startswith("data:"):
                    payload = line[5:].strip()
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("event") == "connected":
                        sess_holder["id"] = obj["session_id"]

    t = threading.Thread(target=presence, daemon=True)
    t.start()
    for _ in range(100):
        if "id" in sess_holder:
            break
        time.sleep(0.1)
    if "id" not in sess_holder:
        print("[ERROR] 세션 연결 실패", file=sys.stderr)
        return 1
    sid = sess_holder["id"]
    print(f"[SESSION] {sid}")

    body = {
        "query": query,
        "general_settings": {
            "llm_model": llm,
            "use_guardrails": False,
            "query_mode": "none",
            "sources": ["tax_laws_master", "tax_precedents_master"],
        },
        "rerank_settings": {"enabled": False, "top_k": 3},
        "eval_settings": {"enabled": False},
    }
    headers = {"X-Session-Id": sid, "Content-Type": "application/json"}
    print(f"[ASK] query={query!r} llm={llm}")
    print("-" * 60)
    event_kind = "message"
    code = 0
    with requests.post(f"{BASE}/ask", json=body, headers=headers, stream=True, timeout=600) as r:
        if not r.ok:
            try:
                payload = r.json()
            except ValueError:
                payload = None
            from error_envelope import envelope_code, exit_code_for, unwrap_envelope
            env = unwrap_envelope(payload) if payload else None
            if env:
                err = env.get("error") or {}
                print(f"[요청 거부] {err.get('code')}", file=sys.stderr)
                print(err.get("message") or "", file=sys.stderr)
                stop.set()
                t.join(timeout=2)
                return exit_code_for(envelope_code(env), http_status=r.status_code)
            print(r.text, file=sys.stderr)
            stop.set()
            t.join(timeout=2)
            return 1
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                event_kind = line[6:].strip() or "message"
                continue
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            text = raw
            obj = None
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and "text" in obj:
                    text = obj["text"]
            except json.JSONDecodeError:
                text = raw.replace("<br>", "\n")
            if event_kind == "error":
                env = obj if isinstance(obj, dict) else {}
                err = env.get("error") or {}
                print(f"[요청 거부] {err.get('code') or '?'}")
                print(err.get("message") or text)
                from error_envelope import envelope_code, exit_code_for
                code = exit_code_for(envelope_code(env), http_status=None)
                event_kind = "message"
                continue
            if event_kind == "notice":
                print("[안내] " + text)
                event_kind = "message"
                continue
            if event_kind == "done":
                event_kind = "message"
                continue
            print(text)
            event_kind = "message"
    stop.set()
    t.join(timeout=2)
    print("-" * 60)
    print("[DONE]")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
