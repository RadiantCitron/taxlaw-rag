# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
# --- START OF FILE cs_console.py ---
"""수업용 CS 클라이언트. RAG 서버 HTTP 만 호출한다. 서버 코드를 import 하지 않는다."""
from __future__ import annotations

import json
import os
import sys

import requests

BASE = os.environ.get("RAG_SERVER", "http://127.0.0.1:8000").rstrip("/")


def _req(method: str, path: str, **kwargs):
    url = f"{BASE}{path}"
    resp = requests.request(method, url, timeout=5, **kwargs)
    if not resp.ok:
        detail = resp.text
        try:
            body = resp.json()
            detail = body.get("detail", body)
        except Exception:
            pass
        raise RuntimeError(f"HTTP {resp.status_code}: {detail}")
    if not resp.content:
        return {}
    return resp.json()


def fetch_status() -> dict:
    return _req("GET", "/admin/status")


def print_dashboard(data: dict) -> None:
    mode = "ON" if data.get("login_mode") else "OFF"
    conn = data.get("connected") or {}
    proc = data.get("processing") or {}
    ids = data.get("user_ids") or []
    print()
    print("=" * 60)
    print(f" RAG CS  {BASE}")
    print(f" login_mode={mode}  allow_ip={','.join(data.get('cs_allow_ip') or [])}")
    print(
        f" 접속 {conn.get('current', 0)}/{conn.get('max', '?')}  "
        f"처리중 {proc.get('current', 0)}"
    )
    print(" 접속 아이디: " + (", ".join(ids) if ids else "(없음)"))
    users = data.get("users") or []
    if users:
        print(" 명단:")
        for row in users:
            flag = []
            if row.get("bound"):
                flag.append("접속")
            if row.get("expired"):
                flag.append("만료")
            extra = f" ({', '.join(flag)})" if flag else ""
            print(f"   - {row.get('id')} ttl={row.get('ttl_sec')}{extra}")
    print("=" * 60)


def handle_command(line: str) -> None:
    parts = line.split()
    if not parts:
        return
    cmd = parts[0].lower()
    if cmd in ("q", "quit", "exit"):
        raise SystemExit(0)
    if cmd in ("help", "?"):
        print("  user add <id> [ttl_sec]   명단에 추가 (기본 ttl 3600)")
        print("  user del <id>             명단에서 삭제")
        print("  users                     명단 조회")
        print("  status                    상태 새로고침")
        print("  q                         종료")
        return
    if cmd == "status":
        print_dashboard(fetch_status())
        return
    if cmd == "users":
        data = _req("GET", "/admin/users")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    if cmd == "user" and len(parts) >= 3 and parts[1].lower() == "add":
        uid = parts[2]
        ttl = int(parts[3]) if len(parts) >= 4 else 3600
        data = _req("POST", "/admin/users", json={"id": uid, "ttl_sec": ttl})
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    if cmd == "user" and len(parts) >= 3 and parts[1].lower() in ("del", "rm", "delete"):
        uid = parts[2]
        data = _req("DELETE", f"/admin/users/{uid}")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    print("[Warn] 알 수 없는 명령입니다. help")


def main() -> None:
    print("수업용 CS. Enter=상태 새로고침, /=명령, q=종료")
    try:
        print_dashboard(fetch_status())
    except Exception as e:
        print(f"[Error] 서버에 연결할 수 없습니다: {e}")
        print("api_server.py 가 실행 중인지, CS_ALLOW_IP 가 이 머신인지 확인하세요.")
        sys.exit(1)
    while True:
        try:
            raw = input("\n[Enter 새로고침 / 명령] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw or raw.lower() in ("q", "quit", "exit"):
            if raw.lower() in ("q", "quit", "exit"):
                break
            try:
                print_dashboard(fetch_status())
            except Exception as e:
                print(f"[Error] {e}")
            continue
        if raw == "/":
            try:
                cmd = input("cs> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if not cmd:
                continue
            try:
                handle_command(cmd)
            except SystemExit:
                break
            except Exception as e:
                print(f"[Error] {e}")
            continue
        if raw.startswith("/"):
            try:
                handle_command(raw[1:].strip())
            except SystemExit:
                break
            except Exception as e:
                print(f"[Error] {e}")
            continue
        try:
            handle_command(raw)
        except SystemExit:
            break
        except Exception as e:
            print(f"[Error] {e}")


if __name__ == "__main__":
    main()
