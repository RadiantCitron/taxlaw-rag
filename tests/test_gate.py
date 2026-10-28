# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
"""게이트 세마포어 수용 기준 — DB/LLM 없이 검증."""
import asyncio
import json

import pytest

from local_llm_gate import (
    acquire_ask_slot,
    init_gate,
    release_ask_slot,
)


class FakeRequest:
    def __init__(self, disconnected: bool = False):
        self._disconnected = disconnected

    async def is_disconnected(self) -> bool:
        return self._disconnected


@pytest.mark.asyncio
async def test_slot_acquire_release_roundtrip():
    init_gate(max_concurrent_user=1, max_waiting=1)
    queue: asyncio.Queue = asyncio.Queue()
    assert await acquire_ask_slot(FakeRequest(), queue) is True
    release_ask_slot()
    assert await acquire_ask_slot(FakeRequest(), queue) is True
    release_ask_slot()


@pytest.mark.asyncio
async def test_gate_full_emits_envelope():
    init_gate(max_concurrent_user=1, max_waiting=0)
    queue: asyncio.Queue = asyncio.Queue()
    assert await acquire_ask_slot(FakeRequest(), queue) is True
    assert await acquire_ask_slot(FakeRequest(), queue) is False
    kind, body = queue.get_nowait()
    assert kind == "error"
    payload = json.loads(body)
    assert payload["error"]["code"] == "GATE_FULL"
    release_ask_slot()


@pytest.mark.asyncio
async def test_disconnected_client_does_not_hold_lock():
    init_gate(max_concurrent_user=1, max_waiting=2)
    queue: asyncio.Queue = asyncio.Queue()
    assert await acquire_ask_slot(FakeRequest(), queue) is True
    got = await acquire_ask_slot(FakeRequest(disconnected=True), queue)
    assert got is False
    release_ask_slot()
    assert await acquire_ask_slot(FakeRequest(), queue) is True
    release_ask_slot()


class FakeQuotaRequest:
    def __init__(self, ip: str = "127.0.0.1", headers: dict | None = None):
        self.client = type("Client", (), {"host": ip})()
        self.headers = headers or {}


def test_quota_daily_limit_and_cooldown(monkeypatch, tmp_path):
    import local_llm_gate as gate_mod
    from fastapi import HTTPException

    # 테스트용 임시 파일 및 쿨다운 설정 (쿨다운 10초, 일일 3회)
    test_file = str(tmp_path / "test_quotas.json")
    monkeypatch.setattr(gate_mod, "_quota_file", test_file)
    monkeypatch.setenv("RATE_LIMIT_DAILY", "3")
    monkeypatch.setenv("RATE_LIMIT_COOLDOWN_SEC", "10")
    monkeypatch.setattr(gate_mod, "_quotas_cache", {})

    req = FakeQuotaRequest(ip="192.168.1.100")

    # 1회차 성공
    res1 = gate_mod.check_and_consume_quota(req)
    assert res1["allowed"] is True
    assert res1["used_today"] == 1
    assert res1["remaining_today"] == 2

    # 즉시 2회차 시도 -> 쿨다운(10초)으로 차단 (429 COOLDOWN_ACTIVE)
    with pytest.raises(HTTPException) as exc1:
        gate_mod.check_and_consume_quota(req)
    assert exc1.value.status_code == 429
    assert exc1.value.detail["error"]["code"] == "COOLDOWN_ACTIVE"
    assert "Retry-After" in exc1.value.headers

    # 시간을 11초 뒤로 모사 (쿨다운 해제)
    with gate_mod._quota_lock:
        today = gate_mod._today_kst()
        gate_mod._quotas_cache[today]["192.168.1.100"]["last_ask_ts"] -= 11

    # 2회차 성공
    res2 = gate_mod.check_and_consume_quota(req)
    assert res2["used_today"] == 2
    assert res2["remaining_today"] == 1

    # 다시 시간 11초 경과 모사
    with gate_mod._quota_lock:
        gate_mod._quotas_cache[today]["192.168.1.100"]["last_ask_ts"] -= 11

    # 3회차 성공 (마지막 한도)
    res3 = gate_mod.check_and_consume_quota(req)
    assert res3["used_today"] == 3
    assert res3["remaining_today"] == 0

    # 다시 시간 11초 경과 모사 후 4회차 시도 -> 일일 쿼터 초과 (429 DAILY_QUOTA_EXCEEDED)
    with gate_mod._quota_lock:
        gate_mod._quotas_cache[today]["192.168.1.100"]["last_ask_ts"] -= 11

    with pytest.raises(HTTPException) as exc2:
        gate_mod.check_and_consume_quota(req)
    assert exc2.value.status_code == 429
    assert exc2.value.detail["error"]["code"] == "DAILY_QUOTA_EXCEEDED"


def test_quota_ip_enforcement_prevents_device_spoofing(monkeypatch, tmp_path):
    """시크릿 창이나 로컬스토리지 삭제로 새 Device-Id를 발급받아도 동일 IP면 우회 불가 검증"""
    import local_llm_gate as gate_mod

    test_file = str(tmp_path / "test_quotas2.json")
    monkeypatch.setattr(gate_mod, "_quota_file", test_file)
    monkeypatch.setenv("RATE_LIMIT_DAILY", "2")
    monkeypatch.setenv("RATE_LIMIT_COOLDOWN_SEC", "0")
    monkeypatch.setattr(gate_mod, "_quotas_cache", {})

    req_dev_a = FakeQuotaRequest(ip="203.0.113.1", headers={"x-device-id": "dev-AAAA"})
    req_dev_b = FakeQuotaRequest(ip="203.0.113.1", headers={"x-device-id": "dev-BBBB"})

    # 디바이스 A로 2회 사용 -> 오늘 한도 소진
    gate_mod.check_and_consume_quota(req_dev_a)
    gate_mod.check_and_consume_quota(req_dev_a)

    # 공격자가 브라우저를 시크릿 모드로 켜서 디바이스 B로 위장해도 같은 IP이므로 즉시 차단
    status_b = gate_mod.get_quota_status(req_dev_b)
    assert status_b["used_today"] == 2
    assert status_b["remaining_today"] == 0
    assert status_b["allowed"] is False
    assert status_b["reason"] == "DAILY_QUOTA_EXCEEDED"


def test_quota_persistence_across_restart(monkeypatch, tmp_path):
    import local_llm_gate as gate_mod

    test_file = str(tmp_path / "test_quotas_persist.json")
    monkeypatch.setattr(gate_mod, "_quota_file", test_file)
    monkeypatch.setenv("RATE_LIMIT_DAILY", "5")
    monkeypatch.setenv("RATE_LIMIT_COOLDOWN_SEC", "0")
    monkeypatch.setattr(gate_mod, "_quotas_cache", {})

    req = FakeQuotaRequest(ip="10.0.0.5")
    gate_mod.check_and_consume_quota(req)
    gate_mod.check_and_consume_quota(req)

    # 서버 재시작 모사: 캐시를 날리고 init_gate() 호출
    monkeypatch.setattr(gate_mod, "_quotas_cache", {})
    gate_mod.init_gate()

    status = gate_mod.get_quota_status(req)
    assert status["used_today"] == 2
    assert status["remaining_today"] == 3


def test_quota_rapid_burst_defense(monkeypatch, tmp_path):
    """프로그램으로 0.001초 간격 10연타 디도스 요청 시 1회만 통과, 9회 전원 즉각 429 입구컷 검증"""
    import local_llm_gate as gate_mod
    from fastapi import HTTPException

    test_file = str(tmp_path / "test_quotas_burst.json")
    monkeypatch.setattr(gate_mod, "_quota_file", test_file)
    monkeypatch.setenv("RATE_LIMIT_DAILY", "5")
    monkeypatch.setenv("RATE_LIMIT_COOLDOWN_SEC", "900")
    monkeypatch.setattr(gate_mod, "_quotas_cache", {})

    req = FakeQuotaRequest(ip="198.51.100.1")

    # 1회차 정상 통과
    res1 = gate_mod.check_and_consume_quota(req)
    assert res1["allowed"] is True
    assert res1["used_today"] == 1

    # 2~10회차 연속 9회 연타 공격 시도 -> 전원 즉시 429 입구컷 차단
    blocked_count = 0
    for _ in range(9):
        try:
            gate_mod.check_and_consume_quota(req)
        except HTTPException as e:
            if e.status_code == 429 and e.detail["error"]["code"] == "COOLDOWN_ACTIVE":
                blocked_count += 1
    assert blocked_count == 9


def test_quota_dynamic_config_override(monkeypatch, tmp_path):
    """24시간 내 최대 5번, 30분(1800초) 쿨다운 등 환경변수 오버라이드 유연성 검증"""
    import local_llm_gate as gate_mod

    test_file = str(tmp_path / "test_quotas_override.json")
    monkeypatch.setattr(gate_mod, "_quota_file", test_file)
    monkeypatch.setenv("RATE_LIMIT_DAILY", "5")
    monkeypatch.setenv("RATE_LIMIT_COOLDOWN_SEC", "1800")  # 30분 쿨다운
    monkeypatch.setattr(gate_mod, "_quotas_cache", {})

    req = FakeQuotaRequest(ip="198.51.100.2")
    gate_mod.check_and_consume_quota(req)

    status = gate_mod.get_quota_status(req)
    assert status["daily_limit"] == 5
    assert status["cooldown_sec"] == 1800
    assert status["cooldown_remaining_sec"] > 0
    assert status["allowed"] is False
    assert status["reason"] == "COOLDOWN_ACTIVE"


