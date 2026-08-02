"""The abuse guardrails (apps/api/limits.py) — the code between the open internet and
the solver. Covers the four behaviors the site's safety actually rests on:

  * client IP resolution and its trust order (CF header > XFF > socket > local)
  * the sliding-window rate limit, including window expiry and the stale-IP sweep
  * the in-flight cap: concurrent gating, release on success AND on exception
  * guard_new_session: per-IP eviction (oldest first, on_evict called), global refusal,
    and the local/in-process bypass

Pure logic tests — the middleware runs under a real FastAPI app via TestClient, the
rest calls the functions directly. No solver, no nets, no sockets."""
from __future__ import annotations

import threading
import time as real_time

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from apps.api import limits


# ── plumbing ─────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Each test gets empty maps, deterministic knobs, and a neutral sweep clock."""
    limits._hits.clear()
    limits._inflight.clear()
    limits._last_sweep = 0.0
    monkeypatch.setattr(limits, "RATE_LIMIT_RPM", 5)
    monkeypatch.setattr(limits, "MAX_INFLIGHT_PER_IP", 2)
    monkeypatch.setattr(limits, "MAX_SESSIONS_TOTAL", 10)
    monkeypatch.setattr(limits, "MAX_SESSIONS_PER_IP", 3)
    yield
    limits._hits.clear()
    limits._inflight.clear()


class _FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _FakeClock()
    monkeypatch.setattr(limits.time, "time", c)
    return c


class _Req:
    """Just enough of a starlette Request for client_ip()."""
    def __init__(self, headers=None, host="9.9.9.9"):
        self.headers = headers or {}
        self.client = type("A", (), {"host": host})() if host else None


# ── client_ip: the trust order ──────────────────────────────────────────────────

def test_client_ip_prefers_cloudflare_header():
    req = _Req({"cf-connecting-ip": " 1.2.3.4 ", "x-forwarded-for": "5.6.7.8"})
    assert limits.client_ip(req) == "1.2.3.4"


def test_client_ip_falls_back_to_first_xff_hop():
    req = _Req({"x-forwarded-for": "5.6.7.8, 10.0.0.1"})
    assert limits.client_ip(req) == "5.6.7.8"


def test_client_ip_socket_then_unknown():
    assert limits.client_ip(_Req()) == "9.9.9.9"
    assert limits.client_ip(_Req(host=None)) == "unknown"


def test_client_ip_none_request_is_local_bypass():
    assert limits.client_ip(None) == "local"


# ── check_rate: sliding window ──────────────────────────────────────────────────

def test_rate_allows_up_to_limit_then_refuses(clock):
    assert all(limits.check_rate("a") for _ in range(5))
    assert not limits.check_rate("a")
    assert limits.check_rate("b")          # other IPs unaffected


def test_rate_window_slides(clock):
    for _ in range(5):
        limits.check_rate("a")
    assert not limits.check_rate("a")
    clock.now += 61.0                       # everything falls out of the window
    assert limits.check_rate("a")


def test_rate_disabled_by_zero_knob(monkeypatch, clock):
    monkeypatch.setattr(limits, "RATE_LIMIT_RPM", 0)
    assert all(limits.check_rate("a") for _ in range(1000))
    assert not limits._hits                 # disabled layer records nothing


def test_sweep_drops_quiet_ips(clock):
    for ip in ("a", "b", "c"):
        limits.check_rate(ip)
    assert len(limits._hits) == 3
    clock.now += 400.0                      # > window AND > sweep cadence
    limits.check_rate("d")                  # any request runs the amortized sweep
    assert set(limits._hits) == {"d"}


def test_sweep_spares_ips_with_work_in_flight(clock):
    limits.check_rate("a")
    limits._inflight["a"] = 1               # request still running
    clock.now += 400.0
    limits.check_rate("d")
    assert "a" in limits._hits              # kept: in-flight work pins the entry


# ── the middleware: in-flight cap, release on every path ────────────────────────

def _mini_app():
    app = FastAPI()
    app.middleware("http")(limits.limit_middleware)
    gate = threading.Event()

    @app.post("/api/play/move")
    def slow_heavy():
        gate.wait(timeout=5.0)
        return {"ok": True}

    @app.post("/api/play/new")
    def boom():
        raise RuntimeError("kaboom")

    @app.get("/api/light")
    def light():
        return {"ok": True}

    return app, gate


def test_inflight_cap_gates_concurrent_heavy_requests():
    app, gate = _mini_app()
    headers = {"cf-connecting-ip": "7.7.7.7"}
    results = []
    with TestClient(app) as client:
        threads = [threading.Thread(
            target=lambda: results.append(
                client.post("/api/play/move", headers=headers).status_code))
            for _ in range(2)]
        for t in threads:
            t.start()
        for _ in range(100):                 # wait until both hold an in-flight slot
            with limits._lock:
                if limits._inflight.get("7.7.7.7", 0) >= 2:
                    break
            real_time.sleep(0.02)
        assert limits._inflight.get("7.7.7.7", 0) == 2
        # the 3rd concurrent heavy request must bounce, and cheap ones must not
        assert client.post("/api/play/move", headers=headers).status_code == 429
        assert client.get("/api/light", headers=headers).status_code == 200
        gate.set()
        for t in threads:
            t.join()
    assert results == [200, 200]
    assert limits._inflight == {}            # fully released, entry dropped


def test_inflight_released_when_the_endpoint_raises():
    app, _gate = _mini_app()
    headers = {"cf-connecting-ip": "8.8.8.8"}
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/api/play/new", headers=headers).status_code == 500
    assert limits._inflight == {}            # the finally path cleaned up


def test_rate_limit_answers_429_with_retry_after():
    app, _gate = _mini_app()
    headers = {"cf-connecting-ip": "6.6.6.6"}
    with TestClient(app) as client:
        for _ in range(5):
            assert client.get("/api/light", headers=headers).status_code == 200
        resp = client.get("/api/light", headers=headers)
    assert resp.status_code == 429
    assert resp.headers["retry-after"]


def test_health_is_exempt():
    app, _gate = _mini_app()

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    headers = {"cf-connecting-ip": "5.5.5.5"}
    with TestClient(app) as client:
        for _ in range(50):                  # way past RATE_LIMIT_RPM=5
            assert client.get("/api/health", headers=headers).status_code == 200


# ── guard_new_session ───────────────────────────────────────────────────────────

class _Sess:
    def __init__(self, owner, last_touch):
        self.owner_ip = owner
        self.last_touch = last_touch


def _owner(s):
    return s.owner_ip


def test_guard_local_caller_bypasses_everything():
    sessions = {i: _Sess("x", i) for i in range(50)}    # way over every cap
    assert limits.guard_new_session(None, sessions, _owner) == "local"
    assert len(sessions) == 50                          # nothing evicted


def test_guard_global_cap_refuses():
    sessions = {i: _Sess(f"ip{i}", i) for i in range(10)}   # at MAX_SESSIONS_TOTAL
    with pytest.raises(HTTPException) as e:
        limits.guard_new_session(_Req({"cf-connecting-ip": "9.9.9.9"}), sessions, _owner)
    assert e.value.status_code == 429


def test_guard_per_ip_cap_evicts_oldest_and_releases_resources():
    evicted = []
    sessions = {
        "old":   _Sess("1.1.1.1", 100.0),
        "mid":   _Sess("1.1.1.1", 200.0),
        "new":   _Sess("1.1.1.1", 300.0),
        "other": _Sess("2.2.2.2", 50.0),     # older than all, but a DIFFERENT ip
    }
    ip = limits.guard_new_session(_Req({"cf-connecting-ip": "1.1.1.1"}), sessions,
                                  _owner, on_evict=evicted.append)
    assert ip == "1.1.1.1"
    assert "old" not in sessions             # own oldest went
    assert "other" in sessions               # someone else's session never touched
    assert len(evicted) == 1 and evicted[0].last_touch == 100.0


def test_guard_under_cap_touches_nothing():
    sessions = {"a": _Sess("1.1.1.1", 1.0)}
    limits.guard_new_session(_Req({"cf-connecting-ip": "1.1.1.1"}), sessions, _owner)
    assert set(sessions) == {"a"}


def test_guard_disabled_by_zero_knob(monkeypatch):
    monkeypatch.setattr(limits, "RATE_LIMIT_RPM", 0)
    sessions = {i: _Sess("1.1.1.1", i) for i in range(50)}
    limits.guard_new_session(_Req({"cf-connecting-ip": "1.1.1.1"}), sessions, _owner)
    assert len(sessions) == 50
