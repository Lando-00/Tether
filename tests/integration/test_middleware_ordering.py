"""FIX 1 — middleware add-order vs runtime order.

Phase 7 RD followup. Locks the contract that with all 4 middlewares enabled:

  Runtime order (outermost → innermost):
    RequestId → TrustedHost → CORS → CSRF → handler

So:
  * A request with a bad ``Host`` header is rejected by TrustedHost (400)
    BEFORE CSRF runs — no CSRF logging, no token comparison — and the
    response still carries ``X-Request-ID`` because RequestId is outermost.
  * A request with a good Host but missing CSRF token is rejected by CSRF
    (403) — TrustedHost let it through, CSRF rejected — and the response
    still carries ``X-Request-ID``.
  * A fully-valid request returns 200 and carries ``X-Request-ID``.

The bug being fixed: previously the middlewares were added in source order
``TrustedHost → CORS → CSRF → RequestId``. Starlette interprets last-added
as OUTERMOST, which inverted the runtime order to
``RequestId → CSRF → CORS → TrustedHost`` — meaning CSRF processed bad-Host
requests before TrustedHost rejected them, and the 400-from-TrustedHost
response wrapped only RequestId (correct), but the conceptual security
posture was inverted vs the docstring's claim.

These tests use a minimal FastAPI app that mirrors the conditional
``app.add_middleware(...)`` wiring in :func:`create_app` — directly using
``create_app`` would require building a fully-initialised Engine which
isn't relevant to middleware ordering.
"""
from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.testclient import TestClient

from tether.app.http.csrf_middleware import CSRFTokenMiddleware
from tether.app.http.middleware import RequestIdMiddleware
from tether.config.settings import (
    CORSSettings,
    CSRFTokenSettings,
    TrustedHostSettings,
)


def _make_app_all_enabled(
    *,
    csrf: CSRFTokenSettings,
    cors: CORSSettings,
    trusted_host: TrustedHostSettings,
) -> FastAPI:
    """Mirror the production ordering in :func:`create_app`.

    Source-add order (CSRF first, RequestId last) ⇒ runtime order
    (RequestId outermost, CSRF innermost).
    """
    app = FastAPI()

    if csrf.enabled:
        app.add_middleware(CSRFTokenMiddleware, settings=csrf)

    if cors.enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors.allow_origins,
            allow_methods=cors.allow_methods,
            allow_headers=cors.allow_headers,
            allow_credentials=cors.allow_credentials,
        )

    if trusted_host.enabled:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=list(trusted_host.allowed_hosts),
        )

    # Outermost: RequestId.
    app.add_middleware(RequestIdMiddleware)

    @app.post("/api/v1/chat/stream")
    def chat():
        return {"ok": True}

    @app.get("/api/v1/chat/stream")
    def chat_get():
        return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# Test 1: TrustedHost runs OUTSIDE CSRF — bad Host gets 400, no CSRF logs.
# ---------------------------------------------------------------------------


def test_trusted_host_rejection_carries_request_id_and_skips_csrf(caplog):
    """Bad Host header → 400 from TrustedHost; CSRF middleware is never
    reached; X-Request-ID still present (RequestId is outermost)."""
    csrf = CSRFTokenSettings(enabled=True, token="t-tok")
    cors = CORSSettings(enabled=True, allow_origins=["http://good.com"])
    th = TrustedHostSettings(enabled=True, allowed_hosts=["good.com"])

    app = _make_app_all_enabled(csrf=csrf, cors=cors, trusted_host=th)

    with caplog.at_level(logging.DEBUG, logger="tether.app.http.csrf_middleware"):
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/api/v1/chat/stream",
                headers={"Host": "evil.com"},
            )

    # TrustedHost rejected: 400. CSRF never had a chance to return 403.
    assert resp.status_code == 400, (
        f"Expected 400 from TrustedHost (bad Host), got {resp.status_code}"
    )
    # RequestId is OUTERMOST: even the 400 carries X-Request-ID.
    assert "x-request-id" in resp.headers, (
        "TrustedHost-rejected response must carry X-Request-ID — "
        "RequestIdMiddleware must be outermost (added last)."
    )
    assert resp.headers["x-request-id"].startswith("req-")

    # CSRF must NOT have run for a TrustedHost-rejected request. The CSRF
    # logger emits records with names beginning with "csrf." for any path it
    # processes (token_generated on init, plus per-request decisions). Since
    # init happens once at app construction it will always log token_generated;
    # what we assert is that NO PER-REQUEST csrf event log fired.
    csrf_request_records = [
        r
        for r in caplog.records
        if r.name == "tether.app.http.csrf_middleware"
        and "csrf.token_generated" not in r.getMessage()
    ]
    assert csrf_request_records == [], (
        "CSRF middleware should not process bad-Host requests — they must be "
        f"rejected by TrustedHost first. Got: {[r.getMessage() for r in csrf_request_records]}"
    )


# ---------------------------------------------------------------------------
# Test 2: CSRF rejection on good Host carries X-Request-ID.
# ---------------------------------------------------------------------------


def test_csrf_rejection_on_good_host_carries_request_id():
    """Good Host but missing CSRF token → 403; X-Request-ID still present.
    TrustedHost let it through (path correct), CSRF rejected with 403."""
    csrf = CSRFTokenSettings(enabled=True, token="t-tok")
    cors = CORSSettings(enabled=True, allow_origins=["http://good.com"])
    th = TrustedHostSettings(enabled=True, allowed_hosts=["good.com", "testserver"])

    app = _make_app_all_enabled(csrf=csrf, cors=cors, trusted_host=th)

    with TestClient(app) as client:
        # Default Host is "testserver" which is in allowed_hosts; omit
        # X-Tether-CSRF so CSRF rejects.
        resp = client.post("/api/v1/chat/stream")

    assert resp.status_code == 403
    body = resp.json()
    assert body["error"] == "csrf_token_missing"
    # Outermost middleware still annotated the response.
    assert "x-request-id" in resp.headers
    assert resp.headers["x-request-id"].startswith("req-")


# ---------------------------------------------------------------------------
# Test 3: Happy path — valid Host + valid token → 200 + X-Request-ID.
# ---------------------------------------------------------------------------


def test_happy_path_passes_all_middlewares_with_request_id():
    """Valid Host AND valid CSRF token → 200; response carries X-Request-ID
    AND CORS allow-origin headers (when Origin matches)."""
    csrf = CSRFTokenSettings(enabled=True, token="t-tok")
    cors = CORSSettings(enabled=True, allow_origins=["http://good.com"])
    th = TrustedHostSettings(enabled=True, allowed_hosts=["good.com", "testserver"])

    app = _make_app_all_enabled(csrf=csrf, cors=cors, trusted_host=th)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/chat/stream",
            headers={"X-Tether-CSRF": "t-tok"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert "x-request-id" in resp.headers
    assert resp.headers["x-request-id"].startswith("req-")
