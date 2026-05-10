"""Integration tests for CSRFTokenMiddleware.

Phase 7 step 79. Tests that the middleware correctly enforces CSRF tokens
on mutating requests when enabled, and is fully transparent when disabled.
"""
from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tether_service.app.http.csrf_middleware import CSRFTokenMiddleware
from tether_service.config.settings import CSRFTokenSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(settings: CSRFTokenSettings) -> FastAPI:
    """Minimal FastAPI app, conditionally adding CSRFTokenMiddleware.

    Mirrors the api.py factory: middleware is only wired when enabled=True.
    """
    app = FastAPI()
    if settings.enabled:
        app.add_middleware(CSRFTokenMiddleware, settings=settings)

    @app.get("/test")
    def get_test():
        return {"ok": True}

    @app.post("/test")
    def post_test():
        return {"ok": True}

    @app.get("/api/v1/healthz")
    def healthz_get():
        return {"ok": True}

    @app.post("/api/v1/healthz")
    def healthz_post():
        return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_default_off_post_passes_without_header():
    """CSRF disabled by default; POST without X-Tether-CSRF succeeds."""
    settings = CSRFTokenSettings(enabled=False)
    with TestClient(_make_app(settings), raise_server_exceptions=True) as client:
        resp = client.post("/test")
    assert resp.status_code == 200


def test_enabled_no_header_returns_403_missing():
    """CSRF enabled; POST without header → 403 csrf_token_missing."""
    settings = CSRFTokenSettings(enabled=True, token="test-token-abc")
    with TestClient(_make_app(settings)) as client:
        resp = client.post("/test")
    assert resp.status_code == 403
    body = resp.json()
    assert body["error"] == "csrf_token_missing"
    assert body["header"] == "X-Tether-CSRF"


def test_enabled_wrong_token_returns_403_invalid():
    """CSRF enabled; POST with wrong header value → 403 csrf_token_invalid."""
    settings = CSRFTokenSettings(enabled=True, token="correct-token")
    with TestClient(_make_app(settings)) as client:
        resp = client.post("/test", headers={"X-Tether-CSRF": "wrong-token"})
    assert resp.status_code == 403
    assert resp.json()["error"] == "csrf_token_invalid"


def test_enabled_correct_token_passes():
    """CSRF enabled; POST with correct token → 200."""
    settings = CSRFTokenSettings(enabled=True, token="correct-token-xyz")
    with TestClient(_make_app(settings)) as client:
        resp = client.post("/test", headers={"X-Tether-CSRF": "correct-token-xyz"})
    assert resp.status_code == 200


def test_get_exempt_even_when_enabled():
    """GET requests are always exempt from CSRF checks."""
    settings = CSRFTokenSettings(enabled=True, token="some-token")
    with TestClient(_make_app(settings)) as client:
        resp = client.get("/test")
    assert resp.status_code == 200


def test_exempt_path_post_passes_without_header():
    """POST to an exempt path bypasses CSRF check."""
    settings = CSRFTokenSettings(enabled=True, token="some-token")
    with TestClient(_make_app(settings)) as client:
        resp = client.post("/api/v1/healthz")
    assert resp.status_code == 200


def test_static_token_from_settings():
    """When token is set in settings, it is used directly (not generated)."""
    settings = CSRFTokenSettings(enabled=True, token="my-static-token")
    # Instantiate directly to inspect state without making HTTP requests.
    from unittest.mock import MagicMock
    mw = CSRFTokenMiddleware(app=MagicMock(), settings=settings)
    assert mw._token == "my-static-token"
    assert mw._token_source == "configured"


def test_generated_token_when_no_static_token():
    """When token=None, a fresh token is generated and source is 'generated'."""
    settings = CSRFTokenSettings(enabled=True, token=None)
    from unittest.mock import MagicMock
    mw = CSRFTokenMiddleware(app=MagicMock(), settings=settings)
    assert mw._token_source == "generated"
    assert len(mw._token) > 10  # token_urlsafe(32) is ~43 chars


def test_generated_token_logged_once(caplog):
    """When token=None and enabled=True, startup logs the generated token."""
    settings = CSRFTokenSettings(enabled=True, token=None)
    app = _make_app(settings)
    with caplog.at_level(logging.INFO, logger="tether_service.app.http.csrf_middleware"):
        with TestClient(app) as client:
            # First request triggers middleware stack instantiation.
            client.get("/test")
    assert any("csrf.token_generated" in r.message for r in caplog.records)


def test_custom_header_name():
    """A custom header_name is honored for both missing and present checks."""
    settings = CSRFTokenSettings(enabled=True, token="tok", header_name="X-My-CSRF")
    with TestClient(_make_app(settings)) as client:
        # Without the custom header → missing error
        resp = client.post("/test")
        assert resp.status_code == 403
        assert resp.json()["header"] == "X-My-CSRF"

        # With the custom header → passes
        resp = client.post("/test", headers={"X-My-CSRF": "tok"})
        assert resp.status_code == 200
