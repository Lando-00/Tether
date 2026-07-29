"""Integration tests for CORSMiddleware wired via SecuritySettings.

Phase 7 step 79. Tests that CORS headers are absent when disabled (default)
and correctly added when enabled with matching origins.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from tether.config.settings import CORSSettings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(settings: CORSSettings) -> FastAPI:
    """Minimal FastAPI app, conditionally adding CORSMiddleware."""
    app = FastAPI()
    if settings.enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allow_origins,
            allow_methods=settings.allow_methods,
            allow_headers=settings.allow_headers,
            allow_credentials=settings.allow_credentials,
        )

    @app.get("/test")
    def get_test():
        return {"ok": True}

    @app.options("/test")
    def options_test():
        return {}

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_default_off_no_acao_header():
    """CORS disabled by default — no Access-Control-Allow-Origin on response."""
    settings = CORSSettings(enabled=False)
    with TestClient(_make_app(settings)) as client:
        resp = client.options(
            "/test",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
    assert "access-control-allow-origin" not in resp.headers


def test_enabled_matching_origin_gets_acao():
    """CORS enabled, matching Origin → Access-Control-Allow-Origin present."""
    settings = CORSSettings(
        enabled=True,
        allow_origins=["http://localhost:3000"],
    )
    with TestClient(_make_app(settings)) as client:
        resp = client.options(
            "/test",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_enabled_non_matching_origin_no_acao():
    """CORS enabled but origin not in allowlist → no ACAO header."""
    settings = CORSSettings(
        enabled=True,
        allow_origins=["http://localhost:3000"],
    )
    with TestClient(_make_app(settings)) as client:
        resp = client.options(
            "/test",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
    acao = resp.headers.get("access-control-allow-origin", "")
    assert acao != "http://evil.example.com"


def test_wildcard_origin_accepts_any():
    """allow_origins=['*'] — any Origin is reflected."""
    settings = CORSSettings(
        enabled=True,
        allow_origins=["*"],
    )
    with TestClient(_make_app(settings)) as client:
        resp = client.options(
            "/test",
            headers={
                "Origin": "http://anything.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
    assert resp.headers.get("access-control-allow-origin") == "*"


def test_cors_does_not_break_normal_get():
    """CORS enabled does not break regular GET requests."""
    settings = CORSSettings(
        enabled=True,
        allow_origins=["http://localhost:3000"],
    )
    with TestClient(_make_app(settings)) as client:
        resp = client.get("/test", headers={"Origin": "http://localhost:3000"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
