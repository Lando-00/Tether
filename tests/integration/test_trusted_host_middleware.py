"""Integration tests for TrustedHostMiddleware wired via SecuritySettings.

Phase 7 step 79. Tests that untrusted Host headers are rejected when the
middleware is enabled, and pass through when disabled (default).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.testclient import TestClient

from tether.config.settings import TrustedHostSettings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(settings: TrustedHostSettings) -> FastAPI:
    """Minimal FastAPI app, conditionally adding TrustedHostMiddleware."""
    app = FastAPI()
    if settings.enabled:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=list(settings.allowed_hosts),
        )

    @app.get("/test")
    def get_test():
        return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_explicit_disabled_any_host_passes():
    """TrustedHost explicitly disabled — any Host header passes through.

    P0-B2 (Phase 9) flipped ``enabled`` to default-on; this test now
    constructs ``enabled=False`` explicitly to exercise the disabled path.
    """
    settings = TrustedHostSettings(enabled=False)
    with TestClient(_make_app(settings)) as client:
        resp = client.get("/test", headers={"Host": "evil.example.com"})
    assert resp.status_code == 200


def test_enabled_trusted_host_passes():
    """TrustedHost enabled; Host: localhost → 200."""
    settings = TrustedHostSettings(
        enabled=True,
        allowed_hosts=["localhost", "127.0.0.1"],
    )
    with TestClient(_make_app(settings)) as client:
        resp = client.get("/test", headers={"Host": "localhost"})
    assert resp.status_code == 200


def test_enabled_untrusted_host_rejected():
    """TrustedHost enabled; Host: evil.com → 400."""
    settings = TrustedHostSettings(
        enabled=True,
        allowed_hosts=["localhost", "127.0.0.1"],
    )
    with TestClient(_make_app(settings), raise_server_exceptions=False) as client:
        resp = client.get("/test", headers={"Host": "evil.com"})
    assert resp.status_code == 400


def test_wildcard_allowed_hosts_accepts_any():
    """allowed_hosts=['*'] — any Host header is accepted."""
    settings = TrustedHostSettings(
        enabled=True,
        allowed_hosts=["*"],
    )
    with TestClient(_make_app(settings)) as client:
        resp = client.get("/test", headers={"Host": "anything.example.com"})
    assert resp.status_code == 200


def test_enabled_127_passes():
    """TrustedHost enabled; Host: 127.0.0.1 is in default allowed list → 200."""
    settings = TrustedHostSettings(
        enabled=True,
        allowed_hosts=["localhost", "127.0.0.1"],
    )
    with TestClient(_make_app(settings)) as client:
        resp = client.get("/test", headers={"Host": "127.0.0.1"})
    assert resp.status_code == 200
