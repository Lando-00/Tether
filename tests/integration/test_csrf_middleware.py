"""Integration tests for CSRFTokenMiddleware.

Phase 7 step 79. Tests that the middleware correctly enforces CSRF tokens
on mutating requests when enabled, and is fully transparent when disabled.
"""
from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tether.app.http.csrf_middleware import CSRFTokenMiddleware
from tether.app.http.middleware import RequestIdMiddleware
from tether.config.settings import CSRFTokenSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(settings: CSRFTokenSettings, *, with_request_id: bool = False) -> FastAPI:
    """Minimal FastAPI app, conditionally adding CSRFTokenMiddleware.

    Mirrors the api.py factory: security middleware is only wired when
    enabled=True; RequestIdMiddleware is added LAST (outermost) when
    ``with_request_id=True`` so ordering tests can assert every response
    carries X-Request-ID.
    """
    app = FastAPI()
    if settings.enabled:
        app.add_middleware(CSRFTokenMiddleware, settings=settings)
    if with_request_id:
        # Added last = outermost, mirroring the fixed api.py ordering.
        app.add_middleware(RequestIdMiddleware)

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


def test_generated_token_logged_without_value(caplog):
    """Structured log has source/token_chars but NOT the raw token value.

    The token must not appear in the log record — it's a long-lived secret
    and the JSON log file is append-only.  The actual token should only
    appear in stderr (tested separately).
    """
    settings = CSRFTokenSettings(enabled=True, token=None)
    app = _make_app(settings)
    with caplog.at_level(logging.INFO, logger="tether.app.http.csrf_middleware"):
        with TestClient(app) as client:
            # First request triggers middleware stack instantiation.
            client.get("/test")

    log_records = [r for r in caplog.records if "csrf.token_generated" in r.message]
    assert log_records, "Expected csrf.token_generated log record"
    record = log_records[0]
    # Metadata present
    assert record.__dict__.get("source") == "secrets.token_urlsafe(32)"
    assert record.__dict__.get("token_chars") == 43  # token_urlsafe(32) → 43 chars
    # Raw token value must NOT appear in the log
    assert "token" not in record.__dict__ or record.__dict__.get("token") is None


def test_generated_token_persisted_to_file(tmp_path):
    """P0-B3: the actual token value is written to a 0600 file (not stderr).

    Pre-P0-B3 this was printed to stderr; ADR-0012 promised a token
    file but didn't ship one. The persisted file is now the bootstrap
    contract for CLI clients (see ``tests/unit/app/test_csrf_token_file.py``
    for the atomic-write / 0o600 / no-residue regressions).
    """
    token_file = tmp_path / "csrf_token"
    settings = CSRFTokenSettings(enabled=True, token=None, token_file=token_file)
    from unittest.mock import MagicMock
    mw = CSRFTokenMiddleware(app=MagicMock(), settings=settings)
    assert token_file.exists(), "Token file must be written on startup"
    assert token_file.read_text(encoding="utf-8").strip() == mw._token


def test_generated_token_stderr_fallback_on_write_failure(capsys, monkeypatch):
    """If the token-file write raises ``OSError``, fall back to stderr.

    Defense in depth: an unwritable filesystem (read-only mount, denied
    ACL) must never silently lose the token, otherwise the operator has
    no way to recover the CSRF secret.
    """
    from tether.app.http import csrf_middleware as mw_mod

    def _boom(path, token):
        raise OSError("simulated read-only fs")

    monkeypatch.setattr(mw_mod, "_atomic_write_token", _boom)

    settings = CSRFTokenSettings(enabled=True, token=None)
    from unittest.mock import MagicMock
    mw = mw_mod.CSRFTokenMiddleware(app=MagicMock(), settings=settings)
    captured = capsys.readouterr()
    assert mw._token in captured.err, "Token must appear in stderr fallback"
    assert "[Tether] CSRF token generated:" in captured.err
    assert "X-Tether-CSRF" in captured.err  # header name hint


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


def test_csrf_403_carries_x_request_id():
    """CSRF 403 responses include X-Request-ID (RequestId is outermost).

    Locks the middleware ordering fix: RequestIdMiddleware must wrap CSRF
    so that even rejected requests get a correlation ID. Phase 7 step 79.
    """
    settings = CSRFTokenSettings(enabled=True, token="some-tok")
    with TestClient(_make_app(settings, with_request_id=True)) as client:
        # Missing token → 403
        resp = client.post("/test")
    assert resp.status_code == 403
    assert "x-request-id" in resp.headers, (
        "403 response must carry X-Request-ID — "
        "RequestIdMiddleware must be outermost (added last)."
    )
    assert resp.headers["x-request-id"].startswith("req-")

