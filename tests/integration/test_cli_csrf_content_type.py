"""Integration test: CLI mutating headers clear the middleware stack.

Closes the gap that let Phase-9 P0-B2's ``RequireJsonContentTypeMiddleware``
silently break ``tether-cli connect`` / ``logout`` / ``\\new`` / ``\\unload``.

Background
----------

The CLI's :func:`tether.cli.main._mutating_headers` returns the HTTP
headers attached to every POST/PUT/PATCH/DELETE request. Before this
fix it only set ``X-Tether-CSRF`` (when configured) and relied on the
``requests`` library to auto-set ``Content-Type: application/json`` —
which it only does when a ``json=`` body is supplied. Every CLI call
that POSTs with no body (login/begin, logout, /sessions, /unload) was
therefore rejected with HTTP 415 by
:class:`tether.app.http.content_type_middleware.RequireJsonContentTypeMiddleware`.

The existing CLI tests mock ``requests.post`` directly so the middleware
was never on the path; this test mounts the **real** middleware stack
around a small FastAPI app that pretends to be ``/api/v1/connectors``
and asserts that a no-body ``POST`` carrying the CLI's
``_mutating_headers()`` value lands clean (NOT 415).

Citations: plan §14; ``src/tether/cli/main.py:54`` (_mutating_headers);
``src/tether/app/http/content_type_middleware.py:22`` (middleware).
"""
from __future__ import annotations

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from tether.app.http.content_type_middleware import RequireJsonContentTypeMiddleware
from tether.cli.main import _mutating_headers


def _build_app_with_content_type_middleware() -> TestClient:
    """Minimal FastAPI app: a single POST route + the real middleware.

    No connectors, no engine, no auth — just enough to verify the
    middleware accepts the CLI's mutating headers. The route returns
    200 with a static body so we can distinguish "middleware passed"
    (200) from "middleware rejected" (415).
    """
    app = FastAPI()
    router = APIRouter(prefix="/api/v1/connectors")

    @router.post("/{connector_id}/login/begin")
    async def begin_login_stub(connector_id: str) -> dict:
        return {"ok": True, "id": connector_id}

    @router.post("/{connector_id}/logout")
    async def logout_stub(connector_id: str) -> dict:
        return {"ok": True, "id": connector_id}

    app.include_router(router)
    app.add_middleware(RequireJsonContentTypeMiddleware)
    return TestClient(app)


def test_mutating_headers_includes_application_json_content_type() -> None:
    """The CLI helper must set ``Content-Type: application/json``.

    Direct unit assertion — the cheapest possible regression guard.
    Without this, every no-body mutating CLI call fails 415.
    """
    headers = _mutating_headers()
    assert headers.get("Content-Type") == "application/json", (
        f"_mutating_headers() must default Content-Type to application/json; "
        f"got headers={headers!r}"
    )


def test_mutating_headers_extra_overrides_content_type() -> None:
    """Per-call ``extra`` wins (e.g. the chat stream uses NDJSON ``Accept``)."""
    headers = _mutating_headers({"Accept": "application/x-ndjson; version=1.0"})
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "application/x-ndjson; version=1.0"


def test_cli_login_begin_post_passes_content_type_middleware() -> None:
    """End-to-end: CLI's mutating POST clears the middleware (NOT 415).

    Mirrors the path the user hits when running
    ``tether-cli connect whatsapp``: a no-body POST to
    ``/api/v1/connectors/{id}/login/begin``. Before the fix this was a
    deterministic 415; after the fix it returns the route's 200.
    """
    client = _build_app_with_content_type_middleware()

    response = client.post(
        "/api/v1/connectors/whatsapp/login/begin",
        headers=_mutating_headers(),
    )

    assert response.status_code != 415, (
        f"CLI mutating headers must clear RequireJsonContentTypeMiddleware; "
        f"got {response.status_code} with body {response.text!r}"
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True, "id": "whatsapp"}


def test_cli_logout_post_passes_content_type_middleware() -> None:
    """The other broken-pre-fix endpoint: POST /logout with no body."""
    client = _build_app_with_content_type_middleware()

    response = client.post(
        "/api/v1/connectors/whatsapp/logout",
        headers=_mutating_headers(),
    )

    assert response.status_code != 415
    assert response.status_code == 200


def test_unset_content_type_yields_415_baseline() -> None:
    """Sanity check the middleware is actually active.

    If this asserts a 200, the middleware is bypassed and the
    happy-path tests above are vacuous. The expected 415 confirms
    the middleware really is on the path and that the fix's value
    comes from the header injection, not from middleware absence.
    """
    client = _build_app_with_content_type_middleware()

    response = client.post(
        "/api/v1/connectors/whatsapp/login/begin",
        # No Content-Type header — should trip the middleware.
    )

    assert response.status_code == 415, (
        f"Baseline: middleware should reject no-Content-Type mutating POST; "
        f"got {response.status_code}. If this fails, the test app is "
        f"misconfigured and the happy-path tests above are not actually "
        f"exercising the middleware."
    )
