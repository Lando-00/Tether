"""Integration tests for the connector HTTP routes (Phase 4.5 step 47e).

Per connector spec §3.8; synthesis §4 Phase 4.5. Verifies the six
routes mounted under ``/api/v1/connectors`` behave as documented.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from tether_service.app.http.routers.connectors import router as connectors_router
from tether_service.connectors.base import Connector
from tether_service.connectors.types import (
    AuthStatus,
    ConnectorState,
    HealthStatus,
    InboundEvent,
    LoginContinueResult,
    LoginPrompt,
)
from tether_service.core.connector_registry import ConnectorRegistry
from tether_service.core.interfaces import Tool
from tether_service.engine import Engine


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _StubTool(Tool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> Dict[str, Any]:
        return {"name": self._name, "parameters": {"type": "object"}}

    async def invoke(self, args, *, context=None):  # type: ignore[no-untyped-def]
        return None

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


# Per-test connector ids to avoid Connector.__init_subclass__ collisions.
class _RouteFakeConnector(Connector):
    id = "route_fake"

    def __init__(
        self,
        *,
        auth_state: ConnectorState = ConnectorState.READY,
        login_result_state: ConnectorState = ConnectorState.READY,
        next_prompt: Optional[LoginPrompt] = None,
    ) -> None:
        self._auth_state = auth_state
        self._login_result_state = login_result_state
        self._next_prompt = next_prompt

        self.start_mock = AsyncMock()
        self.stop_mock = AsyncMock()
        self.logout_mock = AsyncMock()
        self.begin_login_mock = AsyncMock(
            return_value=LoginPrompt(kind="url", payload="https://login.example/x")
        )
        self.complete_login_mock = AsyncMock(
            return_value=LoginContinueResult(
                state=login_result_state,
                detail="login complete" if login_result_state is ConnectorState.READY else "auth needed",
                next_prompt=next_prompt,
            )
        )

    async def start(self) -> None:
        await self.start_mock()

    async def stop(self) -> None:
        await self.stop_mock()

    async def logout(self) -> None:
        await self.logout_mock()

    async def health(self) -> HealthStatus:
        return HealthStatus(state=self._auth_state, detail="hello")

    async def auth_status(self) -> AuthStatus:
        return AuthStatus(state=self._auth_state, user_id="user@example.com")

    async def begin_login(self) -> LoginPrompt:
        return await self.begin_login_mock()

    async def complete_login(
        self, *, payload: Dict[str, Any]
    ) -> LoginContinueResult:
        return await self.complete_login_mock(payload=payload)

    def tools(self) -> Dict[str, Tool]:
        return {"route_fake_tool": _StubTool("route_fake_tool")}

    async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]


def _make_app(connector: Optional[_RouteFakeConnector]) -> tuple[FastAPI, ConnectorRegistry]:
    """Build a FastAPI app wired to a ConnectorRegistry.

    Mounts only the connectors router under ``/api/v1`` (other routers
    aren't needed for these tests). Returns ``(app, registry)`` so tests
    can manipulate ``registry.oauth_state`` directly.
    """
    from tether_service.protocol.parsers.sliding import SlidingParser

    registry = ConnectorRegistry(
        [connector] if connector is not None else [], data_dir=None
    )
    engine = Engine(
        provider=AsyncMock(),
        parser=SlidingParser(),
        session_store=AsyncMock(),
        tools=dict(registry.aggregate_tools()),
        system_prompt="",
        connector_registry=registry,
    )
    app = FastAPI()
    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(connectors_router)
    app.include_router(v1)
    app.state.gen_svc = engine
    return app, registry


# ---------------------------------------------------------------------------
# A2.1 — GET /connectors lists state
# ---------------------------------------------------------------------------


def test_get_list_returns_connector_states():
    conn = _RouteFakeConnector(auth_state=ConnectorState.READY)
    app, _ = _make_app(conn)
    client = TestClient(app)

    resp = client.get("/api/v1/connectors")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 1
    entry = body[0]
    assert entry["id"] == "route_fake"
    assert entry["health"]["state"] == "ready"
    assert entry["health"]["detail"] == "hello"
    assert entry["auth"]["state"] == "ready"
    assert entry["auth"]["user_id"] == "user@example.com"


def test_get_list_empty_registry():
    app, _ = _make_app(None)
    client = TestClient(app)
    resp = client.get("/api/v1/connectors")
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# A2.2 — GET /connectors/{id}/inbox returns 501
# ---------------------------------------------------------------------------


def test_get_inbox_returns_501():
    conn = _RouteFakeConnector()
    app, _ = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/api/v1/connectors/route_fake/inbox")
    assert resp.status_code == 501
    body = resp.json()
    assert "not implemented" in body["detail"].lower()


def test_get_inbox_unknown_connector_returns_404():
    conn = _RouteFakeConnector()
    app, _ = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/api/v1/connectors/does_not_exist/inbox")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# A2.3 — POST /connectors/{id}/login/begin
# ---------------------------------------------------------------------------


def test_post_login_begin_returns_login_prompt():
    conn = _RouteFakeConnector()
    app, _ = _make_app(conn)
    client = TestClient(app)
    resp = client.post("/api/v1/connectors/route_fake/login/begin")
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "url"
    assert body["payload"] == "https://login.example/x"
    conn.begin_login_mock.assert_awaited_once()


def test_post_login_begin_unknown_connector_returns_404():
    conn = _RouteFakeConnector()
    app, _ = _make_app(conn)
    client = TestClient(app)
    resp = client.post("/api/v1/connectors/missing/login/begin")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Phase 4.5 follow-up (F5): /login/begin populates oauth_state cache when
# the connector emits an ``oauth_state`` token in ``LoginPrompt.extra``.
# Per rubber-duck consensus (1m CONCERN); enables Phase 2b Gmail OAuth.
# ---------------------------------------------------------------------------


class _OAuthBeginConnector(Connector):
    """Connector fake whose ``begin_login`` emits ``oauth_state`` in extra."""

    id = "oauth_fake"

    def __init__(self, *, state_token: Optional[str] = "abc123") -> None:
        self._state_token = state_token
        extra: Dict[str, Any] = {}
        if state_token is not None:
            extra["oauth_state"] = state_token
            extra["pkce_verifier"] = "verifier-xyz"
        self._prompt = LoginPrompt(
            kind="url",
            payload="https://accounts.example.com/o/oauth2/auth?...",
            extra=extra,
        )
        self.complete_login_mock = AsyncMock(
            return_value=LoginContinueResult(state=ConnectorState.READY)
        )
        self.start_mock = AsyncMock()

    async def start(self) -> None:
        await self.start_mock()

    async def stop(self) -> None:
        return None

    async def logout(self) -> None:
        return None

    async def health(self) -> HealthStatus:
        return HealthStatus(state=ConnectorState.UNCONFIGURED)

    async def auth_status(self) -> AuthStatus:
        return AuthStatus(state=ConnectorState.UNCONFIGURED)

    async def begin_login(self) -> LoginPrompt:
        return self._prompt

    async def complete_login(
        self, *, payload: Dict[str, Any]
    ) -> LoginContinueResult:
        return await self.complete_login_mock(payload=payload)

    def tools(self) -> Dict[str, Tool]:
        return {}

    async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]


def test_login_begin_populates_oauth_state():
    """F5: when the connector returns a ``LoginPrompt`` with an
    ``oauth_state`` key in ``extra``, the route must store it in the
    registry's TTL cache so the matching ``/oauth/callback`` can
    validate the round-trip.

    Before the fix, the registry's oauth_state cache was never populated
    by ``/login/begin``, so any subsequent ``/oauth/callback`` returned
    400 on the missing-state guard.
    """
    conn = _OAuthBeginConnector(state_token="state-token-1")
    app, registry = _make_app(conn)
    client = TestClient(app)

    resp = client.post("/api/v1/connectors/oauth_fake/login/begin")
    assert resp.status_code == 200

    # The state must now live in the registry's TTL cache.
    cached = registry.oauth_state.get("state-token-1")
    assert cached is not None, "oauth_state was not populated by /login/begin"
    assert cached["connector_id"] == "oauth_fake"
    # The full ``extra`` dict (including PKCE verifier etc.) is forwarded
    # so complete_login on callback can use connector-defined data.
    assert cached["extra"] == {
        "oauth_state": "state-token-1",
        "pkce_verifier": "verifier-xyz",
    }


def test_login_begin_no_oauth_state_no_op():
    """F5: a ``LoginPrompt.extra`` without an ``oauth_state`` key must
    leave the registry's oauth_state cache untouched.

    Echo / QR-flow connectors don't emit oauth_state; the populator
    must not invent one for them.
    """
    conn = _OAuthBeginConnector(state_token=None)
    app, registry = _make_app(conn)
    client = TestClient(app)

    resp = client.post("/api/v1/connectors/oauth_fake/login/begin")
    assert resp.status_code == 200
    # Cache stays empty.
    assert len(registry.oauth_state) == 0


def test_oauth_callback_uses_cached_state():
    """F5: end-to-end — ``/login/begin`` populates oauth_state, then
    ``/oauth/callback?state=...`` consumes it and forwards to
    ``complete_login`` with the cached payload.

    Verifies the begin → callback handshake actually closes (the bug
    1m flagged: producer + consumer were disconnected, so callback
    always 400'd).
    """
    conn = _OAuthBeginConnector(state_token="round-trip-state")
    app, registry = _make_app(conn)
    client = TestClient(app)

    # Step 1: begin_login populates state.
    resp = client.post("/api/v1/connectors/oauth_fake/login/begin")
    assert resp.status_code == 200
    assert registry.oauth_state.get("round-trip-state") is not None

    # Step 2: callback consumes state + forwards to complete_login.
    resp = client.get(
        "/api/v1/connectors/oauth_fake/oauth/callback",
        params={"state": "round-trip-state", "code": "auth-code-Y"},
    )
    assert resp.status_code == 200
    assert resp.json()["state"] == "ready"

    # State token consumed (single-use).
    assert registry.oauth_state.get("round-trip-state") is None

    conn.complete_login_mock.assert_awaited_once()
    payload = conn.complete_login_mock.await_args.kwargs["payload"]
    assert payload["state"] == "round-trip-state"
    assert payload["code"] == "auth-code-Y"
    assert payload["state_payload"]["connector_id"] == "oauth_fake"
    assert payload["state_payload"]["extra"]["oauth_state"] == "round-trip-state"

    # On READY, start_connector ran.
    conn.start_mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# A2.4 — POST /connectors/{id}/login/complete
# ---------------------------------------------------------------------------


def test_post_login_complete_then_start():
    """On READY result, registry.start_connector(id) is invoked."""
    conn = _RouteFakeConnector(login_result_state=ConnectorState.READY)
    app, registry = _make_app(conn)
    client = TestClient(app)

    resp = client.post(
        "/api/v1/connectors/route_fake/login/complete",
        json={"payload": {"code": "abc123"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "ready"

    # complete_login was called with payload={'code':'abc123'}
    conn.complete_login_mock.assert_awaited_once_with(
        payload={"code": "abc123"}
    )
    # start_connector(id) was called → connector.start() invoked.
    conn.start_mock.assert_awaited_once()


def test_post_login_complete_authenticating_does_not_start():
    """If complete_login returns AUTHENTICATING (e.g. MFA next step),
    start_connector is NOT called."""
    next_p = LoginPrompt(kind="code", payload="enter SMS code")
    conn = _RouteFakeConnector(
        login_result_state=ConnectorState.AUTHENTICATING,
        next_prompt=next_p,
    )
    app, _ = _make_app(conn)
    client = TestClient(app)
    resp = client.post(
        "/api/v1/connectors/route_fake/login/complete",
        json={"payload": {"code": "wrong"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "authenticating"
    assert body["next_prompt"]["kind"] == "code"
    conn.start_mock.assert_not_awaited()


def test_post_login_complete_unknown_connector_returns_404():
    conn = _RouteFakeConnector()
    app, _ = _make_app(conn)
    client = TestClient(app)
    resp = client.post(
        "/api/v1/connectors/missing/login/complete",
        json={"payload": {}},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# A2.5 — GET /connectors/{id}/oauth/callback
# ---------------------------------------------------------------------------


def test_get_oauth_callback_consumes_state():
    """Pre-set state in registry.oauth_state → callback consumes it,
    forwards to complete_login, returns success."""
    conn = _RouteFakeConnector(login_result_state=ConnectorState.READY)
    app, registry = _make_app(conn)
    client = TestClient(app)

    registry.oauth_state.set("statetoken-1", {"connector_id": "route_fake"})

    resp = client.get(
        "/api/v1/connectors/route_fake/oauth/callback",
        params={"state": "statetoken-1", "code": "auth-code-X"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "ready"

    # The state token is consumed (popped).
    assert registry.oauth_state.get("statetoken-1") is None

    # complete_login received {state, code, state_payload}.
    conn.complete_login_mock.assert_awaited_once()
    call_kwargs = conn.complete_login_mock.await_args.kwargs
    payload = call_kwargs["payload"]
    assert payload["state"] == "statetoken-1"
    assert payload["code"] == "auth-code-X"
    assert payload["state_payload"] == {"connector_id": "route_fake"}

    # On READY, start_connector(id) ran.
    conn.start_mock.assert_awaited_once()


def test_get_oauth_callback_invalid_state_returns_400():
    """Missing/expired state → 400, complete_login NOT called."""
    conn = _RouteFakeConnector()
    app, _ = _make_app(conn)
    client = TestClient(app)
    resp = client.get(
        "/api/v1/connectors/route_fake/oauth/callback",
        params={"state": "never-set", "code": "X"},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert "state" in body["detail"].lower()
    conn.complete_login_mock.assert_not_awaited()


def test_get_oauth_callback_unknown_connector_returns_404():
    conn = _RouteFakeConnector()
    app, _ = _make_app(conn)
    client = TestClient(app)
    resp = client.get(
        "/api/v1/connectors/missing/oauth/callback",
        params={"state": "x", "code": "y"},
    )
    assert resp.status_code == 404


def test_get_oauth_callback_missing_query_params_returns_422():
    """FastAPI validates Query(...) requireds → 422 on missing."""
    conn = _RouteFakeConnector()
    app, _ = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/api/v1/connectors/route_fake/oauth/callback")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# A2.6 — POST /connectors/{id}/logout
# ---------------------------------------------------------------------------


def test_post_logout_calls_logout():
    conn = _RouteFakeConnector()
    app, _ = _make_app(conn)
    client = TestClient(app)
    resp = client.post("/api/v1/connectors/route_fake/logout")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["id"] == "route_fake"
    # Phase 4.5 follow-up (F2): the route now reports the LOGGED_OUT
    # state and also stops the connector via registry.stop_connector.
    assert body["state"] == ConnectorState.LOGGED_OUT.value
    conn.logout_mock.assert_awaited_once()
    conn.stop_mock.assert_awaited_once()


def test_post_logout_unknown_connector_returns_404():
    conn = _RouteFakeConnector()
    app, _ = _make_app(conn)
    client = TestClient(app)
    resp = client.post("/api/v1/connectors/missing/logout")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Edge case: engine without ConnectorRegistry → 503
# ---------------------------------------------------------------------------


def test_routes_without_connector_registry_return_503():
    """An Engine constructed without ``connector_registry=...`` (legacy
    direct-constructor path) lacks the attribute; routes return 503."""
    from tether_service.protocol.parsers.sliding import SlidingParser

    engine = Engine(
        provider=AsyncMock(),
        parser=SlidingParser(),
        session_store=AsyncMock(),
        tools={},
        system_prompt="",
    )
    assert engine.connector_registry is None

    app = FastAPI()
    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(connectors_router)
    app.include_router(v1)
    app.state.gen_svc = engine
    client = TestClient(app)

    for path, method in (
        ("/api/v1/connectors", "GET"),
        ("/api/v1/connectors/x/inbox", "GET"),
        ("/api/v1/connectors/x/login/begin", "POST"),
        ("/api/v1/connectors/x/logout", "POST"),
    ):
        resp = client.request(method, path)
        assert resp.status_code == 503, f"{method} {path} expected 503, got {resp.status_code}"
