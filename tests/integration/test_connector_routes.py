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

from tether.app.http.routers.connectors import router as connectors_router
from tether.connectors.base import Connector
from tether.connectors.types import (
    AuthStatus,
    ConnectorState,
    HealthStatus,
    InboundEvent,
    LoginContinueResult,
    LoginPrompt,
)
from tether.core.connector_registry import ConnectorRegistry
from tether.core.interfaces import Tool
from tether.engine import Engine


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


def _make_app(
    connector: Optional[_RouteFakeConnector],
    *,
    inbox=None,
    tmp_path=None,
) -> tuple[FastAPI, ConnectorRegistry]:
    """Build a FastAPI app wired to a ConnectorRegistry.

    Mounts only the connectors router under ``/api/v1`` (other routers
    aren't needed for these tests). Returns ``(app, registry)`` so tests
    can manipulate ``registry.oauth_state`` directly.

    Phase 6.5 step 66h: ``inbox`` may be provided explicitly. When
    omitted but ``tmp_path`` is given, a fresh
    :class:`tether.context.inbox_store.SqliteInbox` is constructed
    against ``tmp_path / "inbox.db"`` so tests that exercise the
    inbox routes don't need to manually wire one. Passing
    ``inbox=None`` (and ``tmp_path=None``) reproduces the legacy
    "no inbox configured" path which now surfaces as 503.
    """
    from tether.protocol.parsers.sliding import SlidingParser

    registry = ConnectorRegistry(
        [connector] if connector is not None else [],
        data_dir=None,
        inbox=inbox,
    )
    engine = Engine(
        provider=AsyncMock(),
        parser=SlidingParser(),
        session_store=AsyncMock(),
        tools=dict(registry.aggregate_tools()),
        system_prompt="",
        connector_registry=registry,
        inbox=inbox,
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
# A2.2 — GET /connectors/{id}/inbox (Phase 6.5)
# ---------------------------------------------------------------------------


def _build_inbox(tmp_path):
    """Construct a fresh :class:`SqliteInbox` rooted at ``tmp_path``."""
    from tether.context.inbox_store import SqliteInbox

    db_path = (tmp_path / "inbox.db").as_posix()
    return SqliteInbox(f"sqlite:///{db_path}")


def _seed_events(inbox, events):
    """Helper to run ``inbox.append_many`` synchronously in a private loop."""
    import asyncio

    async def _go():
        await inbox.connect()
        try:
            return await inbox.append_many(list(events))
        finally:
            await inbox.aclose()

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_go())
    finally:
        loop.close()


def test_get_inbox_returns_503_when_inbox_unconfigured():
    """When the engine was built without an inbox, the route surfaces 503.

    Phase 6.5 (replaces the prior 501 stub): the route now actually
    queries an inbox; if none is wired the response is 503 "Inbox not
    configured" rather than the old 501 "not implemented yet".
    """
    conn = _RouteFakeConnector()
    app, _ = _make_app(conn)
    client = TestClient(app)
    resp = client.get("/api/v1/connectors/route_fake/inbox")
    assert resp.status_code == 503
    body = resp.json()
    assert "inbox" in body["detail"].lower()


def test_get_inbox_unknown_connector_returns_404(tmp_path):
    """Unknown connector id 404s before the inbox is queried."""
    conn = _RouteFakeConnector()
    inbox = _build_inbox(tmp_path)
    app, _ = _make_app(conn, inbox=inbox)
    client = TestClient(app)
    resp = client.get("/api/v1/connectors/does_not_exist/inbox")
    assert resp.status_code == 404


def test_get_inbox_unread_returns_only_unread_events(tmp_path):
    """``unread=true`` returns only inbox_seen=0 events, oldest first."""
    import datetime

    from tether.connectors.types import InboundEvent

    conn = _RouteFakeConnector()
    inbox = _build_inbox(tmp_path)
    now = datetime.datetime.now(datetime.timezone.utc)
    events = [
        InboundEvent(
            event_id=f"e{i}",
            connector_id="route_fake",
            kind="msg",
            received_at=now + datetime.timedelta(seconds=i),
            payload={"i": i},
            summary=f"summary {i}",
        )
        for i in range(3)
    ]
    inserted = _seed_events(inbox, events)
    assert inserted == 3

    app, _ = _make_app(conn, inbox=inbox)
    client = TestClient(app)
    resp = client.get(
        "/api/v1/connectors/route_fake/inbox",
        params={"unread": "true", "limit": "50"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 3
    # Oldest unread first.
    assert [e["event_id"] for e in body] == ["e0", "e1", "e2"]
    # Wire-shape sanity check.
    first = body[0]
    assert first["connector_id"] == "route_fake"
    assert first["kind"] == "msg"
    assert first["payload"] == {"i": 0}
    assert first["summary"] == "summary 0"
    assert "received_at" in first


def test_get_inbox_default_returns_recent_newest_first(tmp_path):
    """``unread=false`` (default) returns all events, newest first."""
    import datetime

    from tether.connectors.types import InboundEvent

    conn = _RouteFakeConnector()
    inbox = _build_inbox(tmp_path)
    now = datetime.datetime.now(datetime.timezone.utc)
    events = [
        InboundEvent(
            event_id=f"r{i}",
            connector_id="route_fake",
            kind="msg",
            received_at=now + datetime.timedelta(seconds=i),
            payload={"i": i},
            summary=None,
        )
        for i in range(3)
    ]
    _seed_events(inbox, events)

    app, _ = _make_app(conn, inbox=inbox)
    client = TestClient(app)
    resp = client.get("/api/v1/connectors/route_fake/inbox")
    assert resp.status_code == 200
    body = resp.json()
    assert [e["event_id"] for e in body] == ["r2", "r1", "r0"]


def test_get_inbox_limit_query_param(tmp_path):
    """``limit`` clamps the response size."""
    import datetime

    from tether.connectors.types import InboundEvent

    conn = _RouteFakeConnector()
    inbox = _build_inbox(tmp_path)
    now = datetime.datetime.now(datetime.timezone.utc)
    events = [
        InboundEvent(
            event_id=f"l{i}",
            connector_id="route_fake",
            kind="msg",
            received_at=now + datetime.timedelta(seconds=i),
            payload={"i": i},
        )
        for i in range(5)
    ]
    _seed_events(inbox, events)

    app, _ = _make_app(conn, inbox=inbox)
    client = TestClient(app)
    resp = client.get(
        "/api/v1/connectors/route_fake/inbox", params={"limit": "2"}
    )
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_post_mark_seen_flips_inbox_seen(tmp_path):
    """``POST /inbox/mark-seen`` flips inbox_seen and is idempotent."""
    import datetime

    from tether.connectors.types import InboundEvent

    conn = _RouteFakeConnector()
    inbox = _build_inbox(tmp_path)
    now = datetime.datetime.now(datetime.timezone.utc)
    events = [
        InboundEvent(
            event_id=f"m{i}",
            connector_id="route_fake",
            kind="msg",
            received_at=now + datetime.timedelta(seconds=i),
            payload={"i": i},
        )
        for i in range(3)
    ]
    _seed_events(inbox, events)

    app, _ = _make_app(conn, inbox=inbox)
    client = TestClient(app)

    # Confirm 3 unread.
    resp = client.get(
        "/api/v1/connectors/route_fake/inbox", params={"unread": "true"}
    )
    assert len(resp.json()) == 3

    # Mark first 2 seen.
    resp = client.post(
        "/api/v1/connectors/route_fake/inbox/mark-seen",
        json={"event_ids": ["m0", "m1"]},
    )
    assert resp.status_code == 200
    assert resp.json() == {"affected": 2}

    # Only m2 should remain unread.
    resp = client.get(
        "/api/v1/connectors/route_fake/inbox", params={"unread": "true"}
    )
    body = resp.json()
    assert [e["event_id"] for e in body] == ["m2"]

    # Re-marking same ids is idempotent — affected count is 0 the
    # second time because inbox_seen is already 1.
    resp = client.post(
        "/api/v1/connectors/route_fake/inbox/mark-seen",
        json={"event_ids": ["m0", "m1"]},
    )
    assert resp.json() == {"affected": 0}


def test_post_mark_seen_unknown_connector_returns_404(tmp_path):
    """Unknown connector id 404s before the inbox is touched."""
    conn = _RouteFakeConnector()
    inbox = _build_inbox(tmp_path)
    app, _ = _make_app(conn, inbox=inbox)
    client = TestClient(app)
    resp = client.post(
        "/api/v1/connectors/missing/inbox/mark-seen",
        json={"event_ids": ["x"]},
    )
    assert resp.status_code == 404


def test_post_mark_seen_empty_body_is_noop(tmp_path):
    """Empty event_ids list returns affected=0 without error."""
    conn = _RouteFakeConnector()
    inbox = _build_inbox(tmp_path)
    app, _ = _make_app(conn, inbox=inbox)
    client = TestClient(app)
    resp = client.post(
        "/api/v1/connectors/route_fake/inbox/mark-seen",
        json={"event_ids": []},
    )
    assert resp.status_code == 200
    assert resp.json() == {"affected": 0}


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
        # Phase 8 RD Fix 2: track auth state so ``auth_status`` reflects
        # post-``complete_login`` reality. Real OAuth connectors transition
        # to ``READY`` inside ``complete_login`` before returning the
        # ``LoginContinueResult``; the registry's pre-check
        # (``start_connector`` / ``start_all``) reads ``auth_status`` to
        # decide whether to call ``start()``. Without this counter, the
        # fake stays ``UNCONFIGURED`` forever and the pre-check skips
        # ``start()`` even after a successful login.
        self._auth_state: ConnectorState = ConnectorState.UNCONFIGURED
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
        return HealthStatus(state=self._auth_state)

    async def auth_status(self) -> AuthStatus:
        return AuthStatus(state=self._auth_state)

    async def begin_login(self) -> LoginPrompt:
        return self._prompt

    async def complete_login(
        self, *, payload: Dict[str, Any]
    ) -> LoginContinueResult:
        result = await self.complete_login_mock(payload=payload)
        # Mirror real-connector behaviour: transition internal state to
        # match the returned ``LoginContinueResult.state``. Required for
        # the registry's ``start_connector`` pre-check (Phase 8 RD Fix 2).
        self._auth_state = result.state
        return result

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
    from tether.protocol.parsers.sliding import SlidingParser

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
