"""Phase 4.5 follow-up (rubber-duck consensus, gpt-5.5 BLOCKING #1):
verify the FastAPI lifespan enters and exits the :class:`Engine` async
context, runs tool ``startup``, and schedules ``start_connector`` for
every READY connector.

Before the fix, the lifespan was just ``yield`` + ``aclose`` — it never
called ``__aenter__``, so in production HTTP serving every web_search
call returned ``"web_search not initialised"`` (BraveSearchClient never
opened) and READY connectors never auto-started after a server restart.
The pre-existing ``test_lifespan_calls_engine_aclose`` only verified
``aclose`` was awaited, missing the BLOCKING bug.

CRITICAL test discipline: every test in this file uses
``with TestClient(app) as client:`` (the context manager triggers
lifespan). Bare ``TestClient(app)`` does NOT, so it would not catch
the bug.

Synthesis §4 Phase 4.5 follow-up; connector spec §3.3 step 4.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict
from unittest.mock import AsyncMock, patch

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from tether.app.http.api import create_app, lifespan
from tether.app.http.routers.health import router as health_router
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
from tether.protocol.parsers.sliding import SlidingParser
from tether.providers.dummy.provider import DummyProvider

# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------


class _StartupSpyTool(Tool):
    """Tool fake that flips ``startup_called``/``shutdown_called`` flags."""

    def __init__(self, name: str = "spy_tool") -> None:
        self._name = name
        self.startup_called = False
        self.shutdown_called = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> Dict[str, Any]:
        return {"name": self._name, "parameters": {"type": "object"}}

    async def invoke(self, args, *, context=None):  # type: ignore[no-untyped-def]
        return None

    async def startup(self) -> None:
        self.startup_called = True

    async def shutdown(self) -> None:
        self.shutdown_called = True


class _ReadyConnector(Connector):
    """Connector fake reporting READY in ``auth_status``; tracks start/stop."""

    id = "ready_fake"

    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0

    async def start(self) -> None:
        self.start_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1

    async def logout(self) -> None:
        return None

    async def health(self) -> HealthStatus:
        return HealthStatus(state=ConnectorState.READY)

    async def auth_status(self) -> AuthStatus:
        return AuthStatus(state=ConnectorState.READY)

    async def begin_login(self) -> LoginPrompt:
        return LoginPrompt(kind="url", payload="x")

    async def complete_login(self, *, payload: Dict[str, Any]) -> LoginContinueResult:
        return LoginContinueResult(state=ConnectorState.READY)

    def tools(self) -> Dict[str, Tool]:
        return {}

    async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
        if False:  # pragma: no cover - empty stream
            yield  # type: ignore[unreachable]


def _build_lifespan_app(engine: Engine) -> FastAPI:
    """Wire a minimal FastAPI app with our lifespan + a fake engine.

    Avoids ``create_app()`` so tests don't depend on default.yml's
    real provider/tool stack — we only want to assert the lifespan
    semantics around ``app.state.gen_svc``.
    """
    app = FastAPI(lifespan=lifespan)
    app.state.gen_svc = engine
    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(health_router)
    app.include_router(v1)
    return app


# ---------------------------------------------------------------------------
# A1.1 — lifespan enters Engine __aenter__ exactly once
# ---------------------------------------------------------------------------


def test_lifespan_enters_engine() -> None:
    """``with TestClient(app)`` must invoke ``Engine.__aenter__`` once on
    context start, before any HTTP request is served.

    This is the load-bearing assertion for the Phase 4.5 BLOCKING fix:
    the previous lifespan body skipped ``__aenter__`` entirely.
    """
    engine = Engine(
        provider=DummyProvider(),
        parser=SlidingParser(),
        session_store=AsyncMock(),
        tools={},
        system_prompt="",
    )

    aenter_mock = AsyncMock(return_value=engine)
    aexit_mock = AsyncMock(return_value=None)

    with patch.object(Engine, "__aenter__", aenter_mock), patch.object(
        Engine, "__aexit__", aexit_mock
    ):
        app = _build_lifespan_app(engine)
        with TestClient(app):
            # Lifespan startup has run; aenter must have been called once.
            aenter_mock.assert_awaited_once()

    # On context exit, aexit also runs exactly once.
    aexit_mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# A1.2 — tool startup actually fires during lifespan
# ---------------------------------------------------------------------------


def test_lifespan_runs_tool_startup() -> None:
    """A fake tool with a ``startup_called`` flag must have it flipped
    during the TestClient lifespan.

    Before the fix, ``startup_all`` was never invoked by the production
    HTTP path — every tool that opened resources in ``startup`` (e.g.,
    :class:`BraveSearchClient`) silently stayed unintialised.
    """
    spy = _StartupSpyTool(name="spy_tool")
    engine = Engine(
        provider=DummyProvider(),
        parser=SlidingParser(),
        session_store=AsyncMock(),
        tools={"spy_tool": spy},
        system_prompt="",
    )

    app = _build_lifespan_app(engine)
    assert spy.startup_called is False

    with TestClient(app):
        # By the time TestClient is inside the ``with``, lifespan startup
        # has completed → ``__aenter__`` ran ``startup_all`` → the spy
        # tool's ``startup`` was awaited.
        assert spy.startup_called is True
        assert spy.shutdown_called is False

    # On exit, shutdown_all fired (via __aexit__ → aclose).
    assert spy.shutdown_called is True


# ---------------------------------------------------------------------------
# A1.3 — lifespan exits via __aexit__ (which routes through aclose)
# ---------------------------------------------------------------------------


def test_lifespan_exits_engine() -> None:
    """``aclose`` must be awaited on lifespan teardown (via ``__aexit__``).

    Preserves the contract of the existing
    ``test_lifespan_calls_engine_aclose`` test: even though the lifespan
    now calls ``__aexit__`` (which internally calls ``aclose``), the
    end-state behaviour — engine torn down on app shutdown — must hold.
    """
    engine = Engine(
        provider=DummyProvider(),
        parser=SlidingParser(),
        session_store=AsyncMock(),
        tools={},
        system_prompt="",
    )
    aclose_mock = AsyncMock()
    engine.aclose = aclose_mock  # type: ignore[method-assign]

    app = _build_lifespan_app(engine)
    with TestClient(app):
        aclose_mock.assert_not_awaited()

    aclose_mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# A1.4 — READY connectors get start_connector during lifespan
# ---------------------------------------------------------------------------


def test_lifespan_starts_ready_connectors(tmp_path) -> None:
    """A connector reporting READY in ``auth_status`` must be auto-started
    by ``Engine.__aenter__`` (Phase 4.5 step 47d), so the production
    HTTP path doesn't strand READY connectors behind a manual
    ``/login/complete`` after every restart.
    """
    conn = _ReadyConnector()
    registry = ConnectorRegistry([conn], data_dir=tmp_path)
    engine = Engine(
        provider=DummyProvider(),
        parser=SlidingParser(),
        session_store=AsyncMock(),
        tools={},
        system_prompt="",
        connector_registry=registry,
    )

    app = _build_lifespan_app(engine)
    with TestClient(app):
        # ``__aenter__`` schedules start_connector as a background task;
        # block briefly to let the task settle before asserting.
        import time

        for _ in range(20):
            if conn.start_calls >= 1:
                break
            time.sleep(0.05)
        assert conn.start_calls == 1, (
            f"Expected start_connector to be called once during lifespan, "
            f"got {conn.start_calls}"
        )

    # On exit, stop_all ran via Engine.aclose.
    assert conn.stop_calls >= 1


# ---------------------------------------------------------------------------
# A1.5 — create_app() integration — the production path actually wires
# the new lifespan and the engine reaches __aenter__ + __aexit__.
# ---------------------------------------------------------------------------


def test_create_app_lifespan_calls_aenter_and_aexit() -> None:
    """Smoke test: the real ``create_app()`` factory wires the new
    lifespan; entering the TestClient context fires both ``__aenter__``
    and ``__aexit__`` on the engine.

    Patches both methods to no-ops so the test doesn't need MLC /
    Brave secrets to be present.
    """
    aenter_mock = AsyncMock()
    aexit_mock = AsyncMock(return_value=None)

    with patch.object(Engine, "__aenter__", aenter_mock), patch.object(
        Engine, "__aexit__", aexit_mock
    ):
        app = create_app()
        # __aenter__ should return the engine for ``async with`` semantics;
        # the lifespan stores its result via ``await engine.__aenter__()``.
        aenter_mock.return_value = app.state.gen_svc

        with TestClient(app):
            aenter_mock.assert_awaited_once()

    aexit_mock.assert_awaited_once()
