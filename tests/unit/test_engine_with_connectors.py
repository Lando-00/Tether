"""Tests for ``Engine`` × ``ConnectorRegistry`` wiring (Phase 4.5 step 47d).

Per ``_synthesis.md`` §4 Phase 4.5 + connector spec §3.3. Verifies:

* ``Engine.from_settings`` builds a :class:`ConnectorRegistry` from
  ``settings.connectors.registry`` (skipping ``enabled=False`` entries).
* The aggregated ``tools`` dict the orchestrator sees is the union of
  in-tree tools and connector tools.
* ``Engine.__aenter__`` schedules ``start_connector(id)`` only for
  connectors whose ``auth_status()`` reports READY.
* ``Engine.aclose`` cancels still-pending start tasks, then stops
  connectors, then tools, then watchdog/provider — in that order.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from tether import Engine
from tether.config.settings import Settings
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
from tether.runtime.hw_watchdog import HardwareWatchdog


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _StubTool(Tool):
    """Minimal Tool with no-op startup/shutdown so ``startup_all`` accepts it."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> Dict[str, Any]:
        return {"name": self._name, "parameters": {"type": "object"}}

    async def invoke(
        self, args: Dict[str, Any], *, context: Any = None
    ) -> Any:
        return None

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


# Each fake connector subclass needs a unique ``id`` to coexist in a
# single ConnectorRegistry. We build the subclass dynamically per call
# so the ``id`` ClassVar can be injected without polluting the module.

# Module-level registry of fake-connector classes keyed by id; reused so
# repeated calls with the same id return instances of the same class
# (avoids ``__init_subclass__`` name-conflict noise).
_FAKE_CONNECTOR_CLS_CACHE: Dict[str, type] = {}


def _fake_connector_cls(connector_id: str) -> type:
    """Return (or build + cache) a Connector subclass with the given id."""
    if connector_id in _FAKE_CONNECTOR_CLS_CACHE:
        return _FAKE_CONNECTOR_CLS_CACHE[connector_id]

    class _FakeConnector(Connector):
        id = connector_id

        def __init__(
            self,
            *,
            tool_names: tuple[str, ...] = (),
            auth_state: ConnectorState = ConnectorState.READY,
            start_mock: AsyncMock | None = None,
            stop_mock: AsyncMock | None = None,
        ) -> None:
            self._tools = {n: _StubTool(n) for n in tool_names}
            self._auth_state = auth_state
            self._start = start_mock if start_mock is not None else AsyncMock()
            self._stop = stop_mock if stop_mock is not None else AsyncMock()

        async def start(self) -> None:
            await self._start()

        async def stop(self) -> None:
            await self._stop()

        async def logout(self) -> None:
            return None

        async def health(self) -> HealthStatus:
            return HealthStatus(state=self._auth_state)

        async def auth_status(self) -> AuthStatus:
            return AuthStatus(state=self._auth_state)

        async def begin_login(self) -> LoginPrompt:
            return LoginPrompt(kind="url", payload="https://x")

        async def complete_login(
            self, *, payload: Dict[str, Any]
        ) -> LoginContinueResult:
            return LoginContinueResult(state=ConnectorState.READY)

        def tools(self) -> Dict[str, Tool]:
            return dict(self._tools)

        async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
            if False:  # pragma: no cover
                yield  # type: ignore[unreachable]

    _FAKE_CONNECTOR_CLS_CACHE[connector_id] = _FakeConnector
    return _FakeConnector


def _fake_connector(
    connector_id: str,
    *,
    tool_names: tuple[str, ...] = (),
    auth_state: ConnectorState = ConnectorState.READY,
    start_mock: AsyncMock | None = None,
    stop_mock: AsyncMock | None = None,
) -> Connector:
    cls = _fake_connector_cls(connector_id)
    return cls(
        tool_names=tool_names,
        auth_state=auth_state,
        start_mock=start_mock,
        stop_mock=stop_mock,
    )


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------


def _settings_dict(
    tmp_db: str,
    *,
    connectors_registry: Dict[str, Dict[str, Any]] | None = None,
) -> dict:
    """Minimal Settings dict using DummyProvider + sqlite session store."""
    return {
        "system": {"prompt": "test-prompt"},
        "providers": {
            "model": {
                "impl": "tether.providers.dummy.provider.DummyProvider",
                "args": {},
            },
            "parser": {
                "impl": "tether.protocol.parsers.sliding.SlidingParser",
                "args": {},
            },
            "session_store": {
                "impl": "tether.context.sqlite_store.SqliteSessionStore",
                "args": {},
            },
        },
        "storage": {"sqlite": {"dsn": f"sqlite:///{tmp_db}"}},
        "tools": {
            "registry": [],
            "enabled": [],
            "disabled": ["time", "weather", "forecast", "web_search"],
        },
        "orchestrator": {
            "registry": {
                "chat": "tether.protocol.orchestration.chatty.ChattyAgentOrchestrator",
            },
        },
        "connectors": {"registry": connectors_registry or {}},
    }


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ===========================================================================
# F1. Engine.from_settings builds ConnectorRegistry
# ===========================================================================


def test_engine_from_settings_no_connectors_registry_empty(tmp_path):
    """Empty connectors.registry → engine.connector_registry is non-None
    ConnectorRegistry with 0 connectors."""
    settings = Settings.model_validate(_settings_dict(str(tmp_path / "a.db")))
    engine = Engine.from_settings(settings)
    assert engine.connector_registry is not None
    assert isinstance(engine.connector_registry, ConnectorRegistry)
    assert engine.connector_registry.all() == []
    assert engine.connector_registry.aggregate_tools() == {}


# Module-level fake connector class for ``load(..)`` to import. ``load``
# uses dotted-path import, so the impl path must resolve to a real
# class. Define one here that the test settings can reference.

class _LoadableEnabledConnector(Connector):
    id = "loadable_enabled"

    def __init__(self) -> None:
        self._started = AsyncMock()
        self._stopped = AsyncMock()

    async def start(self) -> None:
        await self._started()

    async def stop(self) -> None:
        await self._stopped()

    async def logout(self) -> None:
        return None

    async def health(self) -> HealthStatus:
        return HealthStatus(state=ConnectorState.READY)

    async def auth_status(self) -> AuthStatus:
        return AuthStatus(state=ConnectorState.READY)

    async def begin_login(self) -> LoginPrompt:
        return LoginPrompt(kind="url", payload="x")

    async def complete_login(
        self, *, payload: Dict[str, Any]
    ) -> LoginContinueResult:
        return LoginContinueResult(state=ConnectorState.READY)

    def tools(self) -> Dict[str, Tool]:
        return {"loadable_enabled_send": _StubTool("loadable_enabled_send")}

    async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]


class _LoadableDisabledConnector(Connector):
    id = "loadable_disabled"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def logout(self) -> None:
        return None

    async def health(self) -> HealthStatus:
        return HealthStatus(state=ConnectorState.READY)

    async def auth_status(self) -> AuthStatus:
        return AuthStatus(state=ConnectorState.READY)

    async def begin_login(self) -> LoginPrompt:
        return LoginPrompt(kind="url", payload="y")

    async def complete_login(
        self, *, payload: Dict[str, Any]
    ) -> LoginContinueResult:
        return LoginContinueResult(state=ConnectorState.READY)

    def tools(self) -> Dict[str, Tool]:
        return {"loadable_disabled_x": _StubTool("loadable_disabled_x")}

    async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]


_THIS_MODULE = "tests.unit.test_engine_with_connectors"


def test_engine_from_settings_with_one_connector(tmp_path):
    """Settings with one enabled connector entry → engine.connector_registry
    contains it."""
    settings = Settings.model_validate(
        _settings_dict(
            str(tmp_path / "b.db"),
            connectors_registry={
                "loadable_enabled": {
                    "impl": f"{_THIS_MODULE}._LoadableEnabledConnector",
                    "args": {},
                    "enabled": True,
                },
            },
        )
    )
    engine = Engine.from_settings(settings)
    assert engine.connector_registry is not None
    connectors = engine.connector_registry.all()
    assert len(connectors) == 1
    assert connectors[0].id == "loadable_enabled"


def test_engine_from_settings_skips_disabled_connectors(tmp_path):
    """``enabled: False`` entries are not instantiated into the registry."""
    settings = Settings.model_validate(
        _settings_dict(
            str(tmp_path / "c.db"),
            connectors_registry={
                "loadable_enabled": {
                    "impl": f"{_THIS_MODULE}._LoadableEnabledConnector",
                    "args": {},
                    "enabled": True,
                },
                "loadable_disabled": {
                    "impl": f"{_THIS_MODULE}._LoadableDisabledConnector",
                    "args": {},
                    "enabled": False,
                },
            },
        )
    )
    engine = Engine.from_settings(settings)
    cids = [c.id for c in engine.connector_registry.all()]
    assert cids == ["loadable_enabled"]
    assert "loadable_disabled" not in cids


def test_engine_from_settings_aggregates_tools(tmp_path):
    """The flat ``tools`` dict the ToolRunner sees contains BOTH in-tree
    tools (none in this minimal config) AND connector tools."""
    settings = Settings.model_validate(
        _settings_dict(
            str(tmp_path / "d.db"),
            connectors_registry={
                "loadable_enabled": {
                    "impl": f"{_THIS_MODULE}._LoadableEnabledConnector",
                    "args": {},
                    "enabled": True,
                },
            },
        )
    )
    engine = Engine.from_settings(settings)
    # Connector tool present in the aggregated dict.
    assert "loadable_enabled_send" in engine.tools
    # ToolRunner sees the same dict (same identity, not a copy).
    assert "loadable_enabled_send" in engine.tool_runner.tools


# ===========================================================================
# F2. Engine.__aenter__ starts only READY connectors
# ===========================================================================


@pytest.mark.anyio
async def test_engine_aenter_starts_ready_connectors(tmp_path):
    """Two connectors registered: one READY, one UNCONFIGURED. Only the
    READY one's ``start()`` is called by ``__aenter__``."""
    ready_start = AsyncMock()
    unconf_start = AsyncMock()
    ready = _fake_connector(
        "aenter_ready",
        tool_names=("aenter_ready_t",),
        auth_state=ConnectorState.READY,
        start_mock=ready_start,
    )
    unconf = _fake_connector(
        "aenter_unconf",
        tool_names=("aenter_unconf_t",),
        auth_state=ConnectorState.UNCONFIGURED,
        start_mock=unconf_start,
    )
    registry = ConnectorRegistry([ready, unconf], data_dir=tmp_path)

    engine = Engine(
        provider=MagicMock(),
        parser=MagicMock(),
        session_store=MagicMock(),
        tools={},
        system_prompt="",
        connector_registry=registry,
    )

    async with engine:
        # Yield to the event loop so the create_task scheduled inside
        # __aenter__ actually runs to completion.
        await asyncio.sleep(0)
        # Drain any tasks the engine scheduled.
        if engine._connector_start_tasks:
            await asyncio.gather(
                *engine._connector_start_tasks, return_exceptions=True
            )

    ready_start.assert_awaited_once()
    unconf_start.assert_not_awaited()


@pytest.mark.anyio
async def test_engine_aenter_skips_connectors_with_failing_auth_status(tmp_path):
    """auth_status() exception → connector is skipped, others proceed."""
    ready_start = AsyncMock()

    bad = _fake_connector(
        "aenter_bad",
        tool_names=("aenter_bad_t",),
        auth_state=ConnectorState.READY,
    )
    # Monkey-patch its auth_status to raise.

    async def _raise_auth() -> AuthStatus:
        raise RuntimeError("auth probe failed")

    bad.auth_status = _raise_auth  # type: ignore[method-assign]

    good = _fake_connector(
        "aenter_good",
        tool_names=("aenter_good_t",),
        auth_state=ConnectorState.READY,
        start_mock=ready_start,
    )

    registry = ConnectorRegistry([bad, good], data_dir=tmp_path)
    engine = Engine(
        provider=MagicMock(),
        parser=MagicMock(),
        session_store=MagicMock(),
        tools={},
        system_prompt="",
        connector_registry=registry,
    )

    async with engine:
        await asyncio.sleep(0)
        if engine._connector_start_tasks:
            await asyncio.gather(
                *engine._connector_start_tasks, return_exceptions=True
            )

    ready_start.assert_awaited_once()


# ===========================================================================
# F3. Engine.aclose stops connectors before tools
# ===========================================================================


@pytest.mark.anyio
async def test_engine_aclose_stops_all_connectors(tmp_path):
    """``aclose`` calls ``ConnectorRegistry.stop_all`` exactly once."""
    stop_a = AsyncMock()
    stop_b = AsyncMock()
    a = _fake_connector(
        "aclose_a", tool_names=("aclose_a_t",), stop_mock=stop_a
    )
    b = _fake_connector(
        "aclose_b", tool_names=("aclose_b_t",), stop_mock=stop_b
    )
    registry = ConnectorRegistry([a, b], data_dir=tmp_path)

    engine = Engine(
        provider=MagicMock(),
        parser=MagicMock(),
        session_store=MagicMock(),
        tools={},
        system_prompt="",
        connector_registry=registry,
    )
    await engine.aclose()
    stop_a.assert_awaited_once()
    stop_b.assert_awaited_once()


@pytest.mark.anyio
async def test_engine_aenter_awaits_start_tasks(tmp_path):
    """P0-F / Tribunal P0-07 (A2-F2): ``__aenter__`` MUST await connector
    start tasks before returning, so the first ``chat()`` cannot land on
    a half-initialized connector. Consequently, by the time ``aclose``
    runs the start-tasks list is already drained and the cancel-pending
    branch is a no-op (kept as defence-in-depth).
    """

    started = asyncio.Event()
    finished = asyncio.Event()

    async def _slow_start() -> None:
        started.set()
        # Brief, bounded sleep so __aenter__ can demonstrate it waits
        # without hanging the test if the contract regresses.
        await asyncio.sleep(0.05)
        finished.set()

    start_mock = AsyncMock(side_effect=_slow_start)
    conn = _fake_connector(
        "cancel_pending",
        tool_names=("cancel_pending_t",),
        auth_state=ConnectorState.READY,
        start_mock=start_mock,
    )
    registry = ConnectorRegistry([conn], data_dir=tmp_path)
    engine = Engine(
        provider=MagicMock(),
        parser=MagicMock(),
        session_store=MagicMock(),
        tools={},
        system_prompt="",
        connector_registry=registry,
    )

    await engine.__aenter__()
    # __aenter__ must not return until start() finished (P0-F).
    assert started.is_set()
    assert finished.is_set(), (
        "__aenter__ returned before start() completed — P0-F regression"
    )
    # Pending tasks list cleared by __aenter__ — aclose has nothing to cancel.
    assert engine._connector_start_tasks == []

    await engine.aclose()
    assert engine._connector_start_tasks == []


@pytest.mark.anyio
async def test_engine_aclose_order(tmp_path):
    """Verify the documented order:
        connectors stop → tool shutdown → watchdog/provider shutdown.

    Uses a shared call list to record each step.
    """
    call_order: List[str] = []

    # Shutdown-recording stub tool.
    class _RecorderTool(Tool):
        @property
        def name(self) -> str:
            return "order_tool"

        @property
        def schema(self) -> Dict[str, Any]:
            return {"name": self.name, "parameters": {"type": "object"}}

        async def invoke(self, args, *, context=None):  # type: ignore[no-untyped-def]
            return None

        async def startup(self) -> None:
            return None

        async def shutdown(self) -> None:
            call_order.append("tool.shutdown")

    # Connector with a recording stop().
    async def _record_stop() -> None:
        call_order.append("connector.stop")

    stop_mock = AsyncMock(side_effect=_record_stop)
    conn = _fake_connector(
        "order_conn", tool_names=(), stop_mock=stop_mock
    )
    registry = ConnectorRegistry([conn], data_dir=tmp_path)

    # Watchdog with a recording shutdown_all().
    fake_watchdog = MagicMock(spec=HardwareWatchdog)

    def _record_watchdog():
        call_order.append("watchdog.shutdown_all")

    fake_watchdog.shutdown_all.side_effect = _record_watchdog

    tool = _RecorderTool()
    engine = Engine(
        provider=MagicMock(),
        parser=MagicMock(),
        session_store=MagicMock(),
        tools={"order_tool": tool},
        system_prompt="",
        connector_registry=registry,
        hw_watchdog=fake_watchdog,
    )

    await engine.aclose()

    assert call_order == [
        "connector.stop",
        "tool.shutdown",
        "watchdog.shutdown_all",
    ], f"unexpected order: {call_order}"


@pytest.mark.anyio
async def test_engine_aclose_no_connector_registry(tmp_path):
    """Engine without a connector_registry → aclose still works (back-compat
    with the legacy direct-constructor path)."""
    fake_provider = MagicMock()
    fake_provider.shutdown_all = MagicMock()
    engine = Engine(
        provider=fake_provider,
        parser=MagicMock(),
        session_store=MagicMock(),
        tools={},
        system_prompt="",
    )
    assert engine.connector_registry is None
    await engine.aclose()
    fake_provider.shutdown_all.assert_called_once()
