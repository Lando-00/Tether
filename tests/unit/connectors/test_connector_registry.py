"""Unit tests for :class:`tether_service.core.connector_registry.ConnectorRegistry`.

Per connector spec §3.3 (validation + lifecycle), §3.6 (data layout), §3.8
(OAuth callback). Synthesis §4 Phase 4.5 steps 47b-47c, §13.4 M5.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict
from unittest.mock import AsyncMock

import pytest

from tether_service.connectors.base import Connector
from tether_service.connectors.types import (
    AuthStatus,
    ConnectorState,
    HealthStatus,
    InboundEvent,
    LoginContinueResult,
    LoginPrompt,
)
from tether_service.core.connector_registry import (
    ConnectorRegistry,
    _OAuthStateCache,
)
from tether_service.core.interfaces import Tool


# ---------------------------------------------------------------------------
# anyio config
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------


class _StubTool(Tool):
    """Minimal Tool whose name is configurable per-instance."""

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


def _make_fake_connector(
    *,
    connector_id: str,
    tool_names: tuple[str, ...] = (),
    start: AsyncMock | None = None,
    stop: AsyncMock | None = None,
) -> Connector:
    """Build a fully-stubbed Connector instance with the given id and tools.

    ``start`` / ``stop`` may be supplied as mocks so individual tests can
    assert on call count / configure delays / configure raises.
    """

    class _FakeConnector(Connector):
        id = connector_id

        def __init__(self) -> None:
            self._tools = {n: _StubTool(n) for n in tool_names}
            self._start = start if start is not None else AsyncMock()
            self._stop = stop if stop is not None else AsyncMock()

        async def start(self) -> None:
            await self._start()

        async def stop(self) -> None:
            await self._stop()

        async def logout(self) -> None:
            return None

        async def health(self) -> HealthStatus:
            return HealthStatus(state=ConnectorState.READY)

        async def auth_status(self) -> AuthStatus:
            return AuthStatus(state=ConnectorState.READY)

        async def begin_login(self) -> LoginPrompt:
            return LoginPrompt(kind="url", payload="https://example.com")

        async def complete_login(
            self, *, payload: Dict[str, Any]
        ) -> LoginContinueResult:
            return LoginContinueResult(state=ConnectorState.READY)

        def tools(self) -> Dict[str, Tool]:
            return dict(self._tools)

        async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
            if False:  # pragma: no cover - never reached
                yield  # type: ignore[unreachable]

    return _FakeConnector()


# ===========================================================================
# A1. Construction + validation
# ===========================================================================


def test_construction_empty(tmp_path: Path) -> None:
    registry = ConnectorRegistry([], set(), data_dir=tmp_path)
    assert registry.all() == []
    assert registry.aggregate_tools() == {}
    assert registry.names() == set()


def test_construction_one_connector(tmp_path: Path) -> None:
    conn = _make_fake_connector(
        connector_id="echo", tool_names=("echo_send",)
    )
    registry = ConnectorRegistry([conn], data_dir=tmp_path)

    assert registry.get("echo") is conn
    aggregated = registry.aggregate_tools()
    assert "echo_send" in aggregated
    assert aggregated["echo_send"].name == "echo_send"
    assert registry.names() == {"echo_send"}
    # all() returns a list copy.
    assert registry.all() == [conn]


def test_construction_unique_id_collision(tmp_path: Path) -> None:
    a = _make_fake_connector(connector_id="echo", tool_names=("echo_a",))
    b = _make_fake_connector(connector_id="echo", tool_names=("echo_b",))
    with pytest.raises(ValueError, match=r"Duplicate connector id: 'echo'"):
        ConnectorRegistry([a, b], data_dir=tmp_path)


def test_construction_empty_id_rejected(tmp_path: Path) -> None:
    """Connector ABC permits the default empty id (intermediate
    abstracts), but the registry must reject any concrete instance whose
    id is still empty."""

    class _BadConnector(Connector):
        # No id override; ``__init_subclass__`` allows this for abstract
        # intermediates but the registry treats it as a misconfiguration.
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
            return LoginPrompt(kind="url", payload="x")

        async def complete_login(
            self, *, payload: Dict[str, Any]
        ) -> LoginContinueResult:
            return LoginContinueResult(state=ConnectorState.READY)

        def tools(self) -> Dict[str, Tool]:
            return {}

        async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
            if False:  # pragma: no cover
                yield  # type: ignore[unreachable]

    with pytest.raises(ValueError, match=r"empty id"):
        ConnectorRegistry([_BadConnector()], data_dir=tmp_path)


def test_construction_missing_prefix_fails(tmp_path: Path) -> None:
    """A1 / A4: M5 prefix violation must fail construction with a clear
    error that bubbles up via ``raise ... from`` (so the original M5
    message is preserved)."""
    conn = _make_fake_connector(
        connector_id="echo", tool_names=("unprefixed",)
    )
    with pytest.raises(ValueError) as excinfo:
        ConnectorRegistry([conn], data_dir=tmp_path)
    msg = str(excinfo.value)
    # Wrapping message names the offending connector.
    assert "echo" in msg
    # M5 talks about prefixes — verify the underlying error chained through.
    assert "prefix" in msg.lower() or "echo_" in msg
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_construction_collision_with_in_tree_tool(tmp_path: Path) -> None:
    """A connector tool whose name is already in the in-tree tool registry
    is rejected at boot."""
    conn = _make_fake_connector(
        connector_id="echo", tool_names=("echo_send",)
    )
    with pytest.raises(ValueError) as excinfo:
        ConnectorRegistry([conn], tool_names={"echo_send"}, data_dir=tmp_path)
    msg = str(excinfo.value)
    assert "echo" in msg
    assert "forbidden" in msg.lower()


def test_construction_cross_connector_collision(tmp_path: Path) -> None:
    """Two connectors cannot expose the same tool name even if both
    satisfy their individual prefix constraints."""
    a = _make_fake_connector(
        connector_id="echo", tool_names=("echo_send",)
    )
    # A second connector whose id is also a prefix of 'echo_send' would
    # technically satisfy require_prefix, but cross-connector collision
    # must still trip the forbidden set.
    b = _make_fake_connector(
        connector_id="echo_send", tool_names=("echo_send_now",)
    )
    # That construction is fine. Build a real collision:
    c = _make_fake_connector(
        connector_id="echo", tool_names=("echo_send",)
    )
    # First, the duplicate-id case for c is independent (covered above);
    # here we want a same-tool different-id collision. Build a connector
    # whose id is a prefix of someone else's tool:
    d = _make_fake_connector(
        connector_id="echo_send", tool_names=("echo_send_now",)
    )
    # And a second connector that also exposes 'echo_send_now'. Its id
    # 'other' won't satisfy the prefix, so the failure cause is the
    # prefix check, not the forbidden one. Build instead one whose id
    # IS 'echo_send_now':
    # Simplest cross-connector tool-name collision: both connectors named
    # differently expose the SAME tool name, both prefixed correctly:
    a2 = _make_fake_connector(
        connector_id="ab", tool_names=("ab_x",)
    )
    b2 = _make_fake_connector(
        connector_id="ab_x", tool_names=("ab_x_run",)
    )
    # No collision yet: ab_x is the tool name of a2 AND the connector id
    # of b2; b2's tool 'ab_x_run' satisfies prefix 'ab_x_'. But the
    # forbidden set after a2 contains 'ab_x' which is NOT 'ab_x_run', so
    # b2 is fine — that's the legitimate "ids may overlap with tool
    # names of other connectors" case.
    ConnectorRegistry([a2, b2], data_dir=tmp_path)  # OK

    # Now build a real collision: two connectors expose the same tool
    # name, and the second one's prefix lets it. Use connector ids
    # 'foo_bar' and 'foo'; both produce a tool named 'foo_bar_baz' (the
    # second connector's tool is 'foo_bar_baz', prefix 'foo_' matches;
    # the first connector's tool 'foo_bar_baz' has prefix 'foo_bar_'
    # which also matches its own id).
    e = _make_fake_connector(
        connector_id="foo_bar", tool_names=("foo_bar_baz",)
    )
    f = _make_fake_connector(
        connector_id="foo", tool_names=("foo_bar_baz",)
    )
    with pytest.raises(ValueError) as excinfo:
        ConnectorRegistry([e, f], data_dir=tmp_path)
    msg = str(excinfo.value)
    assert "foo" in msg
    assert "forbidden" in msg.lower()


def test_construction_logs_summary(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    conn = _make_fake_connector(
        connector_id="echo", tool_names=("echo_send",)
    )
    with caplog.at_level("INFO", logger="tether_service.core.connector_registry"):
        ConnectorRegistry([conn], data_dir=tmp_path)
    assert any(
        "1 connector" in r.message and "1 aggregated tool" in r.message
        for r in caplog.records
    )


def test_get_unknown_raises_keyerror(tmp_path: Path) -> None:
    registry = ConnectorRegistry([], data_dir=tmp_path)
    with pytest.raises(KeyError, match=r"connector not registered"):
        registry.get("nope")


def test_aggregate_tools_returns_copy(tmp_path: Path) -> None:
    conn = _make_fake_connector(
        connector_id="echo", tool_names=("echo_send",)
    )
    registry = ConnectorRegistry([conn], data_dir=tmp_path)
    aggregated = registry.aggregate_tools()
    aggregated["sneaky"] = _StubTool("sneaky")
    assert "sneaky" not in registry.aggregate_tools()
    assert "sneaky" not in registry.names()


# ===========================================================================
# A2. Lifecycle
# ===========================================================================


@pytest.mark.anyio
async def test_start_connector_calls_start(tmp_path: Path) -> None:
    start_mock = AsyncMock()
    conn = _make_fake_connector(connector_id="echo", start=start_mock)
    registry = ConnectorRegistry([conn], data_dir=tmp_path)
    await registry.start_connector("echo")
    start_mock.assert_awaited_once()


@pytest.mark.anyio
async def test_start_connector_creates_data_dir(tmp_path: Path) -> None:
    conn = _make_fake_connector(connector_id="echo")
    registry = ConnectorRegistry([conn], data_dir=tmp_path)
    expected = tmp_path / "echo"
    assert not expected.exists()
    await registry.start_connector("echo")
    assert expected.is_dir()


@pytest.mark.anyio
async def test_start_connector_idempotent_data_dir(tmp_path: Path) -> None:
    """``start_connector`` re-call does not raise on existing dir."""
    conn = _make_fake_connector(connector_id="echo")
    registry = ConnectorRegistry([conn], data_dir=tmp_path)
    await registry.start_connector("echo")
    await registry.start_connector("echo")  # must not raise


@pytest.mark.anyio
async def test_stop_connector_within_budget(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    stop_mock = AsyncMock()
    conn = _make_fake_connector(connector_id="echo", stop=stop_mock)
    registry = ConnectorRegistry([conn], data_dir=tmp_path)
    with caplog.at_level("WARNING", logger="tether_service.core.connector_registry"):
        await registry.stop_connector("echo", timeout_sec=2.0)
    stop_mock.assert_awaited_once()
    # No timeout warning should be logged.
    assert not any(
        "exceeded" in r.message and "cooperative budget" in r.message
        for r in caplog.records
    )


@pytest.mark.anyio
async def test_stop_connector_timeout_logs_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    async def _slow_stop() -> None:
        await asyncio.sleep(2.0)

    stop_mock = AsyncMock(side_effect=_slow_stop)
    conn = _make_fake_connector(connector_id="echo", stop=stop_mock)
    registry = ConnectorRegistry([conn], data_dir=tmp_path)

    t0 = time.monotonic()
    with caplog.at_level("WARNING", logger="tether_service.core.connector_registry"):
        await registry.stop_connector("echo", timeout_sec=0.1)
    elapsed = time.monotonic() - t0

    # Wall time ≈ timeout_sec, NOT the 2 s sleep — registry abandoned.
    assert elapsed < 0.5, f"stop_connector waited too long: {elapsed:.3f}s"
    assert any(
        "exceeded" in r.message and "cooperative budget" in r.message
        for r in caplog.records
    )


@pytest.mark.anyio
async def test_stop_connector_exception_logged_not_raised(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    stop_mock = AsyncMock(side_effect=RuntimeError("boom"))
    conn = _make_fake_connector(connector_id="echo", stop=stop_mock)
    registry = ConnectorRegistry([conn], data_dir=tmp_path)
    with caplog.at_level("ERROR", logger="tether_service.core.connector_registry"):
        # Must not raise.
        await registry.stop_connector("echo")
    assert any("boom" in r.message or "boom" in str(r.exc_info) for r in caplog.records)


@pytest.mark.anyio
async def test_start_all_returns_failures_dict(tmp_path: Path) -> None:
    ok_start = AsyncMock()
    bad_start = AsyncMock(side_effect=RuntimeError("nope"))
    a = _make_fake_connector(
        connector_id="ok", tool_names=("ok_x",), start=ok_start
    )
    b = _make_fake_connector(
        connector_id="bad", tool_names=("bad_y",), start=bad_start
    )
    registry = ConnectorRegistry([a, b], data_dir=tmp_path)
    results = await registry.start_all()
    assert results["ok"] is None
    assert isinstance(results["bad"], RuntimeError)
    assert "nope" in str(results["bad"])
    ok_start.assert_awaited_once()
    bad_start.assert_awaited_once()


@pytest.mark.anyio
async def test_start_all_empty_registry(tmp_path: Path) -> None:
    registry = ConnectorRegistry([], data_dir=tmp_path)
    assert await registry.start_all() == {}


@pytest.mark.anyio
async def test_stop_all_concurrent(tmp_path: Path) -> None:
    """Three connectors each sleep 0.5 s in stop(); concurrent gather
    means total wall time stays under 1 s, not 1.5 s."""

    async def _slow_stop() -> None:
        await asyncio.sleep(0.5)

    conns = [
        _make_fake_connector(
            connector_id=f"c{i}",
            tool_names=(f"c{i}_x",),
            stop=AsyncMock(side_effect=_slow_stop),
        )
        for i in range(3)
    ]
    registry = ConnectorRegistry(conns, data_dir=tmp_path)

    t0 = time.monotonic()
    await registry.stop_all(timeout_sec=2.0)
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0, f"stop_all not concurrent: {elapsed:.3f}s"


@pytest.mark.anyio
async def test_stop_all_empty_registry(tmp_path: Path) -> None:
    registry = ConnectorRegistry([], data_dir=tmp_path)
    await registry.stop_all()  # must not raise


# ===========================================================================
# A3. OAuth state cache
# ===========================================================================


def test_oauth_state_set_get() -> None:
    cache = _OAuthStateCache()
    cache.set("abc", {"connector_id": "gmail"})
    assert cache.get("abc") == {"connector_id": "gmail"}


def test_oauth_state_unknown_key_returns_none() -> None:
    cache = _OAuthStateCache()
    assert cache.get("missing") is None
    assert cache.pop("missing") is None


def test_oauth_state_ttl() -> None:
    cache = _OAuthStateCache(maxsize=4, ttl=0.1)
    cache.set("k", "v")
    time.sleep(0.2)
    assert cache.get("k") is None


def test_oauth_state_maxsize() -> None:
    cache = _OAuthStateCache(maxsize=8, ttl=300.0)
    for i in range(9):
        # Tiny gap so monotonic timestamps differ; otherwise eviction of
        # "oldest" picks an arbitrary tied entry.
        cache.set(f"k{i}", i)
        time.sleep(0.001)
    # k0 should have been evicted as oldest when k8 went in.
    assert cache.get("k0") is None
    assert cache.get("k8") == 8
    assert len(cache) == 8


def test_oauth_state_pop() -> None:
    cache = _OAuthStateCache()
    cache.set("abc", "value")
    assert cache.pop("abc") == "value"
    assert cache.get("abc") is None
    # Pop again returns None.
    assert cache.pop("abc") is None


def test_oauth_state_overwrite_existing() -> None:
    cache = _OAuthStateCache(maxsize=2, ttl=300.0)
    cache.set("a", 1)
    cache.set("a", 2)
    assert cache.get("a") == 2
    assert len(cache) == 1


def test_oauth_state_invalid_args() -> None:
    with pytest.raises(ValueError):
        _OAuthStateCache(maxsize=0)
    with pytest.raises(ValueError):
        _OAuthStateCache(ttl=0)


def test_registry_exposes_oauth_state(tmp_path: Path) -> None:
    registry = ConnectorRegistry([], data_dir=tmp_path)
    assert isinstance(registry.oauth_state, _OAuthStateCache)
    registry.oauth_state.set("state-token", {"cid": "gmail"})
    assert registry.oauth_state.get("state-token") == {"cid": "gmail"}


# ===========================================================================
# A8 (smoke-style) — defaults
# ===========================================================================


def test_default_data_dir_resolution() -> None:
    """When no ``data_dir`` is given, the registry resolves a sane
    fallback. ``platformdirs`` may or may not be installed, so we just
    assert the resolved path is a Path and (when platformdirs is absent)
    is the in-repo ``data/connectors`` fallback."""
    registry = ConnectorRegistry([])
    assert isinstance(registry.data_dir, Path)
    # Either a platformdirs path or the in-tree fallback. Both end in
    # 'connectors'.
    assert registry.data_dir.name == "connectors"


def test_tool_names_default_none(tmp_path: Path) -> None:
    """Calling ``ConnectorRegistry([])`` (no tool_names arg) must work."""
    ConnectorRegistry([], data_dir=tmp_path)  # smoke


# ===========================================================================
# A4. Phase 4.5 follow-up (rubber-duck consensus): tools() called once
# ===========================================================================


def test_tools_called_once_during_construction(tmp_path: Path) -> None:
    """F3: a connector's ``tools()`` must be invoked exactly ONCE during
    ``ConnectorRegistry.__init__`` (validation + aggregation share a
    single cached call).

    Before the fix, ``tools()`` was called twice — once for M5 validation
    and once again for aggregation. A non-idempotent ``tools()`` could
    pass validation then yield a different dict, producing inconsistent
    registry state silently.
    """
    call_counter = {"count": 0}
    fixed_tools: Dict[str, Tool] = {"echo_send": _StubTool("echo_send")}

    class _CountingConnector(Connector):
        id = "echo"

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
            return LoginPrompt(kind="url", payload="x")

        async def complete_login(
            self, *, payload: Dict[str, Any]
        ) -> LoginContinueResult:
            return LoginContinueResult(state=ConnectorState.READY)

        def tools(self) -> Dict[str, Tool]:
            call_counter["count"] += 1
            return dict(fixed_tools)

        async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
            if False:  # pragma: no cover
                yield  # type: ignore[unreachable]

    conn = _CountingConnector()
    registry = ConnectorRegistry([conn], data_dir=tmp_path)

    # Construction must have called tools() exactly once.
    assert call_counter["count"] == 1, (
        f"tools() called {call_counter['count']} times during construction; "
        f"expected exactly 1 (Phase 4.5 follow-up F3)."
    )

    # The aggregated tool set still reflects what tools() returned.
    aggregated = registry.aggregate_tools()
    assert "echo_send" in aggregated
    # aggregate_tools() does NOT re-invoke tools() — uses cached dict.
    assert call_counter["count"] == 1


def test_tools_raise_during_construction(tmp_path: Path) -> None:
    """F3: a connector whose ``tools()`` raises must produce a
    ``ValueError`` chained from the original exception, with a message
    that names the offending connector.
    """

    class _BadToolsConnector(Connector):
        id = "broken"

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
            return LoginPrompt(kind="url", payload="x")

        async def complete_login(
            self, *, payload: Dict[str, Any]
        ) -> LoginContinueResult:
            return LoginContinueResult(state=ConnectorState.READY)

        def tools(self) -> Dict[str, Tool]:
            raise RuntimeError("boom from tools()")

        async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
            if False:  # pragma: no cover
                yield  # type: ignore[unreachable]

    with pytest.raises(ValueError) as excinfo:
        ConnectorRegistry([_BadToolsConnector()], data_dir=tmp_path)

    msg = str(excinfo.value)
    assert "broken" in msg
    assert "tools()" in msg
    # Underlying RuntimeError chained via ``raise ... from``.
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "boom" in str(excinfo.value.__cause__)
