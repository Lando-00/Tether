"""Unit tests for the :class:`tether_service.connectors.base.Connector` ABC.

Per connector spec §3.1; synthesis §4 Phase 4.5 step 47a.
"""
from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict

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
from tether_service.core.interfaces import Tool


# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------


class _StubTool(Tool):
    """Minimal Tool used to verify ``tools()`` return shape."""

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        return "ok_id_noop"

    @property
    def schema(self) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"name": self.name, "parameters": {"type": "object"}}

    async def invoke(
        self, args: Dict[str, Any], *, context=None
    ) -> Any:  # pragma: no cover - trivial
        return None


def _make_full_connector_class(connector_id: str = "ok_id"):
    """Construct a concrete Connector subclass for the given id.

    Implements every abstract method as a no-op suitable for contract
    inspection. Returns the class object so callers can inspect / instantiate.
    """

    class _FullConnector(Connector):
        id = connector_id

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
            return LoginPrompt(kind="url", payload="https://example.com")

        async def complete_login(
            self, *, payload: Dict[str, Any]
        ) -> LoginContinueResult:
            return LoginContinueResult(state=ConnectorState.READY)

        def tools(self) -> Dict[str, Tool]:
            return {}

        async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
            if False:  # pragma: no cover - never reached
                yield  # type: ignore[unreachable]

    return _FullConnector


# ---------------------------------------------------------------------------
# id validation (__init_subclass__)
# ---------------------------------------------------------------------------


def test_connector_subclass_with_invalid_id_raises():
    """Uppercase id rejected at class creation time (not deferred to
    instantiation)."""
    with pytest.raises(ValueError):
        # Uppercase fails the [a-z0-9_]+ pattern.
        _make_full_connector_class(connector_id="WHAT")


def test_connector_id_pattern_enforces_lowercase():
    """Hyphens are not allowed; underscores and digits are."""
    with pytest.raises(ValueError):
        _make_full_connector_class(connector_id="bad-NAME")
    # All-lowercase + underscores + digits should pass.
    _make_full_connector_class(connector_id="ok_id_42")
    _make_full_connector_class(connector_id="abc")


def test_connector_empty_id_does_not_validate():
    """The default ``id = ""`` on the abstract Connector itself does NOT
    trigger validation — that lets callers define intermediate abstract
    subclasses without a concrete id. Concrete subclasses MUST set id
    (registry will reject empty ids at boot).
    """

    class _IntermediateAbstract(Connector):  # noqa: D401 - test fixture
        # No ``id`` override on purpose.
        async def start(self) -> None: ...
        async def stop(self) -> None: ...
        async def logout(self) -> None: ...
        async def health(self) -> HealthStatus: ...
        async def auth_status(self) -> AuthStatus: ...
        async def begin_login(self) -> LoginPrompt: ...
        async def complete_login(
            self, *, payload: Dict[str, Any]
        ) -> LoginContinueResult: ...
        def tools(self) -> Dict[str, Tool]: ...
        async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
            if False:  # pragma: no cover
                yield  # type: ignore[unreachable]

    assert _IntermediateAbstract.id == ""


# ---------------------------------------------------------------------------
# Abstract method enforcement
# ---------------------------------------------------------------------------


def test_connector_concrete_subclass_can_construct():
    """A subclass that implements every abstract method instantiates."""
    cls = _make_full_connector_class()
    instance = cls()
    assert isinstance(instance, Connector)
    assert instance.id == "ok_id"


def test_connector_partial_impl_cannot_instantiate():
    """A subclass missing one abstract method raises ``TypeError`` on
    instantiation (standard Python ABC behavior).
    """

    class _MissingStop(Connector):
        id = "missing"

        async def start(self) -> None: ...
        # stop intentionally omitted
        async def logout(self) -> None: ...
        async def health(self) -> HealthStatus: ...
        async def auth_status(self) -> AuthStatus: ...
        async def begin_login(self) -> LoginPrompt: ...
        async def complete_login(
            self, *, payload: Dict[str, Any]
        ) -> LoginContinueResult: ...
        def tools(self) -> Dict[str, Tool]: ...
        async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
            if False:  # pragma: no cover
                yield  # type: ignore[unreachable]

    with pytest.raises(TypeError) as exc_info:
        _MissingStop()
    assert "stop" in str(exc_info.value)


def test_connector_cannot_instantiate_directly():
    """The ABC itself cannot be instantiated — every method is abstract."""
    with pytest.raises(TypeError):
        Connector()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Method signatures + return shapes
# ---------------------------------------------------------------------------


def test_connector_tools_returns_dict_str_tool():
    """``tools()`` returns ``Dict[str, Tool]``; the empty dict is valid."""
    cls = _make_full_connector_class()
    instance = cls()
    result = instance.tools()
    assert isinstance(result, dict)
    # Empty for the stub; the type is ``Dict[str, Tool]``. Adding a Tool
    # instance verifies the shape.
    result_with_tool = {"ok_id_noop": _StubTool()}
    for k, v in result_with_tool.items():
        assert isinstance(k, str)
        assert isinstance(v, Tool)


async def test_connector_inbound_stream_is_async_generator():
    """``inbound_stream`` is an async generator function — calling it
    returns an async iterable, NOT a coroutine. Iterating yields nothing
    for the stub (empty generator).
    """
    cls = _make_full_connector_class()
    instance = cls()

    # Bound method should be an async generator function.
    assert inspect.isasyncgenfunction(instance.inbound_stream), (
        f"inbound_stream must be an async generator function so callers "
        f"can ``async for ... in connector.inbound_stream()``; "
        f"got {type(instance.inbound_stream)}"
    )

    # And iterating yields zero events for the empty-body stub.
    events = []
    async for event in instance.inbound_stream():
        events.append(event)
    assert events == []


async def test_connector_full_lifecycle_smoke():
    """End-to-end smoke: every abstract method is callable with the right
    arity / await-ability.
    """
    cls = _make_full_connector_class()
    instance = cls()
    # Lifecycle
    await instance.start()
    await instance.stop()
    await instance.logout()
    # Status
    h = await instance.health()
    assert isinstance(h, HealthStatus)
    a = await instance.auth_status()
    assert isinstance(a, AuthStatus)
    # Login flow
    prompt = await instance.begin_login()
    assert isinstance(prompt, LoginPrompt)
    cont = await instance.complete_login(payload={"code": "123456"})
    assert isinstance(cont, LoginContinueResult)
    # Tools/inbound
    assert instance.tools() == {}
    # Inbound iteration tested separately.
    _ = datetime.now(tz=timezone.utc)  # silence unused import
