"""P0-F regression: failed connector start must be visible and isolated.

Tribunal §3 P0-07 / A2-F2.

Before P0-F, ``Engine.__aenter__`` scheduled ``start_connector(cid)``
with ``asyncio.create_task`` and returned without awaiting it. Failures
were silently reaped by ``aclose()``'s ``gather(return_exceptions=True)``
and the first ``chat()`` could land on a half-initialized connector.

This test asserts the new contract:

* ``__aenter__`` completes successfully even when a connector's
  ``start()`` raises.
* The failing connector is removed from the registry so subsequent
  tool dispatch sees a deterministic
  :class:`tether.core.errors.ConnectorNotConfiguredError`.
* ``Engine._connector_start_failures`` records the failed id (so
  ``/readyz`` can surface it).
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict

import pytest

from tether import Engine
from tether.config.settings import ConnectorSpec, Settings
from tether.connectors.base import Connector
from tether.connectors.types import (
    AuthStatus,
    ConnectorState,
    HealthStatus,
    InboundEvent,
    LoginContinueResult,
    LoginPrompt,
)
from tether.core.interfaces import Tool


class FailingConnector(Connector):
    """Connector whose ``start()`` raises but reports READY at config time."""

    id = "failing_test"

    def __init__(self) -> None:
        # No super().__init__() — Connector has no __init__ of its own.
        pass

    async def start(self) -> None:
        raise RuntimeError("synthetic startup failure")

    async def stop(self) -> None:
        return None

    async def logout(self) -> None:
        return None

    async def health(self) -> HealthStatus:
        return HealthStatus(state=ConnectorState.ERROR, detail="startup raised")

    async def auth_status(self) -> AuthStatus:
        # Claim READY so the engine schedules start_connector(cid).
        return AuthStatus(state=ConnectorState.READY)

    async def begin_login(self) -> LoginPrompt:
        raise NotImplementedError

    async def complete_login(
        self, *, payload: Dict[str, Any]
    ) -> LoginContinueResult:
        raise NotImplementedError

    def tools(self) -> Dict[str, Tool]:
        return {}

    async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
        if False:  # pragma: no cover - empty async generator
            yield  # type: ignore[unreachable]


def _settings_with_failing_connector(tmp_db: str) -> Settings:
    """Build a Settings with the FailingConnector wired in."""
    return Settings.model_validate(
        {
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
                    "args": {"dsn": f"sqlite:///{tmp_db}"},
                },
            },
            "tools": {
                "registry": [],
                "enabled": [],
                "disabled": ["time", "weather", "forecast", "web_search"],
            },
            "connectors": {
                "registry": {
                    "failing_test": ConnectorSpec(
                        impl=f"{FailingConnector.__module__}."
                        f"{FailingConnector.__name__}",
                        args={},
                        enabled=True,
                    ).model_dump(),
                }
            },
        }
    )


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_failing_connector_does_not_break_engine_aenter(tmp_path):
    """A connector whose ``start()`` raises must NOT prevent
    ``Engine.__aenter__`` from completing, and the failing connector
    must be removed from the registry.
    """
    settings = _settings_with_failing_connector(str(tmp_path / "p0f.db"))
    engine = Engine.from_settings(settings)

    # Sanity: registry has the failing connector before __aenter__ runs.
    assert engine.connector_registry is not None
    pre_ids = {c.id for c in engine.connector_registry.all()}
    assert "failing_test" in pre_ids

    async with engine as eng:
        # Engine entered successfully despite the failure.
        assert eng.connector_registry is not None
        cids = {c.id for c in eng.connector_registry.all()}
        assert "failing_test" not in cids, (
            "Failing connector was not removed from the registry after "
            "start failure"
        )
        # Failure surface for /readyz.
        assert "failing_test" in eng._connector_start_failures
