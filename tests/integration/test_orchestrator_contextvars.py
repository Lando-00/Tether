"""Orchestrator contextvar binding tests.

Phase 7 step 69. After ``bind_contextvars(turn_id=turn_id)`` is called
in :meth:`ChattyAgentOrchestrator.run`, every structlog event emitted
during the turn automatically carries ``turn_id`` (via the
``merge_contextvars`` processor configured in ``core/logging.py``).

Tests verify:
  - ``turn_id`` is present in structlog contextvars while the generator
    is running (checked inside the ``async for`` loop).
  - ``turn_id`` is absent from structlog contextvars after the generator
    finishes (unbind in the ``finally`` block).
  - Two sequential turns bind distinct ``turn_id`` values.

Synthesis §3.5; §4 Phase 7 step 69.
"""
from __future__ import annotations

import pytest
import structlog.contextvars

from tether_service.config.settings import Settings
from tether_service.engine import Engine
from tether_service.protocol.wire.events import MessageStart

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _settings(tmp_path) -> Settings:
    return Settings.model_validate(
        {
            "system": {"prompt": "sys"},
            "providers": {
                "model": {
                    "impl": "tether_service.providers.dummy.provider.DummyProvider",
                    "args": {},
                },
                "parser": {
                    "impl": "tether_service.protocol.parsers.sliding.SlidingParser",
                    "args": {},
                },
                "session_store": {
                    "impl": "tether_service.context.sqlite_store.SqliteSessionStore",
                    "args": {"dsn": f"sqlite:///{tmp_path}/ctxvars.db"},
                },
            },
            "tools": {
                "registry": [],
                "enabled": [],
                "disabled": ["time", "weather", "forecast", "web_search"],
            },
        }
    )


async def test_turn_id_bound_during_run(tmp_path):
    """``turn_id`` is in structlog contextvars while the orchestrator runs.

    The bind happens before the first ``yield MessageStart``, so at the
    first iteration of the ``async for`` the contextvar is already set.
    Phase 7 step 69.
    """
    eng = Engine.from_settings(_settings(tmp_path))
    sess = await eng.create_session()

    found_during = False
    async for event in eng.chat(
        session_id=sess["session_id"],
        prompt="hi",
        model_name="dummy",
    ):
        ctx = structlog.contextvars.get_contextvars()
        if "turn_id" in ctx:
            found_during = True
        break  # only need the first event to check bind happened

    assert found_during, (
        "turn_id was not in structlog contextvars during Orchestrator.run(); "
        "bind_contextvars(turn_id=...) may be missing or placed after the first yield"
    )


async def test_turn_id_unbound_after_run(tmp_path):
    """After :meth:`Engine.chat` is fully consumed, ``turn_id`` is unbound.

    The ``finally`` block in :meth:`ChattyAgentOrchestrator.run` calls
    ``unbind_contextvars('turn_id')`` to prevent leaking across turns on
    the same asyncio task. Phase 7 step 69.
    """
    eng = Engine.from_settings(_settings(tmp_path))
    sess = await eng.create_session()

    async for _ in eng.chat(
        session_id=sess["session_id"],
        prompt="hi",
        model_name="dummy",
    ):
        pass

    ctx = structlog.contextvars.get_contextvars()
    assert "turn_id" not in ctx, (
        f"turn_id leaked into structlog contextvars after run() exited: {ctx}"
    )


async def test_two_turns_have_different_turn_ids(tmp_path):
    """Two sequential turns bind distinct ``turn_id`` values.

    The ``turn_id`` in each event's envelope (``event.turn_id``) must
    differ between turns — confirming a fresh UUID is generated per turn
    and the contextvar isn't shared across turns. Phase 7 step 69.
    """
    eng = Engine.from_settings(_settings(tmp_path))
    sess = await eng.create_session()

    turn_ids: list[str] = []

    for turn in range(2):
        first_event = None
        async for event in eng.chat(
            session_id=sess["session_id"],
            prompt=f"turn {turn}",
            model_name="dummy",
        ):
            first_event = event
            break  # only need the MessageStart envelope

        assert first_event is not None
        assert isinstance(first_event, MessageStart)
        turn_ids.append(first_event.turn_id)

    assert len(turn_ids) == 2
    assert turn_ids[0] != turn_ids[1], (
        f"Both turns shared the same turn_id {turn_ids[0]!r}; "
        "each turn must generate a fresh UUID"
    )
