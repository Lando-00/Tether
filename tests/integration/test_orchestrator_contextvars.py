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


# ---------------------------------------------------------------------------
# FIX 4 — RD followup: turn_id contextvar must NOT leak when an awaitable
# called BEFORE the try block (e.g. ``_seed_history`` or ``store.start_turn``)
# raises. The previous arrangement bound turn_id BEFORE the try, so any
# exception from those calls propagated WITHOUT running the unbind in finally.
# After the fix, both calls run inside the try and the unbind always fires.
# ---------------------------------------------------------------------------


async def test_turn_id_unbound_when_start_turn_raises(tmp_path):
    """If ``store.start_turn`` raises, ``turn_id`` must still be unbound.

    The orchestrator catches generic ``Exception`` and emits an
    ``Error`` + ``MessageStop`` instead of re-raising. Either way, the
    finally block must run the unbind. Before FIX 4 the bind happened
    BEFORE the try, so any awaitable raised between bind and the start
    of the try would leak the contextvar.
    """
    from tests.golden.conftest import MinimalMemoryStore  # noqa: WPS433

    from tether_service.core.types import OrchestratorConfig
    from tether_service.protocol.orchestration.chatty import (
        ChattyAgentOrchestrator,
    )
    from tether_service.protocol.orchestration.policies import (
        LoopLimitPolicy,
        ToolErrorPolicy,
    )
    from tether_service.protocol.orchestration.tool_runner import ToolRunner
    from tether_service.protocol.parsers.sliding import SlidingParser
    from tether_service.core.interfaces import ModelProvider
    from tether_service.protocol.wire.events import Error, MessageStop

    class _NeverProvider(ModelProvider):
        async def stream(self, model_name, messages, tools=None, **kwargs):
            if False:
                yield ""

        def list_models(self):
            return ["scripted"]

        def unload_model(self, model_name):
            return True

        def get_context_window(self, model_name):
            return 4096

    class _FaultyStore(MinimalMemoryStore):
        async def start_turn(
            self, session_id, turn_id, *, model_name=None
        ) -> None:
            raise RuntimeError("deliberate start_turn failure")

    # Pre-condition: contextvars are clean before we start.
    structlog.contextvars.clear_contextvars()

    orch = ChattyAgentOrchestrator(
        provider=_NeverProvider(),
        parser=SlidingParser(),
        store=_FaultyStore(),
        tools={},
        system_prompt="sys",
        config=OrchestratorConfig(
            max_tool_loops=3,
            auto_reload_on_fatal_error=False,
            save_thinking=False,
            include_thinking_in_history=False,
            loop_limit_policy=LoopLimitPolicy.EMIT_LIMIT_EVENT,
            tool_error_policy=ToolErrorPolicy.FEED_BACK_TO_MODEL,
        ),
        tool_runner=ToolRunner({}, timeout_sec=5),
    )

    events = []
    async for evt in orch.run(
        session_id="sid-faulty",
        prompt="hi",
        model_name="scripted",
    ):
        events.append(evt)

    # The orchestrator turns the failure into an Error event and a
    # MessageStop(stop_reason="error") via its catch-all Exception handler.
    error_events = [e for e in events if isinstance(e, Error)]
    assert error_events, (
        f"Expected an Error event from start_turn failure; got {[type(e).__name__ for e in events]}"
    )
    assert "deliberate start_turn failure" in error_events[0].message

    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1
    assert stops[-1].stop_reason == "error"

    # The actual FIX 4 assertion: contextvars are clean post-run.
    # Note: the orchestrator's post-finally `yield MessageStop(...)` calls
    # `_envelope()` which rebinds `seq` for the final event — that's a
    # pre-existing incidental leak orthogonal to FIX 4 (which is about
    # `turn_id`). The matching test_turn_id_unbound_after_run also only
    # checks `turn_id`.
    ctx = structlog.contextvars.get_contextvars()
    assert "turn_id" not in ctx, (
        f"turn_id leaked after start_turn raised: {ctx}. "
        "FIX 4 regression — bind/unbind must be symmetric inside the try block."
    )
