"""Concurrent ``Engine.chat`` invocations must use isolated parser instances.

Phase 5 followups F1 (rubber-duck review by gpt-5.5): a single shared
``SlidingParser`` cross-contaminates state across requests. After the
fix, ``Engine.chat`` constructs a fresh parser per turn via the
configured ``parser_factory``.

Synthesis §3.5; briefing §2 Seam B.
"""
from __future__ import annotations

import asyncio
from typing import List

import pytest

from tether_service.config.settings import Settings
from tether_service.core.interfaces import StreamParser
from tether_service.engine import Engine
from tether_service.protocol.parsers.sliding import SlidingParser
from tether_service.protocol.wire.events import MessageStart, MessageStop


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
                    "args": {"dsn": f"sqlite:///{tmp_path}/concurrent.db"},
                },
            },
            "tools": {
                "registry": [],
                "enabled": [],
                "disabled": ["time", "weather", "forecast", "web_search"],
            },
        }
    )


@pytest.mark.anyio
async def test_engine_from_settings_uses_per_turn_parser(tmp_path):
    """``Engine.from_settings`` builds a parser FACTORY rather than a single
    instance — every call to :meth:`Engine.chat` produces a fresh parser.
    """
    eng = Engine.from_settings(_settings(tmp_path))
    sess = await eng.create_session()

    # Snap the parser produced at construction (back-compat field).
    p_at_construct = eng.parser

    # Drive a turn; the orchestrator's parser is the one returned by the
    # factory at chat() time, NOT the back-compat ``self.parser``.
    factory_outputs: List[StreamParser] = []
    original_factory = eng._parser_factory

    def tracking_factory() -> StreamParser:
        instance = original_factory()
        factory_outputs.append(instance)
        return instance

    eng._parser_factory = tracking_factory

    async for _ in eng.chat(
        session_id=sess["session_id"], prompt="hi", model_name="dummy"
    ):
        pass
    async for _ in eng.chat(
        session_id=sess["session_id"], prompt="bye", model_name="dummy"
    ):
        pass

    assert len(factory_outputs) == 2, (
        "Engine.chat must call parser_factory once per turn"
    )
    assert factory_outputs[0] is not factory_outputs[1], (
        "Each turn must get a distinct parser instance"
    )
    # The construction-time parser is also distinct from per-turn parsers.
    assert p_at_construct is not factory_outputs[0]


@pytest.mark.anyio
async def test_two_concurrent_chats_get_separate_parsers(tmp_path):
    """Two concurrent :meth:`Engine.chat` invocations on the same session
    each produce a well-formed (``message_start`` ... ``message_stop``)
    event sequence — proving they did not corrupt each other's parser
    state.
    """
    eng = Engine.from_settings(_settings(tmp_path))
    sess = await eng.create_session()

    # Track parser identities to PROVE isolation: each chat call should
    # see a distinct parser instance.
    seen_parsers: List[StreamParser] = []
    original_factory = eng._parser_factory

    def tracking_factory() -> StreamParser:
        instance = original_factory()
        seen_parsers.append(instance)
        return instance

    eng._parser_factory = tracking_factory

    async def collect(prompt: str) -> list:
        events = []
        async for ev in eng.chat(
            session_id=sess["session_id"],
            prompt=prompt,
            model_name="dummy",
        ):
            events.append(ev)
        return events

    events_a, events_b = await asyncio.gather(
        collect("alpha"),
        collect("beta"),
    )

    # Each chat() got its own parser.
    assert len(seen_parsers) == 2
    assert seen_parsers[0] is not seen_parsers[1]
    assert isinstance(seen_parsers[0], SlidingParser)
    assert isinstance(seen_parsers[1], SlidingParser)

    # Both turns produced well-formed event sequences.
    for evs, label in ((events_a, "alpha"), (events_b, "beta")):
        assert len(evs) >= 2, f"{label}: expected at least start+stop"
        assert isinstance(evs[0], MessageStart), (
            f"{label}: first event should be MessageStart, got "
            f"{type(evs[0]).__name__}"
        )
        assert isinstance(evs[-1], MessageStop), (
            f"{label}: last event should be MessageStop, got "
            f"{type(evs[-1]).__name__}"
        )
        # All events must share the same turn_id (no cross-contamination
        # of envelopes from the other turn).
        turn_ids = {getattr(e, "turn_id", None) for e in evs}
        assert len(turn_ids) == 1, (
            f"{label}: events from a single turn must share one turn_id; "
            f"saw {turn_ids}"
        )

    # The two turns have distinct turn_ids — no MessageStart/Stop from
    # one turn leaked into the other's stream.
    turn_a = events_a[0].turn_id
    turn_b = events_b[0].turn_id
    assert turn_a != turn_b
