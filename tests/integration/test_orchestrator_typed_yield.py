"""Integration tests for :meth:`Engine.chat` yielding typed
:class:`WireEvent` and :meth:`Engine.stream` keeping bytes UNCHANGED.

Synthesis §3.4 (Engine.chat library API).
"""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest

from tether.config.settings import Settings
from tether.core.interfaces import ModelProvider
from tether.engine import Engine
from tether.protocol.orchestration.cancel import AsyncEventCancelToken
from tether.protocol.wire.events import (
    MessageStart,
    MessageStop,
    TextDelta,
    WireEvent,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _settings(tmp_path) -> Settings:
    return Settings.model_validate(
        {
            "system": {"prompt": "sys"},
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
            "storage": {"sqlite": {"dsn": f"sqlite:///{tmp_path}/typed.db"}},
            "tools": {
                "registry": [],
                "enabled": [],
                "disabled": ["time", "weather", "forecast", "web_search"],
            },
        }
    )


@pytest.mark.anyio
async def test_engine_chat_yields_wire_event(tmp_path):
    """Each item yielded by :meth:`Engine.chat` is a typed
    :class:`WireEvent` Pydantic instance — never bytes / dict."""
    eng = Engine.from_settings(_settings(tmp_path))
    sess = await eng.create_session()

    events: List[Any] = []
    async for evt in eng.chat(
        session_id=sess["session_id"],
        prompt="hi",
        model_name="dummy",
    ):
        events.append(evt)

    assert len(events) >= 2
    for e in events:
        assert not isinstance(e, (bytes, dict))
        # WireEvent is Annotated[Union[...], discriminator]; instances
        # are concrete Pydantic models.
        from tether.protocol.wire.events import _Base

        assert isinstance(e, _Base)


@pytest.mark.anyio
async def test_engine_chat_first_event_is_message_start(tmp_path):
    """First yielded event is :class:`MessageStart`."""
    eng = Engine.from_settings(_settings(tmp_path))
    sess = await eng.create_session()

    first = None
    async for evt in eng.chat(
        session_id=sess["session_id"],
        prompt="hi",
        model_name="dummy",
    ):
        first = evt
        break

    assert isinstance(first, MessageStart)


@pytest.mark.anyio
async def test_engine_chat_last_event_is_message_stop(tmp_path):
    """Last yielded event is :class:`MessageStop`."""
    eng = Engine.from_settings(_settings(tmp_path))
    sess = await eng.create_session()

    events = []
    async for evt in eng.chat(
        session_id=sess["session_id"],
        prompt="hi",
        model_name="dummy",
    ):
        events.append(evt)

    assert isinstance(events[-1], MessageStop)


@pytest.mark.anyio
async def test_engine_chat_accepts_cancel_token(tmp_path):
    """Engine.chat accepts a CancelToken; setting it stops the stream
    cleanly with MessageStop(cancelled)."""
    eng = Engine.from_settings(_settings(tmp_path))
    sess = await eng.create_session()
    token = AsyncEventCancelToken()
    token.set()  # cancel immediately

    events = []
    async for evt in eng.chat(
        session_id=sess["session_id"],
        prompt="hi",
        model_name="dummy",
        cancel_token=token,
    ):
        events.append(evt)

    assert isinstance(events[-1], MessageStop)
    assert events[-1].stop_reason == "cancelled"


@pytest.mark.anyio
async def test_engine_stream_still_yields_bytes(tmp_path):
    """:meth:`Engine.stream` keeps yielding v0-compat NDJSON bytes
    (existing behaviour preserved). Each line decodes to a v0 dict
    with ``type`` from the legacy vocabulary."""
    eng = Engine.from_settings(_settings(tmp_path))
    sess = await eng.create_session()

    events_dicts: List[dict] = []
    async for raw in eng.stream(
        session_id=sess["session_id"],
        prompt="hi",
        model_name="dummy",
    ):
        assert isinstance(raw, bytes)
        line = raw.decode("utf-8").strip()
        if line:
            events_dicts.append(json.loads(line))

    # Every dict carries a v0 ``type`` field; vocabulary is the legacy set.
    legacy_vocab = {
        "text",
        "think",
        "tool_started",
        "tool_completed",
        "tool_error",
        "error",
        "done",
        "info",
        "loop_limit_reached",
    }
    for d in events_dicts:
        assert d["type"] in legacy_vocab, f"Unknown v0 type: {d['type']}"

    # Last event is ``done``.
    assert events_dicts[-1]["type"] == "done"


@pytest.mark.anyio
async def test_engine_stream_does_not_emit_v2_vocab_on_bytes_wire(tmp_path):
    """:meth:`Engine.stream` MUST NOT emit v2 vocabulary on the bytes
    wire (cutover happens in p5-cutover-a/c). No ``message_start``,
    ``text_delta``, ``message_stop`` should appear."""
    eng = Engine.from_settings(_settings(tmp_path))
    sess = await eng.create_session()

    blob = b""
    async for raw in eng.stream(
        session_id=sess["session_id"],
        prompt="hi",
        model_name="dummy",
    ):
        blob += raw

    text = blob.decode("utf-8")
    forbidden = ["text_delta", "message_start", "message_stop", "thinking_delta"]
    for tok in forbidden:
        assert tok not in text, (
            f"v2 vocabulary token {tok!r} leaked onto v0 bytes wire: {text[:200]}"
        )
