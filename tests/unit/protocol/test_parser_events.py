"""Unit tests for :mod:`tether_service.protocol.parsers.events`.

Phase 5 step 49: parser-level event dataclasses (internal to parser).

Synthesis §4 Phase 5 steps 49 + 51.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tether_service.protocol.parsers.events import (
    PParseError,
    PStreamEnd,
    PText,
    PThink,
    PToolCallDetected,
    PToolCallParsed,
)


def test_parser_text_event():
    ev = PText(text="hi")
    assert ev.text == "hi"


def test_parser_text_event_frozen():
    ev = PText(text="hi")
    with pytest.raises(FrozenInstanceError):
        ev.text = "mutated"  # type: ignore[misc]


def test_parser_think_event():
    ev = PThink(text="thought...")
    assert ev.text == "thought..."


def test_parser_think_event_frozen():
    ev = PThink(text="x")
    with pytest.raises(FrozenInstanceError):
        ev.text = "y"  # type: ignore[misc]


def test_parser_tool_call_detected():
    ev = PToolCallDetected()
    assert isinstance(ev, PToolCallDetected)


def test_parser_tool_call_detected_equals_singleton_value():
    """Two PToolCallDetected instances are equal (no fields)."""
    assert PToolCallDetected() == PToolCallDetected()


def test_parser_tool_call_parsed():
    ev = PToolCallParsed(
        tool_call_id="call-1",
        name="time",
        arguments={"timezone": "UTC"},
    )
    assert ev.tool_call_id == "call-1"
    assert ev.name == "time"
    assert ev.arguments == {"timezone": "UTC"}


def test_parser_tool_call_parsed_default_args():
    ev = PToolCallParsed(tool_call_id="c", name="t")
    assert ev.arguments == {}


def test_parser_tool_call_parsed_frozen():
    ev = PToolCallParsed(tool_call_id="c", name="t")
    with pytest.raises(FrozenInstanceError):
        ev.name = "other"  # type: ignore[misc]


def test_parser_parse_error():
    ev = PParseError(message="bad json", raw="{ broken")
    assert ev.message == "bad json"
    assert ev.raw == "{ broken"


def test_parser_parse_error_default_raw():
    ev = PParseError(message="bad")
    assert ev.raw is None


def test_parser_parse_error_frozen():
    ev = PParseError(message="x")
    with pytest.raises(FrozenInstanceError):
        ev.message = "y"  # type: ignore[misc]


def test_parser_stream_end():
    ev = PStreamEnd()
    assert isinstance(ev, PStreamEnd)


def test_parser_stream_end_equals():
    assert PStreamEnd() == PStreamEnd()


@pytest.mark.parametrize(
    "factory,attr",
    [
        (lambda: PText(text="x"), "text"),
        (lambda: PThink(text="x"), "text"),
        (lambda: PToolCallParsed(tool_call_id="c", name="t"), "name"),
        (lambda: PParseError(message="m"), "message"),
    ],
)
def test_all_frozen(factory, attr):
    ev = factory()
    with pytest.raises(FrozenInstanceError):
        setattr(ev, attr, "mutated")
