"""
Tests that SlidingParser.finalize() does NOT emit a terminal-done event.

Acceptance A3: Synthesis §6 row 3 / A4 PAIN-4.

The orchestrator owns terminal DONE emission (orchestrator.py line 224).
finalize() must only flush residual text/think/error — never a tool-call
event and never a terminal "done".

Phase 5 ``p5-parser-typed-events`` rewrites the assertions to use typed
:class:`tether.protocol.parsers.events.ParserEvent` discriminants
rather than dict-shape ``StreamEvent`` lookups. Because the parser's
typed event vocabulary has no PStreamEnd-or-equivalent emitted by
``finalize()``, the "no DONE" guarantee is now tighter: finalize() must
not emit any tool-call event either.
"""
import pytest
from tether.protocol.parsers.sliding import SlidingParser
from tether.protocol.parsers.events import (
    PParseError,
    PText,
    PThink,
    PToolCallDetected,
    PToolCallParsed,
)


def _has_terminal_event(events) -> bool:
    """The parser's typed-event vocabulary has no terminal event; finalize
    must never emit a tool-call event either (those are stream-time only)."""
    return any(isinstance(e, (PToolCallDetected, PToolCallParsed)) for e in events)


class TestFinalizeNoDone:
    """finalize() must not emit any terminal/tool-call event in any state."""

    def test_empty_buffer_text_mode(self):
        """mode=text, empty buf: finalize returns []."""
        parser = SlidingParser()
        events = parser.finalize()
        assert events == [], f"Unexpected events from empty-buf finalize: {events}"
        assert not _has_terminal_event(events)

    def test_text_in_buffer(self):
        """mode=text, buf has trailing text: returns PText, not a tool/terminal event."""
        parser = SlidingParser()
        # Feed a short string that stays in the overlap buffer
        parser.feed("Hi")
        events = parser.finalize()
        assert not _has_terminal_event(events), f"Tool/terminal event found: {events}"
        # Should have flushed the buffered text
        text_events = [e for e in events if isinstance(e, PText)]
        assert text_events, "Expected a PText event for buffered 'Hi'"
        assert "Hi" in "".join(e.text for e in text_events)

    def test_think_mode(self):
        """mode=think, buf has partial think text: returns PThink, not a tool/terminal event."""
        parser = SlidingParser()
        # Enter think mode by feeding opening tag without closing tag
        parser.feed("<think>some thoughts")
        assert parser.mode == "think"
        events = parser.finalize()
        assert not _has_terminal_event(events), f"Tool/terminal event found: {events}"

    def test_await_payload_mode(self):
        """mode=await_payload (marker detected but no JSON): returns PParseError, not a tool/terminal event."""
        parser = SlidingParser()
        # Feed marker without payload
        parser.feed("<<function_call>>")
        assert parser.mode == "await_payload"
        events = parser.finalize()
        assert not _has_terminal_event(events), f"Tool/terminal event found: {events}"
        # Should emit an error for missing payload
        error_events = [e for e in events if isinstance(e, PParseError)]
        assert error_events, "Expected a PParseError event for missing tool payload"

    def test_incomplete_json_mode(self):
        """mode=json, buf has partial JSON: returns PParseError, not a tool/terminal event."""
        parser = SlidingParser()
        # Get into json mode: feed marker + opening brace without closing
        parser.feed('<<function_call>> {"name": "time_tool", "arg')
        # May be in json mode if the buf has partial JSON
        events = parser.finalize()
        assert not _has_terminal_event(events), f"Tool/terminal event found: {events}"

    def test_finalize_after_complete_tool_call(self):
        """finalize after a complete tool call: returns nothing (or text), never a terminal event."""
        parser = SlidingParser()
        parser.feed('<<function_call>> {"name": "t", "arguments": {}}')
        # Complete tool call — parser returns to text mode
        assert parser.mode == "text"
        events = parser.finalize()
        assert not _has_terminal_event(events), f"Tool/terminal event found: {events}"

    def test_finalize_idempotent_no_done(self):
        """Calling finalize twice does not produce a terminal/tool-call event either time."""
        parser = SlidingParser()
        parser.feed("Hello")
        e1 = parser.finalize()
        e2 = parser.finalize()
        assert not _has_terminal_event(e1), f"Terminal event in first finalize: {e1}"
        assert not _has_terminal_event(e2), f"Terminal event in second finalize: {e2}"
