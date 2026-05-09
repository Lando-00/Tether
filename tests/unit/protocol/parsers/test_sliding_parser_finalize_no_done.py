"""
Tests that SlidingParser.finalize() does NOT emit a DONE event.

Acceptance A3: Synthesis §6 row 3 / A4 PAIN-4.

The orchestrator owns terminal DONE emission (orchestrator.py line 224).
finalize() must only flush residual text/think/error — never DONE.
"""
import pytest
from tether_service.protocol.parsers.sliding import SlidingParser
from tether_service.core.types import StreamEvent


def _has_done(events) -> bool:
    return any(e.get("type") == StreamEvent.DONE for e in events)


class TestFinalizeNoDone:
    """finalize() must not include StreamEvent.DONE in any terminal state."""

    def test_empty_buffer_text_mode(self):
        """mode=text, empty buf: finalize returns [] (no DONE)."""
        parser = SlidingParser()
        events = parser.finalize()
        assert not _has_done(events), f"DONE found in empty-buf finalize: {events}"

    def test_text_in_buffer(self):
        """mode=text, buf has trailing text: returns TEXT event, not DONE."""
        parser = SlidingParser()
        # Feed a short string that stays in the overlap buffer
        parser.feed("Hi")
        events = parser.finalize()
        types = [e["type"] for e in events]
        assert StreamEvent.DONE not in types, f"DONE found: {events}"
        # Should have flushed the buffered text
        text_events = [e for e in events if e["type"] == StreamEvent.TEXT]
        assert text_events, "Expected a TEXT event for buffered 'Hi'"
        assert "Hi" in "".join(e["data"]["delta"] for e in text_events)

    def test_think_mode(self):
        """mode=think, buf has partial think text: returns THINK event, not DONE."""
        parser = SlidingParser()
        # Enter think mode by feeding opening tag without closing tag
        parser.feed("<think>some thoughts")
        assert parser.mode == "think"
        events = parser.finalize()
        types = [e["type"] for e in events]
        assert StreamEvent.DONE not in types, f"DONE found: {events}"

    def test_await_payload_mode(self):
        """mode=await_payload (marker detected but no JSON): returns ERROR, not DONE."""
        parser = SlidingParser()
        # Feed marker without payload
        parser.feed("<<function_call>>")
        assert parser.mode == "await_payload"
        events = parser.finalize()
        types = [e["type"] for e in events]
        assert StreamEvent.DONE not in types, f"DONE found: {events}"
        # Should emit an error for missing payload
        error_events = [e for e in events if e["type"] == StreamEvent.ERROR]
        assert error_events, "Expected an ERROR event for missing tool payload"

    def test_incomplete_json_mode(self):
        """mode=json, buf has partial JSON: returns ERROR, not DONE."""
        parser = SlidingParser()
        # Get into json mode: feed marker + opening brace without closing
        parser.feed('<<function_call>> {"name": "time_tool", "arg')
        # May be in json mode if the buf has partial JSON
        events = parser.finalize()
        types = [e["type"] for e in events]
        assert StreamEvent.DONE not in types, f"DONE found: {events}"

    def test_finalize_after_complete_tool_call(self):
        """finalize after a complete tool call: returns nothing (or text), never DONE."""
        parser = SlidingParser()
        parser.feed('<<function_call>> {"name": "t", "arguments": {}}')
        # Complete tool call — parser returns to text mode
        assert parser.mode == "text"
        events = parser.finalize()
        types = [e["type"] for e in events]
        assert StreamEvent.DONE not in types, f"DONE found: {events}"

    def test_finalize_idempotent_no_done(self):
        """Calling finalize twice does not produce DONE either time."""
        parser = SlidingParser()
        parser.feed("Hello")
        e1 = parser.finalize()
        e2 = parser.finalize()
        assert not _has_done(e1), f"DONE in first finalize: {e1}"
        assert not _has_done(e2), f"DONE in second finalize: {e2}"
