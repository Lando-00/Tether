"""
Unit tests for the SlidingParser with focus on tool call detection.
Tests chunk boundaries, newlines, balanced JSON, and quoted braces.
"""
import pytest
from tether_service.protocol.parsers.sliding import SlidingParser
from tether_service.protocol.parsers.events import (
    PParseError,
    PText,
    PThink,
    PToolCallDetected,
    PToolCallParsed,
)


class TestSlidingParserBasics:
    """Test basic text streaming and parser state management."""

    def test_simple_text(self):
        parser = SlidingParser()
        events = parser.feed("Hello world")
        # Short text is held in overlap buffer until finalize
        assert len(events) == 0 or (len(events) == 1 and isinstance(events[0], PText))

        # Finalize to get remaining text
        final_events = parser.finalize()
        combined = events + final_events
        text_events = [e for e in combined if isinstance(e, PText)]
        assert len(text_events) >= 1
        text_content = "".join(e.text for e in text_events)
        assert "Hello world" in text_content

    def test_text_with_overlap_buffer(self):
        parser = SlidingParser()
        # Feed a chunk longer than OVERLAP
        long_text = "a" * 100
        events = parser.feed(long_text)
        # Should emit all but OVERLAP chars
        assert len(events) == 1
        assert isinstance(events[0], PText)
        emitted_len = len(events[0].text)
        assert emitted_len == len(long_text) - parser.OVERLAP

    def test_empty_chunk(self):
        parser = SlidingParser()
        events = parser.feed("")
        assert events == []


class TestSlidingParserToolCallMarker:
    """Test detection of <<function_call>> marker across various scenarios."""

    def test_marker_in_single_chunk(self):
        parser = SlidingParser()
        events = parser.feed("Here is a tool call: <<function_call>>")

        # Should emit text before marker and PToolCallDetected
        assert len(events) == 2
        assert isinstance(events[0], PText)
        assert "Here is a tool call: " in events[0].text
        assert isinstance(events[1], PToolCallDetected)

    def test_marker_split_across_chunks(self):
        parser = SlidingParser()
        # Split marker across chunks: "<<func" + "tion_call>>"
        events1 = parser.feed("Some text <<func")
        # Should keep overlap, no PToolCallDetected yet
        assert all(not isinstance(e, PToolCallDetected) for e in events1)

        events2 = parser.feed("tion_call>>")
        # Now should detect marker
        assert any(isinstance(e, PToolCallDetected) for e in events2)

    def test_marker_with_leading_newline(self):
        parser = SlidingParser()
        events = parser.feed("\n<<function_call>>")

        # Should emit newline as text, then PToolCallDetected
        found_marker = any(isinstance(e, PToolCallDetected) for e in events)
        assert found_marker

    def test_marker_with_trailing_newline(self):
        parser = SlidingParser()
        events = parser.feed("<<function_call>>\n")

        # Should detect marker
        assert any(isinstance(e, PToolCallDetected) for e in events)


class TestSlidingParserJsonPayload:
    """Test JSON payload parsing after tool call marker."""

    def test_simple_json_payload(self):
        parser = SlidingParser()
        input_text = '<<function_call>> {"name":"get_current_time","arguments":{"timezone":"UTC"}}'
        events = parser.feed(input_text)

        # Should have PToolCallDetected and PToolCallParsed
        assert any(isinstance(e, PToolCallDetected) for e in events)

        tool_complete = next((e for e in events if isinstance(e, PToolCallParsed)), None)
        assert tool_complete is not None
        assert tool_complete.name == "get_current_time"
        assert tool_complete.arguments["timezone"] == "UTC"
        assert tool_complete.tool_call_id.startswith("call-")

    def test_json_with_quoted_braces(self):
        parser = SlidingParser()
        input_text = '<<function_call>> {"name":"test","arguments":{"msg":"Hello {world}"}}'
        events = parser.feed(input_text)

        tool_complete = next((e for e in events if isinstance(e, PToolCallParsed)), None)
        assert tool_complete is not None
        assert tool_complete.arguments["msg"] == "Hello {world}"

    def test_json_with_escaped_quotes(self):
        parser = SlidingParser()
        input_text = r'<<function_call>> {"name":"test","arguments":{"msg":"He said \"hi\""}}'
        events = parser.feed(input_text)

        tool_complete = next((e for e in events if isinstance(e, PToolCallParsed)), None)
        assert tool_complete is not None
        assert tool_complete.arguments["msg"] == 'He said "hi"'

    def test_json_split_across_chunks(self):
        parser = SlidingParser()
        # Split JSON across multiple chunks
        parser.feed('<<function_call>> {"name":"get_')
        parser.feed('current_time","arg')
        events = parser.feed('uments":{"timezone":"UTC"}}')

        tool_complete = next((e for e in events if isinstance(e, PToolCallParsed)), None)
        assert tool_complete is not None
        assert tool_complete.name == "get_current_time"

    def test_json_with_nested_objects(self):
        parser = SlidingParser()
        input_text = '<<function_call>> {"name":"test","arguments":{"nested":{"level2":{"level3":"value"}}}}'
        events = parser.feed(input_text)

        tool_complete = next((e for e in events if isinstance(e, PToolCallParsed)), None)
        assert tool_complete is not None
        assert tool_complete.arguments["nested"]["level2"]["level3"] == "value"

    def test_json_with_arrays(self):
        parser = SlidingParser()
        input_text = '<<function_call>> {"name":"test","arguments":{"items":[1,2,3]}}'
        events = parser.feed(input_text)

        tool_complete = next((e for e in events if isinstance(e, PToolCallParsed)), None)
        assert tool_complete is not None
        assert tool_complete.arguments["items"] == [1, 2, 3]

    def test_json_with_whitespace_after_marker(self):
        parser = SlidingParser()
        input_text = '<<function_call>>   \n  {"name":"test","arguments":{}}'
        events = parser.feed(input_text)

        tool_complete = next((e for e in events if isinstance(e, PToolCallParsed)), None)
        assert tool_complete is not None
        assert tool_complete.name == "test"

    def test_tool_call_id_is_unique_across_consecutive_parses(self):
        """Phase 5 ``p5-parser-typed-events`` A3:
        ``PToolCallParsed.tool_call_id`` is generated per parse and unique
        across consecutive tool calls so downstream consumers can treat it
        as a primary key for the call."""
        parser = SlidingParser()
        events = parser.feed('<<function_call>> {"name":"a","arguments":{}}')
        events += parser.feed('<<function_call>> {"name":"b","arguments":{}}')

        parsed = [e for e in events if isinstance(e, PToolCallParsed)]
        assert len(parsed) == 2
        for evt in parsed:
            assert evt.tool_call_id.startswith("call-")
            # 12 hex chars after "call-"
            assert len(evt.tool_call_id) == len("call-") + 12
        assert parsed[0].tool_call_id != parsed[1].tool_call_id


class TestSlidingParserErrorCases:
    """Test error handling for malformed inputs."""

    def test_incomplete_json_payload(self):
        parser = SlidingParser()
        parser.feed('<<function_call>> {"name":"test","arguments":{')
        events = parser.finalize()

        # Should emit error about incomplete payload
        error = next((e for e in events if isinstance(e, PParseError)), None)
        assert error is not None
        assert "incomplete" in error.message

    def test_missing_json_payload(self):
        parser = SlidingParser()
        parser.feed('<<function_call>>')
        events = parser.finalize()

        # Should emit error about missing payload
        error = next((e for e in events if isinstance(e, PParseError)), None)
        assert error is not None
        assert "missing" in error.message

    def test_invalid_json_syntax(self):
        parser = SlidingParser()
        events = parser.feed('<<function_call>> {name:"test"}')  # Missing quotes

        # Should emit error about parse failure
        error = next((e for e in events if isinstance(e, PParseError)), None)
        assert error is not None
        assert "parse" in error.message

    def test_non_json_after_marker(self):
        parser = SlidingParser()
        events = parser.feed('<<function_call>> this is not json')

        # Parser should detect marker, then revert to text mode when it sees non-JSON
        # Since "this is not json" is short, it might be in overlap buffer
        final_events = parser.finalize()
        all_events = events + final_events

        # Should have either emitted text or held it until finalize
        text_events = [e for e in all_events if isinstance(e, PText)]
        # The non-JSON part should eventually be treated as text
        assert len(text_events) > 0 or any(isinstance(e, PToolCallDetected) for e in all_events)

    def test_truncated_large_json(self):
        parser = SlidingParser(max_tool_chars=100)
        # Create JSON that won't close within the limit - missing closing braces
        large_json_incomplete = '<<function_call>> {"name":"test","arguments":{"data":"' + ('x' * 150)
        events = parser.feed(large_json_incomplete)

        # Should emit truncation error because buffer exceeds max_tool_chars
        error = next((e for e in events if isinstance(e, PParseError)), None)
        assert error is not None
        assert "truncated" in error.message


class TestSlidingParserThinkMode:
    """Test <think>...</think> tag handling."""

    def test_simple_think_block(self):
        parser = SlidingParser()
        events = parser.feed("<think>reasoning here</think>")

        think_events = [e for e in events if isinstance(e, PThink)]
        assert len(think_events) == 1
        assert "reasoning" in think_events[0].text

    def test_think_split_across_chunks(self):
        parser = SlidingParser()
        parser.feed("<think>start of ")
        events = parser.feed("reasoning</think>")

        think_events = [e for e in events if isinstance(e, PThink)]
        assert len(think_events) >= 1


class TestSlidingParserFinalize:
    """Test finalize behavior."""

    def test_finalize_with_remaining_text(self):
        parser = SlidingParser()
        parser.feed("Some text")
        events = parser.finalize()

        # F3 (synthesis §6 row 3): finalize flushes residual text only.
        # DONE is emitted once by the orchestrator, not by finalize.
        assert any(isinstance(e, PText) for e in events)
        # The parser never emits a DONE-equivalent ParserEvent; ensure no
        # tool-call events leak from finalize either.
        assert not any(isinstance(e, (PToolCallDetected, PToolCallParsed)) for e in events)

    def test_finalize_resets_state(self):
        parser = SlidingParser()
        parser.feed('<<function_call>> {"name":"test"}')
        parser.finalize()

        # After finalize, parser should be reset
        assert parser.mode == "text"
        assert parser.buf == ""
        assert not parser._tool_started

    def test_finalize_emits_no_terminal_event(self):
        # F3 (synthesis §6 row 3): finalize does NOT emit a terminal event.
        # The orchestrator is the sole emitter of the terminal done event.
        parser = SlidingParser()
        events = parser.finalize()

        # No events at all from a fresh parser's finalize
        assert events == []


class TestSlidingParserRealWorldScenarios:
    """Test realistic streaming scenarios."""

    def test_text_then_tool_call(self):
        parser = SlidingParser()
        events1 = parser.feed("Let me check the time for you. ")
        events2 = parser.feed('<<function_call>> {"name":"get_current_time","arguments":{"timezone":"Europe/Dublin","format":"human"}}')

        all_events = events1 + events2

        # Should have text, PToolCallDetected, and PToolCallParsed
        assert any(isinstance(e, PText) for e in all_events)
        assert any(isinstance(e, PToolCallDetected) for e in all_events)
        assert any(isinstance(e, PToolCallParsed) for e in all_events)

    def test_multiple_text_chunks_then_tool(self):
        parser = SlidingParser()
        events = []
        events += parser.feed("I'll ")
        events += parser.feed("help ")
        events += parser.feed("you with that. ")
        events += parser.feed('<<function_call>> {"name":"test","arguments":{}}')

        assert any(isinstance(e, PToolCallParsed) for e in events)

    def test_tool_call_at_chunk_boundary(self):
        parser = SlidingParser()
        # Simulate marker exactly at chunk boundary
        events1 = parser.feed("Some text here<<")
        events2 = parser.feed("function_call>>")
        events3 = parser.feed(' {"name":"test","arguments":{}}')

        all_events = events1 + events2 + events3
        assert any(isinstance(e, PToolCallDetected) for e in all_events)
        assert any(isinstance(e, PToolCallParsed) for e in all_events)
