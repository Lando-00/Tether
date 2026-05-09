"""
Unit tests for the SlidingParser with focus on tool call detection.
Tests chunk boundaries, newlines, balanced JSON, and quoted braces.
"""
import pytest
from tether_service.protocol.parsers.sliding import SlidingParser
from tether_service.core.types import StreamEvent


class TestSlidingParserBasics:
    """Test basic text streaming and parser state management."""
    
    def test_simple_text(self):
        parser = SlidingParser()
        events = parser.feed("Hello world")
        # Short text is held in overlap buffer until finalize
        assert len(events) == 0 or (len(events) == 1 and events[0]["type"] == StreamEvent.TEXT)
        
        # Finalize to get remaining text
        final_events = parser.finalize()
        combined = events + final_events
        text_events = [e for e in combined if e["type"] == StreamEvent.TEXT]
        assert len(text_events) >= 1
        text_content = "".join(e["data"]["delta"] for e in text_events)
        assert "Hello world" in text_content
    
    def test_text_with_overlap_buffer(self):
        parser = SlidingParser()
        # Feed a chunk longer than OVERLAP
        long_text = "a" * 100
        events = parser.feed(long_text)
        # Should emit all but OVERLAP chars
        assert len(events) == 1
        emitted_len = len(events[0]["data"]["delta"])
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
        
        # Should emit text before marker and TOOL_STARTED
        assert len(events) == 2
        assert events[0]["type"] == StreamEvent.TEXT
        assert "Here is a tool call: " in events[0]["data"]["delta"]
        assert events[1]["type"] == StreamEvent.TOOL_STARTED
    
    def test_marker_split_across_chunks(self):
        parser = SlidingParser()
        # Split marker across chunks: "<<func" + "tion_call>>"
        events1 = parser.feed("Some text <<func")
        # Should keep overlap, no TOOL_STARTED yet
        assert all(e["type"] != StreamEvent.TOOL_STARTED for e in events1)
        
        events2 = parser.feed("tion_call>>")
        # Now should detect marker
        assert any(e["type"] == StreamEvent.TOOL_STARTED for e in events2)
    
    def test_marker_with_leading_newline(self):
        parser = SlidingParser()
        events = parser.feed("\n<<function_call>>")
        
        # Should emit newline as text, then TOOL_STARTED
        found_marker = any(e["type"] == StreamEvent.TOOL_STARTED for e in events)
        assert found_marker
    
    def test_marker_with_trailing_newline(self):
        parser = SlidingParser()
        events = parser.feed("<<function_call>>\n")
        
        # Should detect marker
        assert any(e["type"] == StreamEvent.TOOL_STARTED for e in events)


class TestSlidingParserJsonPayload:
    """Test JSON payload parsing after tool call marker."""
    
    def test_simple_json_payload(self):
        parser = SlidingParser()
        input_text = '<<function_call>> {"name":"get_current_time","arguments":{"timezone":"UTC"}}'
        events = parser.feed(input_text)
        
        # Should have TOOL_STARTED and TOOL_COMPLETE
        assert any(e["type"] == StreamEvent.TOOL_STARTED for e in events)
        
        tool_complete = next((e for e in events if e["type"] == StreamEvent.TOOL_COMPLETE), None)
        assert tool_complete is not None
        assert tool_complete["data"]["tool_name"] == "get_current_time"
        assert tool_complete["data"]["tool_args"]["timezone"] == "UTC"
    
    def test_json_with_quoted_braces(self):
        parser = SlidingParser()
        input_text = '<<function_call>> {"name":"test","arguments":{"msg":"Hello {world}"}}'
        events = parser.feed(input_text)
        
        tool_complete = next((e for e in events if e["type"] == StreamEvent.TOOL_COMPLETE), None)
        assert tool_complete is not None
        assert tool_complete["data"]["tool_args"]["msg"] == "Hello {world}"
    
    def test_json_with_escaped_quotes(self):
        parser = SlidingParser()
        input_text = r'<<function_call>> {"name":"test","arguments":{"msg":"He said \"hi\""}}'
        events = parser.feed(input_text)
        
        tool_complete = next((e for e in events if e["type"] == StreamEvent.TOOL_COMPLETE), None)
        assert tool_complete is not None
        assert tool_complete["data"]["tool_args"]["msg"] == 'He said "hi"'
    
    def test_json_split_across_chunks(self):
        parser = SlidingParser()
        # Split JSON across multiple chunks
        parser.feed('<<function_call>> {"name":"get_')
        parser.feed('current_time","arg')
        events = parser.feed('uments":{"timezone":"UTC"}}')
        
        tool_complete = next((e for e in events if e["type"] == StreamEvent.TOOL_COMPLETE), None)
        assert tool_complete is not None
        assert tool_complete["data"]["tool_name"] == "get_current_time"
    
    def test_json_with_nested_objects(self):
        parser = SlidingParser()
        input_text = '<<function_call>> {"name":"test","arguments":{"nested":{"level2":{"level3":"value"}}}}'
        events = parser.feed(input_text)
        
        tool_complete = next((e for e in events if e["type"] == StreamEvent.TOOL_COMPLETE), None)
        assert tool_complete is not None
        assert tool_complete["data"]["tool_args"]["nested"]["level2"]["level3"] == "value"
    
    def test_json_with_arrays(self):
        parser = SlidingParser()
        input_text = '<<function_call>> {"name":"test","arguments":{"items":[1,2,3]}}'
        events = parser.feed(input_text)
        
        tool_complete = next((e for e in events if e["type"] == StreamEvent.TOOL_COMPLETE), None)
        assert tool_complete is not None
        assert tool_complete["data"]["tool_args"]["items"] == [1, 2, 3]
    
    def test_json_with_whitespace_after_marker(self):
        parser = SlidingParser()
        input_text = '<<function_call>>   \n  {"name":"test","arguments":{}}'
        events = parser.feed(input_text)
        
        tool_complete = next((e for e in events if e["type"] == StreamEvent.TOOL_COMPLETE), None)
        assert tool_complete is not None
        assert tool_complete["data"]["tool_name"] == "test"


class TestSlidingParserErrorCases:
    """Test error handling for malformed inputs."""
    
    def test_incomplete_json_payload(self):
        parser = SlidingParser()
        parser.feed('<<function_call>> {"name":"test","arguments":{')
        events = parser.finalize()
        
        # Should emit error about incomplete payload
        error = next((e for e in events if e["type"] == StreamEvent.ERROR), None)
        assert error is not None
        assert "incomplete" in error["data"]["message"]
    
    def test_missing_json_payload(self):
        parser = SlidingParser()
        parser.feed('<<function_call>>')
        events = parser.finalize()
        
        # Should emit error about missing payload
        error = next((e for e in events if e["type"] == StreamEvent.ERROR), None)
        assert error is not None
        assert "missing" in error["data"]["message"]
    
    def test_invalid_json_syntax(self):
        parser = SlidingParser()
        events = parser.feed('<<function_call>> {name:"test"}')  # Missing quotes
        
        # Should emit error about parse failure
        error = next((e for e in events if e["type"] == StreamEvent.ERROR), None)
        assert error is not None
        assert "parse" in error["data"]["message"]
    
    def test_non_json_after_marker(self):
        parser = SlidingParser()
        events = parser.feed('<<function_call>> this is not json')
        
        # Parser should detect marker, then revert to text mode when it sees non-JSON
        # Since "this is not json" is short, it might be in overlap buffer
        final_events = parser.finalize()
        all_events = events + final_events
        
        # Should have either emitted text or held it until finalize
        text_events = [e for e in all_events if e["type"] == StreamEvent.TEXT]
        # The non-JSON part should eventually be treated as text
        assert len(text_events) > 0 or any(e["type"] == StreamEvent.TOOL_STARTED for e in all_events)
    
    def test_truncated_large_json(self):
        parser = SlidingParser(max_tool_chars=100)
        # Create JSON that won't close within the limit - missing closing braces
        large_json_incomplete = '<<function_call>> {"name":"test","arguments":{"data":"' + ('x' * 150)
        events = parser.feed(large_json_incomplete)
        
        # Should emit truncation error because buffer exceeds max_tool_chars
        error = next((e for e in events if e["type"] == StreamEvent.ERROR), None)
        assert error is not None
        assert "truncated" in error["data"]["message"]


class TestSlidingParserThinkMode:
    """Test <think>...</think> tag handling."""
    
    def test_simple_think_block(self):
        parser = SlidingParser()
        events = parser.feed("<think>reasoning here</think>")
        
        think_events = [e for e in events if e["type"] == StreamEvent.THINK]
        assert len(think_events) == 1
        assert "reasoning" in think_events[0]["data"]["delta"]
    
    def test_think_split_across_chunks(self):
        parser = SlidingParser()
        parser.feed("<think>start of ")
        events = parser.feed("reasoning</think>")
        
        think_events = [e for e in events if e["type"] == StreamEvent.THINK]
        assert len(think_events) >= 1


class TestSlidingParserFinalize:
    """Test finalize behavior."""
    
    def test_finalize_with_remaining_text(self):
        parser = SlidingParser()
        parser.feed("Some text")
        events = parser.finalize()
        
        # Should emit remaining text and DONE
        assert any(e["type"] == StreamEvent.TEXT for e in events)
        assert any(e["type"] == StreamEvent.DONE for e in events)
    
    def test_finalize_resets_state(self):
        parser = SlidingParser()
        parser.feed('<<function_call>> {"name":"test"}')
        parser.finalize()
        
        # After finalize, parser should be reset
        assert parser.mode == "text"
        assert parser.buf == ""
        assert not parser._tool_started
    
    def test_finalize_emits_done(self):
        parser = SlidingParser()
        events = parser.finalize()
        
        assert any(e["type"] == StreamEvent.DONE for e in events)


class TestSlidingParserRealWorldScenarios:
    """Test realistic streaming scenarios."""
    
    def test_text_then_tool_call(self):
        parser = SlidingParser()
        events1 = parser.feed("Let me check the time for you. ")
        events2 = parser.feed('<<function_call>> {"name":"get_current_time","arguments":{"timezone":"Europe/Dublin","format":"human"}}')
        
        all_events = events1 + events2
        
        # Should have text, TOOL_STARTED, and TOOL_COMPLETE
        assert any(e["type"] == StreamEvent.TEXT for e in all_events)
        assert any(e["type"] == StreamEvent.TOOL_STARTED for e in all_events)
        assert any(e["type"] == StreamEvent.TOOL_COMPLETE for e in all_events)
    
    def test_multiple_text_chunks_then_tool(self):
        parser = SlidingParser()
        events = []
        events += parser.feed("I'll ")
        events += parser.feed("help ")
        events += parser.feed("you with that. ")
        events += parser.feed('<<function_call>> {"name":"test","arguments":{}}')
        
        assert any(e["type"] == StreamEvent.TOOL_COMPLETE for e in events)
    
    def test_tool_call_at_chunk_boundary(self):
        parser = SlidingParser()
        # Simulate marker exactly at chunk boundary
        events1 = parser.feed("Some text here<<")
        events2 = parser.feed("function_call>>")
        events3 = parser.feed(' {"name":"test","arguments":{}}')
        
        all_events = events1 + events2 + events3
        assert any(e["type"] == StreamEvent.TOOL_STARTED for e in all_events)
        assert any(e["type"] == StreamEvent.TOOL_COMPLETE for e in all_events)
