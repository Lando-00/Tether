import pytest
import json
from llm_service.protocol.orchestration.emitter import NdjsonEventEmitter

@pytest.fixture
def emitter():
    """Provides an NdjsonEventEmitter instance for tests."""
    return NdjsonEventEmitter()

def _load_json_event(byte_string):
    """Helper to decode and parse a single NDJSON event."""
    return json.loads(byte_string.decode('utf-8').strip())

def test_emit_token(emitter):
    """Tests the token event emission."""
    text = "Hello"
    event_bytes = emitter.token(text)
    event = _load_json_event(event_bytes)
    
    assert event == {
        "type": "token",
        "content": text,
        "stream_event": "text"
    }

def test_emit_hidden_thought(emitter):
    """Tests the hidden_thought event emission."""
    text = "Thinking about the next step."
    phase = "pre_tool"
    event_bytes = emitter.hidden_thought(text, phase)
    event = _load_json_event(event_bytes)
    
    assert event == {
        "type": "hidden_thought",
        "content": text,
        "phase": phase,
        "stream_event": "think_stream"
    }

def test_emit_tool_start(emitter):
    """Tests the tool_start event emission."""
    tc_id = "tc_123"
    name = "__tool_get_weather"
    event_bytes = emitter.tool_start(tc_id, name)
    event = _load_json_event(event_bytes)
    
    assert event == {
        "type": "tool_start",
        "id": tc_id,
        "name": name,
        "stream_event": "tool_started"
    }

def test_emit_tool_end(emitter):
    """Tests the tool_end event emission."""
    tc_id = "tc_123"
    name = "__tool_get_weather"
    result = {"temperature": "25°C"}
    event_bytes = emitter.tool_end(tc_id, name, result)
    event = _load_json_event(event_bytes)
    
    assert event == {
        "type": "tool_end",
        "id": tc_id,
        "name": name,
        "result": result,
        "stream_event": "tool_complete"
    }

def test_emit_done(emitter):
    """Tests the done event emission."""
    event_bytes = emitter.done()
    event = _load_json_event(event_bytes)
    
    assert event == {"type": "done", "stream_event": "done"}

def test_emit_error(emitter):
    """Tests the error event emission."""
    message = "An error occurred."
    code = "E101"
    event_bytes = emitter.error(message, code)
    event = _load_json_event(event_bytes)
    
    assert event == {
        "type": "error",
        "message": message,
        "code": code,
        "stream_event": "error"
    }

def test_emit_error_no_code(emitter):
    """Tests the error event emission without an error code."""
    message = "A simple error."
    event_bytes = emitter.error(message)
    event = _load_json_event(event_bytes)
    
    assert event == {
        "type": "error",
        "message": message,
        "stream_event": "error"
    }

def test_emit_cancelled(emitter):
    """Tests the cancelled event emission."""
    event_bytes = emitter.cancelled()
    event = _load_json_event(event_bytes)
    
    assert event == {"type": "cancelled", "stream_event": "cancelled"}

def test_serialization_error_handling(emitter):
    """Tests that a serialization error is handled gracefully."""
    # Sets cannot be serialized to JSON by default
    unserializable_result = {"data", "is", "a", "set"}
    
    event_bytes = emitter.tool_end("tc_456", "bad_tool", unserializable_result)
    event = _load_json_event(event_bytes)
    
    assert event["type"] == "error"
    assert "error" in event
    assert "Failed to serialize tool_end event" in event["error"]
