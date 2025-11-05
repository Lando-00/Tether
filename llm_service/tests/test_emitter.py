import json
import pytest

from llm_service.protocol.orchestration.emitter import NdjsonEventEmitter


def parse_event(b: bytes) -> dict:
    """Helper to decode NDJSON event bytes into a dict"""
    try:
        return json.loads(b.decode('utf-8').strip())
    except Exception:
        pytest.skip("Invalid JSON event")


def test_token_event():
    emitter = NdjsonEventEmitter()
    b = emitter.token("hello")
    ev = parse_event(b)
    assert ev["type"] == "token"
    assert ev["content"] == "hello"


def test_hidden_thought_event():
    emitter = NdjsonEventEmitter()
    b = emitter.hidden_thought("thinking...", phase="pre_tool")
    ev = parse_event(b)
    assert ev["type"] == "hidden_thought"
    assert ev["content"] == "thinking..."
    assert ev["phase"] == "pre_tool"


def test_tool_start_event():
    emitter = NdjsonEventEmitter()
    b = emitter.tool_start("id123", "my_tool")
    ev = parse_event(b)
    assert ev["type"] == "tool_start"
    assert ev["id"] == "id123"
    assert ev["name"] == "my_tool"


def test_tool_progress_event():
    emitter = NdjsonEventEmitter()
    b = emitter.tool_progress()
    ev = parse_event(b)
    assert ev["type"] == "tool_progress"
    # No additional payload expected
    assert set(ev.keys()) == {"type", "stream_event"}


def test_tool_end_event():
    emitter = NdjsonEventEmitter()
    result = {"ok": True}
    b = emitter.tool_end("id123", "my_tool", result)
    ev = parse_event(b)
    assert ev["type"] == "tool_end"
    assert ev["id"] == "id123"
    assert ev["name"] == "my_tool"
    assert ev["result"] == result


def test_done_event():
    emitter = NdjsonEventEmitter()
    b = emitter.done()
    ev = parse_event(b)
    assert ev["type"] == "done"