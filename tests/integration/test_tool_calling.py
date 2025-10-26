"""
Integration test for tool calling in tether_service.
Mocks the model stream to emit a tool call and verifies execution.
"""
import pytest
import anyio
from typing import List, Dict, Any, cast
from tether_service.protocol.orchestration.orchestrator import orchestrate
from tether_service.core.interfaces import ModelProvider, StreamParser, SessionStore, Tool
from tether_service.protocol.parsers.sliding import SlidingParser
from tether_service.tools.time_tool import TimeTool


class MockModelProvider(ModelProvider):
    """Mock provider that emits a pre-defined tool call stream."""
    
    def __init__(self, response_chunks):
        self.response_chunks = response_chunks
    
    def stream(self, model_name: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None):
        async def _gen():
            for chunk in self.response_chunks:
                yield chunk
        return _gen()
    
    def list_models(self) -> List[str]:
        return ["test_model"]
    
    def unload_model(self, model_name: str) -> bool:
        return True


class MockSessionStore(SessionStore):
    """Mock session store for testing."""
    
    def __init__(self):
        self.sessions = {}
    
    async def create_session(self, session_id: str, created_at: int):
        pass
    
    async def list_sessions(self):
        return list(self.sessions.keys())
    
    async def delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            self.sessions.pop(session_id)
            return True
        return False
    
    async def delete_all_sessions(self) -> int:
        count = len(self.sessions)
        self.sessions.clear()
        return count
    
    async def clear_history(self, session_id: str):
        self.sessions[session_id] = []
    
    async def ensure_system_prompt(self, session_id: str, prompt: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = [{"role": "system", "content": prompt}]
    
    async def add_user(self, session_id: str, text: str):
        self.sessions.setdefault(session_id, []).append({"role": "user", "content": text})
    
    async def add_assistant_text(self, session_id: str, text: str):
        self.sessions.setdefault(session_id, []).append({"role": "assistant", "content": text})
    
    async def add_assistant_toolcall(self, session_id: str, tool_name: str, args: dict):
        self.sessions.setdefault(session_id, []).append({
            "role": "assistant",
            "tool_calls": [{"function": {"name": tool_name, "arguments": args}}]
        })
    
    async def add_tool_result(self, session_id: str, tool_name: str, result: Any):
        self.sessions.setdefault(session_id, []).append({
            "role": "tool",
            "name": tool_name,
            "content": str(result)
        })
    
    async def get_history(self, session_id: str):
        return self.sessions.get(session_id, [])


@pytest.mark.anyio
async def test_simple_tool_call_execution():
    """Test that a simple tool call is detected and executed."""
    
    # Setup: Create a mock stream that emits a tool call
    chunks = [
        "Let me check the time for you. ",
        '<<function_call>> {"name":"time","arguments":{"timezone":"Europe/Dublin","format":"human"}}'
    ]
    
    provider = MockModelProvider(chunks)
    parser = SlidingParser()
    store = MockSessionStore()
    
    # Create a real TimeTool instance
    time_tool = TimeTool()
    tools: Dict[str, Tool] = cast(Dict[str, Tool], {"time": time_tool})
    
    system_prompt = "You are a helpful assistant."
    
    # Execute orchestration
    results = []
    async for event_bytes in orchestrate(
        session_id="test_session",
        prompt="What time is it?",
        model_name="test_model",
        provider=provider,
        parser=parser,
        store=store,
        tools=tools,
        system_prompt=system_prompt,
    ):
        results.append(event_bytes)
    
    # Parse the NDJSON results
    import json
    events = []
    for result in results:
        try:
            event = json.loads(result.decode('utf-8'))
            events.append(event)
        except:
            pass
    
    # Verify: Should have text events, tool_started, and tool_completed
    event_types = [e.get("type") for e in events]
    
    assert "text" in event_types, "Should have text events"
    assert "tool_started" in event_types, "Should have tool_started event"
    assert "tool_completed" in event_types, "Should have tool_completed event"
    
    # Verify tool was executed with correct args
    tool_started = next((e for e in events if e.get("type") == "tool_started"), None)
    assert tool_started is not None
    assert tool_started["data"]["tool_name"] == "time"
    assert tool_started["data"]["tool_args"]["timezone"] == "Europe/Dublin"
    
    # Verify tool result was returned
    tool_completed = next((e for e in events if e.get("type") == "tool_completed"), None)
    assert tool_completed is not None
    assert "tool_result" in tool_completed["data"]


@pytest.mark.anyio
async def test_tool_call_with_chunk_boundary():
    """Test tool call detection when marker is split across chunks."""
    
    chunks = [
        "Sure, ",
        "<<function_",
        "call>> ",
        '{"name":"time",',
        '"arguments":{"timezone":"UTC","format":"iso"}}'
    ]
    
    provider = MockModelProvider(chunks)
    parser = SlidingParser()
    store = MockSessionStore()
    
    time_tool = TimeTool()
    tools: Dict[str, Tool] = cast(Dict[str, Tool], {"time": time_tool})
    
    system_prompt = "You are a helpful assistant."
    
    results = []
    async for event_bytes in orchestrate(
        session_id="test_session_2",
        prompt="What time is it?",
        model_name="test_model",
        provider=provider,
        parser=parser,
        store=store,
        tools=tools,
        system_prompt=system_prompt,
    ):
        results.append(event_bytes)
    
    import json
    events = []
    for result in results:
        try:
            event = json.loads(result.decode('utf-8'))
            events.append(event)
        except:
            pass
    
    event_types = [e.get("type") for e in events]
    assert "tool_completed" in event_types


@pytest.mark.anyio
async def test_no_tool_call_just_text():
    """Test that normal text responses work without tool calls."""
    
    chunks = [
        "Hello! ",
        "I'm a helpful assistant. ",
        "How can I help you today?"
    ]
    
    provider = MockModelProvider(chunks)
    parser = SlidingParser()
    store = MockSessionStore()
    
    time_tool = TimeTool()
    tools: Dict[str, Tool] = cast(Dict[str, Tool], {"time": time_tool})
    
    system_prompt = "You are a helpful assistant."
    
    results = []
    async for event_bytes in orchestrate(
        session_id="test_session_3",
        prompt="Hello",
        model_name="test_model",
        provider=provider,
        parser=parser,
        store=store,
        tools=tools,
        system_prompt=system_prompt,
    ):
        results.append(event_bytes)
    
    import json
    events = []
    for result in results:
        try:
            event = json.loads(result.decode('utf-8'))
            events.append(event)
        except:
            pass
    
    event_types = [e.get("type") for e in events]
    
    # Should have text events but no tool events
    assert "text" in event_types
    assert "tool_started" not in event_types
    assert "tool_completed" not in event_types


@pytest.mark.anyio
async def test_tool_call_with_newlines():
    """Test tool call detection with newlines around marker and JSON."""
    
    chunks = [
        "Let me help you.\n",
        "<<function_call>>\n",
        '{"name":"time","arguments":{"timezone":"America/New_York","format":"human"}}'
    ]
    
    provider = MockModelProvider(chunks)
    parser = SlidingParser()
    store = MockSessionStore()
    
    time_tool = TimeTool()
    tools: Dict[str, Tool] = cast(Dict[str, Tool], {"time": time_tool})
    
    system_prompt = "You are a helpful assistant."
    
    results = []
    async for event_bytes in orchestrate(
        session_id="test_session_4",
        prompt="What time is it in NY?",
        model_name="test_model",
        provider=provider,
        parser=parser,
        store=store,
        tools=tools,
        system_prompt=system_prompt,
    ):
        results.append(event_bytes)
    
    import json
    events = []
    for result in results:
        try:
            event = json.loads(result.decode('utf-8'))
            events.append(event)
        except:
            pass
    
    event_types = [e.get("type") for e in events]
    assert "tool_completed" in event_types
    
    # Verify correct args
    tool_started = next((e for e in events if e.get("type") == "tool_started"), None)
    assert tool_started is not None
    assert tool_started["data"]["tool_args"]["timezone"] == "America/New_York"
