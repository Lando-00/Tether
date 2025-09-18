import pytest
import json
from fastapi.testclient import TestClient

# Dummy model that simulates a tool call then final response
class DummyModelWithTool:
    def __init__(self, dist_path):
        self._calls = 0

    def get_available_models(self):
        return [{"model_name": "dummy", "model_dir": "/tmp"}]

    def unload_model(self, name, device):
        return True

    def generate(self, model_name, messages, device, dll_path, max_tokens, temperature, top_p, tools):
        # First call: simulate tool call
        if self._calls == 0:
            # Simulate tool call
            class FuncCall:
                name = "get_current_time"
                arguments = {"timezone": "Asia/Tokyo"}
            class ToolCallObj:
                function = FuncCall()
                id = "1"
            class MsgCall:
                role = "assistant"
                content = None
                tool_calls = [ToolCallObj()]
            class ChoiceCall:
                finish_reason = "tool_call"
                message = MsgCall()
            class ToolResponse:
                choices = [ChoiceCall()]
                usage = None
            self._calls += 1
            return ToolResponse()
        else:
            # Return final assistant message
            class MsgResp:
                role = "assistant"
                content = "Time in Tokyo is 12:00 PM"
                tool_calls = []
            class ChoiceResp:
                finish_reason = "stop"
                message = MsgResp()
            class FinalResponse:
                choices = [ChoiceResp()]
                usage = None
            return FinalResponse()

    def clear_engine_cache(self):
        pass

@pytest.fixture(autouse=True)
def patch_model_and_tool(monkeypatch):
    # Patch ModelComponent
    monkeypatch.setattr("llm_service.app.ModelComponent", DummyModelWithTool)
    # Patch execute_tool to return a mock tool result
    from llm_service.protocol.api import execute_tool
    monkeypatch.setattr("llm_service.protocol.api.execute_tool", lambda name, args: "2025-09-18T03:00:00Z")

@pytest.fixture
def client():
    from llm_service.app import create_mcp_app
    app = create_mcp_app(dist_path="dist", database_url="sqlite:///:memory:")
    # Use context manager so ASGI lifespan shutdown is called after tests
    with TestClient(app) as c:
        yield c


def test_stream_with_tool_call(client):
    # Create a session
    res = client.post("/sessions")
    session_id = res.json()["session_id"]

    # Stream with prompt that triggers a tool call
    payload = {"session_id": session_id, "prompt": "What is the time in Tokyo?", "model_name": "dummy"}
    with client.stream("POST", "/generate_stream", json=payload) as resp:
        assert resp.status_code == 200
        events = [json.loads(line) for line in resp.iter_lines() if line]

    # Check that a tool start and end event occurred for get_current_time
    assert any(e["event"] == "tool_start" and e["tool"] == "get_current_time" for e in events)
    assert any(e["event"] == "tool_end" and e.get("tool") == "get_current_time" for e in events)

    # Check that token events produce the final message
    tokens = [e.get("data") for e in events if e.get("event") == "token"]
    text = ''.join(tokens)
    assert "Time in Tokyo is" in text
