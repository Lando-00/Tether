import pytest
import json
from fastapi.testclient import TestClient

# Dummy model that emits hidden-thought tokens
class DummyModelHidden:
    def __init__(self, dist_path):
        self._calls = 0

    def get_available_models(self):
        return [{"model_name": "dummy_hidden", "model_dir": "/tmp"}]

    def unload_model(self, name, device):
        return True

    def generate(self, model_name, messages, device, dll_path, max_tokens, temperature, top_p, tools, stream):
        # Simulate streaming of tokens with hidden-thought blocks
        from types import SimpleNamespace
        tokens = ["Visible1 ", "<think>", "secret1", "</think>", "Visible2"]
        for tok in tokens:
            # Create a fake chunk with delta.content
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=tok), finish_reason=None)])
        # Final chunk to signal end of stream
        yield SimpleNamespace(choices=[SimpleNamespace(delta=None, finish_reason="stop")])

    def clear_engine_cache(self):
        pass

@pytest.fixture(autouse=True)
def patch_model_and_tool(monkeypatch):
    # Patch ModelComponent to use our dummy hidden model
    monkeypatch.setattr("llm_service.app.ModelComponent", DummyModelHidden)
    # Patch execute_tool to no-op (no tools in this test)
    from llm_service.protocol.api import execute_tool
    monkeypatch.setattr("llm_service.protocol.api.execute_tool", lambda name, args: "")

@pytest.fixture
def client():
    from llm_service.app import create_mcp_app
    app = create_mcp_app(dist_path="dist", database_url="sqlite:///:memory:")
    with TestClient(app) as c:
        yield c


def test_hidden_thinking_stream(client):
    # Create a session
    res = client.post("/sessions")
    assert res.status_code == 200
    session_id = res.json()["session_id"]

    # Stream with dummy model that emits hidden blocks
    payload = {"session_id": session_id, "prompt": "Test hidden", "model_name": "dummy_hidden"}
    with client.stream("POST", "/generate_stream", json=payload) as resp:
        assert resp.status_code == 200
        events = [json.loads(line) for line in resp.iter_lines() if line]

    # Collect hidden_thought events
    hidden_events = [e for e in events if e.get("event") == "hidden_thought"]
    assert hidden_events, "Expected hidden_thought events"
    # Check that secret1 is among hidden data
    assert any("secret1" in e.get("data", "") for e in hidden_events)
    # Ensure default phase is pre_tool since no tools invoked
    assert all(e.get("phase") == "pre_tool" for e in hidden_events)

    # Collect visible token events
    token_events = [e for e in events if e.get("event") == "token"]
    assert token_events, "Expected token events"
    concatenated = ''.join([e.get("data", "") for e in token_events])
    assert "Visible1 " in concatenated
    assert "Visible2" in concatenated
