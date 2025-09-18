import pytest
import json
from fastapi.testclient import TestClient

# Dummy model to replace real ModelComponent
class DummyModel:
    def __init__(self, dist_path):
        pass

    def get_available_models(self):
        return [{"model_name": "dummy", "model_dir": "/tmp"}]

    def unload_model(self, name, device):
        return True

    def generate(self, model_name, messages, device, dll_path, max_tokens, temperature, top_p, tools):
        # Simulate LLM generate() returning a response object with no tool calls
        class Msg:
            role = "assistant"
            content = "Hello from mock"
            tool_calls = []  # no tool calls
        class Choice:
            finish_reason = "stop"
            message = Msg()
        class FakeResponse:
            choices = [Choice()]
            usage = None
        return FakeResponse()

    def clear_engine_cache(self):
        pass

@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    # override ModelComponent in app
    monkeypatch.setattr("llm_service.app.ModelComponent", DummyModel)

@pytest.fixture
def client():
    # create app with in-memory DB and dummy model
    from llm_service.app import create_mcp_app
    app = create_mcp_app(dist_path="dist", database_url="sqlite:///:memory:")
    # Use context manager to trigger shutdown events
    with TestClient(app) as c:
        yield c


def test_generate_endpoint(client):
    # create session
    res = client.post("/sessions")
    assert res.status_code == 200
    session_id = res.json()["session_id"]

    payload = {"session_id": session_id, "prompt": "Hi", "model_name": "dummy"}
    res = client.post("/generate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["reply"] == "Hello from mock"
    # messages include user and assistant
    msgs = data.get("messages", [])
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant"]


def test_generate_stream_endpoint(client):
    # create session
    res = client.post("/sessions")
    session_id = res.json()["session_id"]
    payload = {"session_id": session_id, "prompt": "Hi", "model_name": "dummy"}

    with client.stream("POST", "/generate_stream", json=payload) as resp:
        assert resp.status_code == 200
        tokens = []
        for line in resp.iter_lines():
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("event") == "token":
                tokens.append(obj.get("data"))
        assert ''.join(tokens) == "Hello from mock"
