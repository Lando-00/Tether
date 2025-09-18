import pytest
import os
import json
from fastapi.testclient import TestClient
from llm_service.app import create_mcp_app

# Integration test using the real Qwen2.5 model
# WARNING: This test will load the actual model and may be slow.
# Ensure the model is available under the 'dist' directory.

@pytest.fixture(scope="module")
def client():
    # Use actual dist path and in-memory DB
    from llm_service.app import create_mcp_app
    app = create_mcp_app(dist_path="dist", database_url="sqlite:///:memory:")
    # Use context manager to ensure shutdown events fire
    with TestClient(app) as c:
        yield c

@pytest.mark.integration
def test_qwen2_5_generate(client):
    # Create a session
    res = client.post("/sessions")
    assert res.status_code == 200
    session_id = res.json()["session_id"]

    # Find Qwen2.5 model from available models
    models = client.get("/models").json().get("models", [])
    qwen_models = [m for m in models if "Qwen2.5-7B" in m.get("model_name", "")]
    if not qwen_models:
        pytest.skip("Qwen2.5-7B model not found in 'dist'")
    model_name = qwen_models[0]["model_name"]

    # Perform a simple generation
    payload = {
        "session_id": session_id,
        "prompt": "Hello, how are you?",
        "model_name": model_name
    }
    res = client.post("/generate", json=payload)
    assert res.status_code == 200
    data = res.json()
    reply = data.get("reply", "")
    assert isinstance(reply, str) and reply.strip(), "Expected non-empty reply"

    # Basic content check
    assert "" in reply  # placeholder, adjust based on expected output

@pytest.mark.integration
def test_qwen2_5_stream(client):
    # Create a new session
    res = client.post("/sessions")
    session_id = res.json()["session_id"]

    # Use same model as above
    models = client.get("/models").json().get("models", [])
    model_name = [m for m in models if "Qwen2.5-7B" in m.get("model_name", "")][0]["model_name"]

    payload = {
        "session_id": session_id,
        "prompt": "Test streaming output.",
        "model_name": model_name
    }
    with client.stream("POST", "/generate_stream", json=payload) as resp:
        assert resp.status_code == 200
        tokens = []
        for line in resp.iter_lines():
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("event") == "token":
                tokens.append(obj.get("data"))
        text = ''.join(tokens)
        assert text.strip(), "Expected non-empty streamed text"

@pytest.mark.integration
def test_qwen2_5_stream_with_tool(client):
    # Create a new session
    res = client.post("/sessions")
    assert res.status_code == 200
    session_id = res.json()["session_id"]

    # Identify Qwen2.5 model
    models = client.get("/models").json().get("models", [])
    qwen_models = [m for m in models if "Qwen2.5-7B" in m.get("model_name", "")]
    if not qwen_models:
        pytest.skip("Qwen2.5-7B model not found; skipping tool integration test")
    model_name = qwen_models[0]["model_name"]

    # Fetch available tools definitions
    tools_defs = client.get("/tools").json().get("tools", [])

    # Stream with prompt that should trigger get_current_time tool
    payload = {
        "session_id": session_id,
        "prompt": "What is the time in Tokyo?",
        "model_name": model_name,
        "tools": tools_defs
    }
    with client.stream("POST", "/generate_stream", json=payload) as resp:
        assert resp.status_code == 200
        events = [json.loads(line) for line in resp.iter_lines() if line]

    # Confirm a tool call event for get_current_time
    assert any(e.get("event") == "tool_start" and e.get("tool") == "get_current_time" for e in events)
    assert any(e.get("event") == "tool_end" and e.get("tool") == "get_current_time" for e in events)

    # Assemble token data into text and check for expected prefix
    tokens = [e.get("data") for e in events if e.get("event") == "token"]
    text = ''.join(tokens)
    assert "T" in text, f"Expected time string in response, got {text!r}"
