# Move test_mlc_service_advanced.py to tests/ and update import for package structure.
from llm_service.mlc_service_advanced import *
"""
test_mlc_service_advanced.py
Unit tests for mlc_service_advanced.py using FastAPI's TestClient and pytest.
Mocks HTTP layer for true unit tests.
Use `pytest tests/test_mlc_service_advanced.py -v` from root directory
"""

import pytest
from fastapi.testclient import TestClient
from llm_service import mlc_service_advanced

@pytest.fixture(scope="module")
def client():
    with TestClient(mlc_service_advanced.app) as c:
        yield c

def test_list_models(client: TestClient):
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert isinstance(data["models"], list)

def test_create_and_list_session(client: TestClient):
    resp = client.post("/sessions")
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    assert session_id
    # List sessions
    resp = client.get("/sessions")
    assert resp.status_code == 200
    sessions = resp.json()
    assert any(s["session_id"] == session_id for s in sessions)
    # Do not return values from test functions

def test_generate_and_stream(client: TestClient):
    # Create session
    resp = client.post("/sessions")
    session_id = resp.json()["session_id"]
    # Get a model
    models = client.get("/models").json()["models"]
    if not models:
        pytest.skip("No models available for testing.")
    model_name = models[0]["model_name"]
    payload = {
        "session_id": session_id,
        "prompt": "Hello!",
        "model_name": model_name
    }
    # Test /generate
    resp = client.post("/generate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == session_id
    assert "reply" in data
    # Test /generate_stream
    with client.stream("POST", "/generate_stream", json=payload) as resp:
        assert resp.status_code == 200
        chunks = list(resp.iter_text())
        assert any(chunks)

def test_session_messages_and_delete(client: TestClient):
    # Create session
    resp = client.post("/sessions")
    session_id = resp.json()["session_id"]
    # Get messages (should be empty)
    resp = client.get(f"/sessions/{session_id}/messages")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
    # Delete session
    resp = client.delete(f"/sessions/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["detail"] == "Session deleted"

def test_unload_model(client: TestClient):
    """
    Tests that a model can be successfully unloaded.
    """
    # First, ensure a model is loaded by creating a session and generating a message
    # This will place the model in the cache.
    model_name = "DeepSeek-R1-Distill-Qwen-1.5B-q4f16_0-MLC" # Use a model you have locally
    session_res = client.post("/sessions", json={"model_name": model_name, "device": "auto"})
    assert session_res.status_code == 200
    session_id = session_res.json()["session_id"]

    gen_res = client.post(
        "/generate",
        json={
            "session_id": session_id,
            "prompt": "hello",
            "model_name": model_name,
        },
    )
    # This might fail if the model isn't fully set up, but it will still trigger the cache load.
    # We don't need to assert 200 here, just that the attempt was made.

    # Now, unload the model
    unload_res = client.post("/models/unload", json={"model_name": model_name, "device": "opencl"})
    assert unload_res.status_code == 200
    assert "unloaded successfully" in unload_res.json()["detail"]

    # Try to unload it again, which should fail
    unload_again_res = client.post("/models/unload", json={"model_name": model_name, "device": "opencl"})
    assert unload_again_res.status_code == 404
    assert "not found in cache" in unload_again_res.json()["detail"]