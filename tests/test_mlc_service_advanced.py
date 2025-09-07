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
