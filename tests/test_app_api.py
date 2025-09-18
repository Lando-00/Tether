import pytest
import os
from fastapi.testclient import TestClient
from llm_service.app import create_mcp_app

@pytest.fixture(scope="module")
def client():
	# Setup a file-based SQLite DB for testing
	db_path = os.path.abspath("test_app.db")
	# Remove existing test DB if it exists
	try:
		os.remove(db_path)
	except OSError:
		pass
	database_url = f"sqlite:///{db_path}"
	# Create the FastAPI app with test DB
	app = create_mcp_app(dist_path="dist", database_url=database_url)
	with TestClient(app) as c:
		yield c
	# Teardown test DB
	try:
		os.remove(db_path)
	except OSError:
		pass

def test_healthz(client):
	resp = client.get("/healthz")
	assert resp.status_code == 200
	assert resp.json() == {"ok": True}

def test_list_models(client):
	resp = client.get("/models")
	assert resp.status_code == 200
	data = resp.json()
	assert "models" in data
	assert isinstance(data["models"], list)

def test_list_tools(client):
	resp = client.get("/tools")
	assert resp.status_code == 200
	data = resp.json()
	assert "tools" in data
	assert isinstance(data["tools"], list)

def test_session_lifecycle(client):
	# Create session
	resp = client.post("/sessions")
	assert resp.status_code == 200
	data = resp.json()
	assert "session_id" in data
	session_id = data["session_id"]

	# List sessions
	resp = client.get("/sessions")
	assert resp.status_code == 200
	sessions = resp.json()
	assert any(s.get("session_id") == session_id for s in sessions)

	# Get session messages (no messages) should return 404
	resp = client.get(f"/sessions/{session_id}/messages")
	assert resp.status_code == 404

	# Delete session
	resp = client.delete(f"/sessions/{session_id}")
	assert resp.status_code == 200
	assert resp.json().get("detail") == "Session deleted"

	# Delete all sessions
	resp = client.delete("/sessions")
	assert resp.status_code == 200
	assert "Deleted" in resp.json().get("detail", "")

def test_generate_validation(client):
	# Missing payload should return 422
	resp = client.post("/generate", json={})
	assert resp.status_code == 422

def test_unload_model_always_succeeds(client):
	# Unloading any model returns success
	payload = {"model_name": "nonexistent", "device": "auto"}
	resp = client.post("/models/unload", json=payload)
	assert resp.status_code == 200
	assert "unloaded successfully" in resp.json().get("detail", "")

def test_generate_stream_validation(client):
	# Missing payload should return 422
	resp = client.post("/generate_stream", json={})
	assert resp.status_code == 422
