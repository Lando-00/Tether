"""RequestIdMiddleware integration tests.

Phase 7 step 68: every HTTP response carries X-Request-ID; inbound
X-Request-ID echoes back; missing/invalid generates a new one.
"""
import re

import pytest
from fastapi.testclient import TestClient

from tether_service.app.http.api import create_app


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


def test_response_carries_x_request_id(client):
    """Default request gets a generated X-Request-ID on the response."""
    resp = client.get("/api/v1/protocol/schema")
    assert resp.status_code == 200
    rid = resp.headers.get("x-request-id")
    assert rid is not None
    assert rid.startswith("req-")
    assert len(rid) == 4 + 12  # "req-" + 12 hex


def test_inbound_x_request_id_echoed(client):
    """Inbound X-Request-ID with valid format is echoed back."""
    resp = client.get(
        "/api/v1/protocol/schema",
        headers={"X-Request-ID": "client-supplied-12345"},
    )
    assert resp.headers.get("x-request-id") == "client-supplied-12345"


def test_inbound_invalid_x_request_id_replaced(client):
    """Invalid X-Request-ID (e.g., empty, too short, special chars) is replaced."""
    for bad_id in ["", "abc", "x" * 200, "id with spaces", "id@with#special"]:
        resp = client.get(
            "/api/v1/protocol/schema",
            headers={"X-Request-ID": bad_id},
        )
        rid = resp.headers.get("x-request-id")
        assert rid != bad_id, f"Expected replacement for {bad_id!r}, got {rid}"
        assert rid.startswith("req-"), f"Expected req- prefix, got {rid}"


def test_two_requests_have_different_ids(client):
    """Two consecutive requests without inbound ID get distinct generated IDs."""
    r1 = client.get("/api/v1/protocol/schema")
    r2 = client.get("/api/v1/protocol/schema")
    assert r1.headers["x-request-id"] != r2.headers["x-request-id"]


def test_request_id_format_pattern(client):
    """Generated request_id matches the expected pattern."""
    resp = client.get("/api/v1/protocol/schema")
    rid = resp.headers["x-request-id"]
    assert re.match(r"^req-[a-f0-9]{12}$", rid), f"Pattern mismatch: {rid}"


def test_chat_endpoint_also_gets_request_id(client):
    """The /chat/stream endpoint (POST + streaming response) also gets X-Request-ID."""
    resp = client.post(
        "/api/v1/chat/stream",
        json={"session_id": "rid-test", "prompt": "hi", "model_name": "dummy"},
    )
    assert resp.status_code == 200
    rid = resp.headers.get("x-request-id")
    assert rid is not None
    assert rid.startswith("req-")
