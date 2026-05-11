"""P0-B2 regression: mutating routes require application/json."""
from fastapi.testclient import TestClient

from tether.app.http.api import create_app


def test_post_without_content_type_rejected_415(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "test")
    app = create_app()
    # TestClient default Host is "testserver" which TrustedHost (default-on
    # since P0-B2) rejects with 400; override to a localhost value so the
    # request reaches the Content-Type middleware under test.
    with TestClient(app, base_url="http://localhost") as client:
        # Build a JSON body but send with text/plain (or no content-type).
        r = client.post(
            "/api/v1/sessions",
            content=b'{}',
            headers={"Content-Type": "text/plain"},
        )
        assert r.status_code == 415
        assert "Content-Type" in r.json().get("detail", "")


def test_post_with_application_json_passes(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "test")
    app = create_app()
    with TestClient(app, base_url="http://localhost") as client:
        r = client.post("/api/v1/sessions", json={})
        assert r.status_code != 415
