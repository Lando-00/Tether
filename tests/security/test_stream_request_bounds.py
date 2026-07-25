"""
Security tests: Pydantic body bounds on StreamRequest.

§4 Phase 0A: StreamRequest must validate inputs at the FastAPI layer so that
invalid or malicious values never reach the orchestrator or provider.

Bounds (per acceptance criteria A2):
  - prompt: min_length=1, max_length=32768
  - session_id: pattern=^[A-Za-z0-9_-]{1,128}$
  - model_name: safe local names plus one ``org/repo`` namespace and optional
    ``:quant`` suffix; traversal/path syntax remains rejected.
"""

import pytest
from pydantic import ValidationError

from tether.app.http.routers.chat import StreamRequest

# ---------------------------------------------------------------------------
# Direct Pydantic validation (no HTTP layer needed for most checks)
# ---------------------------------------------------------------------------

def _valid_payload(**overrides):
    base = {
        "session_id": "session-123",
        "prompt": "Hello world",
        "model_name": "Qwen3-4B-q4f16_0-MLC",
    }
    base.update(overrides)
    return base


def test_valid_request_passes():
    """A well-formed request must validate without errors."""
    req = StreamRequest(**_valid_payload())
    assert req.session_id == "session-123"
    assert req.prompt == "Hello world"
    assert req.model_name == "Qwen3-4B-q4f16_0-MLC"


# -- prompt bounds -----------------------------------------------------------

def test_prompt_empty_rejected():
    """Empty prompt (min_length=1) must raise ValidationError."""
    with pytest.raises(ValidationError):
        StreamRequest(**_valid_payload(prompt=""))


def test_prompt_too_long_rejected():
    """Prompt exceeding 32768 chars must raise ValidationError."""
    with pytest.raises(ValidationError):
        StreamRequest(**_valid_payload(prompt="x" * 32769))


def test_prompt_at_max_length_accepted():
    """Prompt exactly 32768 chars must pass."""
    StreamRequest(**_valid_payload(prompt="x" * 32768))


def test_prompt_single_char_accepted():
    """Single-character prompt (min_length boundary) must pass."""
    StreamRequest(**_valid_payload(prompt="a"))


# -- session_id pattern -------------------------------------------------------

@pytest.mark.parametrize("bad_session_id", [
    "",                        # empty
    "a" * 129,                 # too long
    "has space",               # space
    "has/slash",               # path separator
    "has\\backslash",          # backslash
    "has:colon",               # colon
    "has@at",                  # @
    "has.dot",                 # dot not in pattern
    "has!excl",                # !
])
def test_session_id_invalid_rejected(bad_session_id):
    """Invalid session_id values must raise ValidationError."""
    with pytest.raises(ValidationError):
        StreamRequest(**_valid_payload(session_id=bad_session_id))


@pytest.mark.parametrize("good_session_id", [
    "abc",
    "session-123",
    "Session_456",
    "a" * 128,                 # exactly 128 — boundary
    "A1B2C3",
    "abc-def_ghi",
])
def test_session_id_valid_accepted(good_session_id):
    """Valid session_id values must pass."""
    StreamRequest(**_valid_payload(session_id=good_session_id))


# -- model_name pattern -------------------------------------------------------

@pytest.mark.parametrize("bad_model_name", [
    "",                        # empty
    "a" * 257,                 # too long
    "../etc/passwd",           # path traversal
    "..\\windows",             # backslash traversal
    "/etc/passwd",             # absolute path
    "has space",               # space
    "has\\backslash",          # backslash
    "org//repo",               # empty namespace segment
    "org/repo/extra",          # more than one namespace separator
    "org/../repo",             # traversal segment
    ":quant",                  # missing model segment
    "model:",                  # missing quant segment
    "model::quant",            # duplicate quant separator
    "has@at",                  # @
    "has!excl",                # !
])
def test_model_name_invalid_rejected(bad_model_name):
    """Invalid model_name values including traversal attempts must be rejected."""
    with pytest.raises(ValidationError):
        StreamRequest(**_valid_payload(model_name=bad_model_name))


@pytest.mark.parametrize("good_model_name", [
    "Qwen3-4B-q4f16_0-MLC",
    "Qwen2.5-7B-q4f16_0-MLC-adreno",
    "my-model",
    "model123",
    "a" * 128,
    "unsloth/Qwen3-1.7B-GGUF:Q4_0",
    "org/repo",
    "model:Q4_0",
])
def test_model_name_valid_accepted(good_model_name):
    """Valid model names must pass."""
    StreamRequest(**_valid_payload(model_name=good_model_name))


# ---------------------------------------------------------------------------
# HTTP layer: 422 responses via FastAPI TestClient
# ---------------------------------------------------------------------------

def _make_test_app():
    """Build a minimal FastAPI app that mounts only the chat router."""
    from fastapi import FastAPI

    from tether.app.http.routers.chat import router as chat_router

    app = FastAPI()

    class FakeGenService:
        async def stream(self, **kwargs):
            yield b'{"type":"done"}\n'

    app.state.gen_svc = FakeGenService()
    app.include_router(chat_router, prefix="/api/v1")
    return app


def test_http_empty_prompt_returns_422():
    from fastapi.testclient import TestClient
    app = _make_test_app()
    client = TestClient(app)
    resp = client.post("/api/v1/chat/stream", json={
        "session_id": "s1",
        "prompt": "",
        "model_name": "Qwen3-4B-q4f16_0-MLC",
    })
    assert resp.status_code == 422


def test_http_prompt_too_long_returns_422():
    from fastapi.testclient import TestClient
    app = _make_test_app()
    client = TestClient(app)
    resp = client.post("/api/v1/chat/stream", json={
        "session_id": "s1",
        "prompt": "x" * 32769,
        "model_name": "Qwen3-4B-q4f16_0-MLC",
    })
    assert resp.status_code == 422


def test_http_invalid_session_id_returns_422():
    from fastapi.testclient import TestClient
    app = _make_test_app()
    client = TestClient(app)
    resp = client.post("/api/v1/chat/stream", json={
        "session_id": "has/slash",
        "prompt": "hello",
        "model_name": "Qwen3-4B-q4f16_0-MLC",
    })
    assert resp.status_code == 422


def test_http_model_name_with_traversal_returns_422():
    from fastapi.testclient import TestClient
    app = _make_test_app()
    client = TestClient(app)
    resp = client.post("/api/v1/chat/stream", json={
        "session_id": "s1",
        "prompt": "hello",
        "model_name": "../etc/passwd",
    })
    assert resp.status_code == 422


def test_http_valid_request_reaches_handler():
    from fastapi.testclient import TestClient
    app = _make_test_app()
    client = TestClient(app)
    resp = client.post("/api/v1/chat/stream", json={
        "session_id": "session-123",
        "prompt": "Hello",
        "model_name": "Qwen3-4B-q4f16_0-MLC",
    })
    # 200 means the handler was reached (our stub returns 200)
    assert resp.status_code == 200


def test_http_namespaced_quantized_model_reaches_handler():
    """A GenieX org/repo:quant identifier is safe and reaches the handler."""
    from fastapi.testclient import TestClient

    app = _make_test_app()
    client = TestClient(app)
    resp = client.post("/api/v1/chat/stream", json={
        "session_id": "session-123",
        "prompt": "Hello",
        "model_name": "unsloth/Qwen3-1.7B-GGUF:Q4_0",
    })
    assert resp.status_code == 200
