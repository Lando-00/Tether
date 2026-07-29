"""Comprehensive unit tests for OllamaNativeClient (ADR-0022 §3).

Tests cover the NDJSON streaming client against the contract defined in
ADR-0022-contract-stubs.md §3.  All tests are gated with pytest.importorskip —
they skip cleanly on this branch (mp-ol-2c-tests) and activate automatically
once Phase-2.A modules land at the Phase-3.1 merge.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Phase-3.1 gate: skip entire file until native_client + client_base exist.
# Remove these importorskip calls after the 2.A merge.
# ---------------------------------------------------------------------------
pytest.importorskip("tether.providers.ollama.client")
pytest.importorskip("tether.providers.ollama.client")

from tether.providers.ollama.client import (
    OllamaNativeClient,  # noqa: E402
    OllamaStreamEvent,  # noqa: E402
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ndjson(*dicts: dict) -> bytes:
    """Encode dicts as NDJSON bytes (one JSON object per line)."""
    return ("\n".join(json.dumps(d) for d in dicts) + "\n").encode()


def _make_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _make_client(handler, base_url: str = "http://test") -> OllamaNativeClient:
    """Build OllamaNativeClient backed by an httpx MockTransport."""
    transport = _make_transport(handler)
    http = httpx.AsyncClient(transport=transport, base_url=base_url)
    return OllamaNativeClient(base_url, http_client=http)


# Simple chat response NDJSON for reuse
_CHAT_LINES = [
    {"model": "qwen3:8b", "message": {"role": "assistant", "content": "Hi"}, "done": False},
    {"model": "qwen3:8b", "message": {"role": "assistant", "content": " there!"}, "done": False},
    {"model": "qwen3:8b", "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
]


# ---------------------------------------------------------------------------
# Endpoint shape tests
# ---------------------------------------------------------------------------


async def test_version_endpoint_shape() -> None:
    """version() GETs /api/version and returns the parsed JSON body."""
    requests_seen: list[httpx.Request] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.6.0"})
        return httpx.Response(404)

    client = _make_client(_handler)
    result = await client.version()

    assert len(requests_seen) == 1
    assert requests_seen[0].method == "GET"
    assert "/api/version" in str(requests_seen[0].url)
    assert result.get("version") == "0.6.0"


async def test_list_models_returns_models_array() -> None:
    """list_models() GETs /api/tags and returns the list under the 'models' key."""

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "qwen3:8b", "size": 123},
                        {"name": "llama3.1:8b", "size": 456},
                    ]
                },
            )
        return httpx.Response(404)

    client = _make_client(_handler)
    models = await client.list_models()

    assert isinstance(models, list)
    assert len(models) == 2
    names = [m["name"] for m in models]
    assert "qwen3:8b" in names
    assert "llama3.1:8b" in names


async def test_show_model_posts_name() -> None:
    """show_model() POSTs {"name": model} to /api/show."""
    bodies_seen: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show" and request.method == "POST":
            bodies_seen.append(json.loads(request.content))
            return httpx.Response(200, json={"modelfile": "FROM qwen3:8b"})
        return httpx.Response(404)

    client = _make_client(_handler)
    result = await client.show_model("qwen3:8b")

    assert len(bodies_seen) == 1
    assert bodies_seen[0].get("name") == "qwen3:8b"
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# stream_chat request body tests
# ---------------------------------------------------------------------------


async def test_stream_chat_posts_correct_body() -> None:
    """stream_chat POSTs a body with model, messages, stream=True, think, tools, options."""
    bodies_seen: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat" and request.method == "POST":
            body = json.loads(request.content)
            bodies_seen.append(body)
            lines = [
                json.dumps({"model": "qwen3:8b", "message": {"content": "ok"}, "done": False}),
                json.dumps({"model": "qwen3:8b", "done": True, "done_reason": "stop"}),
            ]
            return httpx.Response(
                200,
                content=("\n".join(lines) + "\n").encode(),
                headers={"content-type": "application/x-ndjson"},
            )
        return httpx.Response(404)

    client = _make_client(_handler)
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    options = {"temperature": 0.7}

    # Consume the stream; we only care about the request body the
    # client emits (asserted below via ``bodies_seen``).
    _ = [
        e
        async for e in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "Hi"}],
            tools=tools,
            think=True,
            options=options,
        )
    ]

    assert len(bodies_seen) == 1
    body = bodies_seen[0]
    assert body["model"] == "qwen3:8b"
    assert body["messages"] == [{"role": "user", "content": "Hi"}]
    assert body["stream"] is True
    assert body["think"] is True
    assert body["tools"] == tools
    assert body["options"] == options


# ---------------------------------------------------------------------------
# stream_chat event yield tests
# ---------------------------------------------------------------------------


async def test_stream_chat_yields_text_events() -> None:
    """feed mocked /api/chat NDJSON; client yields OllamaStreamEvent(kind='text') in order."""

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_ndjson(*_CHAT_LINES),
            headers={"content-type": "application/x-ndjson"},
        )

    client = _make_client(_handler)
    events = [
        e
        async for e in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "Hello"}],
        )
    ]

    text_events = [e for e in events if e.kind == "text"]
    assert text_events, "Expected at least one text event"
    assert text_events[0].text == "Hi"
    assert text_events[1].text == " there!"


async def test_stream_chat_yields_thinking_events() -> None:
    """NDJSON with message.thinking → OllamaStreamEvent(kind='thinking')."""
    thinking_lines = [
        {"model": "qwen3:8b", "message": {"thinking": "Let me reason...", "content": ""}, "done": False},
        {"model": "qwen3:8b", "message": {"thinking": "", "content": "Answer"}, "done": False},
        {"model": "qwen3:8b", "message": {"content": ""}, "done": True, "done_reason": "stop"},
    ]

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_ndjson(*thinking_lines),
            headers={"content-type": "application/x-ndjson"},
        )

    client = _make_client(_handler)
    events = [
        e
        async for e in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "Explain entanglement"}],
            think=True,
        )
    ]

    think_events = [e for e in events if e.kind == "thinking"]
    assert think_events, "Expected at least one thinking event"
    assert think_events[0].text == "Let me reason..."


async def test_stream_chat_yields_tool_call_events_with_existing_id() -> None:
    """Ollama tool_call with an 'id' field → id is preserved in OllamaStreamEvent."""
    tool_line = {
        "model": "qwen3:8b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "preserved-id-123",
                    "function": {"name": "get_weather", "arguments": {"location": "London"}},
                }
            ],
        },
        "done": True,
        "done_reason": "stop",
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_ndjson(tool_line),
            headers={"content-type": "application/x-ndjson"},
        )

    client = _make_client(_handler)
    events = [
        e
        async for e in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "Weather?"}],
        )
    ]

    tool_events = [e for e in events if e.kind == "tool_call"]
    assert tool_events, "Expected a tool_call event"
    assert tool_events[0].tool_call is not None
    assert tool_events[0].tool_call["id"] == "preserved-id-123"
    assert tool_events[0].tool_call["function"]["name"] == "get_weather"


async def test_stream_chat_yields_tool_call_events_synthesises_id() -> None:
    """Ollama tool_call without 'id' → client synthesises a 12-char hex uuid."""
    tool_line = {
        "model": "qwen3:8b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                # Note: no "id" key
                {"function": {"name": "get_weather", "arguments": {"location": "Paris"}}},
            ],
        },
        "done": True,
        "done_reason": "stop",
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_ndjson(tool_line),
            headers={"content-type": "application/x-ndjson"},
        )

    client = _make_client(_handler)
    events = [
        e
        async for e in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "Weather in Paris?"}],
        )
    ]

    tool_events = [e for e in events if e.kind == "tool_call"]
    assert tool_events
    synthesised_id = tool_events[0].tool_call["id"]
    assert len(synthesised_id) >= 8, "Synthesised id must be at least 8 hex chars"
    # Should be hex-like (uuid4().hex[:12] pattern)
    assert all(c in "0123456789abcdef" for c in synthesised_id), (
        f"Expected hex id, got: {synthesised_id!r}"
    )


async def test_stream_chat_tool_call_arguments_serialised_to_string() -> None:
    """When Ollama returns arguments as a dict, client serialises to JSON string
    (MLC-style canonical: 'arguments' is always a str)."""
    tool_line = {
        "model": "qwen3:8b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "London", "unit": "celsius"},
                    }
                }
            ],
        },
        "done": True,
        "done_reason": "stop",
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_ndjson(tool_line))

    client = _make_client(_handler)
    events = [
        e
        async for e in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "Weather?"}],
        )
    ]

    tool_events = [e for e in events if e.kind == "tool_call"]
    assert tool_events
    args = tool_events[0].tool_call["function"]["arguments"]
    # arguments must be a JSON string, not a dict
    assert isinstance(args, str), f"Expected str, got {type(args).__name__}: {args!r}"
    parsed = json.loads(args)
    assert parsed["location"] == "London"


async def test_stream_chat_done_event_breaks_loop() -> None:
    """done=True line terminates the generator; subsequent lines are ignored."""

    def _handler(request: httpx.Request) -> httpx.Response:
        lines = [
            json.dumps({"model": "qwen3:8b", "message": {"content": "hello"}, "done": False}),
            json.dumps({"model": "qwen3:8b", "done": True, "done_reason": "stop"}),
            # This line comes AFTER done=True; must not produce events.
            json.dumps({"model": "qwen3:8b", "message": {"content": "SHOULD NOT APPEAR"}, "done": False}),
        ]
        return httpx.Response(200, content=("\n".join(lines) + "\n").encode())

    client = _make_client(_handler)
    events = [
        e
        async for e in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "Hi"}],
        )
    ]

    texts = [e.text for e in events if e.kind == "text"]
    assert "SHOULD NOT APPEAR" not in texts
    done_events = [e for e in events if e.kind == "done"]
    assert len(done_events) == 1
    assert done_events[0].stop_reason == "stop"


async def test_stream_chat_skips_malformed_lines() -> None:
    """A non-JSON line emits a WARNING log and stream continues without raising.

    Tether uses structlog; we use structlog.testing.capture_logs which is
    independent of caplog / capsys and not affected by other tests' structlog
    config.
    """
    import structlog

    def _handler(request: httpx.Request) -> httpx.Response:
        lines = [
            "THIS IS NOT JSON",  # malformed
            json.dumps({"model": "qwen3:8b", "message": {"content": "ok"}, "done": False}),
            json.dumps({"model": "qwen3:8b", "done": True, "done_reason": "stop"}),
        ]
        return httpx.Response(200, content=("\n".join(lines) + "\n").encode())

    client = _make_client(_handler)
    with structlog.testing.capture_logs() as records:
        events = [
            e
            async for e in client.stream_chat(
                model="qwen3:8b",
                messages=[{"role": "user", "content": "Hi"}],
            )
        ]

    text_events = [e for e in events if e.kind == "text"]
    assert text_events, "Stream must continue past malformed line"
    assert text_events[0].text == "ok"
    # Warning emitted via structlog.
    assert any(
        "malformed" in (r.get("event", "")).lower() or "ndjson" in (r.get("event", "")).lower()
        for r in records
    ), f"Expected malformed/ndjson warning; got records: {records}"


async def test_stream_chat_uses_aiter_lines_correctly() -> None:
    """Response body with embedded newlines in content is correctly split into
    NDJSON lines (validates that the client uses line-based parsing, not
    raw byte iteration)."""

    def _handler(request: httpx.Request) -> httpx.Response:
        # Two valid NDJSON lines in a single response body
        line1 = json.dumps({"model": "qwen3:8b", "message": {"content": "line1"}, "done": False})
        line2 = json.dumps({"model": "qwen3:8b", "message": {"content": "line2"}, "done": False})
        line3 = json.dumps({"model": "qwen3:8b", "done": True, "done_reason": "stop"})
        body = (line1 + "\n" + line2 + "\n" + line3 + "\n").encode()
        return httpx.Response(200, content=body)

    client = _make_client(_handler)
    events = [
        e
        async for e in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
        )
    ]

    texts = [e.text for e in events if e.kind == "text"]
    assert "line1" in texts
    assert "line2" in texts


# ---------------------------------------------------------------------------
# Error mapping tests
# ---------------------------------------------------------------------------


async def test_404_model_not_found_maps_to_actionable_runtime_error() -> None:
    """HTTP 404 with 'model ... not found' body → RuntimeError mentioning
    model name and 'ollama pull'."""

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"error": "model 'qwen3:72b' not found, try pulling it first"},
        )

    client = _make_client(_handler)
    with pytest.raises(RuntimeError) as exc_info:
        async for _ in client.stream_chat(
            model="qwen3:72b",
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass

    msg = str(exc_info.value)
    assert "qwen3:72b" in msg
    assert "ollama pull" in msg.lower() or "pull" in msg.lower()


async def test_connection_refused_maps_to_actionable_runtime_error() -> None:
    """httpx.ConnectError → RuntimeError mentioning the base_url."""

    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    client = _make_client(_handler, base_url="http://192.168.1.99:11434")
    with pytest.raises(RuntimeError) as exc_info:
        async for _ in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass

    assert "192.168.1.99" in str(exc_info.value)


async def test_timeout_maps_to_actionable_runtime_error() -> None:
    """httpx.TimeoutException → RuntimeError mentioning timeout."""

    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("Read timed out")

    client = _make_client(_handler)
    with pytest.raises(RuntimeError) as exc_info:
        async for _ in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass

    msg = str(exc_info.value).lower()
    assert "timeout" in msg or "timed out" in msg


# ---------------------------------------------------------------------------
# Cancel token test
# ---------------------------------------------------------------------------


async def test_cancel_token_breaks_stream_loop() -> None:
    """Flipping a cancel token mid-stream causes the generator to exit cleanly."""

    class _FakeCancel:
        def __init__(self) -> None:
            self._set = False

        def is_set(self) -> bool:
            return self._set

        def set(self) -> None:
            self._set = True

    cancel = _FakeCancel()

    def _handler(request: httpx.Request) -> httpx.Response:
        # Many lines; cancel token will fire before all are consumed.
        lines = [
            json.dumps({"model": "qwen3:8b", "message": {"content": f"chunk{i}"}, "done": False})
            for i in range(10)
        ] + [json.dumps({"model": "qwen3:8b", "done": True, "done_reason": "stop"})]
        return httpx.Response(200, content=("\n".join(lines) + "\n").encode())

    client = _make_client(_handler)

    events_seen: list[OllamaStreamEvent] = []
    async for event in client.stream_chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "hi"}],
        cancel_token=cancel,
    ):
        events_seen.append(event)
        if len(events_seen) >= 3:
            cancel.set()  # cancel mid-stream

    # Generator must exit cleanly (no exception) and not have consumed all 10 chunks.
    assert len(events_seen) < 12, "Generator should have stopped early after cancel"


# ---------------------------------------------------------------------------
# Client lifecycle tests (owned vs borrowed)
# ---------------------------------------------------------------------------


async def test_aclose_closes_owned_client() -> None:
    """Client constructed without http_client creates and owns its httpx client;
    aclose() closes it."""
    mock_instance = AsyncMock()

    with patch("httpx.AsyncClient") as mock_http_cls:
        mock_http_cls.return_value = mock_instance
        # Construct standalone — no injected http_client
        client = OllamaNativeClient(base_url="http://localhost:11434")

    await client.aclose()
    mock_instance.aclose.assert_called_once()


async def test_aclose_skips_borrowed_client() -> None:
    """Client constructed WITH an injected http_client borrows it; aclose() is a no-op."""
    borrowed = AsyncMock()
    client = OllamaNativeClient("http://test", http_client=borrowed)

    await client.aclose()
    borrowed.aclose.assert_not_called()
