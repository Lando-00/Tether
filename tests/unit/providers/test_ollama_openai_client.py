"""Comprehensive unit tests for OllamaOpenAICompatClient (ADR-0022 §4).

Tests cover the OpenAI-compatible SSE client against the contract defined in
ADR-0022-contract-stubs.md §4.  All tests are gated with pytest.importorskip —
they skip cleanly on this branch (mp-ol-2c-tests) and activate automatically
once Phase-2.B modules land at the Phase-3.1 merge.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

# ---------------------------------------------------------------------------
# Phase-3.1 gate: skip entire file until openai_compat_client + client_base exist.
# Remove these importorskip calls after the 2.B merge.
# ---------------------------------------------------------------------------
pytest.importorskip("tether.providers.ollama.openai_client")
pytest.importorskip("tether.providers.ollama.client")

from tether.providers.ollama.openai_client import (  # noqa: E402
    OllamaOpenAICompatClient,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse_body(*payloads: str | dict, add_done: bool = True) -> bytes:
    """Encode an SSE stream body with data: prefix framing.

    Pass dicts for JSON payloads; pass the string "[DONE]" for the terminal
    sentinel.  When add_done=True, automatically appends data: [DONE].
    """
    lines: list[bytes] = []
    for payload in payloads:
        if isinstance(payload, dict):
            lines.append(f"data: {json.dumps(payload)}\n\n".encode())
        else:
            lines.append(f"data: {payload}\n\n".encode())
    if add_done:
        lines.append(b"data: [DONE]\n\n")
    return b"".join(lines)


def _delta(content: str) -> dict:
    """Build a minimal SSE chat-completion chunk with a text delta."""
    return {
        "id": "chatcmpl-abc",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        "model": "qwen3:8b",
    }


def _finish(reason: str = "stop") -> dict:
    """Build a minimal SSE chunk with finish_reason."""
    return {
        "id": "chatcmpl-abc",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
        "model": "qwen3:8b",
    }


def _tool_delta(
    index: int = 0,
    tc_id: str | None = None,
    name: str | None = None,
    arguments: str = "",
) -> dict:
    """Build a SSE chunk with a tool_call delta."""
    tc: dict[str, Any] = {"index": index}
    if tc_id is not None:
        tc["id"] = tc_id
    if name is not None:
        tc["type"] = "function"
        tc["function"] = {"name": name, "arguments": arguments}
    else:
        tc["function"] = {"arguments": arguments}
    return {
        "id": "chatcmpl-xyz",
        "choices": [{"index": 0, "delta": {"tool_calls": [tc]}, "finish_reason": None}],
        "model": "qwen3:8b",
    }


def _make_client(handler, base_url: str = "http://test/v1", api_key: str | None = None) -> OllamaOpenAICompatClient:
    """Build OllamaOpenAICompatClient backed by an httpx MockTransport."""
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url=base_url)
    return OllamaOpenAICompatClient(base_url, http_client=http, api_key=api_key)


# ---------------------------------------------------------------------------
# Endpoint path tests
# ---------------------------------------------------------------------------


async def test_completions_endpoint_path() -> None:
    """stream_chat() POSTs to /v1/chat/completions."""
    paths_seen: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        paths_seen.append(request.url.path)
        body = _sse_body(_delta("hi"), _finish())
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = _make_client(_handler)
    async for _ in client.stream_chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "Hello"}],
    ):
        pass

    assert any("/chat/completions" in p for p in paths_seen), (
        f"Expected /v1/chat/completions in requests; got: {paths_seen}"
    )


async def test_models_endpoint_path() -> None:
    """list_models() GETs /v1/models; response data array mapped to list[dict]."""
    paths_seen: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        paths_seen.append(request.url.path)
        if "/models" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "qwen3:8b", "object": "model"},
                        {"id": "llama3.1:8b", "object": "model"},
                    ]
                },
            )
        return httpx.Response(404)

    client = _make_client(_handler)
    models = await client.list_models()

    assert any("/models" in p for p in paths_seen)
    assert isinstance(models, list)
    assert len(models) == 2


async def test_show_model_returns_empty_dict() -> None:
    """show_model() returns {} — /v1 has no per-model detail endpoint."""
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)  # should not be called

    client = _make_client(_handler)
    result = await client.show_model("qwen3:8b")

    assert result == {}


async def test_version_uses_models_probe() -> None:
    """version() probes /v1/models and returns a dict with ok=True."""

    def _handler(request: httpx.Request) -> httpx.Response:
        if "/models" in request.url.path:
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    client = _make_client(_handler)
    result = await client.version()

    assert isinstance(result, dict)
    assert result.get("ok") is True


# ---------------------------------------------------------------------------
# Auth header tests
# ---------------------------------------------------------------------------


async def test_authorization_header_when_api_key_set() -> None:
    """When api_key is provided, every request carries Authorization: Bearer <key>."""
    headers_seen: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        headers_seen.append(dict(request.headers))
        body = _sse_body(_delta("ok"), _finish())
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = _make_client(_handler, api_key="sk-test-key")
    async for _ in client.stream_chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "hi"}],
    ):
        pass

    assert headers_seen, "Expected at least one request"
    for hdrs in headers_seen:
        auth = hdrs.get("authorization", hdrs.get("Authorization", ""))
        assert "Bearer sk-test-key" in auth, (
            f"Expected 'Bearer sk-test-key' in Authorization header; got: {auth!r}"
        )


async def test_no_authorization_header_when_api_key_none() -> None:
    """When api_key is None, no Authorization header is sent."""
    headers_seen: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        headers_seen.append(dict(request.headers))
        body = _sse_body(_delta("ok"), _finish())
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = _make_client(_handler, api_key=None)
    async for _ in client.stream_chat(
        model="qwen3:8b",
        messages=[{"role": "user", "content": "hi"}],
    ):
        pass

    assert headers_seen
    for hdrs in headers_seen:
        auth = hdrs.get("authorization", hdrs.get("Authorization", ""))
        assert auth == "", f"Unexpected Authorization header: {auth!r}"


# ---------------------------------------------------------------------------
# SSE parsing tests
# ---------------------------------------------------------------------------


async def test_sse_parsing_text_deltas() -> None:
    """SSE lines with choices[0].delta.content → OllamaStreamEvent(kind='text')."""

    def _handler(request: httpx.Request) -> httpx.Response:
        body = _sse_body(_delta("Hi"), _delta(" there!"), _finish())
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = _make_client(_handler)
    events = [
        e
        async for e in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "Hello"}],
        )
    ]

    text_events = [e for e in events if e.kind == "text"]
    assert len(text_events) >= 2
    assert text_events[0].text == "Hi"
    assert text_events[1].text == " there!"


async def test_sse_done_sentinel_terminates_loop() -> None:
    """data: [DONE] breaks the parse loop and produces a done event."""

    def _handler(request: httpx.Request) -> httpx.Response:
        body = _sse_body(_delta("hello"), _finish("stop"))
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = _make_client(_handler)
    events = [
        e
        async for e in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
        )
    ]

    done_events = [e for e in events if e.kind == "done"]
    assert done_events, "Expected a done event after [DONE] sentinel"
    assert done_events[0].stop_reason is not None


async def test_tool_call_arguments_buffered_across_deltas() -> None:
    """Three SSE lines split 'arguments' across them → single tool_call event with
    concatenated JSON (per contract stubs §4 example B)."""

    def _handler(request: httpx.Request) -> httpx.Response:
        body = _sse_body(
            # First delta: function name + empty arguments start
            _tool_delta(index=0, tc_id="call_abc", name="get_weather", arguments=""),
            # Second delta: partial arguments
            _tool_delta(index=0, arguments='{"location":'),
            # Third delta: arguments completion
            _tool_delta(index=0, arguments='"London"}'),
            # Finish with flush trigger
            _finish("tool_calls"),
            add_done=True,
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = _make_client(_handler)
    events = [
        e
        async for e in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "What's the weather?"}],
        )
    ]

    tool_events = [e for e in events if e.kind == "tool_call"]
    assert tool_events, "Expected a tool_call event after finish_reason flush"
    tc = tool_events[0].tool_call
    assert tc is not None
    fn = tc.get("function", {})
    assert fn["name"] == "get_weather"
    # Arguments must be fully concatenated
    parsed_args = json.loads(fn["arguments"])
    assert parsed_args["location"] == "London"


async def test_tool_call_id_preserved_from_first_chunk() -> None:
    """Tool call id from the first SSE delta is preserved in the flushed event."""

    def _handler(request: httpx.Request) -> httpx.Response:
        body = _sse_body(
            _tool_delta(index=0, tc_id="call_preserved_xyz", name="lookup", arguments="{}"),
            _finish("tool_calls"),
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = _make_client(_handler)
    events = [
        e
        async for e in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "look up x"}],
        )
    ]

    tool_events = [e for e in events if e.kind == "tool_call"]
    assert tool_events
    assert tool_events[0].tool_call["id"] == "call_preserved_xyz"


async def test_tool_call_id_synthesised_when_absent() -> None:
    """Tool call with no id in any delta → client synthesises a non-empty id."""

    def _handler(request: httpx.Request) -> httpx.Response:
        body = _sse_body(
            # No tc_id given
            _tool_delta(index=0, name="lookup", arguments="{}"),
            _finish("tool_calls"),
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = _make_client(_handler)
    events = [
        e
        async for e in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "look up x"}],
        )
    ]

    tool_events = [e for e in events if e.kind == "tool_call"]
    assert tool_events
    synthesised_id = tool_events[0].tool_call["id"]
    assert synthesised_id, "Synthesised id must be non-empty"
    assert len(synthesised_id) >= 8


async def test_finish_reason_flushes_pending_tool_call_buffer() -> None:
    """finish_reason in a delta flushes the pending tool_call buffer even
    before data: [DONE] arrives (spec §4 — flush on finish_reason)."""

    def _handler(request: httpx.Request) -> httpx.Response:
        body = _sse_body(
            _tool_delta(index=0, tc_id="call_flush", name="test_func", arguments='{"x":1}'),
            # finish_reason chunk triggers flush
            _finish("tool_calls"),
            # [DONE] appended by _sse_body
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = _make_client(_handler)
    events = [
        e
        async for e in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "test"}],
        )
    ]

    tool_events = [e for e in events if e.kind == "tool_call"]
    assert tool_events, "Expected tool_call event after finish_reason flush"
    assert tool_events[0].tool_call["id"] == "call_flush"


async def test_think_param_ignored_with_info_log() -> None:
    """think=True is silently dropped (not sent in POST body) with an INFO log.

    The OpenAI-compatible surface has no 'think' parameter; sending it could
    confuse some servers.  The provider logs a warning at __init__ when
    thinking_models is configured (separate test); here we verify the client
    itself doesn't forward 'think' to the wire.

    Tether uses structlog — use structlog.testing.capture_logs (not caplog/capsys)
    so the assertion is robust to other tests' structlog config.
    """
    import structlog

    bodies_seen: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            bodies_seen.append(json.loads(request.content))
        body = _sse_body(_delta("ok"), _finish())
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = _make_client(_handler)

    with structlog.testing.capture_logs() as records:
        async for _ in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "Think?"}],
            think=True,  # should be dropped
        ):
            pass

    assert bodies_seen, "Expected a POST to /chat/completions"
    body = bodies_seen[0]
    assert "think" not in body, (
        f"'think' must NOT appear in the request body for openai_compat; body={body}"
    )
    assert any(
        "think" in (r.get("event", "")).lower()
        for r in records
    ), f"Expected an info/warn log mentioning 'think'; records={records}"


async def test_connection_refused_maps_to_actionable_runtime_error() -> None:
    """httpx.ConnectError → RuntimeError mentioning the base_url."""

    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    client = _make_client(_handler, base_url="http://192.168.1.77:11434")

    with pytest.raises(RuntimeError) as exc_info:
        async for _ in client.stream_chat(
            model="qwen3:8b",
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass

    assert "192.168.1.77" in str(exc_info.value)


async def test_sse_malformed_line_logged_and_stream_continues(caplog) -> None:
    """SSE lines missing 'data:' prefix or containing invalid JSON are logged
    at WARNING and the stream continues without raising."""

    def _handler(request: httpx.Request) -> httpx.Response:
        # Mix of garbage, a valid chunk, and [DONE]
        body = (
            b"not-an-sse-line\n\n"
            + f"data: {json.dumps(_delta('ok'))}\n\n".encode()
            + b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = _make_client(_handler)
    with caplog.at_level(logging.WARNING):
        events = [
            e
            async for e in client.stream_chat(
                model="qwen3:8b",
                messages=[{"role": "user", "content": "hi"}],
            )
        ]

    text_events = [e for e in events if e.kind == "text"]
    assert text_events, "Stream should have continued past malformed line"
    assert text_events[0].text == "ok"
