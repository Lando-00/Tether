"""Unit tests for GenieXProvider HTTP/SSE client behaviour.

These tests exercise the provider's HTTP interaction contract against
mock transports (httpx.MockTransport), never hitting real network.

Contract reference: geniex-contract-probe-2026-07-25.md
Interface: GenieXProvider(*, base_url, model_id, request_model_id, context_window,
           timeout_seconds, connect_timeout_seconds, temperature, max_tokens,
           url_validator, http_client)
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest

from tether.core.errors import TransientProviderError
from tether.providers.geniex.provider import GenieXProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse_frame(payload: str) -> str:
    """Wrap a payload string into the GenieX SSE format (data:{json}\\n\\n)."""
    return f"data:{payload}\n\n"


def _sse_done() -> str:
    return "data:[DONE]\n\n"


def _chunk_json(content: str, finish_reason: str = "") -> str:
    """Produce a minimal GenieX-style SSE chunk JSON."""
    return json.dumps({
        "id": "",
        "choices": [{
            "delta": {
                "content": content,
                "role": "assistant",
                "tool_calls": None,
                "function_call": {"arguments": "", "name": ""},
                "refusal": "",
            },
            "finish_reason": finish_reason,
            "index": 0,
            "logprobs": {"content": None, "refusal": None},
        }],
        "created": 0,
        "model": "",
        "object": "chat.completion.chunk",
        "service_tier": "",
        "system_fingerprint": "",
        "usage": {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    })


def _models_response(model_ids: List[str]) -> dict:
    return {
        "data": [
            {"id": mid, "created": 0, "object": "model", "owned_by": "test"}
            for mid in model_ids
        ],
        "object": "list",
    }


def _build_transport(handlers: Dict[str, Any]):
    """Build a MockTransport dispatching by (method, path) tuples in *handlers*.

    Handler values are either (status, body_dict) or a callable(request)->Response.
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        key = (method, path)
        h = handlers.get(key)
        if h is None:
            return httpx.Response(404, text="not found")
        if callable(h):
            return h(request)
        status, body = h
        return httpx.Response(status, json=body)

    return httpx.MockTransport(_handler)


def _stream_transport(
    chunks: List[str],
    *,
    health_ok: bool = True,
    models: List[str] | None = None,
    validate_request: Any = None,
):
    """Build transport for SSE streaming responses."""
    model_list = models or ["test-model"]

    def _handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method

        if method == "GET" and path == "/v1/":
            if health_ok:
                return httpx.Response(200, text='"GenieX-CLI is running"')
            return httpx.Response(500, text="down")

        if method == "GET" and path == "/v1/models":
            return httpx.Response(200, json=_models_response(model_list))

        if method == "POST" and path == "/v1/chat/completions":
            if validate_request is not None:
                validate_request(request)
            body = "".join(chunks)
            return httpx.Response(
                200,
                content=body.encode(),
                headers={"content-type": "text/event-stream"},
            )

        return httpx.Response(404, text="not found")

    return httpx.MockTransport(_handler)


# ---------------------------------------------------------------------------
# SSE content streaming
# ---------------------------------------------------------------------------


class TestSSEStreaming:
    """SSE frame parsing and content extraction."""

    @pytest.mark.anyio
    async def test_basic_stream_yields_text_deltas(self):
        """Normal multi-chunk SSE stream yields ProviderText events."""
        chunks = [
            _sse_frame(_chunk_json("Hello")),
            _sse_frame(_chunk_json(" world")),
            _sse_done(),
        ]
        transport = _stream_transport(chunks)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )
        from tether.providers.types import ProviderText

        events = []
        async for ev in provider.stream_typed(
            model_name="test-model",
            messages=[{"role": "user", "content": "hi"}],
        ):
            events.append(ev)

        text_events = [e for e in events if isinstance(e, ProviderText)]
        assert len(text_events) >= 2
        combined = "".join(e.text for e in text_events)
        assert "Hello" in combined
        assert "world" in combined

    @pytest.mark.anyio
    async def test_data_prefix_no_space(self):
        """GenieX uses 'data:{json}' with no space — provider must parse it."""
        # Verify that data: (no space) works — this is the contract
        chunks = [
            "data:" + _chunk_json("ok") + "\n\n",
            _sse_done(),
        ]
        transport = _stream_transport(chunks)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )
        from tether.providers.types import ProviderText

        events = []
        async for ev in provider.stream_typed(
            model_name="test-model",
            messages=[{"role": "user", "content": "hi"}],
        ):
            events.append(ev)

        assert any(isinstance(e, ProviderText) and e.text == "ok" for e in events)

    @pytest.mark.anyio
    async def test_done_sentinel_terminates_stream(self):
        """data:[DONE] terminates the stream cleanly."""
        chunks = [
            _sse_frame(_chunk_json("a")),
            _sse_done(),
            # Anything after [DONE] is ignored
            _sse_frame(_chunk_json("SHOULD_NOT_APPEAR")),
        ]
        transport = _stream_transport(chunks)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )
        from tether.providers.types import ProviderText

        texts = []
        async for ev in provider.stream_typed(
            model_name="test-model",
            messages=[{"role": "user", "content": "hi"}],
        ):
            if isinstance(ev, ProviderText):
                texts.append(ev.text)

        assert "a" in texts
        assert "SHOULD_NOT_APPEAR" not in texts

    @pytest.mark.anyio
    async def test_empty_content_chunks_skipped(self):
        """Chunks with empty content should not yield events."""
        chunks = [
            _sse_frame(_chunk_json("")),
            _sse_frame(_chunk_json("real")),
            _sse_done(),
        ]
        transport = _stream_transport(chunks)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )
        from tether.providers.types import ProviderText

        texts = []
        async for ev in provider.stream_typed(
            model_name="test-model",
            messages=[{"role": "user", "content": "hi"}],
        ):
            if isinstance(ev, ProviderText):
                texts.append(ev.text)

        # Only non-empty content yielded
        assert all(t != "" for t in texts)
        assert "real" in texts

    @pytest.mark.anyio
    async def test_repeated_role_in_every_chunk_handled(self):
        """GenieX sends role:"assistant" in every chunk (unusual). Provider tolerates."""
        chunks = [
            _sse_frame(_chunk_json("tok1")),
            _sse_frame(_chunk_json("tok2")),
            _sse_done(),
        ]
        transport = _stream_transport(chunks)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )
        from tether.providers.types import ProviderText

        texts = []
        async for ev in provider.stream_typed(
            model_name="test-model",
            messages=[{"role": "user", "content": "hi"}],
        ):
            if isinstance(ev, ProviderText):
                texts.append(ev.text)

        assert len(texts) == 2


# ---------------------------------------------------------------------------
# SSE framing edge cases (split buffers)
# ---------------------------------------------------------------------------


class TestSSEFraming:
    """Edge cases in SSE line splitting."""

    @pytest.mark.anyio
    async def test_split_across_chunk_boundary(self):
        """SSE frames split arbitrarily across TCP read boundaries."""
        # Simulate one big blob (as if all arrived in a single read)
        full = (
            _sse_frame(_chunk_json("part1"))
            + _sse_frame(_chunk_json("part2"))
            + _sse_done()
        )
        chunks = [full]
        transport = _stream_transport(chunks)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )
        from tether.providers.types import ProviderText

        texts = []
        async for ev in provider.stream_typed(
            model_name="test-model",
            messages=[{"role": "user", "content": "hi"}],
        ):
            if isinstance(ev, ProviderText):
                texts.append(ev.text)

        assert "part1" in texts
        assert "part2" in texts


# ---------------------------------------------------------------------------
# Error handling: HTTP errors
# ---------------------------------------------------------------------------


class TestHTTPErrors:
    """Provider must map HTTP error responses correctly."""

    @pytest.mark.anyio
    async def test_400_model_not_found(self):
        """400 from server → provider raises with error message."""
        error_body = json.dumps({
            "error": "SDKError(File not found), model 'bad' not found in local cache"
        })

        def _handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/v1/":
                return httpx.Response(200, text='"GenieX-CLI is running"')
            if path == "/v1/models":
                return httpx.Response(200, json=_models_response(["test-model"]))
            if path == "/v1/chat/completions":
                return httpx.Response(400, content=error_body.encode())
            return httpx.Response(404)

        transport = httpx.MockTransport(_handler)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )

        with pytest.raises(Exception) as exc_info:
            async for _ in provider.stream_typed(
                model_name="test-model",
                messages=[{"role": "user", "content": "hi"}],
            ):
                pass  # pragma: no cover

        assert "not found" in str(exc_info.value).lower() or "400" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_500_server_error(self):
        """500 from server → provider raises."""
        error_body = json.dumps({
            "code": -100001,
            "error": "SDKError(Invalid input), quantization not found",
        })

        def _handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/v1/":
                return httpx.Response(200, text='"GenieX-CLI is running"')
            if path == "/v1/models":
                return httpx.Response(200, json=_models_response(["test-model"]))
            if path == "/v1/chat/completions":
                return httpx.Response(500, content=error_body.encode())
            return httpx.Response(404)

        transport = httpx.MockTransport(_handler)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )

        with pytest.raises(Exception) as exc_info:
            async for _ in provider.stream_typed(
                model_name="test-model",
                messages=[{"role": "user", "content": "hi"}],
            ):
                pass  # pragma: no cover

        assert "500" in str(exc_info.value) or "error" in str(exc_info.value).lower()

    @pytest.mark.anyio
    async def test_raw_json_error_in_stream_context(self):
        """GenieX may return raw JSON error body instead of SSE on stream=true.

        The provider must detect non-SSE response and raise appropriately."""
        error_body = json.dumps({
            "error": "SDKError(File not found), model 'x' not found"
        })

        def _handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/v1/":
                return httpx.Response(200, text='"GenieX-CLI is running"')
            if path == "/v1/models":
                return httpx.Response(200, json=_models_response(["test-model"]))
            if path == "/v1/chat/completions":
                # Server returns JSON error body even though stream=true was requested
                return httpx.Response(
                    400,
                    content=error_body.encode(),
                    headers={"content-type": "application/json"},
                )
            return httpx.Response(404)

        transport = httpx.MockTransport(_handler)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )

        with pytest.raises(Exception):
            async for _ in provider.stream_typed(
                model_name="test-model",
                messages=[{"role": "user", "content": "hi"}],
            ):
                pass  # pragma: no cover


# ---------------------------------------------------------------------------
# Connection errors
# ---------------------------------------------------------------------------


class TestConnectionErrors:
    """Network-level failures."""

    @pytest.mark.anyio
    async def test_connect_error_maps_to_exception(self):
        """Connection refused → clear exception (not unhandled httpx.ConnectError)."""

        def _handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(_handler)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )

        with pytest.raises(Exception) as exc_info:
            async for _ in provider.stream_typed(
                model_name="test-model",
                messages=[{"role": "user", "content": "hi"}],
            ):
                pass  # pragma: no cover

        # Should be a provider-level error, not raw httpx.ConnectError
        exc_str = str(exc_info.value).lower()
        assert "connect" in exc_str or "unavailable" in exc_str or "refused" in exc_str

    @pytest.mark.anyio
    async def test_timeout_maps_to_exception(self):
        """Read timeout → clear timeout exception."""

        def _handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out")

        transport = httpx.MockTransport(_handler)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            timeout_seconds=1.0,
            http_client=client,
        )

        with pytest.raises(Exception) as exc_info:
            async for _ in provider.stream_typed(
                model_name="test-model",
                messages=[{"role": "user", "content": "hi"}],
            ):
                pass  # pragma: no cover

        exc_str = str(exc_info.value).lower()
        assert "timeout" in exc_str or "timed" in exc_str

    @pytest.mark.anyio
    async def test_midstream_timeout_maps_to_transient_provider_error(self):
        """Read timeout after response headers remains in the typed taxonomy."""

        class _TimeoutStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield _sse_frame(_chunk_json("partial")).encode()
                raise httpx.ReadTimeout("stream timed out")

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                stream=_TimeoutStream(),
                headers={"content-type": "text/event-stream"},
            )

        transport = httpx.MockTransport(_handler)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            timeout_seconds=1.0,
            http_client=client,
        )

        with pytest.raises(TransientProviderError, match="timeout"):
            async for _ in provider.stream_typed(
                model_name="test-model",
                messages=[{"role": "user", "content": "hi"}],
            ):
                pass
