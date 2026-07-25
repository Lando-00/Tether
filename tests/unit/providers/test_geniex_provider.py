"""Unit tests for GenieXProvider ModelProvider interface compliance.

Tests cover: request body construction (no tools, enable_think top-level),
model alias (request_model_id), static model list, warmup endpoints,
capabilities, metadata, and provider lifecycle.

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

# Skip cleanly if the provider module is not yet merged.
_mod = pytest.importorskip(
    "tether.providers.geniex.provider", reason="GenieXProvider not merged yet"
)
GenieXProvider = _mod.GenieXProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse_frame(content: str) -> str:
    chunk = json.dumps({
        "id": "",
        "choices": [{
            "delta": {"content": content, "role": "assistant", "tool_calls": None},
            "finish_reason": "",
            "index": 0,
        }],
        "created": 0,
        "model": "",
        "object": "chat.completion.chunk",
    })
    return f"data:{chunk}\n\n"


def _sse_done() -> str:
    return "data:[DONE]\n\n"


def _models_response(ids: List[str]) -> dict:
    return {
        "data": [{"id": mid, "created": 0, "object": "model", "owned_by": "test"} for mid in ids],
        "object": "list",
    }


def _make_transport(
    *,
    model_ids: List[str] | None = None,
    health_ok: bool = True,
    capture_request: dict | None = None,
    sse_body: str | None = None,
):
    """Build a mock transport that captures chat/completions requests."""
    model_list = model_ids or ["test-model"]
    default_sse = _sse_frame("ok") + _sse_done()

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
            if capture_request is not None:
                capture_request["body"] = json.loads(request.content)
                capture_request["headers"] = dict(request.headers)
            body = sse_body or default_sse
            return httpx.Response(
                200,
                content=body.encode(),
                headers={"content-type": "text/event-stream"},
            )

        return httpx.Response(404, text="not found")

    return httpx.MockTransport(_handler)


async def _drain(provider, model_name="test-model", messages=None):
    """Consume a stream_typed generator fully."""
    msgs = messages or [{"role": "user", "content": "hi"}]
    events = []
    async for ev in provider.stream_typed(
        model_name=model_name,
        messages=msgs,
    ):
        events.append(ev)
    return events


# ---------------------------------------------------------------------------
# Request body construction
# ---------------------------------------------------------------------------


class TestRequestBody:
    """Verify the exact request body sent to GenieX."""

    @pytest.mark.anyio
    async def test_no_tools_or_tool_choice_in_request(self):
        """Provider must NOT forward tools/tool_choice to GenieX."""
        captured: Dict[str, Any] = {}
        transport = _make_transport(capture_request=captured)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )

        await _drain(provider)

        body = captured["body"]
        assert "tools" not in body
        assert "tool_choice" not in body
        assert "functions" not in body

    @pytest.mark.anyio
    async def test_tools_arg_in_stream_typed_not_forwarded(self):
        """Even if tools= is passed to stream_typed, provider must strip them."""
        captured: Dict[str, Any] = {}
        transport = _make_transport(capture_request=captured)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )

        async for _ in provider.stream_typed(
            model_name="test-model",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "web_search"}}],
        ):
            pass

        body = captured["body"]
        assert "tools" not in body
        assert "tool_choice" not in body

    @pytest.mark.anyio
    async def test_enable_think_top_level(self):
        """enable_think=false must be top-level, not nested."""
        captured: Dict[str, Any] = {}
        transport = _make_transport(capture_request=captured)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )

        await _drain(provider)

        body = captured["body"]
        assert "enable_think" in body
        assert body["enable_think"] is False
        # Must NOT be nested in extra_body or any sub-object
        assert "extra_body" not in body

    @pytest.mark.anyio
    async def test_stream_true_in_request(self):
        """Provider always requests stream=true."""
        captured: Dict[str, Any] = {}
        transport = _make_transport(capture_request=captured)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )

        await _drain(provider)
        assert captured["body"]["stream"] is True

    @pytest.mark.anyio
    async def test_temperature_and_max_tokens_forwarded(self):
        """Constructor temperature and max_tokens appear in request."""
        captured: Dict[str, Any] = {}
        transport = _make_transport(capture_request=captured)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            temperature=0.8,
            max_tokens=512,
            http_client=client,
        )

        await _drain(provider)

        body = captured["body"]
        assert body["temperature"] == 0.8
        assert body["max_tokens"] == 512

    @pytest.mark.anyio
    async def test_messages_forwarded_correctly(self):
        """Messages array is forwarded without mutation."""
        captured: Dict[str, Any] = {}
        transport = _make_transport(capture_request=captured)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        await _drain(provider, messages=messages)

        body = captured["body"]
        assert body["messages"] == messages


# ---------------------------------------------------------------------------
# Model alias (request_model_id)
# ---------------------------------------------------------------------------


class TestModelAlias:
    """request_model_id allows using a different ID on the wire."""

    @pytest.mark.anyio
    async def test_alias_used_in_request_body(self):
        """When request_model_id is set, that ID appears in the request."""
        captured: Dict[str, Any] = {}
        transport = _make_transport(capture_request=captured)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="display-name",
            request_model_id="unsloth/Qwen3-1.7B-GGUF:Q4_0",
            http_client=client,
        )

        await _drain(provider, model_name="display-name")

        body = captured["body"]
        assert body["model"] == "unsloth/Qwen3-1.7B-GGUF:Q4_0"

    @pytest.mark.anyio
    async def test_no_alias_uses_model_id(self):
        """Without request_model_id, model_id is used on the wire."""
        captured: Dict[str, Any] = {}
        transport = _make_transport(capture_request=captured)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="unsloth/Qwen3-1.7B-GGUF:Q4_0",
            http_client=client,
        )

        await _drain(provider, model_name="unsloth/Qwen3-1.7B-GGUF:Q4_0")

        body = captured["body"]
        assert body["model"] == "unsloth/Qwen3-1.7B-GGUF:Q4_0"


# ---------------------------------------------------------------------------
# Static model list
# ---------------------------------------------------------------------------


class TestStaticModelList:
    """list_models returns statically configured model(s)."""

    def test_list_models_returns_configured_model(self):
        transport = _make_transport()
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="unsloth/Qwen3-1.7B-GGUF:Q4_0",
            http_client=client,
        )

        models = provider.list_models()
        assert "unsloth/Qwen3-1.7B-GGUF:Q4_0" in models

    def test_list_models_with_alias_shows_display_name(self):
        transport = _make_transport()
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="qwen3-npu",
            request_model_id="unsloth/Qwen3-1.7B-GGUF:Q4_0",
            http_client=client,
        )

        models = provider.list_models()
        assert "qwen3-npu" in models


# ---------------------------------------------------------------------------
# Capabilities and metadata
# ---------------------------------------------------------------------------


class TestCapabilities:
    """Provider capability introspection."""

    def test_kind_is_geniex(self):
        transport = _make_transport()
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )
        assert provider.kind == "geniex"

    def test_source_is_remote(self):
        """GenieX is an external HTTP server — source must be 'remote'."""
        transport = _make_transport()
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )
        assert provider.source == "remote"

    def test_capabilities_marker_only(self):
        """GenieX supports marker-based tool calling, NOT native."""
        from tether.providers.types import ProviderCapabilities

        transport = _make_transport()
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )

        caps = provider.capabilities
        assert isinstance(caps, ProviderCapabilities)
        assert caps.streaming is True
        assert caps.tools_native is False
        assert caps.tools_marker is True
        assert caps.warm_up_required is True
        assert caps.multi_model is False

    def test_context_window_returns_configured(self):
        transport = _make_transport()
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            context_window=8192,
            http_client=client,
        )
        assert provider.get_context_window("test-model") == 8192

    def test_default_context_window(self):
        transport = _make_transport()
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )
        assert provider.get_context_window("test-model") == 4096

    def test_unload_model_returns_false(self):
        """GenieX is externally managed — unload is a no-op."""
        transport = _make_transport()
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )
        assert provider.unload_model("test-model") is False

    def test_list_model_info_shape(self):
        """list_model_info returns ModelDetails with correct fields."""
        from tether.providers.types import ModelDetails

        transport = _make_transport()
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            context_window=4096,
            http_client=client,
        )

        infos = provider.list_model_info()
        assert len(infos) >= 1
        info = infos[0]
        assert isinstance(info, ModelDetails)
        assert info.id == "test-model"
        assert info.provider_kind == "geniex"
        assert info.source == "remote"
        assert info.context_window == 4096
        assert info.supports_thinking is False


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


class TestWarmup:
    """warm_up() hits the health endpoint."""

    @pytest.mark.anyio
    async def test_warmup_hits_health_endpoint(self):
        """warm_up should verify connectivity to GenieX."""
        hit_endpoints: List[str] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            hit_endpoints.append(request.url.path)
            if request.url.path == "/v1/":
                return httpx.Response(200, text='"GenieX-CLI is running"')
            if request.url.path == "/v1/models":
                return httpx.Response(200, json=_models_response(["test-model"]))
            return httpx.Response(404)

        transport = httpx.MockTransport(_handler)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )

        await provider.warm_up("test-model")

        # Should have hit at least the health endpoint
        assert "/v1/" in hit_endpoints or "/v1/models" in hit_endpoints

    @pytest.mark.anyio
    async def test_warmup_server_down_raises(self):
        """warm_up with unavailable server should raise."""

        def _handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(_handler)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )

        with pytest.raises(Exception):
            await provider.warm_up("test-model")


# ---------------------------------------------------------------------------
# Legacy stream() interface
# ---------------------------------------------------------------------------


class TestLegacyStream:
    """The legacy stream() method should also work for backward compat."""

    @pytest.mark.anyio
    async def test_legacy_stream_yields_strings(self):
        """Legacy stream() yields str chunks."""
        chunks = [_sse_frame("hello") + _sse_done()]
        transport = _make_transport(sse_body="".join(chunks))
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )

        results = []
        async for chunk in provider.stream(
            model_name="test-model",
            messages=[{"role": "user", "content": "hi"}],
        ):
            results.append(chunk)

        # Legacy stream yields strings
        assert all(isinstance(c, str) for c in results)
        assert any("hello" in c for c in results)


# ---------------------------------------------------------------------------
# URL validator
# ---------------------------------------------------------------------------


class TestURLValidator:
    """url_validator is called on construction to validate base_url."""

    def test_url_validator_called_on_init(self):
        """If url_validator is provided, it's called with base_url."""
        validated_urls: List[str] = []

        def validator(url: str) -> None:
            validated_urls.append(url)

        transport = _make_transport()
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            url_validator=validator,
            http_client=client,
        )

        assert "http://test" in validated_urls

    def test_url_validator_rejects_bad_url(self):
        """If url_validator raises, construction fails."""

        def validator(url: str) -> None:
            raise ValueError(f"Blocked: {url}")

        transport = _make_transport()
        client = httpx.AsyncClient(transport=transport, base_url="http://test")

        with pytest.raises(ValueError, match="Blocked"):
            GenieXProvider(
                base_url="http://evil.com",
                model_id="test-model",
                url_validator=validator,
                http_client=client,
            )


# ---------------------------------------------------------------------------
# Generator closure / cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    """Stream can be closed mid-flight without hanging."""

    @pytest.mark.anyio
    async def test_generator_closure_clean(self):
        """Closing the async generator mid-stream should not raise."""
        chunks = [
            _sse_frame("tok1") + _sse_frame("tok2") + _sse_frame("tok3") + _sse_done()
        ]
        transport = _make_transport(sse_body="".join(chunks))
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )

        gen = provider.stream_typed(
            model_name="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
        # Read one event then close
        first = await gen.__anext__()
        assert first is not None
        await gen.aclose()  # Should not raise


# ---------------------------------------------------------------------------
# aclose lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Provider aclose shuts down client if owned."""

    @pytest.mark.anyio
    async def test_aclose_is_safe_to_call(self):
        """aclose() should be callable without error."""
        transport = _make_transport()
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        provider = GenieXProvider(
            base_url="http://test",
            model_id="test-model",
            http_client=client,
        )
        # Should not raise
        await provider.aclose()
