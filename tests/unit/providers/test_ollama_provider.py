"""Comprehensive unit tests for OllamaProvider (ADR-0022).

Tests are written against the contract defined in ADR-0022 §2 + §6 and
ADR-0022-contract-stubs.md.  All tests are gated with pytest.importorskip —
they skip cleanly on this branch (mp-ol-2c-tests) and activate automatically
once the Phase-2.A / 2.B implementation modules land at the Phase-3.1 merge.
"""
from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Phase-3.1 gate: skip entire file until provider + client_base exist.
# Remove these importorskip calls after the 2.A/2.B merge.
# ---------------------------------------------------------------------------
pytest.importorskip("tether.providers.ollama.provider")
pytest.importorskip("tether.providers.ollama.client")

from tether.providers.ollama.client import OllamaStreamEvent  # noqa: E402
from tether.providers.ollama.provider import OllamaProvider  # noqa: E402
from tether.providers.types import (  # noqa: E402
    ProviderText,
    ProviderThink,
    ProviderToolCall,
)

# ---------------------------------------------------------------------------
# Construction helper
# ---------------------------------------------------------------------------


def _make_provider(
    base_url: str = "http://localhost:11434",
    api_surface: str = "native",
    models: list[str] | None = None,
    default_model: str | None = None,
    thinking_models: list[str] = (),
    context_windows: dict[str, int] | None = None,
    discover_at_startup: bool = False,
    *,
    fake_client: Any = None,
    **extra_kwargs: Any,
) -> OllamaProvider:
    """Construct OllamaProvider with a no-op url_validator and mocked httpx."""
    if models is None:
        models = ["qwen3:8b"]

    with patch("httpx.AsyncClient") as _mock_http_cls:
        _mock_http_cls.return_value = MagicMock()
        provider = OllamaProvider(
            base_url=base_url,
            api_surface=api_surface,  # type: ignore[arg-type]
            models=models,
            default_model=default_model,
            thinking_models=thinking_models,
            context_windows=context_windows,
            discover_at_startup=discover_at_startup,
            url_validator=lambda u: None,  # bypass assert_safe_url
            **extra_kwargs,
        )

    if fake_client is not None:
        provider._client = fake_client

    return provider


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


def test_construct_with_minimal_args() -> None:
    """OllamaProvider constructs without error given base_url + models."""
    provider = _make_provider(base_url="http://localhost:11434", models=["qwen3:8b"])
    assert provider is not None


def test_construct_validates_url() -> None:
    """base_url is passed through url_validator; a bad URL raises ValueError."""

    def _strict(url: str) -> None:
        if "bogus" in url:
            raise ValueError(f"scheme not allowed: {url}")

    with patch("httpx.AsyncClient"):
        with pytest.raises(ValueError, match="invalid base_url"):
            OllamaProvider(
                base_url="bogus://not-real",
                models=["qwen3:8b"],
                url_validator=_strict,
            )


def test_kind_and_source() -> None:
    """kind == 'ollama'; source == 'local' (LAN/on-premises server)."""
    provider = _make_provider(base_url="http://localhost:11434", api_surface="native")
    assert provider.kind == "ollama"
    assert provider.source == "local"


def test_capabilities() -> None:
    """Capabilities match ADR-0022 §2 decision 9 (thinking_channel=True
    when thinking_models is configured)."""
    provider = _make_provider(
        models=["qwen3:8b"],
        thinking_models=["qwen3:8b"],
    )
    caps = provider.capabilities
    assert caps.streaming is True
    assert caps.tools_native is True
    assert caps.tools_marker is False
    assert caps.thinking_channel is True  # non-empty thinking_models
    assert caps.cancel_inflight is True
    assert caps.multi_model is True
    # ADR-0022 P3.2 wants a connectivity probe at engine startup. Post
    # ADR-0021 that intent is carried by warm_up_on_startup; warm_up_required
    # stays False because the Ollama server holds model state, so warming
    # buys the client nothing on the inference path.
    assert caps.warm_up_required is False
    assert caps.warm_up_on_startup is True


def test_capabilities_thinking_channel_off_when_no_thinking_models() -> None:
    """thinking_channel is False when thinking_models is empty (ADR-0022 §6)."""
    provider = _make_provider(models=["qwen3:8b"], thinking_models=())
    assert provider.capabilities.thinking_channel is False


def testapi_surface_native_default() -> None:
    """Default api_surface='native' selects OllamaNativeClient."""
    provider = _make_provider(api_surface="native")
    assert provider.api_surface == "native"
    # OllamaNativeClient class name should contain "Native"
    client_cls_name = type(provider._client).__name__
    assert "native" in client_cls_name.lower() or "ollama" in client_cls_name.lower()


def testapi_surface_openai_compat() -> None:
    """api_surface='openai_compat' selects OllamaOpenAICompatClient."""
    provider = _make_provider(api_surface="openai_compat")
    assert provider.api_surface == "openai_compat"
    client_cls_name = type(provider._client).__name__
    assert (
        "openai" in client_cls_name.lower()
        or "compat" in client_cls_name.lower()
        or "ollama" in client_cls_name.lower()
    )


def testapi_surface_unknown_raises() -> None:
    """api_surface='bogus' raises ValueError or TypeError at construction time."""
    with patch("httpx.AsyncClient"):
        with pytest.raises((ValueError, TypeError)):
            OllamaProvider(
                base_url="http://localhost:11434",
                models=["qwen3:8b"],
                api_surface="bogus",  # type: ignore[arg-type]
                url_validator=lambda u: None,
            )


def test_default_model_returns_explicit_arg() -> None:
    """default_model='qwen3:14b' overrides the first-in-list."""
    provider = _make_provider(
        models=["qwen3:8b", "qwen3:14b"],
        default_model="qwen3:14b",
    )
    assert provider.default_model() == "qwen3:14b"


def test_default_model_falls_back_to_first_model() -> None:
    """No default_model arg → first item in models list is returned."""
    provider = _make_provider(models=["llama3.1:8b", "qwen3:8b"])
    assert provider.default_model() == "llama3.1:8b"


def test_list_model_info_shape() -> None:
    """Every required field is present; provider_id sentinel is '_unwrapped_';
    supports_reasoning_effort is False; supports_thinking is True only for
    models in thinking_models (per-model, not capabilities-level)."""
    provider = _make_provider(
        models=["qwen3:8b", "llama3.1:8b"],
        thinking_models=["qwen3:8b"],
    )
    infos = provider.list_model_info()
    assert len(infos) == 2
    by_id = {i.id: i for i in infos}

    for info in infos:
        assert info.provider_id == "_unwrapped_", (
            f"Sentinel '_unwrapped_' expected; got {info.provider_id!r}"
        )
        assert info.supports_reasoning_effort is False
        assert info.provider_kind == "ollama"
        assert info.source in ("local", "remote")

    # Per-model thinking support (not capabilities-level aggregate).
    assert by_id["qwen3:8b"].supports_thinking is True
    assert by_id["llama3.1:8b"].supports_thinking is False


def test_list_model_info_context_window_from_config() -> None:
    """context_windows arg populates ModelDetails.context_window."""
    provider = _make_provider(
        models=["qwen3:8b", "llama3.1:8b"],
        context_windows={"qwen3:8b": 40960, "llama3.1:8b": 131072},
    )
    by_id = {i.id: i for i in provider.list_model_info()}
    assert by_id["qwen3:8b"].context_window == 40960
    assert by_id["llama3.1:8b"].context_window == 131072


def test_list_model_info_context_window_default() -> None:
    """When a model has no entry in context_windows, fallback is 4096."""
    provider = _make_provider(models=["unknown-model"])
    assert provider.list_model_info()[0].context_window == 4096


def test_unload_model_returns_false() -> None:
    """Ollama auto-unloads via keep_alive; provider claims no ownership."""
    provider = _make_provider()
    assert provider.unload_model("qwen3:8b") is False


def test_lazy_import_no_httpx_at_module_load() -> None:
    """Importing tether.providers.ollama.provider does not perform any
    network I/O.  The module must be safely importable even when no Ollama
    server is reachable; real connections only happen inside stream_chat().
    """
    # Re-verify the module is importable and exposes OllamaProvider.
    import tether.providers.ollama.provider as p_mod  # noqa: F401

    assert hasattr(p_mod, "OllamaProvider")
    assert isinstance(p_mod.OllamaProvider, type)


# ---------------------------------------------------------------------------
# Async tests (asyncio_mode = "auto" in pyproject.toml)
# ---------------------------------------------------------------------------


async def test_warm_up_calls_version() -> None:
    """warm_up() probes liveness via client.version()."""
    fake_client = AsyncMock()
    fake_client.version.return_value = {"version": "0.6.0"}
    fake_client.list_models.return_value = []

    provider = _make_provider(fake_client=fake_client)
    await provider.warm_up("qwen3:8b")

    fake_client.version.assert_called_once()


async def test_warm_up_unreachable_raises_actionable_runtime_error() -> None:
    """ConnectError at warm_up → RuntimeError mentioning base_url and 'unreachable'."""
    import httpx

    fake_client = AsyncMock()
    fake_client.version.side_effect = httpx.ConnectError("Connection refused")

    provider = _make_provider(
        base_url="http://192.168.1.50:11434",
        fake_client=fake_client,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await provider.warm_up("qwen3:8b")

    msg = str(exc_info.value).lower()
    assert "192.168.1.50" in msg
    assert "unreachable" in msg or "reach" in msg or "server" in msg


async def test_warm_up_discovers_models_when_enabled() -> None:
    """discover_at_startup=True: warm_up merges /api/tags results into _models."""
    fake_client = AsyncMock()
    fake_client.version.return_value = {"version": "0.6.0"}
    fake_client.list_models.return_value = [
        {"name": "qwen3:8b"},
        {"name": "llama3.1:8b"},
        {"name": "newly-pulled-model"},
    ]

    provider = _make_provider(
        models=["qwen3:8b"],
        discover_at_startup=True,
        fake_client=fake_client,
    )
    await provider.warm_up("qwen3:8b")

    models_after = provider.list_models()
    assert "qwen3:8b" in models_after  # original preserved
    assert "llama3.1:8b" in models_after  # discovered
    assert "newly-pulled-model" in models_after  # discovered


async def test_warm_up_no_duplicate_models() -> None:
    """Models already in config are not duplicated even if they appear in /api/tags."""
    fake_client = AsyncMock()
    fake_client.version.return_value = {"version": "0.6.0"}
    # /api/tags echoes back qwen3:8b which is already in config
    fake_client.list_models.return_value = [{"name": "qwen3:8b"}]

    provider = _make_provider(
        models=["qwen3:8b"],
        discover_at_startup=True,
        fake_client=fake_client,
    )
    await provider.warm_up("qwen3:8b")

    assert provider.list_models().count("qwen3:8b") == 1


async def test_aclose_closes_client() -> None:
    """aclose() closes the owned httpx.AsyncClient."""
    mock_http = AsyncMock()

    with patch("httpx.AsyncClient") as mock_http_cls:
        mock_http_cls.return_value = mock_http
        provider = OllamaProvider(
            base_url="http://localhost:11434",
            models=["qwen3:8b"],
            url_validator=lambda u: None,
        )

    await provider.aclose()
    mock_http.aclose.assert_called_once()


async def test_stream_forwards_think_for_thinking_models() -> None:
    """think=True is forwarded to client.stream_chat when model is in thinking_models."""
    received_think: list[bool] = []

    async def _fake_stream(**kwargs: Any) -> AsyncIterator[OllamaStreamEvent]:
        received_think.append(kwargs.get("think", False))
        yield OllamaStreamEvent(kind="text", text="hello")
        yield OllamaStreamEvent(kind="done", stop_reason="stop")

    fake_client = MagicMock()
    fake_client.stream_chat = _fake_stream

    provider = _make_provider(
        models=["qwen3:8b"],
        thinking_models=["qwen3:8b"],
        api_surface="native",
        fake_client=fake_client,
    )

    chunks = [
        c
        async for c in provider.stream(
            "qwen3:8b", [{"role": "user", "content": "Hi"}]
        )
    ]

    assert received_think == [True], (
        "Expected think=True forwarded to stream_chat for a thinking model"
    )
    assert "hello" in chunks


async def test_stream_does_not_forward_think_for_non_thinking_model() -> None:
    """think is False (or omitted) for models NOT in thinking_models."""
    received_think: list[bool] = []

    async def _fake_stream(**kwargs: Any) -> AsyncIterator[OllamaStreamEvent]:
        received_think.append(kwargs.get("think", False))
        yield OllamaStreamEvent(kind="text", text="ok")
        yield OllamaStreamEvent(kind="done", stop_reason="stop")

    fake_client = MagicMock()
    fake_client.stream_chat = _fake_stream

    provider = _make_provider(
        models=["llama3.1:8b"],
        thinking_models=[],  # llama3 not in thinking_models
        fake_client=fake_client,
    )

    _ = [
        c
        async for c in provider.stream(
            "llama3.1:8b", [{"role": "user", "content": "Hi"}]
        )
    ]

    assert received_think == [False]


async def test_stream_ignores_reasoning_effort_kwarg() -> None:
    """Provider accepts reasoning_effort kwarg without error (capability=False),
    does NOT forward it to client.stream_chat."""
    forwarded_kwargs: list[dict] = []

    async def _fake_stream(**kwargs: Any) -> AsyncIterator[OllamaStreamEvent]:
        forwarded_kwargs.append(dict(kwargs))
        yield OllamaStreamEvent(kind="text", text="ok")
        yield OllamaStreamEvent(kind="done", stop_reason="stop")

    fake_client = MagicMock()
    fake_client.stream_chat = _fake_stream

    provider = _make_provider(fake_client=fake_client)

    # Must not raise even though capability advertises supports_reasoning_effort=False
    chunks = [
        c
        async for c in provider.stream(
            "qwen3:8b",
            [{"role": "user", "content": "Hi"}],
            reasoning_effort="high",
        )
    ]

    assert "ok" in chunks
    assert forwarded_kwargs, "stream_chat should have been called"
    assert "reasoning_effort" not in forwarded_kwargs[0], (
        "reasoning_effort must NOT be forwarded to client.stream_chat"
    )


async def test_stream_translates_tool_call_event_to_list_chunk() -> None:
    """OllamaStreamEvent(kind='tool_call') → provider yields [tool_call_dict]
    (list-of-one-dict, MLC-style, consumed by _native_tool_call_from_chunk)."""
    tool_call_dict = {
        "id": "abc123def456",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"location":"London"}'},
    }

    async def _fake_stream(**kwargs: Any) -> AsyncIterator[OllamaStreamEvent]:
        yield OllamaStreamEvent(kind="tool_call", tool_call=tool_call_dict)
        yield OllamaStreamEvent(kind="done", stop_reason="stop")

    fake_client = MagicMock()
    fake_client.stream_chat = _fake_stream

    provider = _make_provider(fake_client=fake_client)

    chunks = [
        c
        async for c in provider.stream(
            "qwen3:8b", [{"role": "user", "content": "What's the weather?"}]
        )
    ]

    tool_chunks = [c for c in chunks if isinstance(c, list)]
    assert len(tool_chunks) == 1, "Expected exactly one list-of-dict tool_call chunk"
    assert tool_chunks[0][0]["id"] == "abc123def456"
    assert tool_chunks[0][0]["function"]["name"] == "get_weather"


async def test_stream_typed_yields_provider_events() -> None:
    """stream_typed yields ProviderText, ProviderThink, ProviderToolCall correctly."""

    async def _fake_stream(**kwargs: Any) -> AsyncIterator[OllamaStreamEvent]:
        yield OllamaStreamEvent(kind="thinking", text="Let me think...")
        yield OllamaStreamEvent(kind="text", text="The answer is 42.")
        yield OllamaStreamEvent(
            kind="tool_call",
            tool_call={
                "id": "tc001",
                "type": "function",
                "function": {"name": "calculator", "arguments": '{"expr":"6*7"}'},
            },
        )
        yield OllamaStreamEvent(kind="done", stop_reason="stop")

    fake_client = MagicMock()
    fake_client.stream_chat = _fake_stream

    provider = _make_provider(
        models=["qwen3:8b"],
        thinking_models=["qwen3:8b"],
        fake_client=fake_client,
    )

    events = [
        e
        async for e in provider.stream_typed(
            model_name="qwen3:8b",
            messages=[{"role": "user", "content": "What is 6*7?"}],
        )
    ]

    think_events = [e for e in events if isinstance(e, ProviderThink)]
    text_events = [e for e in events if isinstance(e, ProviderText)]
    tool_events = [e for e in events if isinstance(e, ProviderToolCall)]

    assert think_events, "Expected at least one ProviderThink event"
    assert think_events[0].text == "Let me think..."

    assert text_events, "Expected at least one ProviderText event"
    assert text_events[0].text == "The answer is 42."

    assert tool_events, "Expected at least one ProviderToolCall event"
    assert tool_events[0].tool_call_id == "tc001"
    assert tool_events[0].name == "calculator"
    assert tool_events[0].arguments == {"expr": "6*7"}


async def test_stream_typed_synthesises_tool_call_id_when_absent() -> None:
    """When OllamaStreamEvent tool_call dict has no id, stream_typed synthesises one."""

    async def _fake_stream(**kwargs: Any) -> AsyncIterator[OllamaStreamEvent]:
        yield OllamaStreamEvent(
            kind="tool_call",
            tool_call={
                # Note: no "id" key
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            },
        )
        yield OllamaStreamEvent(kind="done", stop_reason="stop")

    fake_client = MagicMock()
    fake_client.stream_chat = _fake_stream

    provider = _make_provider(fake_client=fake_client)

    events = [
        e
        async for e in provider.stream_typed(
            model_name="qwen3:8b",
            messages=[{"role": "user", "content": "look up x"}],
        )
    ]

    tool_events = [e for e in events if isinstance(e, ProviderToolCall)]
    assert tool_events
    # Synthesised id must be non-empty and unique-ish
    assert len(tool_events[0].tool_call_id) >= 8
