"""Tests for ``MLCProvider``'s v2 typed contract (Phase 3 step 39).

We never construct a real ``AsyncMLCEngine`` here — the v2 wrapper just
adapts whatever the legacy :meth:`stream` yields, so monkey-patching
``stream`` is enough to cover the :class:`ProviderText` / :class:`ProviderToolCall`
dispatch logic.

Synthesis §4 Phase 3 step 39, §6 bug #12.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from tether_service.providers.mlc.provider import MLCProvider
from tether_service.providers.types import (
    ProviderCapabilities,
    ProviderText,
    ProviderToolCall,
)


def _make_provider(tmp_path: Path) -> MLCProvider:
    """Construct an MLCProvider rooted at ``tmp_path``.

    Mirrors the helper in ``test_mlc_provider_hwlifecycle.py``: the
    constructor doesn't load any engines, so as long as we don't trigger
    ``_ensure_engine`` / ``_get_engine`` we don't need real MLC libs.
    """
    dist_root = tmp_path / "dist"
    dist_root.mkdir()
    (dist_root / "libs").mkdir()
    return MLCProvider(dist_root=str(dist_root), device="auto", max_tokens=1024)


# ---------------------------------------------------------------------------
# kind / capabilities introspection
# ---------------------------------------------------------------------------


def test_mlc_kind_property(tmp_path: Path):
    """MLCProvider.kind is the canonical 'mlc' identifier."""
    provider = _make_provider(tmp_path)
    assert provider.kind == "mlc"


def test_mlc_capabilities(tmp_path: Path):
    """MLCProvider.capabilities reflects the runtime: streams, tools both
    native + marker, no thinking channel, cancellable, multi-model, warm-up
    required (engine cold start ~seconds)."""
    provider = _make_provider(tmp_path)
    caps = provider.capabilities

    assert isinstance(caps, ProviderCapabilities)
    assert caps.streaming is True
    assert caps.tools_native is True
    assert caps.tools_marker is True
    assert caps.thinking_channel is False
    assert caps.cancel_inflight is True
    assert caps.multi_model is True
    assert caps.warm_up_required is True


# ---------------------------------------------------------------------------
# stream_typed: text path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mlc_stream_typed_yields_provider_text(tmp_path: Path, monkeypatch):
    """When the legacy stream() yields strings, stream_typed wraps each in
    a ProviderText event."""
    provider = _make_provider(tmp_path)

    async def fake_stream(
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Any, None]:
        for s in ("hello", "world"):
            yield s

    monkeypatch.setattr(provider, "stream", fake_stream)

    events = []
    async for ev in provider.stream_typed(
        model_name="any-model", messages=[{"role": "user", "content": "hi"}]
    ):
        events.append(ev)

    assert len(events) == 2
    assert all(isinstance(ev, ProviderText) for ev in events)
    assert [ev.text for ev in events] == ["hello", "world"]
    assert all(ev.type == "text" for ev in events)


# ---------------------------------------------------------------------------
# stream_typed: native tool_calls path (synthesis §6 bug #12)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mlc_stream_typed_yields_provider_toolcall(tmp_path: Path, monkeypatch):
    """When the legacy stream() yields a list-of-dicts (native MLC
    tool_calls), stream_typed converts each dict to a ProviderToolCall.

    Synthesis §6 bug #12: the legacy orchestrator silently dropped these
    because it only consumed text chunks. The v2 typed path emits them.
    """
    provider = _make_provider(tmp_path)

    # Each item is the shape ``delta.tool_calls[i].model_dump()`` returns
    # for an OpenAI-style tool call: id + type + function{name, arguments}.
    tool_calls_chunk = [
        {
            "id": "call_abc",
            "type": "function",
            "function": {
                "name": "get_time",
                "arguments": {"tz": "UTC"},
            },
        },
        {
            "id": "call_def",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": {"city": "Paris"},
            },
        },
    ]

    async def fake_stream(
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Any, None]:
        yield "thinking..."
        yield tool_calls_chunk

    monkeypatch.setattr(provider, "stream", fake_stream)

    events = []
    async for ev in provider.stream_typed(
        model_name="any-model", messages=[{"role": "user", "content": "hi"}]
    ):
        events.append(ev)

    # 1 ProviderText + 2 ProviderToolCall
    assert len(events) == 3
    assert isinstance(events[0], ProviderText)
    assert events[0].text == "thinking..."

    assert isinstance(events[1], ProviderToolCall)
    assert events[1].tool_call_id == "call_abc"
    assert events[1].name == "get_time"
    assert events[1].arguments == {"tz": "UTC"}
    assert events[1].type == "tool_call"

    assert isinstance(events[2], ProviderToolCall)
    assert events[2].tool_call_id == "call_def"
    assert events[2].name == "get_weather"
    assert events[2].arguments == {"city": "Paris"}


@pytest.mark.asyncio
async def test_mlc_stream_typed_parses_string_arguments(
    tmp_path: Path, monkeypatch
):
    """MLC sometimes returns ``arguments`` as a JSON string; the typed
    adapter must parse it into a dict so consumers don't re-parse."""
    provider = _make_provider(tmp_path)

    tool_calls_chunk = [
        {
            "id": "call_str",
            "type": "function",
            "function": {
                "name": "get_time",
                "arguments": '{"tz": "America/Los_Angeles", "fmt": "iso"}',
            },
        }
    ]

    async def fake_stream(
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Any, None]:
        yield tool_calls_chunk

    monkeypatch.setattr(provider, "stream", fake_stream)

    events = []
    async for ev in provider.stream_typed(
        model_name="any-model", messages=[{"role": "user", "content": "hi"}]
    ):
        events.append(ev)

    assert len(events) == 1
    assert isinstance(events[0], ProviderToolCall)
    assert events[0].arguments == {
        "tz": "America/Los_Angeles",
        "fmt": "iso",
    }


@pytest.mark.asyncio
async def test_mlc_stream_typed_handles_invalid_json_arguments(
    tmp_path: Path, monkeypatch
):
    """Malformed JSON string in ``arguments`` is preserved under ``_raw``
    so consumers can debug it instead of crashing the whole stream."""
    provider = _make_provider(tmp_path)

    tool_calls_chunk = [
        {
            "id": "call_bad",
            "type": "function",
            "function": {
                "name": "get_time",
                "arguments": '{"tz": "UTC"',  # missing closing brace
            },
        }
    ]

    async def fake_stream(
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Any, None]:
        yield tool_calls_chunk

    monkeypatch.setattr(provider, "stream", fake_stream)

    events = []
    async for ev in provider.stream_typed(
        model_name="any-model", messages=[{"role": "user", "content": "hi"}]
    ):
        events.append(ev)

    assert len(events) == 1
    assert isinstance(events[0], ProviderToolCall)
    assert events[0].arguments == {"_raw": '{"tz": "UTC"'}


# ---------------------------------------------------------------------------
# warm_up / aclose
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mlc_warm_up_calls_ensure_engine(tmp_path: Path, monkeypatch):
    """warm_up validates the model name and triggers the cold-start path
    via _ensure_engine."""
    provider = _make_provider(tmp_path)

    ensure_mock = AsyncMock(return_value=object())
    monkeypatch.setattr(provider, "_ensure_engine", ensure_mock)

    await provider.warm_up("Qwen3-4B-q4f16_0-MLC")

    ensure_mock.assert_awaited_once_with("Qwen3-4B-q4f16_0-MLC")


@pytest.mark.asyncio
async def test_mlc_warm_up_validates_model_name(tmp_path: Path, monkeypatch):
    """warm_up enforces the path-traversal guard (security R-pathtraversal)
    before reaching _ensure_engine."""
    provider = _make_provider(tmp_path)

    ensure_mock = AsyncMock(return_value=object())
    monkeypatch.setattr(provider, "_ensure_engine", ensure_mock)

    with pytest.raises(ValueError):
        await provider.warm_up("../escape")

    ensure_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_mlc_aclose_calls_shutdown_all(tmp_path: Path, monkeypatch):
    """aclose delegates to the existing shutdown_all() (Phase 3 step 38
    parallel teardown)."""
    provider = _make_provider(tmp_path)

    shutdown_mock = MagicMock()
    monkeypatch.setattr(provider, "shutdown_all", shutdown_mock)

    await provider.aclose()

    shutdown_mock.assert_called_once_with()
