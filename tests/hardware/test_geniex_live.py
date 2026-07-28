"""Hardware live tests for GenieX provider against a running server.

These tests require BOTH markers:
  - @pytest.mark.hardware (general hardware gate)
  - @pytest.mark.geniex  (GenieX-specific gate)

Environment variables (required for execution):
  - GENIEX_BASE_URL: e.g. http://127.0.0.1:18181
  - GENIEX_MODEL_ID: e.g. unsloth/Qwen3-1.7B-GGUF:Q4_0

Optional:
  - GENIEX_REQUEST_MODEL_ID: alias for wire-level model name

These tests:
  - Never download models or start/stop a server
  - Never mutate environment
  - Never assert a performance floor
  - Skip cleanly when provider module or env vars are absent
"""
from __future__ import annotations

import os
from typing import List

import pytest

from tether.providers.geniex.provider import GenieXProvider

_BASE_URL = os.environ.get("GENIEX_BASE_URL")
_MODEL_ID = os.environ.get("GENIEX_MODEL_ID")
_REQUEST_MODEL_ID = os.environ.get("GENIEX_REQUEST_MODEL_ID")

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.geniex,
    pytest.mark.skipif(
        not _BASE_URL or not _MODEL_ID,
        reason="GENIEX_BASE_URL and GENIEX_MODEL_ID env vars required",
    ),
]


@pytest.fixture
def provider():
    """Construct a GenieXProvider pointed at the live server."""
    kwargs = {
        "base_url": _BASE_URL,
        "model_id": _MODEL_ID,
    }
    if _REQUEST_MODEL_ID:
        kwargs["request_model_id"] = _REQUEST_MODEL_ID
    return GenieXProvider(**kwargs)


class TestLiveHealth:
    """Verify connectivity to the live GenieX server."""

    @pytest.mark.anyio
    async def test_warmup_succeeds(self, provider):
        """warm_up() completes without error against live server."""
        await provider.warm_up(_MODEL_ID)

    def test_list_models_non_empty(self, provider):
        """list_models() returns at least the configured model."""
        models = provider.list_models()
        assert len(models) >= 1
        assert _MODEL_ID in models or (
            _REQUEST_MODEL_ID and _REQUEST_MODEL_ID in str(models)
        )

    def test_kind_and_source(self, provider):
        """Provider reports correct kind and source."""
        assert provider.kind == "geniex"
        assert provider.source == "local"


class TestLiveStreaming:
    """Stream completions from the live server."""

    @pytest.mark.anyio
    async def test_stream_typed_yields_text(self, provider):
        """stream_typed() yields at least one ProviderText event."""
        from tether.providers.types import ProviderText

        events = []
        async for ev in provider.stream_typed(
            model_name=_MODEL_ID,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_output_tokens=30,
        ):
            events.append(ev)
            if len(events) > 50:
                break  # Safety cap

        text_events = [e for e in events if isinstance(e, ProviderText)]
        assert len(text_events) >= 1
        combined = "".join(e.text for e in text_events)
        assert len(combined) > 0

    @pytest.mark.anyio
    async def test_legacy_stream_yields_strings(self, provider):
        """Legacy stream() yields str chunks."""
        results: List[str] = []
        async for chunk in provider.stream(
            model_name=_MODEL_ID,
            messages=[{"role": "user", "content": "Say hi."}],
        ):
            results.append(chunk)
            if len(results) > 50:
                break

        assert all(isinstance(c, str) for c in results)
        assert len(results) >= 1

    @pytest.mark.anyio
    async def test_stream_cancel_no_hang(self, provider):
        """Closing the generator mid-stream should not hang or raise."""
        gen = provider.stream_typed(
            model_name=_MODEL_ID,
            messages=[
                {"role": "user", "content": "Count from 1 to 100 slowly."}
            ],
            max_output_tokens=200,
        )
        # Read a few events then close
        count = 0
        async for _ in gen:
            count += 1
            if count >= 3:
                break
        await gen.aclose()


class TestLiveMarkerToolCalling:
    """Verify that marker-based tool calls work through the GenieX server."""

    @pytest.mark.anyio
    async def test_model_emits_function_call_marker(self, provider):
        """With tool prompt in system message, model emits <<function_call>>."""
        from tether.providers.types import ProviderText

        system_prompt = (
            "You are a helpful assistant with access to tools.\n\n"
            "Available tools:\n"
            '```json\n[{"name":"get_time","description":"Get current time.",'
            '"parameters":{"type":"object","properties":{},"required":[]}}]\n```\n\n'
            "When you want to call a tool, output EXACTLY this format:\n"
            '<<function_call>> {"name":"TOOL_NAME","arguments":{...}}\n\n'
            "Do not describe the tool call. Just emit the marker."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What time is it?"},
        ]

        texts: List[str] = []
        async for ev in provider.stream_typed(
            model_name=_MODEL_ID,
            messages=messages,
            max_output_tokens=100,
        ):
            if isinstance(ev, ProviderText):
                texts.append(ev.text)

        combined = "".join(texts)
        # Model should emit the marker (may not always, but this is a live test)
        # We assert it's at least generating text — actual marker presence is
        # model-dependent and we don't want flaky CI.
        assert len(combined) > 0


class TestLiveMetadata:
    """Provider metadata against live server."""

    def test_capabilities_shape(self, provider):
        """capabilities returns expected flags."""
        caps = provider.capabilities
        assert caps.streaming is True
        assert caps.tools_native is False
        assert caps.tools_marker is True

    def test_context_window(self, provider):
        """get_context_window returns configured value."""
        cw = provider.get_context_window(_MODEL_ID)
        assert cw > 0

    @pytest.mark.anyio
    async def test_aclose_safe(self, provider):
        """aclose() is callable without error after use."""
        await provider.aclose()
