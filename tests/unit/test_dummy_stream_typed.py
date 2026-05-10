"""Tests for ``DummyProvider``'s v2 typed contract (Phase 3 step 39).

DummyProvider has no native handles, so warm_up / aclose use the ABC
defaults (no-op). Capabilities reflect the trivial implementation:
streaming yes, but no tools / no thinking / no cancel.

Synthesis §4 Phase 3 step 39.
"""
from __future__ import annotations

import pytest

from tether.providers.dummy.provider import DummyProvider
from tether.providers.types import ProviderCapabilities, ProviderText


# ---------------------------------------------------------------------------
# kind / capabilities introspection
# ---------------------------------------------------------------------------


def test_dummy_kind_property():
    """DummyProvider.kind is the canonical 'dummy' identifier."""
    provider = DummyProvider()
    assert provider.kind == "dummy"


def test_dummy_capabilities():
    """DummyProvider.capabilities reflects the trivial implementation:
    streaming yes; no tools (native or marker); no thinking; no cancel;
    multi-model yes (no per-instance state); no warm-up needed."""
    provider = DummyProvider()
    caps = provider.capabilities

    assert isinstance(caps, ProviderCapabilities)
    assert caps.streaming is True
    assert caps.tools_native is False
    assert caps.tools_marker is False
    assert caps.thinking_channel is False
    assert caps.cancel_inflight is False
    assert caps.multi_model is True
    assert caps.warm_up_required is False


# ---------------------------------------------------------------------------
# stream_typed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dummy_stream_typed_yields_provider_text():
    """DummyProvider's stream_typed wraps each legacy chunk in a
    ProviderText event. The legacy stream yields three echoed-prompt
    strings, so we expect three ProviderText events."""
    provider = DummyProvider()

    events = []
    async for ev in provider.stream_typed(
        model_name="dummy-model-1",
        messages=[{"role": "user", "content": "hello"}],
    ):
        events.append(ev)

    assert len(events) == 3
    assert all(isinstance(ev, ProviderText) for ev in events)
    assert all(ev.type == "text" for ev in events)
    # DummyProvider.stream() yields ``f"{prompt}-{i}"`` for i in 0..2.
    assert [ev.text for ev in events] == ["hello-0", "hello-1", "hello-2"]


# ---------------------------------------------------------------------------
# warm_up / aclose: ABC defaults (no-op)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dummy_warm_up_default_noop():
    """DummyProvider doesn't override warm_up — the ABC default is a
    no-op coroutine that returns None without error."""
    provider = DummyProvider()
    result = await provider.warm_up("any-model")
    assert result is None


@pytest.mark.asyncio
async def test_dummy_aclose_default_noop():
    """DummyProvider doesn't override aclose — the ABC default is a
    no-op coroutine that returns None without error."""
    provider = DummyProvider()
    result = await provider.aclose()
    assert result is None
