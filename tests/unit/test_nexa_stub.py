"""Tests for NexaProvider stub.

Verifies that:
- The stub correctly implements the Provider v2 typed contract (kind, capabilities).
- Every concrete operation raises NotImplementedError (no accidental no-ops).
- NexaProvider is NOT a HardwareLifecycle (NPU teardown deferred to future session).
- The stub is importable from tether.providers.nexa namespace.

Synthesis §11.3 R20: no NexaProvider impl ships in this refactor; the stub
verifies forward compatibility only. Briefing §12.6 (Seam A).
"""
from __future__ import annotations

import pytest

from tether.core.interfaces import ModelProvider
from tether.providers.hw import HardwareLifecycle
from tether.providers.nexa.provider import NexaProvider


@pytest.fixture
def provider() -> NexaProvider:
    return NexaProvider()


# --- Construction & identity ---


def test_nexa_construction(provider: NexaProvider) -> None:
    assert provider is not None


def test_nexa_implements_modelprovider(provider: NexaProvider) -> None:
    assert isinstance(provider, ModelProvider)


def test_nexa_does_not_implement_hardware_lifecycle(provider: NexaProvider) -> None:
    assert isinstance(provider, HardwareLifecycle) is False


# --- Kind & capabilities ---


def test_nexa_kind(provider: NexaProvider) -> None:
    assert provider.kind == "nexa"


def test_nexa_capabilities(provider: NexaProvider) -> None:
    caps = provider.capabilities
    assert caps.streaming is True
    assert caps.tools_native is True
    assert caps.tools_marker is False
    assert caps.thinking_channel is False
    assert caps.cancel_inflight is True
    assert caps.multi_model is True
    assert caps.warm_up_required is False


# --- Legacy v1 contract raises ---


@pytest.mark.asyncio
async def test_nexa_stream_raises(provider: NexaProvider) -> None:
    with pytest.raises(NotImplementedError):
        await provider.stream("model", [], None).__anext__()


def test_nexa_list_models_raises(provider: NexaProvider) -> None:
    with pytest.raises(NotImplementedError):
        provider.list_models()


def test_nexa_unload_model_raises(provider: NexaProvider) -> None:
    with pytest.raises(NotImplementedError):
        provider.unload_model("model")


def test_nexa_get_context_window_raises(provider: NexaProvider) -> None:
    with pytest.raises(NotImplementedError):
        provider.get_context_window("model")


# --- v2 typed contract raises ---


@pytest.mark.asyncio
async def test_nexa_stream_typed_raises(provider: NexaProvider) -> None:
    with pytest.raises(NotImplementedError):
        await provider.stream_typed(model_name="model", messages=[]).__anext__()


@pytest.mark.asyncio
async def test_nexa_warm_up_raises(provider: NexaProvider) -> None:
    with pytest.raises(NotImplementedError):
        await provider.warm_up("model")


@pytest.mark.asyncio
async def test_nexa_aclose_raises(provider: NexaProvider) -> None:
    with pytest.raises(NotImplementedError):
        await provider.aclose()
