"""Tests for the Provider v2 types module.

Synthesis §4 Phase 3 step 39, §11.3 R21, §6 bug #12.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from pydantic import TypeAdapter, ValidationError

from tether.providers.types import (
    ModelInfo,
    ProviderCapabilities,
    ProviderEvent,
    ProviderText,
    ProviderThink,
    ProviderToolCall,
)

# ---------------------------------------------------------------------------
# ProviderCapabilities (frozen dataclass)
# ---------------------------------------------------------------------------


def test_provider_capabilities_dataclass_frozen():
    """Frozen dataclass: assignment after construction must raise.

    The watchdog and orchestrator pattern-match on capabilities; if a caller
    could mutate flags at runtime, behavior would diverge from
    introspection.
    """
    caps = ProviderCapabilities()
    with pytest.raises(FrozenInstanceError):
        caps.streaming = True  # type: ignore[misc]


def test_provider_capabilities_default_all_false():
    """Conservative defaults: subclasses must explicitly opt in.

    Synthesis §4 Phase 3 step 39: the ABC default returns
    ``ProviderCapabilities()`` so any concrete provider that forgets to
    override has *no* capabilities — fail closed, not open.
    """
    caps = ProviderCapabilities()
    assert caps.streaming is False
    assert caps.tools_native is False
    assert caps.tools_marker is False
    assert caps.thinking_channel is False
    assert caps.cancel_inflight is False
    assert caps.multi_model is False
    assert caps.warm_up_required is False


def test_provider_capabilities_explicit_set():
    """Explicitly setting flags wires through unchanged."""
    caps = ProviderCapabilities(
        streaming=True,
        tools_native=True,
        tools_marker=True,
        thinking_channel=False,
        cancel_inflight=True,
        multi_model=True,
        warm_up_required=True,
    )
    assert caps.streaming is True
    assert caps.tools_native is True
    assert caps.tools_marker is True
    assert caps.thinking_channel is False
    assert caps.cancel_inflight is True
    assert caps.multi_model is True
    assert caps.warm_up_required is True


# ---------------------------------------------------------------------------
# ModelInfo (frozen dataclass)
# ---------------------------------------------------------------------------


def test_model_info_dataclass():
    """Construction with required fields; ``metadata`` defaults to {}."""
    info = ModelInfo(name="m", kind="mlc")
    assert info.name == "m"
    assert info.kind == "mlc"
    assert info.metadata == {}


def test_model_info_with_metadata():
    """Free-form metadata round-trips."""
    info = ModelInfo(
        name="Qwen3-4B-q4f16_0-MLC",
        kind="mlc",
        metadata={"context_window": 40960, "quantization": "q4f16_0"},
    )
    assert info.metadata["context_window"] == 40960
    assert info.metadata["quantization"] == "q4f16_0"


def test_model_info_frozen():
    """ModelInfo is also frozen — same rationale as ProviderCapabilities."""
    info = ModelInfo(name="m", kind="mlc")
    with pytest.raises(FrozenInstanceError):
        info.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ProviderText / ProviderThink / ProviderToolCall (Pydantic v2)
# ---------------------------------------------------------------------------


def test_provider_text_pydantic():
    """ProviderText carries a ``type='text'`` discriminator and a string
    payload. Frozen + ``extra='forbid'`` per the v2 contract."""
    ev = ProviderText(text="hi")
    assert ev.type == "text"
    assert ev.text == "hi"

    # Frozen: assignment raises.
    with pytest.raises(ValidationError):
        ev.text = "other"  # type: ignore[misc]

    # Extra forbidden: unknown fields raise.
    with pytest.raises(ValidationError):
        ProviderText(text="hi", junk="nope")  # type: ignore[call-arg]


def test_provider_think_pydantic():
    """ProviderThink mirrors ProviderText with ``type='think'``."""
    ev = ProviderThink(text="reasoning")
    assert ev.type == "think"
    assert ev.text == "reasoning"

    with pytest.raises(ValidationError):
        ProviderThink(text="x", extra=1)  # type: ignore[call-arg]


def test_provider_toolcall_pydantic():
    """ProviderToolCall carries id + name + parsed arguments dict."""
    ev = ProviderToolCall(
        tool_call_id="id1",
        name="time",
        arguments={"tz": "UTC"},
    )
    assert ev.type == "tool_call"
    assert ev.tool_call_id == "id1"
    assert ev.name == "time"
    assert ev.arguments == {"tz": "UTC"}

    with pytest.raises(ValidationError):
        ProviderToolCall(
            tool_call_id="id1",
            name="time",
            arguments={"tz": "UTC"},
            extra="nope",  # type: ignore[call-arg]
        )


def test_provider_toolcall_arguments_must_be_dict():
    """``arguments`` is typed ``Dict[str, Any]``; passing a string fails
    validation. The MLC adapter parses JSON strings into dicts before
    constructing the event."""
    with pytest.raises(ValidationError):
        ProviderToolCall(
            tool_call_id="id1",
            name="time",
            arguments='{"tz":"UTC"}',  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# ProviderEvent discriminated union (Pydantic v2 TypeAdapter)
# ---------------------------------------------------------------------------


def test_provider_event_union_text():
    """TypeAdapter dispatches ``type='text'`` -> ProviderText."""
    adapter = TypeAdapter(ProviderEvent)
    ev = adapter.validate_python({"type": "text", "text": "hi"})
    assert isinstance(ev, ProviderText)
    assert ev.text == "hi"


def test_provider_event_union_think():
    adapter = TypeAdapter(ProviderEvent)
    ev = adapter.validate_python({"type": "think", "text": "reasoning"})
    assert isinstance(ev, ProviderThink)
    assert ev.text == "reasoning"


def test_provider_event_union_toolcall():
    """TypeAdapter dispatches ``type='tool_call'`` -> ProviderToolCall."""
    adapter = TypeAdapter(ProviderEvent)
    ev = adapter.validate_python(
        {
            "type": "tool_call",
            "tool_call_id": "x",
            "name": "y",
            "arguments": {},
        }
    )
    assert isinstance(ev, ProviderToolCall)
    assert ev.tool_call_id == "x"
    assert ev.name == "y"
    assert ev.arguments == {}


def test_provider_event_union_invalid_type():
    """Unknown ``type`` discriminator fails validation across the union."""
    adapter = TypeAdapter(ProviderEvent)
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "unknown", "text": "hi"})


def test_provider_event_isinstance_dispatch():
    """Consumers pattern-match via ``isinstance``; verify the union members
    are themselves discriminable when constructed directly (no adapter)."""
    text_ev: ProviderEvent = ProviderText(text="hi")
    think_ev: ProviderEvent = ProviderThink(text="reason")
    tc_ev: ProviderEvent = ProviderToolCall(
        tool_call_id="id1", name="time", arguments={}
    )

    assert isinstance(text_ev, ProviderText)
    assert not isinstance(text_ev, ProviderToolCall)
    assert isinstance(think_ev, ProviderThink)
    assert isinstance(tc_ev, ProviderToolCall)
