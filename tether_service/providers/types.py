"""Provider v2 typed contract.

ADDITIVE to the legacy ``stream() / list_models() / unload_model()`` methods
on :class:`ModelProvider`. Phase 5 step 52 will migrate the orchestrator to
consume :func:`stream_typed` directly; until then both contracts coexist.

Synthesis citations:
    - §4 Phase 3 step 39 — Provider v2 contract introduction (this PR).
    - §11.3 R21        — typed events instead of raw chunks.
    - §6 bug #12       — native MLC ``delta.tool_calls`` were silently dropped
      by the orchestrator; the typed path emits :class:`ProviderToolCall`
      events so the bug is fixed in the v2 contract. The legacy ``stream()``
      stays bug-compatible until Phase 5.

Design must support future ``OllamaProvider``, ``NexaProvider`` and
CodeLinaro variants WITHOUT further type changes. Verified by code review:

    OllamaProvider (HTTP /api/chat with ``stream=True``):
        - kind = "ollama"
        - capabilities = ProviderCapabilities(streaming=True,
              tools_native=True,    # Ollama supports tools API
              tools_marker=False,   # marker fallback unnecessary
              thinking_channel=False,
              cancel_inflight=True, # close the HTTP stream
              multi_model=True,     # Ollama serves named models
              warm_up_required=False, # server keeps state
          )
        - warm_up: optional HEAD /api/tags or /api/show
        - aclose:  close the underlying httpx.AsyncClient
        - stream_typed: parse NDJSON; ``message.content`` -> ProviderText;
          ``message.tool_calls`` -> ProviderToolCall; nothing else needed.

    NexaProvider (Snapdragon NPU):
        - kind = "nexa"
        - capabilities differ only in ``warm_up_required=True`` and
          ``multi_model=False`` (single NPU). No new event types needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Union

from pydantic import BaseModel, ConfigDict


# ---------- Provider capabilities (introspection) ----------


@dataclass(frozen=True)
class ProviderCapabilities:
    """Static introspection of what a provider can do.

    Boolean flags; conservative defaults so subclasses must explicitly enable
    each capability. The orchestrator (Phase 5+) will branch on these to
    decide whether to:

    - send ``tools=`` to the model (``tools_native`` or ``tools_marker``)
    - extract ``<think>...</think>`` blocks (``thinking_channel``)
    - cancel in-flight requests on disconnect (``cancel_inflight``)
    - call :meth:`ModelProvider.warm_up` before first inference
      (``warm_up_required``).

    Frozen ``@dataclass`` — kept off Pydantic because §13.4 M6's
    ``StrictModel`` is for *Settings* sub-models specifically. This is a
    runtime-introspection value, not config.
    """

    streaming: bool = False
    """Provider yields chunks via :meth:`stream_typed`."""

    tools_native: bool = False
    """Provider supports OpenAI-style ``tool_calls`` in the response (e.g.,
    MLC's ``delta.tool_calls``). The orchestrator will pass ``tools`` and
    consume native :class:`ProviderToolCall` events instead of relying on
    the text marker."""

    tools_marker: bool = False
    """Provider supports the ``<<function_call>>`` text-marker fallback for
    tool calls (e.g., when ``tools_native`` is False or returns ``None``).
    The orchestrator's parser will detect the marker in
    :class:`ProviderText` events."""

    thinking_channel: bool = False
    """Provider emits a separate thinking-text channel (:class:`ProviderThink`
    events). Currently no provider does; future thinking-supported models
    flip this on."""

    cancel_inflight: bool = False
    """Provider supports aborting an in-flight request (via ``cancel_token``
    in :meth:`stream_typed`; e.g., MLC's ``engine._abort(request_id)``)."""

    multi_model: bool = False
    """Provider can hold multiple loaded models simultaneously (e.g., MLC's
    engine cache; Ollama's named models). Single-model providers (e.g., a
    fixed local server) flip this off."""

    warm_up_required: bool = False
    """Provider needs :meth:`ModelProvider.warm_up` called before first
    inference is fast. MLC: True (engine init takes seconds). Ollama via
    HTTP: False (server holds state). The orchestrator may invoke
    ``warm_up`` in advance."""


@dataclass(frozen=True)
class ModelInfo:
    """Identity + light metadata for a model exposed by a provider.

    Frozen ``@dataclass`` — passed around by value, not validated against
    untrusted YAML. ``metadata`` is intentionally free-form; readers should
    tolerate missing keys.
    """

    name: str
    """Model identifier (passed to :meth:`stream_typed`'s ``model_name``)."""

    kind: str
    """Provider kind that owns this model (e.g., ``"mlc"``, ``"dummy"``,
    ``"ollama"``)."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Free-form metadata (``size_gb``, ``quantization``,
    ``context_window`` …). Keys are not enforced; readers should tolerate
    missing keys."""


# ---------- Provider events (typed stream output) ----------


class ProviderText(BaseModel):
    """A token / text delta from the model. Multiple :class:`ProviderText`
    events are concatenated by the orchestrator (Phase 5+).
    """

    type: Literal["text"] = "text"
    text: str

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProviderThink(BaseModel):
    """A thinking-channel delta (e.g., for models that emit reasoning
    separately). Today no provider emits this; the type is reserved so the
    discriminated union doesn't change shape when a future model adds
    thinking output.
    """

    type: Literal["think"] = "think"
    text: str

    model_config = ConfigDict(frozen=True, extra="forbid")


class ProviderToolCall(BaseModel):
    """A native tool call from the model (NOT the marker-based path).

    Replaces the legacy ``delta.tool_calls`` list-of-dicts shape with a
    typed event. Synthesis §6 bug #12: the legacy orchestrator silently
    dropped these because it only consumed string chunks; the v2 path
    emits a :class:`ProviderToolCall` so :class:`OrchestratorState` (Phase
    5+) can dispatch it.
    """

    type: Literal["tool_call"] = "tool_call"

    tool_call_id: str
    """Provider-emitted unique ID for this tool call (used for matching the
    result back). MLC's ``tool_calls`` have a ``.id`` field."""

    name: str
    """Tool name (function name)."""

    arguments: Dict[str, Any]
    """Parsed arguments dict. The MLC adapter parses the JSON-string
    ``arguments`` into a dict before constructing this event so consumers
    don't have to re-parse."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# Discriminated union for stream_typed's yield type. Pydantic v2 picks the
# right class based on the ``type`` Literal field automatically when used
# via ``TypeAdapter(ProviderEvent).validate_python(...)``. Consumers pattern-
# match on ``isinstance`` for ergonomic dispatch.
ProviderEvent = Union[ProviderText, ProviderThink, ProviderToolCall]


__all__ = [
    "ProviderCapabilities",
    "ModelInfo",
    "ProviderText",
    "ProviderThink",
    "ProviderToolCall",
    "ProviderEvent",
]
