"""NexaProvider stub.

Implements the typed Provider v2 contract (kind, capabilities) but raises
NotImplementedError from every operation. The capabilities flags reflect
the projected Snapdragon NPU profile (briefing §12.6):

    streaming=True            (NPU pipes tokens out as they're generated)
    tools_native=True         (Nexa's chat-completions API supports tools)
    tools_marker=False        (Tether's marker fallback is unnecessary
                               with native tool support)
    thinking_channel=False    (Nexa doesn't emit reasoning separately today)
    cancel_inflight=True      (NPU runtime supports request cancellation)
    multi_model=True          (NPU runtime can hold multiple models)
    warm_up_required=False    (Nexa server holds state; cold-start cost
                               is on the server side, not the client)

When a future session implements this, the NPU-specific path may flip
warm_up_required to True if client-side preloading proves beneficial.

Synthesis briefing §2 Seam A + §3 extension points; §11.3 R20 (no impl
ships in this refactor).
"""
from __future__ import annotations

from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional

from tether.core.interfaces import ModelProvider
from tether.providers.types import ProviderCapabilities, ProviderEvent


class NexaProvider(ModelProvider):
    """Stub provider for Snapdragon NPU via Nexa SDK. Every concrete
    operation raises NotImplementedError. See module docstring for design notes.
    """

    @property
    def kind(self) -> str:
        return "nexa"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=True,
            tools_native=True,
            tools_marker=False,
            thinking_channel=False,
            cancel_inflight=True,
            multi_model=True,
            warm_up_required=False,
        )

    # --- Legacy v1 contract (raises) ---

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[Any, None]:
        raise NotImplementedError(
            "NexaProvider is a stub for forward-compatibility verification only. "
            "Implement against Nexa SDK in a future session."
        )
        # Unreachable but makes this an async generator
        if False:
            yield  # type: ignore[unreachable]

    def list_models(self) -> List[str]:
        raise NotImplementedError(
            "NexaProvider stub: list_models requires Nexa SDK integration."
        )

    def unload_model(self, model_name: str) -> bool:
        raise NotImplementedError(
            "NexaProvider stub: unload_model requires Nexa SDK integration."
        )

    def get_context_window(self, model_name: str) -> int:
        raise NotImplementedError(
            "NexaProvider stub: get_context_window requires Nexa SDK integration."
        )

    # --- v2 typed contract (raises) ---

    async def warm_up(self, model_name: str) -> None:
        raise NotImplementedError(
            "NexaProvider stub: warm_up requires Nexa SDK integration."
        )

    async def aclose(self) -> None:
        # Default ABC behavior is no-op; we raise here so stub-misuse
        # is loud at runtime. If a future session decides aclose should
        # be a no-op for Nexa (HTTP-backed, stateless client), they can
        # restore the ABC default by removing this override.
        raise NotImplementedError(
            "NexaProvider stub: aclose requires Nexa SDK integration."
        )

    async def stream_typed(
        self,
        *,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        cancel_token: Optional[Any] = None,
    ) -> AsyncIterator[ProviderEvent]:
        raise NotImplementedError(
            "NexaProvider stub: stream_typed requires Nexa SDK integration."
        )
        # Unreachable but makes this an async generator
        if False:
            yield  # type: ignore[unreachable]


__all__ = ["NexaProvider"]
