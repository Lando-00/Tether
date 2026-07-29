"""GenieX ModelProvider — marker-only external SSE provider.

Connects to an operator-managed GenieX CLI server. Tool calling uses the
``<<function_call>>`` text-marker path exclusively; no native tool fields
are sent or parsed. See geniex-contract-probe-2026-07-25.md for the full
server contract.

Provider kind: ``"geniex"``
Capabilities: streaming, marker-based tools, cancel via stream close.
"""
from __future__ import annotations

from typing import Any, AsyncGenerator, AsyncIterator, Callable, Dict, List, Optional

import httpx
import structlog

from tether.core.errors import TransientProviderError
from tether.core.interfaces import ModelProvider
from tether.providers.geniex.client import GenieXClient
from tether.providers.types import (
    ModelDetails,
    ProviderCapabilities,
    ProviderEvent,
    ProviderText,
)

_log = structlog.get_logger(__name__)

_CAPABILITIES = ProviderCapabilities(
    streaming=True,
    tools_native=False,
    tools_marker=True,
    thinking_channel=False,
    cancel_inflight=True,
    multi_model=False,
    warm_up_required=True,
    # Warm-up here is a cheap HTTP reachability probe (GET /v1/ + GET
    # /v1/models), not a weight load, so the Engine runs it at boot to
    # classify health for degraded-mode routing.
    warm_up_on_startup=True,
)


class GenieXProvider(ModelProvider):
    """Marker-only SSE provider for an external GenieX CLI server.

    The server is operator-managed — this provider never starts, stops,
    or downloads models. It issues HTTP requests and parses SSE streams.

    Parameters
    ----------
    base_url:
        Server root URL (e.g. ``http://127.0.0.1:18181``).
    model_id:
        Configured model identifier exposed via ``list_models()`` and
        ``list_model_info()`` even when the server is unreachable.
    request_model_id:
        Model ID sent in completion requests. Defaults to ``model_id``
        if not specified. Allows decoupling display ID from wire ID.
    context_window:
        Context window size in tokens for the configured model.
    timeout_seconds:
        Read timeout for streaming completions (seconds).
    connect_timeout_seconds:
        TCP connect timeout (seconds).
    temperature:
        Default sampling temperature.
    max_tokens:
        Default max output tokens.
    url_validator:
        Optional outbound URL validator (e.g. ``assert_safe_url``).
    http_client:
        Optional pre-built httpx.AsyncClient for testing.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        request_model_id: str | None = None,
        context_window: int = 4096,
        timeout_seconds: float = 600.0,
        connect_timeout_seconds: float = 10.0,
        temperature: float = 0.6,
        max_tokens: int = 1024,
        url_validator: Callable[[str], None] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model_id = model_id
        self._request_model_id = request_model_id or model_id
        self._context_window = context_window
        self._temperature = temperature
        self._max_tokens = max_tokens

        self._client = GenieXClient(
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            url_validator=url_validator,
            http_client=http_client,
        )

    # ------------------------------------------------------------------
    # Provider v2 introspection
    # ------------------------------------------------------------------

    @property
    def kind(self) -> str:
        return "geniex"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return _CAPABILITIES

    @property
    def source(self) -> str:
        """Locally hosted on-device inference server."""
        return "local"

    def default_model(self) -> str | None:
        return self._model_id

    # ------------------------------------------------------------------
    # Model listing (static — always returns configured model)
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        """Return configured model ID regardless of server state.

        This enables Engine 503 unhealthy routing: even when the GenieX
        server is down, the provider reports its configured model so the
        engine can surface a meaningful unhealthy status.
        """
        return [self._model_id]

    def list_model_info(self) -> List[ModelDetails]:
        return [
            ModelDetails(
                id=self._model_id,
                provider_kind=self.kind,
                source=self.source,
                context_window=self._context_window,
                supports_thinking=False,
                supports_reasoning_effort=False,
                is_default=True,
            )
        ]

    def get_context_window(self, model_name: str) -> int:
        return self._context_window

    def unload_model(self, model_name: str) -> bool:
        """GenieX models are managed externally; unload is a no-op."""
        return False

    # ------------------------------------------------------------------
    # Warm-up (preflight health + model verification)
    # ------------------------------------------------------------------

    async def warm_up(self, model_name: str) -> None:
        """Verify server health and configured model availability.

        Calls GET /v1/ (health) and GET /v1/models, then checks that
        the configured request_model_id appears in the server's model
        list. Does NOT issue a generation request.

        Raises TransientProviderError if server is unreachable or the
        configured model is not listed.
        """
        healthy = await self._client.health()
        if not healthy:
            raise TransientProviderError(
                "GenieX server health check failed (GET /v1/ did not return 200)"
            )

        try:
            server_models = await self._client.list_models()
        except TransientProviderError:
            raise

        if self._request_model_id not in server_models:
            raise TransientProviderError(
                f"GenieX model {self._request_model_id!r} not listed by server. "
                f"Available: {server_models}"
            )

        _log.info(
            "geniex.warm_up.ok",
            model_id=self._model_id,
            request_model_id=self._request_model_id,
            server_models=server_models,
        )

    # ------------------------------------------------------------------
    # Legacy stream (string chunks)
    # ------------------------------------------------------------------

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Legacy string-chunk stream.

        Deliberately ignores the ``tools`` argument — GenieX uses
        marker-based tool calling only via system prompt instructions.
        Never sends tools/tool_choice/functions to the server.
        """
        async for content in self._client.stream_completion(
            model=self._request_model_id,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        ):
            yield content

    # ------------------------------------------------------------------
    # v2 typed stream
    # ------------------------------------------------------------------

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
        """v2 typed stream yielding ProviderText events.

        Deliberately ignores ``tools`` — marker-only provider.
        Respects ``max_output_tokens`` if provided (capped to context window),
        otherwise uses configured ``max_tokens``.

        Cancel is handled by the caller closing the async generator
        (httpx stream is released in the client's finally block).
        """
        effective_max = min(
            max_output_tokens or self._max_tokens,
            self._context_window,
        )

        async for content in self._client.stream_completion(
            model=self._request_model_id,
            messages=messages,
            temperature=self._temperature,
            max_tokens=effective_max,
        ):
            yield ProviderText(text=content)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
