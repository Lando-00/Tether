"""GenieX ModelProvider — marker-only external SSE provider.

Connects to an operator-managed GenieX CLI server. Tool calling uses the
``<<function_call>>`` text-marker path exclusively; no native tool fields
are sent or parsed. See geniex-contract-probe-2026-07-25.md for the full
server contract.

Provider kind: ``"geniex"``
Capabilities: streaming, marker-based tools, cancel via stream close.
"""
from __future__ import annotations

import time
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Dict, List, Literal, Optional

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

# Minimum seconds between un-forced GET /v1/models refreshes, so a stream of
# requests for unknown model names cannot turn into a stream of round-trips
# to the external server.
_MODEL_REFRESH_MIN_INTERVAL_S = 30.0

# The validated GenieX release accepts `max_tokens` but does not enforce it, so
# the token budget has to be converted into a client-side character bound. Four
# characters per token is a deliberate over-estimate: the bound exists to stop a
# runaway generation, not to trim a legitimate reply, so it must not bite first.
_CHARS_PER_TOKEN_BOUND = 4

_CAPABILITIES = ProviderCapabilities(
    streaming=True,
    tools_native=False,
    tools_marker=True,
    thinking_channel=False,
    cancel_inflight=True,
    # The server can serve several models; which ones is discovered at
    # warm-up rather than fixed by config (see refresh_models).
    multi_model=True,
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

        # Discovery cache (see refresh_models). ``None`` means "the server has
        # never been reached", which is distinct from "the server reported no
        # models" (an empty list) — only the former falls back to the
        # configured id so /readyz can still describe an unreachable server.
        self._discovered: List[str] | None = None
        self._discovered_at: float = 0.0

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
    # Model listing (discovered from the server; see refresh_models)
    # ------------------------------------------------------------------

    def _model_names(self) -> List[str]:
        """Resolve the advertised model list.

        The configured ``model_id`` is always first and always present — it is
        the display alias for ``request_model_id`` and is what
        ``default_model()`` returns. Any *other* model the server advertises is
        appended under its own server-side id.

        When the server has never been reached, this degrades to the configured
        id alone so the Engine can still surface a meaningful unhealthy status
        for a down server rather than an empty provider.
        """
        if self._discovered is None:
            return [self._model_id]
        extra = [
            name
            for name in self._discovered
            if name != self._request_model_id and name != self._model_id
        ]
        return [self._model_id, *extra]

    def _resolve_request_model(self, model_name: str) -> str:
        """Map a caller-facing model name onto the id sent on the wire.

        ``model_id`` is an alias for ``request_model_id``; every other
        discovered name is already a server-side id and passes through. An
        empty/absent name falls back to the configured default so callers that
        do not care which model they get keep working.
        """
        if not model_name or model_name == self._model_id:
            return self._request_model_id
        return model_name

    async def refresh_models(self, *, force: bool = False) -> List[str]:
        """Re-read ``GET /v1/models`` into the discovery cache.

        Throttled to one call per :data:`_MODEL_REFRESH_MIN_INTERVAL_S` unless
        ``force`` is set, so a stream of requests for unknown model names
        cannot turn into a stream of HTTP round-trips.

        Never raises: a failed refresh leaves the previous cache in place (or
        leaves it unpopulated), because discovery is best-effort metadata and
        must not take down request routing.
        """
        now = time.monotonic()
        if (
            not force
            and self._discovered is not None
            and (now - self._discovered_at) < _MODEL_REFRESH_MIN_INTERVAL_S
        ):
            return self._model_names()
        try:
            self._discovered = await self._client.list_models()
            self._discovered_at = now
        except Exception as exc:  # noqa: BLE001 - discovery is best-effort
            _log.warning("geniex.refresh_models.failed", error=str(exc))
        return self._model_names()

    def list_models(self) -> List[str]:
        """Return the models this provider serves.

        Synchronous by contract (:class:`ModelProvider`), so it reads the cache
        populated by :meth:`warm_up` / :meth:`refresh_models` rather than
        issuing HTTP itself.
        """
        return self._model_names()

    def list_model_info(self) -> List[ModelDetails]:
        source: Literal["local", "remote"] = "local"
        return [
            ModelDetails(
                id=name,
                provider_kind=self.kind,
                source=source,
                context_window=self._context_window,
                supports_thinking=False,
                supports_reasoning_effort=False,
                is_default=(name == self._model_id),
            )
            for name in self._model_names()
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

        # Warm-up is the one guaranteed round-trip at boot, so it doubles as
        # the discovery pass that seeds list_models() for routing.
        self._discovered = server_models
        self._discovered_at = time.monotonic()

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
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Legacy string-chunk stream.

        Deliberately ignores ``tools`` and ``reasoning_effort`` — GenieX uses
        marker-based tool calling only via system prompt instructions and does
        not advertise reasoning-effort support, so the hint is accepted for
        interface parity and dropped. Never sends tools/tool_choice/functions
        to the server.
        """
        async for content in self._client.stream_completion(
            model=self._resolve_request_model(model_name),
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            max_output_chars=self._max_tokens * _CHARS_PER_TOKEN_BOUND,
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
            model=self._resolve_request_model(model_name),
            messages=messages,
            temperature=self._temperature,
            max_tokens=effective_max,
            max_output_chars=effective_max * _CHARS_PER_TOKEN_BOUND,
        ):
            yield ProviderText(text=content)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
