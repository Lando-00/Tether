"""
NexaProvider skeleton — HTTP-backed implementation targeting `nexa serve`.

NOTES BEFORE READING:

1. This file lives in `D:\\Dev\\TetherWorkspace\\nexa\\provider\\` (NOT in
   the Tether repo). It's a design exercise only. The repo already has a
   stub at `tether_service/providers/nexa/provider.py` with the right
   capabilities; this fills in the methods.

2. Current upstream blockers (May 2026):
   - Python SDK (`pip install nexaai`) install is broken — Qualcomm S3
     bucket returns 403 for the bridge binary that pip downloads at
     install time. See qualcomm/nexa-sdk#1069, #1071.
   - NPU license server (`lic.nexa.ai:443`) is unreachable from multiple
     networks, blocking `nexa infer` for any NPU model. See #1068, #1072,
     #1073.

3. Therefore this skeleton targets the **OpenAI-compatible HTTP API**
   exposed by `nexa serve` (which works for *non-NPU* GGUF models today,
   and will work for NPU models when the license server comes back).
   Targeting the HTTP boundary instead of the broken Python SDK gives
   us a clean abstraction either way.

4. Fits the existing Tether ABC at
   `tether_service/core/interfaces.py::ModelProvider` (Phase 7
   refactor — confirmed 2026-05-10). The existing stub at
   `tether_service/providers/nexa/provider.py` has the correct
   `capabilities` shape; this file just implements the `stream*`,
   `list_models`, `warm_up`, etc. methods against the Nexa REST API.

5. Wire shape: Nexa's HTTP server speaks OpenAI's
   `/v1/chat/completions` with `stream=true` (SSE). The implementation
   below uses `httpx.AsyncClient`. No Nexa-specific Python deps —
   intentional, so this skeleton is import-safe even when the broken
   `nexaai` package can't be installed.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

# These imports point at the Tether repo. When this skeleton is moved
# into `tether_service/providers/nexa/provider.py`, they resolve cleanly.
# Until then they're "for documentation" — not import-safe outside Tether.
# from tether_service.core.interfaces import ModelProvider
# from tether_service.providers.types import (
#     ProviderCapabilities,
#     ProviderEvent,
#     ProviderText,
#     ProviderToolCall,
# )

_log = logging.getLogger(__name__)


# Type aliases (resolved when the file lives inside Tether).
ModelProvider = object  # type: ignore[misc,assignment]
ProviderCapabilities = object  # type: ignore[misc,assignment]
ProviderEvent = object  # type: ignore[misc,assignment]
ProviderText = object  # type: ignore[misc,assignment]
ProviderToolCall = object  # type: ignore[misc,assignment]


class NexaProvider(ModelProvider):  # type: ignore[misc,valid-type]
    """HTTP-backed provider that talks to ``nexa serve``.

    Tether owns the lifecycle of the ``nexa serve`` process — typically
    started at app boot via a sidecar manager. The provider only needs the
    base URL.

    Args:
        base_url: ``http://127.0.0.1:18181`` (or wherever ``nexa serve``
            is bound). Configurable via Tether settings.
        token: Optional NEXA_TOKEN; the public token from the
            qualcomm/nexa-sdk README is the default for personal use.
        timeout_s: Per-request timeout. Streaming long completions can
            need 60–300 s; default 300.
        default_model: A model name to use as ``model_name`` default;
            overridable per call.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:18181",
        token: Optional[str] = None,
        timeout_s: float = 300.0,
        default_model: Optional[str] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._default_model = default_model
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            headers=self._build_headers(),
        )

    def _build_headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {"User-Agent": "tether/nexa-provider"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    # ------------------------------------------------------------------
    # v2 contract: kind + capabilities
    # ------------------------------------------------------------------

    @property
    def kind(self) -> str:
        return "nexa"

    @property
    def capabilities(self):
        # Lazy-import to mirror DummyProvider's pattern.
        from tether_service.providers.types import ProviderCapabilities  # type: ignore[import-not-found]

        return ProviderCapabilities(
            streaming=True,
            tools_native=True,        # OpenAI-compat tools API
            tools_marker=False,
            thinking_channel=False,   # Future: parse <think>...</think>
            cancel_inflight=True,     # httpx aclose() of the response
            multi_model=True,         # nexa serve can host multiple
            warm_up_required=False,   # Server-side cold-start, not client
        )

    # ------------------------------------------------------------------
    # Legacy v1 stream — kept for one cycle; will be removed when Tether
    # drops legacy stream support entirely.
    # ------------------------------------------------------------------

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ):
        """Yield raw text chunks. Native tool_calls go through the typed
        path only — the legacy contract drops them, matching MLC's
        behaviour during the transition."""
        async for evt in self.stream_typed(
            model_name=model_name,
            messages=messages,
            tools=tools,
            request_id=request_id,
        ):
            from tether_service.providers.types import ProviderText  # type: ignore[import-not-found]

            if isinstance(evt, ProviderText):
                yield evt.text
            # ProviderToolCall and ProviderThink dropped on the legacy path

    # ------------------------------------------------------------------
    # v2 typed stream — the real entry point.
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
    ) -> AsyncIterator[Any]:  # AsyncIterator[ProviderEvent] in Tether
        from tether_service.providers.types import (  # type: ignore[import-not-found]
            ProviderText,
            ProviderToolCall,
        )

        body: Dict[str, Any] = {
            "model": model_name or self._default_model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            body["tools"] = tools
        if max_output_tokens:
            body["max_tokens"] = max_output_tokens

        # Per-request headers (correlation ID for cross-layer logs).
        headers: Dict[str, str] = {}
        if request_id:
            headers["X-Request-ID"] = request_id

        async with self._client.stream(
            "POST", "/v1/chat/completions", json=body, headers=headers
        ) as resp:
            resp.raise_for_status()

            # Accumulate streaming tool-call deltas (OpenAI splits
            # arguments across many SSE events; we coalesce per id).
            tc_buf: Dict[str, Dict[str, Any]] = {}

            async for line in resp.aiter_lines():
                if cancel_token is not None and cancel_token.is_cancelled():
                    # httpx will close the response stream when we leave
                    # the `async with` block.
                    break

                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):].strip()
                if payload == "[DONE]":
                    break

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    _log.warning("nexa.stream.bad_chunk", payload=payload[:120])
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}

                content = delta.get("content")
                if content:
                    yield ProviderText(text=content)

                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index", 0)
                    key = str(idx)
                    buf = tc_buf.setdefault(
                        key,
                        {"id": None, "name": None, "arguments": ""},
                    )
                    if tc_delta.get("id"):
                        buf["id"] = tc_delta["id"]
                    fn = tc_delta.get("function") or {}
                    if fn.get("name"):
                        buf["name"] = fn["name"]
                    if fn.get("arguments"):
                        buf["arguments"] += fn["arguments"]

                finish = choices[0].get("finish_reason")
                if finish == "tool_calls":
                    # Flush coalesced tool calls.
                    for buf in tc_buf.values():
                        try:
                            args = (
                                json.loads(buf["arguments"])
                                if buf["arguments"]
                                else {}
                            )
                        except json.JSONDecodeError:
                            args = {"_raw": buf["arguments"]}
                        yield ProviderToolCall(
                            tool_call_id=buf["id"] or "nexa-tc-0",
                            name=buf["name"] or "",
                            arguments=args,
                        )
                    tc_buf.clear()

    # ------------------------------------------------------------------
    # Lifecycle and introspection.
    # ------------------------------------------------------------------

    async def warm_up(self, model_name: str) -> None:
        """Touch the server's model-load endpoint. ``nexa serve``
        lazy-loads on first inference; we can pre-load via a
        cheap health request."""
        # GET /v1/models (OpenAI-compat) just confirms the server
        # is alive. For aggressive warm-up, send a 1-token completion.
        r = await self._client.get("/v1/models")
        r.raise_for_status()

    async def aclose(self) -> None:
        """Close the httpx client. The ``nexa serve`` process is owned
        by Tether's sidecar manager, not us — we don't kill it here."""
        await self._client.aclose()

    def list_models(self) -> List[str]:
        """Synchronous list — Tether's existing ABC keeps this sync.
        Use a short-lived sync httpx client to avoid blocking the loop;
        list_models is called from setup paths, not the hot path."""
        with httpx.Client(
            base_url=self._base_url,
            timeout=10.0,
            headers=self._build_headers(),
        ) as c:
            r = c.get("/v1/models")
            r.raise_for_status()
            data = r.json()
            return [m["id"] for m in data.get("data", [])]

    def unload_model(self, model_name: str) -> bool:
        """Nexa's OpenAI-compat HTTP API does not expose unload today.
        Return False so callers know the request was a no-op."""
        return False

    def get_context_window(self, model_name: str) -> int:
        """Best-effort. Nexa's OpenAI-compat API doesn't expose model
        metadata reliably yet. Hard-coded fallback to a conservative
        4096 — Tether's orchestrator will use the practical-context
        math (Briefing §2 Seam C) on top regardless."""
        # Future: call /v1/models/{model_name} for richer metadata once
        # Nexa's HTTP API exposes context_window_size.
        return 4096


__all__ = ["NexaProvider"]
