"""Ollama HTTP clients (ADR-0022 Phase 2.A).

This module hosts the **native NDJSON** client for the Ollama ``/api/chat``
surface plus the shared :class:`OllamaClientBase` Protocol and the
:class:`OllamaStreamEvent` value object that both client implementations
emit.  The OpenAI-compatible SSE client lives in
``tether.providers.ollama.openai_client`` and is delivered by Phase 2.B —
this module deliberately does NOT import it; the provider performs a lazy
import only when ``api_surface="openai_compat"`` is selected.

Wire-format references:
    - Native chat stream:  contract stubs §3
    - Tool-call translation: contract stubs §7
    - Error mapping:        contract stubs §8
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
)

import httpx
import structlog

_log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OllamaStreamEvent:
    """Normalised event yielded by every Ollama client implementation.

    Clients translate provider-specific wire formats (NDJSON for native,
    SSE for openai_compat) into this shape; :class:`OllamaProvider` then
    converts these into either legacy ``stream()`` outputs (str / list[dict])
    or v2 typed :class:`tether.providers.types.ProviderEvent` values.
    """

    kind: Literal["text", "thinking", "tool_call", "done"]
    text: str = ""
    tool_call: Optional[dict] = None
    stop_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Protocol — structural interface shared by native + openai-compat clients
# ---------------------------------------------------------------------------


@runtime_checkable
class OllamaClientBase(Protocol):
    """Structural Protocol shared by both Ollama client implementations.

    Phase 2.A ships :class:`OllamaNativeClient`. Phase 2.B will add
    ``OllamaOpenAICompatClient`` in a sibling module; both will satisfy
    this Protocol.
    """

    async def version(self) -> dict: ...

    async def list_models(self) -> list[dict]: ...

    async def show_model(self, model: str) -> dict: ...

    def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        think: bool = False,
        options: Optional[dict] = None,
        keep_alive: Optional[Any] = None,
        cancel_token: Optional[Any] = None,
    ) -> AsyncIterator[OllamaStreamEvent]: ...

    async def aclose(self) -> None: ...


# ---------------------------------------------------------------------------
# Tool-call translation (contract stubs §7)
# ---------------------------------------------------------------------------


def translate_native_tool_calls(ollama_tool_calls: list[dict]) -> list[dict]:
    """Translate Ollama ``message.tool_calls`` into the MLC-style list-of-dicts.

    Input  (per-item):  ``{"id"?: str, "function": {"name": str, "arguments": dict|str}}``
    Output (per-item):  ``{"id": str, "type": "function",
                           "function": {"name": str, "arguments": str}}``

    - ``id`` is preserved if present, otherwise synthesised via uuid4 hex[:12].
    - ``arguments`` is JSON-serialised to a string (the canonical MLC shape).
    - ``type`` is always ``"function"`` (Ollama omits it).
    """
    result: list[dict] = []
    for tc in ollama_tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        raw_args = fn.get("arguments", {})
        if isinstance(raw_args, str):
            args_str = raw_args
        else:
            try:
                args_str = json.dumps(raw_args)
            except (TypeError, ValueError):
                args_str = json.dumps({"_raw": repr(raw_args)})
        tc_id: str = tc.get("id") or uuid.uuid4().hex[:12]
        result.append(
            {
                "id": tc_id,
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            }
        )
    return result


# ---------------------------------------------------------------------------
# Native NDJSON client
# ---------------------------------------------------------------------------


def _cancelled(cancel_token: Any) -> bool:
    """Return True if ``cancel_token`` is set / cancelled.

    Tolerates the duck-typed shapes used across Tether: an
    :class:`asyncio.Event`-like ``is_set()``, a CancelToken-style
    ``cancelled()`` or ``is_cancelled()`` method, or a plain truthy flag.
    """
    if cancel_token is None:
        return False
    for attr in ("is_set", "cancelled", "is_cancelled"):
        fn = getattr(cancel_token, attr, None)
        if callable(fn):
            try:
                if bool(fn()):
                    return True
            except Exception:  # pragma: no cover - defensive
                return False
    return False


class OllamaNativeClient:
    """Native Ollama HTTP client speaking NDJSON on ``/api/chat``.

    Owns its :class:`httpx.AsyncClient` only when one is not injected.
    Tests inject an ``httpx.AsyncClient`` built on top of
    :class:`httpx.MockTransport`; production code lets the constructor
    create a long-lived shared client per provider.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 600.0,
        connect_timeout: float = 10.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = float(timeout)
        self._connect_timeout_seconds = float(connect_timeout)

        if http_client is None:
            timeout_cfg = httpx.Timeout(
                connect=connect_timeout,
                read=timeout,
                write=30.0,
                pool=5.0,
            )
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=timeout_cfg,
            )
            self._owns_client = True
        else:
            self._http = http_client
            self._owns_client = False

    # -- simple JSON endpoints ------------------------------------------------

    async def version(self) -> dict:
        try:
            resp = await self._http.get("/api/version")
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"could not reach Ollama at {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self._connect_timeout_seconds}s; "
                f"increase connect_timeout_seconds"
            ) from exc
        self._raise_for_status(resp, "/api/version")
        return resp.json()

    async def list_models(self) -> list[dict]:
        try:
            resp = await self._http.get("/api/tags")
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"could not reach Ollama at {self._base_url}: {exc}"
            ) from exc
        self._raise_for_status(resp, "/api/tags")
        body = resp.json()
        if isinstance(body, dict):
            return list(body.get("models") or [])
        return []

    async def show_model(self, model: str) -> dict:
        try:
            resp = await self._http.post("/api/show", json={"name": model})
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"could not reach Ollama at {self._base_url}: {exc}"
            ) from exc
        if resp.status_code == 404:
            raise RuntimeError(
                f"model '{model}' not pulled on Ollama; "
                f"run `ollama pull {model}` on the server"
            )
        self._raise_for_status(resp, "/api/show")
        return resp.json()

    # -- streaming chat -------------------------------------------------------

    async def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        think: bool = False,
        options: Optional[dict] = None,
        keep_alive: Optional[Any] = None,
        cancel_token: Optional[Any] = None,
    ) -> AsyncIterator[OllamaStreamEvent]:
        """Open POST /api/chat with ``stream=true`` and yield NDJSON events."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": bool(think),
        }
        if tools:
            body["tools"] = tools
        if options:
            body["options"] = options
        if keep_alive is not None:
            body["keep_alive"] = keep_alive

        try:
            async with self._http.stream(
                "POST", "/api/chat", json=body
            ) as resp:
                if resp.status_code == 404:
                    # Try to surface the not-pulled message; body is short.
                    text = ""
                    try:
                        text = (await resp.aread()).decode("utf-8", "replace")
                    except Exception:  # pragma: no cover
                        pass
                    lowered = text.lower()
                    if "not found" in lowered or "no such model" in lowered:
                        raise RuntimeError(
                            f"model '{model}' not pulled on Ollama; "
                            f"run `ollama pull {model}` on the server"
                        )
                    raise RuntimeError(
                        f"Ollama returned 404 from /api/chat: {text[:200]}"
                    )
                if resp.status_code >= 500:
                    text = ""
                    try:
                        text = (await resp.aread()).decode("utf-8", "replace")
                    except Exception:  # pragma: no cover
                        pass
                    raise RuntimeError(
                        f"Ollama server error {resp.status_code} at /api/chat: "
                        f"{text[:200]}"
                    )
                if resp.status_code >= 400:
                    text = ""
                    try:
                        text = (await resp.aread()).decode("utf-8", "replace")
                    except Exception:  # pragma: no cover
                        pass
                    raise RuntimeError(
                        f"Ollama returned HTTP {resp.status_code} at /api/chat: "
                        f"{text[:200]}"
                    )

                async for line in resp.aiter_lines():
                    if _cancelled(cancel_token):
                        _log.debug("ollama.stream.cancelled")
                        break
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        _log.warning("ollama.ndjson.malformed", line=line[:200])
                        continue
                    if not isinstance(chunk, dict):
                        _log.warning("ollama.ndjson.malformed", line=line[:200])
                        continue

                    msg = chunk.get("message") or {}
                    thinking_text = msg.get("thinking") or ""
                    content_text = msg.get("content") or ""
                    tool_calls = msg.get("tool_calls") or []

                    if thinking_text:
                        yield OllamaStreamEvent(
                            kind="thinking", text=thinking_text
                        )
                    if content_text:
                        yield OllamaStreamEvent(kind="text", text=content_text)
                    if tool_calls:
                        translated = translate_native_tool_calls(tool_calls)
                        for tc in translated:
                            yield OllamaStreamEvent(
                                kind="tool_call", tool_call=tc
                            )

                    if chunk.get("done"):
                        stop_reason = chunk.get("done_reason") or "stop"
                        yield OllamaStreamEvent(
                            kind="done", stop_reason=stop_reason
                        )
                        break
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"could not reach Ollama at {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self._timeout_seconds}s; "
                f"increase timeout_seconds"
            ) from exc

    # -- lifecycle ------------------------------------------------------------

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _raise_for_status(resp: httpx.Response, endpoint: str) -> None:
        if resp.status_code >= 500:
            raise RuntimeError(
                f"Ollama server error {resp.status_code} at {endpoint}"
            )
        if resp.status_code >= 400:
            text = ""
            try:
                text = resp.text
            except Exception:  # pragma: no cover
                pass
            raise RuntimeError(
                f"Ollama returned HTTP {resp.status_code} at {endpoint}: "
                f"{text[:200]}"
            )


__all__ = [
    "OllamaClientBase",
    "OllamaNativeClient",
    "OllamaStreamEvent",
    "translate_native_tool_calls",
]
