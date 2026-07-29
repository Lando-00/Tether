"""
openai_client.py — OpenAI-compatible SSE client for Ollama (ADR-0022 §4).

Implements ``OllamaOpenAICompatClient``, which hits Ollama's /v1/chat/completions
endpoint (OpenAI-compatible SSE surface) and yields ``OllamaStreamEvent`` objects
identical to those produced by the native client.

Phase 2.B of the Ollama provider rollout.  See ADR-0022 §4, §7, §8.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

import httpx
import structlog

from tether.providers.ollama.client import OllamaStreamEvent, _cancelled

_log = structlog.get_logger(__name__)


class _OAIToolCallBuffer:
    """Buffer for OpenAI-compatible streaming tool-call delta fragments.

    Accumulates per-index fragments across SSE events and flushes a complete
    MLC-style list when the stream ends or a finish_reason arrives.

    ADR-0022 §7 specifies the exact buffering behaviour: ``id`` and ``name``
    arrive only in the first delta for a given index; ``arguments`` is
    concatenated across all deltas for that index.
    """

    def __init__(self) -> None:
        self._buf: dict[int, dict] = {}

    def feed(self, tool_call_deltas: list[dict]) -> None:
        """Accumulate delta fragments from one SSE event."""
        for tc in tool_call_deltas:
            idx: int = tc.get("index", 0)
            if idx not in self._buf:
                self._buf[idx] = {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            entry = self._buf[idx]
            if tc.get("id"):
                entry["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                entry["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                entry["function"]["arguments"] += fn["arguments"]

    def flush(self) -> list[dict]:
        """Return complete MLC-style tool-call list sorted by index.

        Assigns a synthetic ID (``uuid.uuid4().hex[:12]``) for any entry
        where the server omitted ``id``.  Clears the buffer after flushing.
        """
        result: list[dict] = []
        for idx in sorted(self._buf):
            entry = self._buf[idx]
            if not entry["id"]:
                entry["id"] = uuid.uuid4().hex[:12]
            result.append(entry)
        self._buf.clear()
        return result

    @property
    def is_empty(self) -> bool:
        return not self._buf


class OllamaOpenAICompatClient:
    """OpenAI-compatible HTTP client for Ollama's /v1 surface.

    Implements the ``OllamaClientBase`` Protocol.  Uses Ollama's
    ``/v1/chat/completions`` SSE endpoint and ``/v1/models`` discovery
    endpoint rather than the native ``/api/chat`` NDJSON surface.

    Constructor
    -----------
    base_url : str
        The /v1 root URL the caller has already constructed, e.g.
        ``"http://192.168.1.50:11434/v1"``.  Do NOT include a trailing
        slash; paths are appended as ``/chat/completions`` etc.
    api_key : str | None
        If provided, every request carries an ``Authorization: Bearer``
        header.  Pass ``None`` for a local Ollama instance (no auth).
    timeout : float
        Read timeout in seconds (default 600 s — LLM streams can be long).
    connect_timeout : float
        Connect timeout in seconds (default 10 s).
    http_client : httpx.AsyncClient | None
        Inject a pre-built ``httpx.AsyncClient`` (e.g. from ``OllamaProvider``
        or from tests).  When ``None`` (default), an owned client is created
        using ``base_url``, ``timeout``, ``connect_timeout``, and ``api_key``.
        ``aclose()`` only closes a client that was created internally; it is
        a no-op when a caller-provided client was injected.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 600.0,
        connect_timeout: float = 10.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._owns_client = http_client is None

        if http_client is not None:
            self._http = http_client
        else:
            headers: dict[str, str] = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(
                    connect=connect_timeout,
                    read=timeout,
                    write=30.0,
                    pool=5.0,
                ),
                headers=headers,
            )

    # ------------------------------------------------------------------
    # OllamaClientBase Protocol implementation
    # ------------------------------------------------------------------

    async def version(self) -> dict:
        """Probe Ollama reachability via GET /v1/models.

        The OpenAI-compatible surface does not expose a ``/v1/version``
        endpoint.  We use ``GET /v1/models`` as a probe-by-proxy and return
        ``{"ok": True, "models_seen": N}`` so callers (e.g. OllamaProvider.warm_up)
        can detect connectivity without inspecting the full model list.
        """
        url = f"{self._base_url}/models"
        req_headers = self._auth_headers()
        try:
            resp = await self._http.get(url, headers=req_headers)
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            return {"ok": True, "models_seen": len(models)}
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise RuntimeError(
                f"could not reach Ollama OpenAI-compat at {self._base_url}; "
                "is the server running?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise RuntimeError(
                f"Ollama server error {status} at {url}"
            ) from exc

    async def list_models(self) -> list[dict]:
        """GET /v1/models → list of model dicts (OpenAI-shape).

        Each dict has at least ``"id"`` and ``"object": "model"``.
        Returns the ``data`` array from the OpenAI-compatible response body.
        """
        url = f"{self._base_url}/models"
        req_headers = self._auth_headers()
        try:
            resp = await self._http.get(url, headers=req_headers)
            resp.raise_for_status()
            return resp.json().get("data", [])
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise RuntimeError(
                f"could not reach Ollama OpenAI-compat at {self._base_url}; "
                "is the server running?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise RuntimeError(
                f"Ollama server error {status} at {url}"
            ) from exc

    async def show_model(self, model: str) -> dict:
        """Return empty dict — /v1 does not expose per-model details.

        The ``/v1/models/{model}`` endpoint is not available on the
        Ollama OpenAI-compatible surface.  Returning ``{}`` signals to
        the caller (OllamaProvider) that no context-window detail is
        available and it should fall back to the value in config.
        """
        return {}

    def stream_chat(  # type: ignore[override]  # Protocol return is AsyncIterator
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        think: bool = False,
        options: dict | None = None,
        keep_alive: Any | None = None,
        cancel_token: Any | None = None,
    ) -> AsyncIterator[OllamaStreamEvent]:
        """Stream a chat completion via POST /v1/chat/completions (SSE).

        Yields ``OllamaStreamEvent`` objects:
        - ``kind="text"`` — content delta
        - ``kind="tool_call"`` — fully assembled tool call (after buffering)
        - ``kind="done"`` — stream complete

        The ``think`` parameter is not forwarded to the OpenAI-compat surface;
        if ``True``, an INFO log is emitted once and the request proceeds
        without it.

        ``keep_alive`` is accepted for ``OllamaClientBase`` parity but ignored —
        the OpenAI-compatible surface has no equivalent parameter. An INFO log
        is emitted once if a non-None value is supplied.

        ``cancel_token`` is checked before each line is processed; the
        :func:`_cancelled` helper duck-types across ``is_set()``,
        ``cancelled()`` and ``is_cancelled()`` so any Tether cancel-token
        shape works.

        NOTE: this method is intentionally NOT ``async def`` — it returns
        the underlying async generator directly so callers can use
        ``async for`` without an extra ``await`` (matching OllamaNativeClient's
        async-generator pattern).
        """
        return self._stream_chat_impl(
            model=model,
            messages=messages,
            tools=tools,
            think=think,
            options=options,
            keep_alive=keep_alive,
            cancel_token=cancel_token,
        )

    async def aclose(self) -> None:
        """Close the owned httpx client.

        No-op when the client was injected by the caller (e.g. OllamaProvider
        owns the lifecycle and calls aclose on the shared httpx.AsyncClient
        itself).
        """
        if self._owns_client:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        """Return ``Authorization`` header dict when api_key is set, else empty."""
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    async def _stream_chat_impl(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        think: bool,
        options: dict | None,
        keep_alive: Any | None,
        cancel_token: Any | None,
    ) -> AsyncIterator[OllamaStreamEvent]:
        if think:
            _log.info(
                "ollama.openai_compat.think_ignored",
                reason="think=True ignored by api_surface='openai_compat'",
            )
        if keep_alive is not None:
            _log.info(
                "ollama.openai_compat.keep_alive_ignored",
                reason="keep_alive ignored by api_surface='openai_compat'",
                value=str(keep_alive),
            )

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            body["tools"] = tools

        url = f"{self._base_url}/chat/completions"
        buf = _OAIToolCallBuffer()

        try:
            async with self._http.stream("POST", url, json=body, headers=self._auth_headers()) as response:
                if response.status_code == 404:
                    text = await response.aread()
                    body_text = text.decode("utf-8", errors="replace")
                    if "not found" in body_text.lower():
                        raise RuntimeError(
                            f"model '{model}' not pulled on Ollama; "
                            f"run `ollama pull {model}` on the server"
                        )
                    raise RuntimeError(
                        f"Ollama server error 404 at {url}: {body_text[:200]}"
                    )
                response.raise_for_status()

                async for line in response.aiter_lines():
                    # Honour cancellation at each chunk boundary.
                    # _cancelled duck-types so any Tether cancel-token shape works.
                    if cancel_token is not None and _cancelled(cancel_token):
                        _log.debug("ollama.stream.cancelled")
                        return

                    if not line:
                        continue

                    # Skip SSE framing lines we don't care about
                    if line.startswith("event:") or line.startswith("id:"):
                        continue

                    if not line.startswith("data:"):
                        continue

                    payload = line[len("data:"):].strip()

                    if payload == "[DONE]":
                        # Flush any buffered tool calls before the done event
                        for tc in buf.flush():
                            yield OllamaStreamEvent(kind="tool_call", tool_call=tc)
                        yield OllamaStreamEvent(kind="done", stop_reason="stop")
                        return

                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        _log.warning("ollama.sse.malformed", line=line)
                        continue

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    finish_reason = choice.get("finish_reason")

                    # Accumulate tool-call deltas
                    tc_deltas = delta.get("tool_calls")
                    if tc_deltas:
                        buf.feed(tc_deltas)

                    # Emit text content delta
                    content = delta.get("content")
                    if content:
                        yield OllamaStreamEvent(kind="text", text=content)

                    # finish_reason signals end of stream — flush buffer and done
                    if finish_reason:
                        for tc in buf.flush():
                            yield OllamaStreamEvent(kind="tool_call", tool_call=tc)
                        yield OllamaStreamEvent(kind="done", stop_reason=finish_reason)
                        return

        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise RuntimeError(
                f"could not reach Ollama OpenAI-compat at {self._base_url}; "
                "is the server running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Ollama timed out streaming from {self._base_url}; "
                "increase timeout_seconds in provider config"
            ) from exc
        except RuntimeError:
            raise
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise RuntimeError(
                f"Ollama server error {status} at {url}"
            ) from exc
