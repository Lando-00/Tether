"""Low-level httpx client for GenieX CLI server communication.

Handles SSE framing quirks (no space after ``data:``, ``[DONE]`` terminal,
raw JSON error responses at stream start) and maps HTTP/connection errors
to Tether's typed exception taxonomy.

Synthesis: geniex-contract-probe-2026-07-25.md §§3,6,7.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import httpx
import structlog

from tether.core.errors import FatalProviderError, TransientProviderError
from tether.providers.geniex.degeneracy import DegenerateOutputGuard

_log = structlog.get_logger(__name__)


class GenieXClient:
    """HTTP client for a single GenieX CLI server instance.

    Parameters
    ----------
    base_url:
        Server root (e.g. ``http://127.0.0.1:18181``). Trailing slash stripped.
    timeout_seconds:
        Read timeout for streaming completions.
    connect_timeout_seconds:
        TCP connect timeout.
    url_validator:
        Optional outbound URL validator (e.g. ``assert_safe_url``).
    http_client:
        Optional pre-built httpx.AsyncClient for testing/injection.
    """

    HEALTH_PATH = "/v1/"
    MODELS_PATH = "/v1/models"
    COMPLETIONS_PATH = "/v1/chat/completions"

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 600.0,
        connect_timeout_seconds: float = 10.0,
        url_validator: Callable[[str], None] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._connect_timeout_seconds = connect_timeout_seconds
        self._url_validator = url_validator
        if self._url_validator is not None:
            self._url_validator(self._base_url)
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                timeout=timeout_seconds,
                connect=connect_timeout_seconds,
            ),
        )

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        url = f"{self._base_url}{path}"
        if self._url_validator is not None:
            self._url_validator(url)
        return url

    # ------------------------------------------------------------------
    # Health / models
    # ------------------------------------------------------------------

    async def health(self) -> bool:
        """GET /v1/ — returns True if server responds 200."""
        try:
            resp = await self._client.get(
                self._url(self.HEALTH_PATH),
                timeout=httpx.Timeout(
                    timeout=self._connect_timeout_seconds,
                    connect=self._connect_timeout_seconds,
                ),
            )
            return resp.status_code == 200
        except httpx.TransportError as exc:
            _log.debug("geniex.health.failed", error=str(exc))
            return False

    async def list_models(self) -> List[str]:
        """GET /v1/models — returns model ID list from server.

        Raises TransientProviderError on connection/timeout failures.
        """
        try:
            resp = await self._client.get(
                self._url(self.MODELS_PATH),
                timeout=httpx.Timeout(
                    timeout=self._connect_timeout_seconds,
                    connect=self._connect_timeout_seconds,
                ),
            )
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except httpx.TimeoutException as exc:
            raise TransientProviderError(
                f"GenieX server timeout at {self._base_url}: {exc}"
            ) from exc
        except httpx.TransportError as exc:
            raise TransientProviderError(
                f"GenieX server unreachable at {self._base_url}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise TransientProviderError(
                f"GenieX /v1/models returned {exc.response.status_code}"
            ) from exc

    # ------------------------------------------------------------------
    # Streaming completions
    # ------------------------------------------------------------------

    async def stream_completion(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        max_output_chars: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """POST /v1/chat/completions with stream=true.

        Yields content-delta strings. Handles SSE framing (``data:{json}``
        with no space), ``[DONE]`` terminal, and raw JSON error responses.

        Two client-side bounds are applied, because the validated GenieX
        release does not enforce ``max_tokens`` server-side:

        * ``max_output_chars`` stops the stream once that much content has been
          emitted, so a runaway generation cannot stream indefinitely.
        * A :class:`DegenerateOutputGuard` aborts the stream if the output
          collapses into pathological repetition (see
          :mod:`tether.providers.geniex.degeneracy`).

        Raises
        ------
        TransientProviderError
            Server unreachable, timed out, or the generation collapsed.
            Collapse is transient by nature — the same request usually
            succeeds on retry.
        FatalProviderError
            Server returned a model/configuration error (400/500).
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "enable_think": False,
        }

        url = self._url(self.COMPLETIONS_PATH)

        try:
            req = self._client.build_request("POST", url, json=payload)
            resp = await self._client.send(req, stream=True)
        except httpx.TimeoutException as exc:
            raise TransientProviderError(
                f"GenieX server timeout at {self._base_url}: {exc}"
            ) from exc
        except httpx.TransportError as exc:
            raise TransientProviderError(
                f"GenieX server unreachable at {self._base_url}: {exc}"
            ) from exc

        try:
            # Check for non-2xx error before entering SSE loop.
            # GenieX may return raw JSON error (not SSE-wrapped) on failures.
            if resp.status_code >= 400:
                body = await resp.aread()
                error_text = self._extract_error(resp.status_code, body)
                if resp.status_code == 400:
                    raise FatalProviderError(
                        f"GenieX request rejected: {error_text}"
                    )
                raise TransientProviderError(
                    f"GenieX server error ({resp.status_code}): {error_text}"
                )

            # Stream SSE lines
            guard = DegenerateOutputGuard()
            emitted_chars = 0
            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    # Detect raw JSON error mid-stream
                    if line.startswith("{"):
                        error_text = self._parse_error_json(line)
                        raise TransientProviderError(
                            f"GenieX stream error: {error_text}"
                        )
                    continue

                payload_str = line[5:]  # Strip "data:" (no space per contract)
                if payload_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    _log.debug("geniex.sse.bad_json", raw=payload_str[:200])
                    continue

                # Extract content delta
                choices = chunk.get("choices")
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if not content:
                    continue

                reason = guard.observe(content)
                if reason is not None:
                    _log.warning(
                        "geniex.stream.degenerate",
                        model=model,
                        chars_emitted=emitted_chars,
                        reason=reason,
                    )
                    raise TransientProviderError(
                        f"GenieX generation collapsed: {reason}. "
                        "This is a known intermittent fault in the NPU "
                        "inference stack; retrying usually succeeds."
                    )

                if max_output_chars is not None and emitted_chars >= max_output_chars:
                    # The validated GenieX release ignores max_tokens, so the
                    # only enforcement point is here.
                    _log.warning(
                        "geniex.stream.output_cap",
                        model=model,
                        max_output_chars=max_output_chars,
                    )
                    break

                emitted_chars += len(content)
                yield content
        except httpx.TimeoutException as exc:
            raise TransientProviderError(
                f"GenieX server timeout at {self._base_url}: {exc}"
            ) from exc
        except httpx.TransportError as exc:
            raise TransientProviderError(
                f"GenieX stream transport failed at {self._base_url}: {exc}"
            ) from exc
        finally:
            await resp.aclose()

    # ------------------------------------------------------------------
    # Error parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_error(status_code: int, body: bytes) -> str:
        """Extract a human-readable error from a GenieX error response."""
        try:
            data = json.loads(body)
            return data.get("error", f"HTTP {status_code}")
        except (json.JSONDecodeError, KeyError):
            text = body.decode("utf-8", errors="replace")[:300]
            return text or f"HTTP {status_code}"

    @staticmethod
    def _parse_error_json(line: str) -> str:
        """Parse a raw JSON error line from the stream."""
        try:
            data = json.loads(line)
            return data.get("error", line[:200])
        except json.JSONDecodeError:
            return line[:200]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying httpx client if we own it."""
        if self._owns_client:
            await self._client.aclose()
