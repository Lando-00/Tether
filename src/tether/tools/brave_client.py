"""
brave_client.py - Brave Search API client wrapper.

This module provides an async HTTP client for the Brave Search API with:
- Separate connect/read timeouts
- Exponential backoff retry logic (429, 5xx only)
- Response normalization to structured format
- Security: No API keys or full responses in logs
- Long-lived shared ``httpx.AsyncClient`` (synthesis §6 row 17): the
  client is opened once via :meth:`aopen` and reused across every
  :meth:`search` call so the TLS handshake + connection pool warm-up
  (~150-300 ms) is paid once instead of per request. The previous
  per-call ``async with httpx.AsyncClient(...)`` pattern was replaced
  in the p4-brave-client-lifecycle PR (synthesis §4 Phase 4 step 44).
"""

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

from tether.core.redact import redact_text

logger = logging.getLogger(__name__)


class BraveSearchClient:
    """
    Async HTTP client for Brave Search API.

    Implements timeout management, retry logic, and response normalization.

    Lifecycle: call :meth:`aopen` before the first :meth:`search` call (or
    use the client as an ``async with`` context manager). :meth:`search`
    reuses a single :class:`httpx.AsyncClient` across calls — see
    synthesis §6 row 17 for the cold-TLS bug this fixes.
    """

    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(
        self,
        api_key: str,
        connect_timeout: float = 2.0,
        read_timeout: float = 6.0,
        total_timeout: float = 15.0,
        max_retries: int = 2,
        backoff_base: float = 0.5,
    ):
        """
        Initialize Brave Search client.

        Args:
            api_key: Brave API subscription token
            connect_timeout: Socket connect timeout in seconds (default: 2s)
            read_timeout: Socket read timeout in seconds (default: 6s)
            total_timeout: Total request timeout including retries (default: 15s)
            max_retries: Maximum retry attempts for 429/5xx (default: 2)
            backoff_base: Base delay for exponential backoff in seconds (default: 0.5s)
        """
        if not api_key:
            raise ValueError("API key cannot be empty")

        self.api_key = api_key
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.total_timeout = total_timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base

        # Create timeout config for httpx
        self.timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=5.0,
            pool=5.0
        )

        # Shared httpx.AsyncClient — populated by :meth:`aopen`. Keeping
        # the connection pool open across calls eliminates the per-call
        # TLS handshake (synthesis §6 row 17).
        self._client: Optional[httpx.AsyncClient] = None
        self._opened: bool = False

    async def aopen(self) -> None:
        """Open the long-lived :class:`httpx.AsyncClient`. Idempotent.

        Calling twice is a no-op. Per synthesis §6 row 17, the same
        client instance is reused across every :meth:`search` call so
        the TLS handshake + connection pool warm-up cost is paid once
        per process rather than once per query.
        """
        if self._opened:
            return
        self._client = httpx.AsyncClient(timeout=self.timeout)
        self._opened = True

    async def aclose(self) -> None:
        """Close the underlying client. Idempotent.

        Safe to call when :meth:`aopen` was never invoked or when
        :meth:`aclose` was already called — both are no-ops.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._opened = False

    async def __aenter__(self) -> "BraveSearchClient":
        """Async context-manager entry — calls :meth:`aopen`."""
        await self.aopen()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Async context-manager exit — calls :meth:`aclose`."""
        await self.aclose()

    async def search(
        self,
        q: str,
        count: int = 10,
        country: str = "us",
        search_lang: str = "en",
        freshness: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a web search query via Brave Search API.

        Args:
            q: Search query string (required)
            count: Number of results to return (1-20)
            country: 2-letter country code (maps to Brave's 'cc' param)
            search_lang: Language code (maps to Brave's 'hl' param)
            freshness: Freshness filter - 'pd' (past day), 'pw' (past week),
                      'pm' (past month), 'py' (past year), or None
            **kwargs: Additional Brave API parameters

        Returns:
            Dict with structured format:
            {
                "results": [{"url", "title", "snippet", "rank"}],
                "meta": {"took_ms", "engine", "query"}
            }

        Raises:
            RuntimeError: If :meth:`aopen` has not been called.
            httpx.HTTPStatusError: For non-retryable errors (4xx except 429)
            asyncio.TimeoutError: If total timeout exceeded
        """
        if not self._opened or self._client is None:
            raise RuntimeError(
                "BraveSearchClient.search called before aopen() — open the "
                "client via 'async with BraveSearchClient(...)' or call "
                "aopen() explicitly. Synthesis §6 row 17."
            )

        import time
        start_time = time.time()

        # Build query params with explicit Brave API param names
        params = {
            "q": q,
            "count": min(count, 20),  # Cap at 20
            "cc": country,  # country → cc (Brave param)
            "hl": search_lang,  # search_lang → hl (Brave param)
        }

        if freshness:
            params["freshness"] = freshness

        # Add any additional kwargs
        params.update(kwargs)

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,  # Auth header
        }

        # Retry loop with exponential backoff
        attempt = 0
        last_exception = None

        while attempt <= self.max_retries:
            # Check if we've exceeded total timeout budget
            elapsed = time.time() - start_time
            if elapsed >= self.total_timeout:
                logger.error(f"Total timeout exceeded ({self.total_timeout}s) after {attempt} attempts")
                raise asyncio.TimeoutError(f"Request exceeded total timeout of {self.total_timeout}s")

            # Calculate remaining timeout for this attempt
            remaining_timeout = self.total_timeout - elapsed
            attempt_timeout = httpx.Timeout(
                connect=min(self.connect_timeout, remaining_timeout),
                read=min(self.read_timeout, remaining_timeout),
                write=5.0,
                pool=5.0
            )

            try:
                # Reuse the shared self._client across retries — no
                # per-call AsyncClient construction (synthesis §6 row 17).
                # Per-attempt timeout shrinks the read budget toward the
                # remaining total_timeout so retries cannot overrun.
                response = await self._client.get(
                    self.BASE_URL,
                    headers=headers,
                    params=params,
                    timeout=attempt_timeout,
                )

                # Log response metadata (NO full response or API key)
                logger.info(
                    f"Brave API response: status={response.status_code}, "
                    f"latency={int((time.time() - start_time) * 1000)}ms, "
                    f"attempt={attempt + 1}"
                )

                # Check for errors
                if response.status_code == 200:
                    # Success - normalize and return
                    return self._normalize_response(response.json(), q, time.time() - start_time)

                elif response.status_code == 429:
                    # Rate limit - retry with backoff
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = self.backoff_base * (2 ** attempt)
                    else:
                        delay = self.backoff_base * (2 ** attempt)

                    logger.warning(
                        f"Rate limit (429) - attempt {attempt + 1}/{self.max_retries + 1}, "
                        f"retrying in {delay:.1f}s"
                    )

                    if attempt < self.max_retries:
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    else:
                        # Max retries exceeded
                        response.raise_for_status()

                elif response.status_code >= 500:
                    # Server error - retry with backoff
                    delay = self.backoff_base * (2 ** attempt)
                    logger.warning(
                        f"Server error ({response.status_code}) - attempt {attempt + 1}/{self.max_retries + 1}, "
                        f"retrying in {delay:.1f}s"
                    )

                    if attempt < self.max_retries:
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    else:
                        # Max retries exceeded
                        response.raise_for_status()

                elif response.status_code in (403, 422):
                    # Auth failure (403) or invalid token (422) - do not retry, provide friendly error
                    logger.error(f"Authentication failed ({response.status_code}) - check BRAVE_API_KEY")
                    raise ValueError(
                        "Brave API authentication failed. Please verify your BRAVE_API_KEY "
                        "is correct and active at https://api-dashboard.search.brave.com/"
                    )

                else:
                    # Other 4xx errors - do not retry
                    logger.error(f"Client error ({response.status_code}): {redact_text(response.text[:100])}")
                    response.raise_for_status()

            except (httpx.TimeoutException, asyncio.TimeoutError) as e:
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries + 1}")
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.backoff_base * (2 ** attempt)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                else:
                    raise

            except httpx.HTTPStatusError:
                # Already logged above, re-raise
                raise

            except Exception as e:
                logger.error(f"Unexpected error: {type(e).__name__}: {str(e)[:100]}")
                raise

        # Should not reach here, but if we do, raise the last exception
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected retry loop exit")

    def _normalize_response(
        self,
        data: Dict[str, Any],
        query: str,
        elapsed_sec: float
    ) -> Dict[str, Any]:
        """
        Normalize Brave API response to structured format.

        Args:
            data: Raw Brave API JSON response
            query: Original query string
            elapsed_sec: Request elapsed time in seconds

        Returns:
            Normalized response with:
            - results: List of structured result dicts
            - meta: Query metadata
            - articles: Deprecated list of formatted strings (for backward compatibility)
        """
        web_results = data.get("web", {}).get("results", [])

        # Build structured results
        results = []
        for idx, item in enumerate(web_results):
            # Extract and clean description/snippet
            description = item.get("description", "")
            # Remove HTML tags (basic cleanup)
            import re
            clean_desc = re.sub(r'<[^>]+>', '', description)

            # Truncate snippet to 360 chars
            snippet = clean_desc
            if len(snippet) > 360:
                snippet = snippet[:360] + "..."

            results.append({
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "snippet": snippet,
                "rank": idx + 1
            })

        # Build metadata
        meta = {
            "took_ms": int(elapsed_sec * 1000),
            "engine": "brave",
            "query": query
        }

        # §13 R5: query already not leaked in results but log level was INFO — demote to DEBUG
        logger.debug(f"Normalized {len(results)} results for query='{query}'")

        return {
            "results": results,
            "meta": meta
        }
