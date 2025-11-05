"""
brave_client.py - Brave Search API client wrapper.

This module provides an async HTTP client for the Brave Search API with:
- Separate connect/read timeouts
- Exponential backoff retry logic (429, 5xx only)
- Response normalization to structured format
- Security: No API keys or full responses in logs
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
import httpx


logger = logging.getLogger(__name__)


class BraveSearchClient:
    """
    Async HTTP client for Brave Search API.
    
    Implements timeout management, retry logic, and response normalization.
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
                "meta": {"took_ms", "engine", "query"},
                "articles": List[str]  # Deprecated, for backward compatibility
            }
        
        Raises:
            httpx.HTTPStatusError: For non-retryable errors (4xx except 429)
            asyncio.TimeoutError: If total timeout exceeded
        """
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
                async with httpx.AsyncClient(timeout=attempt_timeout) as client:
                    response = await client.get(
                        self.BASE_URL,
                        headers=headers,
                        params=params
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
                        logger.error(f"Client error ({response.status_code}): {response.text[:100]}")
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
        
        # Build deprecated articles format for backward compatibility
        articles = []
        for r in results:
            # Format: "{title}: {snippet} - {url}"
            article_str = f"{r['title']}: {r['snippet']} - {r['url']}"
            articles.append(article_str)
        
        # Log result count (NO full results or API key)
        logger.info(f"Normalized {len(results)} results for query='{query}'")
        
        return {
            "results": results,
            "meta": meta,
            "articles": articles  # Deprecated - will be removed in future release
        }
