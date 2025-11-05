"""
web_search_tool.py - Web search tool using Brave Search API.

This module provides web search functionality via the Brave Search API,
replacing the legacy NewsAPI implementation.
"""

import os
import logging
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from tether_service.tools.base import BaseTool
from tether_service.tools.brave_client import BraveSearchClient


load_dotenv()
logger = logging.getLogger(__name__)

# Track if deprecation warning has been shown for 'language' param
_language_deprecation_warned = False


def _get_client() -> BraveSearchClient:
    """
    Get configured Brave Search client.
    
    Reads BRAVE_API_KEY from environment and returns an initialized client
    with default timeout/retry settings.
    
    Returns:
        BraveSearchClient instance
        
    Raises:
        ValueError: If BRAVE_API_KEY is not set or empty
    """
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise ValueError(
            "Environment variable BRAVE_API_KEY not set. "
            "Get your free API key at https://api-dashboard.search.brave.com/"
        )
    
    # Return client with default settings (2s connect, 6s read, 15s total)
    return BraveSearchClient(
        api_key=api_key,
        connect_timeout=2.0,
        read_timeout=6.0,
        total_timeout=15.0,
        max_retries=2,
        backoff_base=0.5
    )


def _validate_count(count: int, max_count: int = 20) -> int:
    """
    Validate and clamp count parameter.
    
    Args:
        count: Requested number of results
        max_count: Maximum allowed (default 20)
        
    Returns:
        Clamped count value (silently clamped if over max)
    """
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    
    # Silently clamp to max_count (don't error)
    if count > max_count:
        logger.debug(f"count={count} clamped to max_count={max_count}")
        return max_count
    
    return count


class WebSearchTool(BaseTool):
    """
    Search the web using Brave Search API.
    
    Provides general web search with country, language, and freshness filters.
    """
    
    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(
        self,
        query: str,
        count: int = 5,
        country: str = "us",
        search_lang: str = "en",
        freshness: Optional[str] = None,
        language: Optional[str] = None  # Deprecated alias
    ) -> Dict[str, Any]:
        """
        Search the web using Brave Search.

        Args:
            query: Search query (required).
            count: Number of results to return (1-20, default 5).
            country: 2-letter country code (default "us"). Maps to Brave's 'cc' param.
            search_lang: 2-letter language code (default "en"). Maps to Brave's 'hl' param.
            freshness: Freshness filter - 'pd' (past day), 'pw' (past week), 'pm' (past month), 'py' (past year), or None (no filter).
            language: DEPRECATED - Use 'search_lang' instead. Kept for backward compatibility.

        Returns:
            Dictionary with structured format:
            {
                "results": [{"url", "title", "snippet", "rank"}],
                "meta": {"took_ms", "engine", "query"},
                "articles": List[str]  # Deprecated, for backward compatibility
            }
            
            Or error format:
            {
                "error": "error message"
            }
        """
        global _language_deprecation_warned
        
        # Handle deprecated 'language' parameter
        if language is not None:
            if not _language_deprecation_warned:
                logger.warning(
                    "Parameter 'language' is deprecated. Use 'search_lang' instead. "
                    "The 'language' parameter will be removed in a future release."
                )
                _language_deprecation_warned = True
            # Map deprecated language to search_lang if search_lang is still default
            if search_lang == "en" and language != "en":
                search_lang = language
        
        # Validate query
        query = query.strip()
        if not query:
            return {"error": "query must be a non-empty string"}
        
        # Validate and clamp count (max 20 as per config plan)
        try:
            count = _validate_count(count, max_count=20)
        except ValueError as e:
            return {"error": str(e)}
        
        # Validate country code (2-letter)
        if not isinstance(country, str) or len(country) != 2:
            return {"error": "country must be a 2-letter ISO code"}
        
        # Validate search_lang (2-letter)
        if not isinstance(search_lang, str) or len(search_lang) != 2:
            return {"error": "search_lang must be a 2-letter ISO code"}
        
        # Validate freshness if provided
        if freshness is not None:
            valid_freshness = {"pd", "pw", "pm", "py"}
            if freshness not in valid_freshness:
                return {"error": f"freshness must be one of {valid_freshness} or None"}
        
        # Execute search
        try:
            client = _get_client()
            result = await client.search(
                q=query,
                count=count,
                country=country,  # Mapped to 'cc' in brave_client.py (line 109)
                search_lang=search_lang,  # Mapped to 'hl' in brave_client.py (line 110)
                freshness=freshness
            )
            return result
            
        except ValueError as e:
            # Friendly errors (e.g., 403 auth failure)
            return {"error": str(e)}
        except Exception as e:
            # Unexpected errors
            logger.error(f"web_search error: {type(e).__name__}: {str(e)}")
            return {"error": f"Search failed: {str(e)}"}