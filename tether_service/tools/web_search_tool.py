"""
web_search_tool.py - Web search tools using NewsAPI.

This module provides news search and source listing tools that fetch live data
from the NewsAPI service.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException
from dotenv import load_dotenv
from tether_service.tools.base import BaseTool


load_dotenv()
logger = logging.getLogger(__name__)


def _get_client() -> NewsApiClient:
    """Get configured NewsAPI client."""
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise ValueError("Environment variable NEWSAPI_KEY not set")
    if len(api_key) != 32:
        raise ValueError("NEWSAPI_KEY must be a 32-character UUID string")
    return NewsApiClient(api_key=api_key)


def _format_article(article: Dict[str, Any]) -> str:
    """Format a single article into a readable string."""
    title = (article.get("title") or "No title").strip()
    src = (article.get("source", {}).get("name") or "Unknown source").strip()
    desc = (article.get("description") or "").strip()
    url = (article.get("url") or "").strip()
    pub = article.get("publishedAt", "")
    date_str = ""
    if pub:
        try:
            date_str = pub.split("T")[0]
            datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            logger.warning(f"Invalid date: {pub}")

    text = title
    if src:
        text += f" [{src}]"
    if desc:
        if len(desc) > 100:
            desc = desc[:97] + "..."
        text += f": {desc}"
    if date_str:
        text += f" ({date_str})"
    if url:
        text += f" - {url}"
    return text


def _validate_common(page_size: Optional[int], page: Optional[int]) -> None:
    """Validate common pagination parameters."""
    if page_size is not None and (not isinstance(page_size, int) or not (1 <= page_size <= 100)):
        raise ValueError("page_size must be an int between 1 and 100")
    if page is not None and (not isinstance(page, int) or page < 1):
        raise ValueError("page must be an int > 0")


class WebSearchTool(BaseTool):
    """Search historical news articles using NewsAPI (/v2/everything endpoint)."""
    
    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(
        self,
        query: str,
        count: int = 5,
        sources: Optional[str] = None,
        domains: Optional[str] = None,
        exclude_domains: Optional[str] = None,
        from_param: Optional[str] = None,
        to: Optional[str] = None,
        language: str = "en",
        sort_by: Literal["relevancy", "popularity", "publishedAt"] = "relevancy",
        page: int = 1
    ) -> Dict[str, Any]:
        """
        Search historical news articles.

        Args:
            query: Search query (required).
            count: Number of results to return (1-100).
            sources: Comma-separated source IDs to filter by.
            domains: Comma-separated domains to filter by.
            exclude_domains: Comma-separated domains to exclude.
            from_param: Start date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).
            to: End date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).
            language: 2-letter ISO language code.
            sort_by: Sort order (relevancy, popularity, or publishedAt).
            page: Page number (>0).

        Returns:
            Dictionary with articles list or error information.
        """
        query = query.strip()
        if not query:
            return {"error": "query must be a non-empty string"}

        try:
            _validate_common(count, page)
            if len(language) != 2:
                return {"error": "language must be a 2-letter ISO code"}

            date_args: Dict[str, str] = {}
            for name, val in (("from_param", from_param), ("to", to)):
                if val:
                    try:
                        _ = datetime.fromisoformat(val.replace("Z", "")) if "T" in val else datetime.strptime(val, "%Y-%m-%d")
                    except Exception:
                        return {"error": f"{name} must be YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS"}
                    date_args[name] = val

            client = _get_client()
            resp = client.get_everything(
                q=query, page_size=count, language=language, sort_by=sort_by,
                page=page, **date_args,
                sources=sources or None,
                domains=domains or None,
                exclude_domains=exclude_domains or None
            )
        except NewsAPIException as e:
            return {"error": f"NewsAPI error: {e}"}
        except ValueError as e:
            return {"error": f"Parameter error: {e}"}
        except Exception as e:
            logger.error(f"web_search error: {e}")
            return {"error": f"Unexpected error: {e}"}

        if resp.get("status") != "ok":
            return {"error": f"NewsAPI error ({resp.get('code')}): {resp.get('message')}"}

        articles = resp.get("articles", [])
        if not articles:
            return {"articles": [], "message": f"No articles for query='{query}'"}

        return {"articles": [_format_article(a) for a in articles[:count]]}