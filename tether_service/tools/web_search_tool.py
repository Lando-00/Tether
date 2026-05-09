"""
web_search_tool.py - Web search tool using Brave Search API.

This module provides web search functionality via the Brave Search API,
replacing the legacy NewsAPI implementation.

Style A migration (synthesis §4 Phase 4 step 43; A2 step 5):
A nested :class:`WebSearchInputs` model declares the validated input
schema; ``BaseTool.invoke`` Pydantic-validates the args dict before
calling :meth:`run`. Hand-rolled validation in the previous ``run()``
body has been deleted — Pydantic enforces the same constraints
declaratively (``min_length=1`` for query, ``ge=1, le=20`` for count,
``pattern`` for country/search_lang, ``Literal`` for freshness).
"""

import os
import logging
from typing import Any, Dict, Literal, Optional

from pydantic import Field, field_validator

from tether_service.tools.base import BaseTool, ToolInputs
from tether_service.tools.brave_client import BraveSearchClient
from tether_service.tools.registration import tool


logger = logging.getLogger(__name__)


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


class WebSearchInputs(ToolInputs):
    """Validated inputs for :class:`WebSearchTool`.

    All bounds and patterns previously enforced by hand inside
    :meth:`WebSearchTool.run` are now declarative Pydantic ``Field``
    constraints — ``BaseTool.invoke`` validates before ``run`` is
    called. Synthesis §4 Phase 4 step 43; A2 step 5.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="Search query (required, non-empty after stripping).",
    )
    count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return (1-20, default 5).",
    )
    country: str = Field(
        default="us",
        min_length=2,
        max_length=2,
        pattern=r"^[A-Za-z]{2}$",
        description=(
            "2-letter ISO country code (default 'us'). "
            "Maps to Brave's 'cc' param."
        ),
    )
    search_lang: str = Field(
        default="en",
        min_length=2,
        max_length=10,
        pattern=r"^[A-Za-z-]+$",
        description=(
            "Language code (default 'en'). Maps to Brave's 'hl' param."
        ),
    )
    freshness: Optional[Literal["pd", "pw", "pm", "py"]] = Field(
        default=None,
        description=(
            "Freshness filter — 'pd' (past day), 'pw' (past week), "
            "'pm' (past month), 'py' (past year), or None (no filter)."
        ),
    )

    @field_validator("query", mode="before")
    @classmethod
    def _strip_query(cls, v: Any) -> Any:
        """Strip surrounding whitespace before ``min_length`` validates,
        so ``query='   '`` is rejected the same as ``query=''``.

        Preserves the previous hand-rolled behavior in ``run()``."""
        if isinstance(v, str):
            return v.strip()
        return v


@tool(name="web_search")
class WebSearchTool(BaseTool):
    """Search the web using the Brave Search API.

    Provides general web search with country, language, and freshness
    filters. Style A: input validation lives in :class:`WebSearchInputs`
    (synthesis §4 Phase 4 step 43; A2 step 5).
    """

    Inputs = WebSearchInputs

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(self, inputs: WebSearchInputs) -> Dict[str, Any]:
        """Execute the search.

        Pydantic has already validated ``inputs`` via the ``Inputs``
        ClassVar dispatch in :meth:`BaseTool.invoke`; ``run`` here
        only handles transport / API-call concerns.

        Returns:
            Structured result dict with ``results`` / ``meta`` /
            ``articles`` keys, or an error dict ``{"error": "..."}``
            for transport failures (so the orchestrator can feed the
            error back to the model rather than crashing the loop).
        """
        try:
            client = _get_client()
            return await client.search(
                q=inputs.query,
                count=inputs.count,
                country=inputs.country,
                search_lang=inputs.search_lang,
                freshness=inputs.freshness,
            )
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"web_search error: {type(e).__name__}: {str(e)}")
            return {"error": f"Search failed: {str(e)}"}