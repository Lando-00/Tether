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

Lifecycle migration (synthesis §4 Phase 4 step 44; §6 row 17):
The tool now owns a long-lived :class:`BraveSearchClient` opened in
:meth:`startup` and closed in :meth:`shutdown`. The legacy
``_get_client()`` helper that constructed a fresh client per call has
been removed — synthesis §6 row 17 (cold TLS per call). The
:class:`SecretsProvider` (default
:class:`tether.core.secrets.EnvFileSecretsProvider`) supplies
the ``BRAVE_API_KEY``; if the secret is missing the tool stays in the
registry (``REQUIRED = False``) but :meth:`run` returns an error dict.
That choice keeps ``/tools`` discoverable and lets the model see a
clear, structured "missing-key" message rather than a silent drop.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

from pydantic import Field, field_validator

from tether.core.secrets import EnvFileSecretsProvider, SecretsProvider
from tether.security.outbound import assert_safe_url
from tether.tools.base import BaseTool, ToolInputs
from tether.tools.brave_client import BraveSearchClient
from tether.tools.registration import tool

if TYPE_CHECKING:
    from tether.config.settings import Settings


logger = logging.getLogger(__name__)


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

    Lifecycle: opens a long-lived :class:`BraveSearchClient` in
    :meth:`startup` and reuses it across every :meth:`run` call. If
    ``BRAVE_API_KEY`` is missing from the :class:`SecretsProvider`,
    :meth:`startup` logs a warning and leaves ``_client`` as ``None``;
    :meth:`run` then returns a structured error dict (synthesis §4
    Phase 4 step 44; §6 row 17).
    """

    Inputs = WebSearchInputs

    def __init__(
        self,
        *,
        secrets: Optional[SecretsProvider] = None,
        settings: Optional["Settings"] = None,
    ):
        """Construct the tool.

        Args:
            secrets: Optional :class:`SecretsProvider` for tests / DI;
                defaults to :class:`EnvFileSecretsProvider` (env-first,
                file-fallback under ``<data_dir>/secrets/<key>``).
                Connector spec §3.5; synthesis §4 Phase 4.5 step 47a.
            settings: Optional :class:`Settings` instance used to enforce
                outbound URL policy via ``assert_safe_url`` (Phase 7 step 78).
                When ``None`` (default), only the always-on scheme + host
                checks apply.
        """
        super().__init__()
        self._secrets: SecretsProvider = secrets or EnvFileSecretsProvider()
        self._settings: Optional["Settings"] = settings
        self._client: Optional[BraveSearchClient] = None

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def startup(self) -> None:
        """Open the shared :class:`BraveSearchClient`.

        Reads ``BRAVE_API_KEY`` via the configured
        :class:`SecretsProvider`. Per synthesis §4 Phase 4 step 41 +
        step 44, ``REQUIRED = False`` (default for :class:`BaseTool`)
        means a missing key does NOT abort engine startup: we log a
        warning, leave ``_client = None``, and let :meth:`run` return
        a structured error so the model gets a clear message rather
        than a silent registry drop. The tool stays discoverable in
        ``/tools`` listings.
        """
        api_key = self._secrets.get("BRAVE_API_KEY")
        if not api_key:
            logger.warning(
                "BRAVE_API_KEY not set; WebSearchTool will return errors "
                "until the key is configured (env var or "
                "<data_dir>/secrets/BRAVE_API_KEY)."
            )
            return

        self._client = BraveSearchClient(
            api_key=api_key,
            connect_timeout=2.0,
            read_timeout=6.0,
            total_timeout=15.0,
            max_retries=2,
            backoff_base=0.5,
        )
        await self._client.aopen()

    async def shutdown(self) -> None:
        """Close the shared :class:`BraveSearchClient` if opened.

        Safe to call when :meth:`startup` skipped client construction
        (missing key) — the ``None`` check makes shutdown idempotent.
        Per synthesis §4 Phase 4 step 41, shutdown failures are caught
        upstream by :func:`tether.tools.lifecycle.shutdown_all`
        and never raised.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def run(self, inputs: WebSearchInputs) -> Dict[str, Any]:
        """Execute the search.

        Pydantic has already validated ``inputs`` via the ``Inputs``
        ClassVar dispatch in :meth:`BaseTool.invoke`; ``run`` here
        only handles transport / API-call concerns.

        Returns:
            Structured result dict with ``results`` / ``meta`` /
            ``articles`` keys, or an error dict ``{"error": "..."}``
            for transport / configuration failures (so the orchestrator
            can feed the error back to the model rather than crashing
            the loop).
        """
        if self._client is None:
            return {
                "error": (
                    "web_search not initialised — BRAVE_API_KEY is not "
                    "configured. Set the env var or write the value to "
                    "<data_dir>/secrets/BRAVE_API_KEY and restart."
                )
            }

        # Phase 7 step 78: assert the fixed Brave API endpoint is safe.
        # With default settings (no allowlist), only scheme + host checks apply.
        # Callers that supply Settings with outbound_allowlist.enabled=True can
        # restrict to specific hosts (e.g., ["api.search.brave.com"]).
        try:
            assert_safe_url(BraveSearchClient.BASE_URL, self._settings)
        except Exception as e:
            return {"error": str(e)}

        try:
            return await self._client.search(
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
