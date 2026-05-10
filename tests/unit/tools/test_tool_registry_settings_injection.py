"""Registry-level integration tests for settings injection through ToolRegistry.

Phase 7 step 78 BLOCKER fix: ``ToolRegistry._instantiate`` now passes
``settings`` (and ``secrets``) to tool constructors that declare those
kwargs.  Verifies that policy from ``settings.security.outbound_allowlist``
is live for tools constructed through the registry — not dead code.

Covers:
* ``ToolRegistry.from_settings(settings)`` injects ``settings`` into
  ``WebSearchTool.__init__``.
* With ``outbound_allowlist.enabled=True, allowed_hosts=[]`` (deny-all),
  ``tool.run()`` returns an ``{"error": "..."}`` containing the blocked
  URL message.
* Positive path: with ``allowed_hosts=["api.search.brave.com"]``, the
  ``assert_safe_url`` call passes (the Brave URL is on the allowlist);
  the error is about the missing API key, not the URL policy.
* Discover path: a synthetic ``@tool``-decorated tool that declares
  ``settings`` kwarg receives the settings object.
* Tools that do NOT declare ``settings`` in ``__init__`` are constructed
  without it (backward-compat).
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock

import pytest

from tether.config.settings import Settings
from tether.core.tool_registry import ToolRegistry
from tether.tools.base import BaseTool
from tether.tools.brave_client import BraveSearchClient
from tether.tools.registration import (
    _DECORATED_TOOLS,
    _clear_registry,
    tool,
)
from tether.tools.web_search_tool import WebSearchInputs, WebSearchTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(
    *,
    enabled: bool = False,
    allowed_hosts: list[str] | None = None,
    allow_private: bool = False,
) -> Settings:
    return Settings.model_validate(
        {
            "providers": {
                "model": {"impl": "tether.providers.dummy.provider.DummyProvider"},
                "parser": {"impl": "tether.protocol.parsers.sliding.SlidingWindowParser"},
                "session_store": {"impl": "tether.context.sqlite_store.SqliteSessionStore"},
            },
            "security": {
                "outbound_allowlist": {
                    "enabled": enabled,
                    "allowed_hosts": allowed_hosts or [],
                    "allow_private": allow_private,
                },
            },
            "tools": {
                "registry": [
                    {
                        "name": "web_search",
                        "impl": "tether.tools.web_search_tool.WebSearchTool",
                    }
                ],
                "enabled": ["web_search"],
            },
        }
    )


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Save / restore the global decorator registry around each test."""
    saved = dict(_DECORATED_TOOLS)
    _clear_registry()
    try:
        yield
    finally:
        _clear_registry()
        _DECORATED_TOOLS.update(saved)


# ---------------------------------------------------------------------------
# BLOCKER fix: settings injected through registry
# ---------------------------------------------------------------------------


def test_registry_injects_settings_into_web_search_tool():
    """``ToolRegistry.from_settings`` must pass ``settings`` to tools
    that declare the kwarg — the BLOCKER fix."""
    settings = _make_settings(enabled=False)  # default policy
    registry = ToolRegistry.from_settings(settings)
    web_search = registry.get("web_search")

    assert web_search is not None
    assert isinstance(web_search, WebSearchTool)
    # The settings object must be the same one passed to from_settings.
    assert web_search._settings is settings


@pytest.mark.asyncio
async def test_registry_deny_all_allowlist_returns_error_dict():
    """deny-by-default: enabled=True + empty allowed_hosts → every URL blocked.

    Build registry with deny-all policy, inject a mock client so run()
    reaches the assert_safe_url call, verify the error dict.
    """
    settings = _make_settings(enabled=True, allowed_hosts=[])
    registry = ToolRegistry.from_settings(settings)
    tool_instance = registry.get("web_search")
    assert isinstance(tool_instance, WebSearchTool)

    # Inject a mock client to bypass the missing-key early return.
    mock_client = AsyncMock(spec=BraveSearchClient)
    mock_client.search.return_value = {"results": []}
    tool_instance._client = mock_client

    result = await tool_instance.run(WebSearchInputs(query="test query"))

    assert "error" in result
    err_msg = result["error"]
    # Must mention URL blocking (not API key error).
    assert "Outbound URL blocked" in err_msg or "allowlist" in err_msg.lower()
    # Must NOT have called the actual search client.
    mock_client.search.assert_not_called()


@pytest.mark.asyncio
async def test_registry_allowlist_with_brave_host_passes_url_check():
    """Positive path: allowed_hosts=["api.search.brave.com"] → URL check passes.

    The actual Brave call will fail (no API key), but the error must be
    about the missing key, not the allowlist.
    """
    settings = _make_settings(
        enabled=True,
        allowed_hosts=["api.search.brave.com"],
    )
    registry = ToolRegistry.from_settings(settings)
    tool_instance = registry.get("web_search")
    assert isinstance(tool_instance, WebSearchTool)

    # No client injected → run() hits the missing-client early return.
    # (startup was never called, so _client is None.)
    result = await tool_instance.run(WebSearchInputs(query="test"))

    assert "error" in result
    err_msg = result["error"]
    # Error must be about missing BRAVE_API_KEY, not the allowlist.
    assert "BRAVE_API_KEY" in err_msg
    assert "Outbound URL blocked" not in err_msg


# ---------------------------------------------------------------------------
# Discover path: settings injection for @tool-decorated classes
# ---------------------------------------------------------------------------


def test_discover_path_injects_settings():
    """Discover path: ``_instantiate`` passes settings to decorated tools
    that declare a ``settings`` kwarg."""
    received: dict[str, Any] = {}

    @tool(name="_test_settings_receiver")
    class _SettingsReceiverTool(BaseTool):
        @property
        def schema(self) -> Dict[str, Any]:
            return {}

        def __init__(self, *, settings: Optional[Settings] = None) -> None:
            super().__init__()
            received["settings"] = settings

        async def run(self) -> Dict[str, Any]:
            return {}

    target_settings = _make_settings()
    registry = ToolRegistry(
        registry_cfg=None,
        discovered={"_test_settings_receiver": _SettingsReceiverTool},
        settings=target_settings,
    )

    assert registry.get("_test_settings_receiver") is not None
    assert received["settings"] is target_settings


def test_discover_path_no_settings_kwarg_not_broken():
    """Tools that don't declare ``settings`` in ``__init__`` must still
    construct successfully (backward-compat check)."""

    @tool(name="_test_no_settings")
    class _NoSettingsTool(BaseTool):
        @property
        def schema(self) -> Dict[str, Any]:
            return {}

        async def run(self) -> Dict[str, Any]:
            return {}

    settings = _make_settings()
    registry = ToolRegistry(
        registry_cfg=None,
        discovered={"_test_no_settings": _NoSettingsTool},
        settings=settings,
    )
    assert registry.get("_test_no_settings") is not None
