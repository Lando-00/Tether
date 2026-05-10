"""Tests for tether_service.security.outbound (assert_safe_url + OutboundUrlBlocked).

Phase 7 step 78. Synthesis Section 3 (security), B5 steps 9-10.

Coverage:
  - Always-on checks: scheme, host presence (no Settings needed).
  - Private/loopback IP blocking (with Settings, default policy).
  - allow_private=True permits private IPs.
  - Allowlist enforcement: match, miss, empty (deny-by-default).
  - Wildcard host matching via fnmatch.
  - Known gap: hostname-based SSRF (e.g., 'localhost') is NOT blocked
    because DNS resolution is not performed — only IP literals are checked.
    See test_localhost_not_blocked_by_ip_check for documentation.
"""
from __future__ import annotations

import pytest

from tether_service.config.settings import (
    OutboundAllowlistSettings,
    SecuritySettings,
    Settings,
)
from tether_service.security.outbound import OutboundUrlBlocked, assert_safe_url


# ---------------------------------------------------------------------------
# Helpers to build minimal Settings with specific allowlist config
# ---------------------------------------------------------------------------


def _make_settings(
    *,
    enabled: bool = False,
    allowed_hosts: list[str] | None = None,
    allow_private: bool = False,
) -> Settings:
    """Build a minimal Settings with the given outbound_allowlist values.

    Uses Settings.model_validate to avoid touching YAML / env / providers.
    """
    return Settings.model_validate(
        {
            "providers": {
                "model": {"impl": "tether_service.providers.dummy.provider.DummyProvider"},
                "parser": {"impl": "tether_service.protocol.parsers.sliding.SlidingWindowParser"},
                "session_store": {"impl": "tether_service.context.sqlite_store.SqliteSessionStore"},
            },
            "security": {
                "outbound_allowlist": {
                    "enabled": enabled,
                    "allowed_hosts": allowed_hosts or [],
                    "allow_private": allow_private,
                },
            },
        }
    )


# Reusable Settings with default policy (enabled=False, allow_private=False).
@pytest.fixture
def settings_default() -> Settings:
    return _make_settings()


@pytest.fixture
def settings_allow_private() -> Settings:
    return _make_settings(allow_private=True)


# ---------------------------------------------------------------------------
# Always-on checks (no Settings required)
# ---------------------------------------------------------------------------


def test_default_off_valid_url_passes():
    """1. Default-off: valid https URL with no Settings — passes."""
    assert_safe_url("https://example.com")  # must not raise


def test_default_off_blocks_file_scheme():
    """2. Default-off blocks bad scheme: file:// raises OutboundUrlBlocked."""
    with pytest.raises(OutboundUrlBlocked) as exc_info:
        assert_safe_url("file:///etc/passwd")
    assert "scheme" in exc_info.value.reason
    assert exc_info.value.url == "file:///etc/passwd"


def test_default_off_blocks_ftp_scheme():
    """Bonus: ftp:// is also rejected by the always-on scheme check."""
    with pytest.raises(OutboundUrlBlocked) as exc_info:
        assert_safe_url("ftp://files.example.com/data.csv")
    assert "scheme" in exc_info.value.reason


def test_default_off_blocks_no_scheme():
    """3. Default-off blocks no-scheme: 'example.com' raises (parsed as path, no scheme)."""
    with pytest.raises(OutboundUrlBlocked) as exc_info:
        assert_safe_url("example.com")
    assert "scheme" in exc_info.value.reason


def test_default_off_blocks_no_host():
    """4. No-host: 'https://' raises OutboundUrlBlocked (empty hostname)."""
    with pytest.raises(OutboundUrlBlocked) as exc_info:
        assert_safe_url("https://")
    assert "host" in exc_info.value.reason


def test_http_scheme_passes():
    """http:// is a valid scheme (not just https)."""
    assert_safe_url("http://example.com/api")  # must not raise


# ---------------------------------------------------------------------------
# Private/loopback IP blocking (Settings-gated, enabled by default policy)
# ---------------------------------------------------------------------------


def test_private_address_blocked_rfc1918_192(settings_default: Settings):
    """5. Private address 192.168.x.x blocked when Settings provided."""
    with pytest.raises(OutboundUrlBlocked) as exc_info:
        assert_safe_url("http://192.168.1.1/admin", settings_default)
    assert "private" in exc_info.value.reason.lower() or "loopback" in exc_info.value.reason.lower()


def test_private_address_blocked_rfc1918_10(settings_default: Settings):
    """RFC 1918 10.0.0.0/8 is blocked."""
    with pytest.raises(OutboundUrlBlocked):
        assert_safe_url("http://10.0.0.1/internal", settings_default)


def test_private_address_blocked_rfc1918_172(settings_default: Settings):
    """RFC 1918 172.16.0.0/12 is blocked."""
    with pytest.raises(OutboundUrlBlocked):
        assert_safe_url("http://172.16.0.1/", settings_default)


def test_loopback_127_blocked(settings_default: Settings):
    """6. Loopback 127.0.0.1 is blocked."""
    with pytest.raises(OutboundUrlBlocked) as exc_info:
        assert_safe_url("http://127.0.0.1/", settings_default)
    assert "127.0.0.1" in exc_info.value.reason


def test_localhost_hostname_not_blocked_by_ip_check(settings_default: Settings):
    """7. 'localhost' (hostname, not IP literal) is NOT blocked.

    Known gap: _is_private_address only checks IP literals. Resolving
    'localhost' via DNS would require a blocking call and is out of scope
    for this MVP. Callers that need hostname-level SSRF protection should
    use an allowlist (enabled=True) and omit 'localhost' from allowed_hosts.
    """
    # Must not raise — the hostname 'localhost' is not an IP literal so
    # _is_private_address returns False.  Document, don't fix here.
    assert_safe_url("http://localhost/", settings_default)  # passes (known gap)


def test_link_local_blocked(settings_default: Settings):
    """Link-local (169.254.x.x) is blocked."""
    with pytest.raises(OutboundUrlBlocked):
        assert_safe_url("http://169.254.169.254/latest/meta-data/", settings_default)


# ---------------------------------------------------------------------------
# allow_private=True permits private IPs
# ---------------------------------------------------------------------------


def test_allow_private_permits_loopback(settings_allow_private: Settings):
    """8. allow_private=True: 127.0.0.1 is allowed."""
    assert_safe_url("http://127.0.0.1/", settings_allow_private)  # must not raise


def test_allow_private_permits_rfc1918(settings_allow_private: Settings):
    """allow_private=True: 192.168.x.x is allowed."""
    assert_safe_url("http://192.168.1.100/api", settings_allow_private)  # must not raise


# ---------------------------------------------------------------------------
# Allowlist enforcement
# ---------------------------------------------------------------------------


def test_allowlist_enabled_matching_host_passes():
    """9. Allowlist enabled + matching host — passes."""
    settings = _make_settings(enabled=True, allowed_hosts=["*.example.com"])
    assert_safe_url("https://api.example.com/", settings)  # must not raise


def test_allowlist_enabled_literal_host_passes():
    """Literal host match (no wildcard) — passes."""
    settings = _make_settings(enabled=True, allowed_hosts=["api.example.com"])
    assert_safe_url("https://api.example.com/search?q=test", settings)


def test_allowlist_enabled_miss_raises():
    """10. Allowlist enabled + host not in list — raises."""
    settings = _make_settings(enabled=True, allowed_hosts=["*.example.com"])
    with pytest.raises(OutboundUrlBlocked) as exc_info:
        assert_safe_url("https://api.evil.com/", settings)
    assert "allowlist" in exc_info.value.reason.lower() or "not in" in exc_info.value.reason.lower()


def test_allowlist_enabled_empty_raises_deny_by_default():
    """11. Allowlist enabled but empty — every URL raises (deny-by-default)."""
    settings = _make_settings(enabled=True, allowed_hosts=[])
    with pytest.raises(OutboundUrlBlocked) as exc_info:
        assert_safe_url("https://example.com/", settings)
    assert "empty" in exc_info.value.reason.lower() or "deny" in exc_info.value.reason.lower()


def test_allowlist_disabled_does_not_check_host():
    """Allowlist disabled (default) — any public host passes."""
    settings = _make_settings(enabled=False)
    assert_safe_url("https://totally-random-host.io/api", settings)  # must not raise


# ---------------------------------------------------------------------------
# Wildcard matching behaviour
# ---------------------------------------------------------------------------


def test_wildcard_matches_subdomain():
    """12a. *.example.com matches api.example.com."""
    settings = _make_settings(enabled=True, allowed_hosts=["*.example.com"])
    assert_safe_url("https://api.example.com/", settings)
    assert_safe_url("https://foo.example.com/", settings)


def test_wildcard_does_not_match_apex():
    """12b. *.example.com does NOT match example.com (fnmatch gotcha).

    fnmatch('example.com', '*.example.com') → False because '*' must match
    at least one character AND there's no preceding dot for the apex domain.
    Callers that need both 'example.com' and '*.example.com' must list both.
    """
    settings = _make_settings(enabled=True, allowed_hosts=["*.example.com"])
    with pytest.raises(OutboundUrlBlocked):
        assert_safe_url("https://example.com/", settings)


def test_allowlist_requires_both_apex_and_wildcard():
    """To allow both apex + subdomains, list both patterns."""
    settings = _make_settings(
        enabled=True,
        allowed_hosts=["example.com", "*.example.com"],
    )
    assert_safe_url("https://example.com/", settings)
    assert_safe_url("https://api.example.com/", settings)


# ---------------------------------------------------------------------------
# OutboundUrlBlocked attributes
# ---------------------------------------------------------------------------


def test_exception_carries_url_and_reason():
    """OutboundUrlBlocked exposes .url and .reason attributes."""
    with pytest.raises(OutboundUrlBlocked) as exc_info:
        assert_safe_url("ftp://example.com/")
    err = exc_info.value
    assert err.url == "ftp://example.com/"
    assert "ftp" in err.reason
    assert "ftp://example.com/" in str(err)


def test_exception_is_tether_error():
    """OutboundUrlBlocked is a TetherError (fits existing error taxonomy)."""
    from tether_service.core.errors import TetherError

    with pytest.raises(TetherError):
        assert_safe_url("file:///etc/shadow")


# ---------------------------------------------------------------------------
# WebSearchTool integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_web_search_assert_safe_url_brave_endpoint_passes():
    """assert_safe_url on BraveSearchClient.BASE_URL passes with default policy."""
    from unittest.mock import AsyncMock

    from tether_service.tools.brave_client import BraveSearchClient
    from tether_service.tools.web_search_tool import WebSearchInputs, WebSearchTool

    class _FakeSecrets:
        def get(self, key: str):
            return None

    tool = WebSearchTool(secrets=_FakeSecrets(), settings=settings_default)
    # _client is None → early return before assert_safe_url; test the helper directly.
    from tether_service.security.outbound import assert_safe_url as _assert
    _assert(BraveSearchClient.BASE_URL)  # must not raise


@pytest.mark.asyncio
async def test_web_search_run_with_allowlist_mismatch_returns_error_dict():
    """When allowlist blocks the Brave endpoint, run() returns an error dict."""
    from unittest.mock import AsyncMock

    from tether_service.tools.brave_client import BraveSearchClient
    from tether_service.tools.web_search_tool import WebSearchInputs, WebSearchTool

    class _FakeSecrets:
        def get(self, key: str):
            return None

    # Settings with allowlist enabled but empty (deny-all) + allow_private to
    # avoid private-IP logic interfering.
    settings = _make_settings(enabled=True, allowed_hosts=["api.evil-host.com"])

    tool = WebSearchTool(secrets=_FakeSecrets(), settings=settings)
    # Inject a mock client so run() reaches the assert_safe_url call.
    mock_client = AsyncMock(spec=BraveSearchClient)
    mock_client.search.return_value = {"results": []}
    tool._client = mock_client

    result = await tool.run(WebSearchInputs(query="test"))
    assert "error" in result
    assert "Outbound URL blocked" in result["error"] or "allowlist" in result["error"].lower() or "not in" in result["error"].lower()
