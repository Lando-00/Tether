"""Outbound URL safety helper.

Phase 7 step 78. Synthesis Section 3 (security), B5 steps 9-10.

Tools that fetch external content (e.g., web_search) call assert_safe_url(url, settings)
before making the request. If the configured policy rejects the URL,
raises OutboundUrlBlocked with a structured reason.

Design notes:
- When ``settings`` is ``None``, only always-on checks apply (scheme + host
  presence). Private-IP and allowlist enforcement require a Settings object.
- DNS resolution is deliberately NOT performed — only IP literals in private
  ranges are blocked. Hostname-based SSRF (e.g., ``http://localhost/``) would
  require DNS-level interception, which is out of scope for MVP.
- ``fnmatch`` is used for wildcard host matching. It treats ``*`` as
  "any characters" within the pattern, so ``*.example.com`` matches
  ``api.example.com`` but NOT ``example.com`` (no leading dot consumed).
"""
from __future__ import annotations

import fnmatch
import ipaddress
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlparse

from tether.core.errors import TetherError

if TYPE_CHECKING:
    from tether.config.settings import Settings


class OutboundUrlBlocked(TetherError):
    """Raised when assert_safe_url rejects an outbound URL.

    Attributes:
        url: The URL that was rejected.
        reason: Human-readable explanation of why the URL was blocked.
    """

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"Outbound URL blocked ({reason}): {url}")
        self.url = url
        self.reason = reason


_ALLOWED_SCHEMES = frozenset({"http", "https"})


def _is_private_address(host: str) -> bool:
    """Return True iff ``host`` is an IP literal in a private/loopback range.

    Returns False for hostnames (no DNS resolution is performed).
    Covers: loopback (127.0.0.0/8), link-local (169.254.0.0/16),
    RFC 1918 (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16),
    reserved, and multicast ranges.
    """
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
    )


def _matches_allowlist(host: str, patterns: list[str]) -> bool:
    """Return True iff ``host`` matches any literal or wildcard pattern.

    Matching is case-insensitive. ``fnmatch`` is used for wildcard
    patterns — ``*`` matches any sequence of characters, so
    ``*.example.com`` matches ``api.example.com`` but NOT ``example.com``.
    """
    host_lower = host.lower()
    for pattern in patterns:
        p = pattern.lower()
        if p == host_lower:
            return True
        if fnmatch.fnmatch(host_lower, p):
            return True
    return False


def assert_safe_url(url: str, settings: Optional["Settings"] = None) -> None:
    """Raise OutboundUrlBlocked if ``url`` violates the configured policy.

    Always-on checks (applied regardless of ``settings``):
      - URL must parse cleanly and have a non-empty host.
      - Scheme must be ``http`` or ``https``.

    Settings-gated checks (only when ``settings`` is provided):
      - Private/loopback IP literals are blocked unless
        ``settings.security.outbound_allowlist.allow_private=True``.
      - When ``settings.security.outbound_allowlist.enabled=True``,
        the host must match one of the patterns in ``allowed_hosts``
        (empty ``allowed_hosts`` with ``enabled=True`` → deny-by-default).

    Args:
        url: The URL to validate (e.g., ``"https://api.example.com/search"``).
        settings: Optional Settings instance. When ``None``, only the
            always-on checks are applied (current Tether default: single-user
            local deployment with no outbound hardening).

    Raises:
        OutboundUrlBlocked: If the URL fails any check.
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise OutboundUrlBlocked(url, f"URL did not parse: {exc}") from exc

    if not parsed.scheme:
        raise OutboundUrlBlocked(url, "no scheme (must be http or https)")
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise OutboundUrlBlocked(url, f"scheme '{parsed.scheme}' not allowed (must be http or https)")
    if not parsed.hostname:
        raise OutboundUrlBlocked(url, "no host")

    host = parsed.hostname

    if settings is None:
        return

    cfg = settings.security.outbound_allowlist

    if not cfg.allow_private and _is_private_address(host):
        raise OutboundUrlBlocked(url, f"private/loopback address '{host}' blocked (set allow_private=True to permit)")

    if cfg.enabled:
        if not cfg.allowed_hosts:
            raise OutboundUrlBlocked(
                url,
                "allowlist enabled but allowed_hosts is empty — deny-by-default",
            )
        if not _matches_allowlist(host, cfg.allowed_hosts):
            raise OutboundUrlBlocked(url, f"host '{host}' not in allowlist")
