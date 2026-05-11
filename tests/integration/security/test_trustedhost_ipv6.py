"""P0-B2 regression: TrustedHost default-on, IPv6 loopback accepted."""
from tether.config.settings import load_settings


def test_trustedhost_enabled_by_default():
    s = load_settings()
    assert s.security.trusted_host.enabled is True


def test_ipv6_loopback_in_default_allowed_hosts():
    s = load_settings()
    hosts = s.security.trusted_host.allowed_hosts
    assert "[::1]" in hosts or "::1" in hosts
