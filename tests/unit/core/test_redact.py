"""RedactingFilter unit tests for known token/secret patterns.

Phase 7 step 67 — tether_service/core/redact.py coverage.
"""
from __future__ import annotations

import pytest

from tether_service.core.redact import redact_text


def test_redact_bearer_token():
    """Bearer tokens are redacted, preserving the 'Bearer ' prefix label."""
    text = "Authorization: Bearer abcdef123456789"
    redacted = redact_text(text)
    assert "abcdef123456789" not in redacted
    assert "Bearer ***REDACTED***" in redacted


def test_redact_url_query_key():
    """?key=value or &key=value redacted; key= prefix preserved."""
    text = "https://api.example.com/search?key=secret_token_xxx&q=hello"
    redacted = redact_text(text)
    assert "secret_token_xxx" not in redacted
    assert "key=***REDACTED***" in redacted
    assert "q=hello" in redacted


def test_redact_json_api_key():
    """JSON 'api_key':'value' field redacted; other fields preserved."""
    text = '{"api_key": "abc123def456ghi789", "model": "test"}'
    redacted = redact_text(text)
    assert "abc123def456ghi789" not in redacted
    assert '"api_key": "***REDACTED***"' in redacted
    assert '"model": "test"' in redacted


def test_redact_brave_api_key_env_style():
    """BRAVE_API_KEY=value (env-var style) redacted."""
    text = "BRAVE_API_KEY=abc123def456ghi789012"
    redacted = redact_text(text)
    assert "abc123def456ghi789012" not in redacted
    assert "BRAVE_API_KEY=***REDACTED***" in redacted


def test_redact_brave_api_key_json():
    """JSON 'BRAVE_API_KEY' field (case-insensitive) redacted."""
    text = '{"BRAVE_API_KEY": "secret-token-99999"}'
    redacted = redact_text(text)
    assert "secret-token-99999" not in redacted


def test_redact_short_bearer_not_matched():
    """Bearer with very short token (< 8 chars) is not redacted."""
    text = "Bearer ab"
    redacted = redact_text(text)
    assert redacted == text


def test_redact_passthrough_non_string():
    """Non-string values pass through unchanged."""
    assert redact_text(42) == 42
    assert redact_text(None) is None
    assert redact_text({"k": "v"}) == {"k": "v"}


def test_redact_no_match_passthrough():
    """Strings with no matching patterns pass through unchanged."""
    text = "ordinary log message with no secrets"
    assert redact_text(text) == text


def test_redact_multiple_patterns_in_one_string():
    """Multiple redaction patterns in one string are all redacted."""
    text = 'Authorization: Bearer secret_token_99999 and api_key="another_secret_88888"'
    redacted = redact_text(text)
    assert "secret_token_99999" not in redacted
    assert "another_secret_88888" not in redacted


def test_redact_idempotent():
    """Redacting an already-redacted string is a no-op (no double-redaction)."""
    text = "Bearer secret_token_99999"
    once = redact_text(text)
    twice = redact_text(once)
    assert once == twice
