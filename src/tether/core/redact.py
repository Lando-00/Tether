"""PII / secret redaction for log records.

Phase 7 step 67. Used by RedactingFilter (stdlib) AND by structlog
processors (configured in configure_logging).

Redaction patterns (initial; expand per B5 review):
  - Brave API key strings (BRAVE_API_KEY=... env-style or "key:..." JSON)
  - Bearer tokens
  - URL query-param api keys

This module is stdlib-only so it can be imported eagerly by core/logging.py.

Citations: _synthesis.md §3 (observability), §4 Phase 7 step 67.
"""
from __future__ import annotations

import re
from typing import Any

# Patterns that match secrets/tokens (broad; err on redacting too much).
_TOKEN_PATTERNS = [
    # OpenAI / Anthropic / generic Bearer tokens (8+ char value)
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{8,}"),
    # API key in URL params: ?key=xxx or &key=xxx
    re.compile(r"([?&](?:key|api_key|apikey|token)=)([A-Za-z0-9._\-]{8,})"),
    # JSON-style key fields: "api_key":"xxx", "BRAVE_API_KEY":"xxx"
    re.compile(
        r'("(?:api_key|apikey|token|brave_api_key|api-key|x_subscription_token)"\s*:\s*")([^"]+)(")',
        re.IGNORECASE,
    ),
    # Env-style: BRAVE_API_KEY=xxx or API_KEY=xxx (no quotes)
    re.compile(r"\b(BRAVE_API_KEY|API_KEY|TOKEN)\s*[:=]\s*([A-Za-z0-9._\-]{8,})"),
    # Bare key=quoted-value: api_key="xxx" or api_key='xxx' (lowercase, any quote style)
    re.compile(
        r"\b((?:api_key|apikey|token|api-key|brave_api_key)\s*=\s*)([\"'])([A-Za-z0-9._\-]{8,})([\"'])",
        re.IGNORECASE,
    ),
]


def _redact_token(text: str) -> str:
    """Apply all token-redaction regex patterns. Returns redacted text."""
    redacted = text
    redacted = _TOKEN_PATTERNS[0].sub("Bearer ***REDACTED***", redacted)
    redacted = _TOKEN_PATTERNS[1].sub(r"\1***REDACTED***", redacted)
    redacted = _TOKEN_PATTERNS[2].sub(r"\1***REDACTED***\3", redacted)
    redacted = _TOKEN_PATTERNS[3].sub(r"\1=***REDACTED***", redacted)
    redacted = _TOKEN_PATTERNS[4].sub(r"\1\2***REDACTED***\4", redacted)
    return redacted


def redact_text(text: Any) -> Any:
    """Redact obvious tokens from a string-ish value.

    Pass-through for non-strings so logging non-string objects (e.g.,
    integers, dicts) doesn't crash.
    """
    if not isinstance(text, str):
        return text
    return _redact_token(text)


def redact_record_message(record: Any) -> None:
    """Mutate a logging.LogRecord's ``msg`` field in-place to redact tokens.

    Used by RedactingFilter (stdlib bridge). Keep idempotent.
    """
    try:
        msg = record.getMessage()
        if isinstance(msg, str):
            redacted = _redact_token(msg)
            if redacted != msg:
                record.msg = redacted
                record.args = ()  # already formatted; no further substitution
    except Exception:
        # Never crash the logger
        pass


__all__ = ["redact_text", "redact_record_message"]
