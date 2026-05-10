"""FIX 6 — stdlib ``logger.exception()`` traceback redaction.

Verifies that secrets embedded in exception messages or tracebacks are scrubbed
before reaching any handler's formatted output. ``RedactingFilter`` only mutates
``record.msg``; the exception text is appended later by
:meth:`logging.Formatter.format` via :meth:`formatException`. The
``_RedactingFormatter`` wrapping subclass redacts the FINAL formatted string so
both message and traceback are scrubbed.

Phase 7 RD followups bundle (FIX 6).
"""
from __future__ import annotations

import io
import logging

import pytest

from tether_service.config.settings import load_settings
from tether_service.core.logging import (
    _RedactingFormatter,
    configure_logging,
    reset_logging_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset the configure_logging idempotency flag before/after each test."""
    reset_logging_for_tests()
    yield
    reset_logging_for_tests()
    # Also clean up any handlers we attached to the root logger.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)


def _make_settings_no_file(tmp_path) -> "object":
    """Build a Settings instance whose logging writes only to stderr.

    Uses env-var overrides because Settings sub-models are pydantic-frozen.
    """
    return load_settings(
        env={
            "TETHER__OBSERVABILITY__LOGS__LEVEL": "DEBUG",
            "TETHER__OBSERVABILITY__LOGS__FORMAT": "text",
            "TETHER__OBSERVABILITY__LOGS__CONSOLE": "true",
            "TETHER__OBSERVABILITY__LOGS__FILE__ENABLED": "false",
        }
    )


def test_redacting_formatter_scrubs_traceback_directly():
    """Unit test on _RedactingFormatter: tracebacks containing a Bearer token
    are redacted in the final formatted output."""
    formatter = _RedactingFormatter("%(message)s")
    logger = logging.getLogger("test_redacting_formatter")
    logger.setLevel(logging.DEBUG)

    try:
        raise RuntimeError("auth failed: Bearer abcdef0123456789zzzz")
    except RuntimeError:
        import sys
        record = logger.makeRecord(
            logger.name,
            logging.ERROR,
            __file__,
            0,
            "request failed",
            None,
            sys.exc_info(),
        )

    formatted = formatter.format(record)

    # The Bearer token is in exc_info, which Formatter appends AFTER msg.
    assert "Bearer abcdef0123456789zzzz" not in formatted
    assert "Bearer ***REDACTED***" in formatted
    # Traceback structure preserved
    assert "Traceback" in formatted
    assert "RuntimeError" in formatted


def test_logger_exception_traceback_redacted_via_handler(monkeypatch, tmp_path):
    """End-to-end: configure_logging installs _RedactingFormatter on a stream
    handler. ``logger.exception(...)`` whose exception carries a secret has
    that secret scrubbed in the captured handler output."""
    settings = _make_settings_no_file(tmp_path)

    # Redirect stderr so configure_logging's StreamHandler writes to a buffer
    # we can inspect. configure_logging captures sys.stderr at call time, so
    # set the patch BEFORE calling it.
    captured = io.StringIO()
    monkeypatch.setattr("sys.stderr", captured)

    configure_logging(settings)

    test_logger = logging.getLogger("test_traceback_redaction")
    try:
        # The URL with api_key= matches a redaction pattern. Raising it from
        # an exception ensures it lands in the traceback (not just the msg).
        raise ValueError(
            "outbound failed: https://api.example.com/v1/search?api_key=SECRETKEY12345"
        )
    except ValueError:
        test_logger.exception("upstream call failed")

    output = captured.getvalue()
    # Sanity: handler ran and emitted something
    assert "upstream call failed" in output
    assert "Traceback" in output
    assert "ValueError" in output
    # Secret scrubbed in BOTH the msg path and the traceback path
    assert "SECRETKEY12345" not in output
    assert "***REDACTED***" in output


def test_logger_error_with_bearer_token_in_message_redacted(monkeypatch, tmp_path):
    """Bearer tokens in the formatted msg (no exc_info) are still redacted —
    sanity-check that adding the formatter didn't break the existing
    RedactingFilter path."""
    settings = _make_settings_no_file(tmp_path)

    captured = io.StringIO()
    monkeypatch.setattr("sys.stderr", captured)

    configure_logging(settings)

    test_logger = logging.getLogger("test_message_redaction")
    test_logger.error("call failed: Bearer abcdef0123456789zzzz")

    output = captured.getvalue()
    assert "Bearer abcdef0123456789zzzz" not in output
    assert "Bearer ***REDACTED***" in output
