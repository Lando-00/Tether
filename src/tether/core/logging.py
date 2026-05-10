"""Structured logging via structlog + stdlib bridge + RedactingFilter.

Phase 7 step 67. Backwards-compat: existing
``from tether.core.logging import logger`` continues to work.
``configure_logging(settings)`` is called once at startup
(``Engine.from_settings``); tests can call it with their own settings.

Library-first invariant: importing ``tether_service`` does NOT
auto-configure logging. The module-level ``logger`` object is a lazy
structlog BoundLogger that works before ``configure_logging`` is called.

Citations: _synthesis.md §3 (observability), §4 Phase 7 step 67.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import structlog

from tether.core.redact import redact_record_message, redact_text

if TYPE_CHECKING:
    from tether.config.settings import Settings


# ---------------------------------------------------------------------------
# Package-level logger
# ---------------------------------------------------------------------------

# Usable BEFORE configure_logging is called (structlog binds lazily to
# whatever logging is configured at log time).
logger = structlog.get_logger("tether")


# ---------------------------------------------------------------------------
# Idempotency guard
# ---------------------------------------------------------------------------

_CONFIGURED = False


# ---------------------------------------------------------------------------
# RedactingFilter
# ---------------------------------------------------------------------------

class RedactingFilter(logging.Filter):
    """Stdlib filter that redacts tokens/secrets from log record messages.

    Applied to ALL handlers (file + stream) so any code path that uses
    stdlib logging.Logger benefits from the same redaction as structlog.

    Citations: _synthesis.md §3 (observability), §4 Phase 7 step 67.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        redact_record_message(record)
        return True


# ---------------------------------------------------------------------------
# RedactingFormatter
# ---------------------------------------------------------------------------

class _RedactingFormatter(logging.Formatter):
    """stdlib Formatter that redacts the FINAL formatted output (incl. tracebacks).

    Phase 7 RD followup (FIX 6): :class:`RedactingFilter` only mutates
    ``record.msg``, but :meth:`logging.Formatter.formatException` is appended
    AFTER the filter runs (in :meth:`format`). Tracebacks frequently embed
    secrets — e.g., ``urllib.error.HTTPError`` includes the offending URL with
    its query params, and downstream library exceptions sometimes carry the
    Bearer token they failed on. Without this formatter, those secrets would
    leak into the file/console handlers despite RedactingFilter.

    This formatter runs :func:`redact_text` on the entire formatted string
    (message + traceback + stack info) so secrets are scrubbed regardless of
    which path placed them there. Idempotent with RedactingFilter (running
    redaction twice on already-redacted text is a no-op).
    """

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return redact_text(formatted)


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------

def configure_logging(settings: Optional["Settings"] = None) -> None:
    """Configure structlog + stdlib logging once.

    Idempotent: calling more than once is safe — subsequent calls are
    ignored. Tests can call ``reset_logging_for_tests()`` then re-invoke.

    The stdlib bridge ensures both ``structlog.get_logger(...)`` AND
    ``logging.getLogger(...)`` calls flow through the same handlers and
    have the same RedactingFilter applied.

    Citations: _synthesis.md §3 (observability), §4 Phase 7 step 67.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Resolve effective settings
    if settings is None:
        from tether.config.settings import load_settings
        settings = load_settings()

    obs = settings.observability
    log_settings = obs.logs

    level_name = log_settings.level.upper()
    level = getattr(logging, level_name, logging.INFO)

    # Resolve file path (None -> platformdirs default)
    file_path: Optional[Path] = None
    if log_settings.file.enabled:
        file_path = _resolve_log_file_path(log_settings.file.path)

    # Configure stdlib root logger — reset any prior config first so this
    # is idempotent within a test suite that resets _CONFIGURED.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    handlers: list[logging.Handler] = []

    # Phase 7 RD followup (FIX 6): use _RedactingFormatter so the final
    # formatted output (including tracebacks appended by formatException)
    # passes through redact_text. RedactingFilter alone only scrubs
    # record.msg; it does not catch exc_info text.
    formatter = _RedactingFormatter("%(message)s")

    if log_settings.console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(level)
        stream_handler.addFilter(RedactingFilter())
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    if file_path is not None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.addFilter(RedactingFilter())
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    for h in handlers:
        root.addHandler(h)

    # structlog configuration.
    # The stdlib LoggerFactory bridges structlog → stdlib so that
    # structlog.get_logger(...).info(...) flows through the handlers above.
    if log_settings.format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True

    # Lazy install OTel adapter if enabled.
    if settings.observability.otel.enabled:
        from tether.observability.otel_adapter import install_otel_adapter
        install_otel_adapter(settings)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_log_file_path(configured: Optional[str]) -> Path:
    """Resolve log file path, defaulting to platformdirs user_log_dir.

    Lazy-imports platformdirs so that ``import tether.core.logging``
    does not trigger it unless actually needed.
    """
    if configured:
        return Path(configured).expanduser().resolve()

    from platformdirs import user_log_dir
    log_dir = Path(user_log_dir("Tether", appauthor=False))
    return log_dir / "tether.jsonl"


def reset_logging_for_tests() -> None:
    """Test-helper: clear the configure_logging idempotency flag.

    Used by test fixtures that need to reconfigure logging with custom
    settings. NOT part of the public library API.
    """
    global _CONFIGURED
    _CONFIGURED = False


__all__ = [
    "logger",
    "configure_logging",
    "RedactingFilter",
    "reset_logging_for_tests",
]
