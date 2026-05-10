"""configure_logging foundation tests.

Phase 7 step 67 — tether_service/core/logging.py coverage.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from tether.core.logging import (
    configure_logging,
    reset_logging_for_tests,
    logger,
)
from tether.config.settings import (
    Settings,
    ObservabilitySettings,
    LogsSettings,
    LogFileSettings,
)


@pytest.fixture(autouse=True)
def _reset_logging():
    """Always reset the idempotency flag before and after each test."""
    reset_logging_for_tests()
    yield
    reset_logging_for_tests()


def _make_settings(tmp_path: Path, *, file_path: str | None = None, console: bool = False, file_enabled: bool = True) -> Settings:
    """Build a minimal Settings object for logging tests."""
    from tether.config.settings import load_settings
    # Use load_settings to get the full defaults, then we test the observability sub-model
    # by constructing ObservabilitySettings directly.
    base = load_settings()
    # Pydantic v2 frozen models: use model_copy to create modified version
    log_file = LogFileSettings(enabled=file_enabled, path=file_path)
    logs = LogsSettings(file=log_file, console=console)
    obs = ObservabilitySettings(logs=logs)
    return base.model_copy(update={"observability": obs})


def test_configure_logging_idempotent(tmp_path: Path) -> None:
    """Calling configure_logging twice is safe — no duplicate handlers added."""
    log_path = tmp_path / "logs" / "tether.jsonl"
    settings = _make_settings(tmp_path, file_path=str(log_path))
    configure_logging(settings)
    handler_count_first = len(logging.getLogger().handlers)

    configure_logging(settings)  # second call must be a no-op
    handler_count_second = len(logging.getLogger().handlers)

    assert handler_count_first == handler_count_second


def test_configure_logging_creates_log_file(tmp_path: Path) -> None:
    """When file.enabled=True, the log file is created on first emission."""
    log_path = tmp_path / "logs" / "tether.jsonl"
    settings = _make_settings(tmp_path, file_path=str(log_path))
    configure_logging(settings)

    logger.info("test_message", foo="bar")

    # Force flush
    for h in logging.getLogger().handlers:
        h.flush()

    assert log_path.exists()
    contents = log_path.read_text(encoding="utf-8")
    assert "test_message" in contents


def test_configure_logging_redacts_secrets(tmp_path: Path) -> None:
    """RedactingFilter scrubs Bearer tokens from emitted log lines."""
    log_path = tmp_path / "logs" / "tether.jsonl"
    settings = _make_settings(tmp_path, file_path=str(log_path))
    configure_logging(settings)

    stdlib_logger = logging.getLogger("tether.test")
    stdlib_logger.info("Authorization: Bearer secret_token_xxxxxxxxxxxxxxx")

    for h in logging.getLogger().handlers:
        h.flush()

    contents = log_path.read_text(encoding="utf-8")
    assert "secret_token_xxxxxxxxxxxxxxx" not in contents
    assert "Bearer ***REDACTED***" in contents


def test_default_path_uses_platformdirs(tmp_path: Path) -> None:
    """When file.path is None, log path resolves via platformdirs (not CWD)."""
    settings = _make_settings(tmp_path, file_path=None)
    configure_logging(settings)

    file_handlers = [h for h in logging.getLogger().handlers if hasattr(h, "baseFilename")]
    assert len(file_handlers) >= 1
    base = file_handlers[0].baseFilename

    assert "Tether" in base
    assert base.endswith("tether.jsonl")


def test_logger_works_before_configure() -> None:
    """Module-level logger import works without prior configure_logging call."""
    # configure_logging has NOT been called (autouse fixture only resets the flag)
    from tether.core.logging import logger as fresh_logger
    # Should not raise — structlog lazy-binds
    fresh_logger.info("pre_config_message")


def test_log_file_disabled_creates_no_file(tmp_path: Path) -> None:
    """When file.enabled=False, no log file is created."""
    log_path = tmp_path / "logs" / "tether.jsonl"
    settings = _make_settings(tmp_path, file_path=str(log_path), file_enabled=False)
    configure_logging(settings)

    logger.info("test_message")

    assert not log_path.exists()
