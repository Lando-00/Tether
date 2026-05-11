"""P0-I regression: OTel adapter requires experimental_acknowledged to install.

Tribunal §3 P0-19 (A9-F2, A9-F3). The Phase 7 OTel adapter emits zero-duration
point spans at ``*.end`` time only and is unsuitable as a tracing-UI source.
Per Defender T2 we gate (don't delete) the adapter behind an explicit ack flag
and emit a startup warning when it installs.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from pydantic import ValidationError


def test_otel_enabled_without_ack_raises_validation_error():
    """observability.otel.enabled=True without experimental_acknowledged=True must fail."""
    from tether.config.settings import OTelSettings

    with pytest.raises((ValidationError, ValueError)) as exc_info:
        OTelSettings(enabled=True, experimental_acknowledged=False)
    msg = str(exc_info.value)
    assert "experimental_acknowledged" in msg, (
        f"Error message must mention the flag: {msg!r}"
    )


def test_otel_enabled_with_ack_succeeds():
    """experimental_acknowledged=True allows enabled=True."""
    from tether.config.settings import OTelSettings

    cfg = OTelSettings(enabled=True, experimental_acknowledged=True)
    assert cfg.enabled is True
    assert cfg.experimental_acknowledged is True


def test_otel_disabled_no_ack_required():
    """When OTel is off, the ack flag is irrelevant."""
    from tether.config.settings import OTelSettings

    cfg = OTelSettings(enabled=False)
    assert cfg.enabled is False
    assert cfg.experimental_acknowledged is False


def test_otel_disabled_explicit_no_ack_still_ok():
    """enabled=False, experimental_acknowledged=False is a valid combination."""
    from tether.config.settings import OTelSettings

    cfg = OTelSettings(enabled=False, experimental_acknowledged=False)
    assert cfg.enabled is False


def test_install_otel_adapter_emits_experimental_warning(capsys, caplog):
    """Calling install_otel_adapter() must log a warning about the limitation."""
    pytest.importorskip("opentelemetry")

    from tether.config.settings import OTelSettings
    from tether.observability.otel_adapter import (
        install_otel_adapter,
        reset_otel_adapter_for_tests,
    )

    reset_otel_adapter_for_tests()

    otel_cfg = OTelSettings(
        enabled=True,
        experimental_acknowledged=True,
        exporter="console",
    )
    obs = SimpleNamespace(otel=otel_cfg)
    settings = SimpleNamespace(observability=obs)

    caplog.set_level(logging.WARNING)
    try:
        install_otel_adapter(settings)
    except Exception:
        # Some envs may not have an exporter wired; we only care about the
        # warning emission, which happens before any exporter setup.
        pass
    finally:
        reset_otel_adapter_for_tests()

    # structlog may render to stdout/stderr (default ConsoleRenderer) OR through
    # the stdlib bridge into caplog depending on how `configure_logging` was set
    # up earlier in the session. Accept either source.
    captured = capsys.readouterr()
    text = (
        captured.out
        + "\n"
        + captured.err
        + "\n"
        + "\n".join(r.getMessage() for r in caplog.records)
    )
    assert (
        "experimental" in text.lower()
        or "zero-duration" in text.lower()
        or "P0-19" in text
        or "otel_adapter.experimental" in text
    ), f"install_otel_adapter() did not emit an experimental-status warning: {text!r}"
