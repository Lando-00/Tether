"""Tests for tether.observability.otel_adapter.

Phase 7 step 76 — Optional OpenTelemetry adapter.

OTel-dependent tests call ``pytest.importorskip("opentelemetry")`` inside each
test function so they're skipped individually on CI without ``tether[otel]``.
Non-OTel tests (default-off, friendly error, settings parsing) always run.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tether.observability.otel_adapter import (
    install_otel_adapter,
    reset_otel_adapter_for_tests,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(otel_overrides: dict[str, Any] | None = None):
    """Return a Settings instance with optional OTel overrides (via env vars).

    P0-I: when ``enabled=True`` is requested, auto-injects
    ``experimental_acknowledged=true`` so legacy adapter tests continue to
    exercise the adapter path without tripping the new validator.
    """
    from tether.config.settings import load_settings

    if not otel_overrides:
        return load_settings(env={})

    # P0-I cross-fix: every test that flips enabled=True must also ack the
    # experimental status, otherwise OTelSettings.model_validator raises.
    if otel_overrides.get("enabled") is True and "experimental_acknowledged" not in otel_overrides:
        otel_overrides = {**otel_overrides, "experimental_acknowledged": True}

    env: dict[str, str] = {}
    for k, v in otel_overrides.items():
        env_key = f"TETHER__OBSERVABILITY__OTEL__{k.upper()}"
        env[env_key] = str(v).lower() if isinstance(v, bool) else str(v)
    return load_settings(env=env)


@pytest.fixture(autouse=True)
def _reset_adapter():
    """Reset adapter init flag before/after each test."""
    reset_otel_adapter_for_tests()
    yield
    reset_otel_adapter_for_tests()


# ---------------------------------------------------------------------------
# Test 1: Default disabled → no-op (no OTel packages needed)
# ---------------------------------------------------------------------------


def test_default_disabled_is_noop():
    """install_otel_adapter with otel.enabled=False must not import OTel."""
    settings = _make_settings()
    assert not settings.observability.otel.enabled

    otel_before = {k for k in sys.modules if k.startswith("opentelemetry")}
    install_otel_adapter(settings)
    otel_after = {k for k in sys.modules if k.startswith("opentelemetry")}

    new_modules = otel_after - otel_before
    assert not new_modules, f"OTel was imported despite being disabled: {new_modules}"

    from tether.observability import otel_adapter
    assert not otel_adapter._initialized


# ---------------------------------------------------------------------------
# Test 2: Enabled but OTel not installed → ImportError with pip hint
# ---------------------------------------------------------------------------


def test_enabled_but_otel_not_installed_raises():
    """When enabled=True but OTel packages missing, raise ImportError with pip hint."""
    settings = _make_settings({"enabled": True})
    assert settings.observability.otel.enabled

    # Simulate missing package: patch the top-level "opentelemetry" entry to None.
    with patch.dict(sys.modules, {"opentelemetry": None}):
        with pytest.raises(ImportError, match="pip install tether\\[otel\\]"):
            install_otel_adapter(settings)


# ---------------------------------------------------------------------------
# Test 3: Enabled with console exporter (skipped without OTel)
# ---------------------------------------------------------------------------


def test_enabled_console_exporter_calls_set_tracer_provider():
    """enabled=True, exporter='console' → set_tracer_provider called with TracerProvider."""
    otel_trace = pytest.importorskip("opentelemetry.trace", reason="pip install tether[otel]")
    from opentelemetry.sdk.trace import TracerProvider

    settings = _make_settings({"enabled": True, "exporter": "console"})

    with patch.object(otel_trace, "set_tracer_provider") as mock_set:
        install_otel_adapter(settings)

    mock_set.assert_called_once()
    assert isinstance(mock_set.call_args[0][0], TracerProvider)


# ---------------------------------------------------------------------------
# Test 4: otlp_http without endpoint → ValueError
# ---------------------------------------------------------------------------


def test_enabled_otlp_http_no_endpoint_raises():
    """enabled=True, exporter='otlp_http', no endpoint → ValueError."""
    pytest.importorskip("opentelemetry", reason="pip install tether[otel]")

    settings = _make_settings({"enabled": True, "exporter": "otlp_http"})
    assert settings.observability.otel.exporter_endpoint is None

    with pytest.raises(ValueError, match="requires exporter_endpoint"):
        install_otel_adapter(settings)


def test_enabled_otlp_http_with_endpoint():
    """enabled=True, exporter='otlp_http', endpoint set → no error."""
    otel_trace = pytest.importorskip("opentelemetry.trace", reason="pip install tether[otel]")
    pytest.importorskip(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        reason="pip install tether[otel]",
    )

    settings = _make_settings(
        {
            "enabled": True,
            "exporter": "otlp_http",
            "exporter_endpoint": "http://localhost:4318/v1/traces",
        }
    )
    with patch.object(otel_trace, "set_tracer_provider"):
        install_otel_adapter(settings)  # must not raise


# ---------------------------------------------------------------------------
# Test 5: Idempotent double call
# ---------------------------------------------------------------------------


def test_idempotent_double_call():
    """Calling install_otel_adapter twice is safe (no second initialization)."""
    otel_trace = pytest.importorskip("opentelemetry.trace", reason="pip install tether[otel]")

    settings = _make_settings({"enabled": True, "exporter": "console"})

    with patch.object(otel_trace, "set_tracer_provider") as mock_set:
        install_otel_adapter(settings)
        install_otel_adapter(settings)  # must be no-op

    assert mock_set.call_count == 1, "set_tracer_provider called more than once"


# ---------------------------------------------------------------------------
# Test 6: Settings parsing (no OTel needed)
# ---------------------------------------------------------------------------


def test_default_settings_otel_disabled():
    """Default settings must have otel.enabled=False and sensible defaults."""
    s = _make_settings()
    assert s.observability.otel.enabled is False
    assert s.observability.otel.exporter == "console"
    assert s.observability.otel.sample_rate == 1.0
    assert s.observability.otel.service_name == "tether"
    assert s.observability.otel.exporter_endpoint is None


def test_yaml_override_settings_parsing(tmp_path: Path):
    """YAML overlay with otel block parses to correct typed values."""
    from importlib import resources

    from tether.config.settings import load_settings

    overlay = tmp_path / "override.yml"
    overlay.write_text(
        "observability:\n"
        "  otel:\n"
        "    enabled: true\n"
        "    experimental_acknowledged: true\n"
        "    sample_rate: 0.1\n"
        "    service_name: my-svc\n",
        encoding="utf-8",
    )
    default_yaml = Path(str(resources.files("tether.config") / "default.yml"))
    cfg = load_settings(default_yaml=default_yaml, overlay_yaml=overlay, env={})

    assert cfg.observability.otel.enabled is True
    assert cfg.observability.otel.sample_rate == pytest.approx(0.1)
    assert cfg.observability.otel.service_name == "my-svc"


# ---------------------------------------------------------------------------
# Test 7: configure_logging with default settings must NOT import OTel
# ---------------------------------------------------------------------------


def test_configure_logging_default_does_not_import_otel():
    """configure_logging(default_settings) must not import any OTel modules."""
    from tether.core.logging import configure_logging, reset_logging_for_tests

    reset_logging_for_tests()

    settings = _make_settings()
    otel_before = {k for k in sys.modules if k.startswith("opentelemetry")}

    configure_logging(settings)

    otel_after = {k for k in sys.modules if k.startswith("opentelemetry")}
    new_modules = otel_after - otel_before
    assert not new_modules, (
        f"OTel was imported by configure_logging despite being disabled: {new_modules}"
    )

    reset_logging_for_tests()
