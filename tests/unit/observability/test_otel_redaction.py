"""FIX 2 — OTel adapter must redact secrets in span attributes.

The structlog OTel processor runs INSIDE structlog (before the stdlib bridge),
so :class:`tether.core.logging.RedactingFilter` doesn't protect span
attributes. Without explicit redaction, a Bearer token in
``tool.error::error_message`` or an api_key in
``provider.stream.error::url`` would land in the OTel collector verbatim.

Phase 7 RD followup (FIX 2). Tests the in-memory ``_make_attrs`` directly
(unit-fast, no OTel SDK required) and end-to-end through an
``InMemorySpanExporter`` (skipped when OTel is absent).
"""
from __future__ import annotations

import pytest

from tether.observability.otel_adapter import _make_attrs


# ---------------------------------------------------------------------------
# Unit tests on _make_attrs — no OTel dependency
# ---------------------------------------------------------------------------


def test_make_attrs_redacts_string_values():
    """String values are passed through redact_text — Bearer tokens scrubbed."""
    attrs = _make_attrs(
        {
            "event": "tool.error",
            "error_message": "auth failed: Bearer abcdef0123456789zzzz",
            "tool_name": "web_search",
        }
    )
    assert "Bearer abcdef0123456789zzzz" not in attrs["error_message"]
    assert "Bearer ***REDACTED***" in attrs["error_message"]
    # Non-secret string is preserved verbatim.
    assert attrs["tool_name"] == "web_search"
    # ``event`` is filtered out (in _SKIP_ATTRS).
    assert "event" not in attrs


def test_make_attrs_redacts_url_query_keys():
    """URLs with ``?api_key=...`` query params are redacted."""
    attrs = _make_attrs(
        {
            "event": "provider.stream.error",
            "url": "https://api.example.com/v1/search?api_key=SECRETKEY12345",
        }
    )
    assert "SECRETKEY12345" not in attrs["url"]
    assert "***REDACTED***" in attrs["url"]


def test_make_attrs_preserves_native_types():
    """OTel-native types (int, float, bool) are kept as-is, not stringified.

    Dashboards rely on numeric attributes for math (avg duration, etc.) so
    coercing everything to ``str`` (the previous behaviour) broke them.
    """
    attrs = _make_attrs(
        {
            "event": "tool.end",
            "duration_ms": 123,
            "p99_latency_sec": 0.45,
            "from_cache": True,
            "result_count": 0,
        }
    )
    assert attrs["duration_ms"] == 123
    assert isinstance(attrs["duration_ms"], int)
    assert attrs["p99_latency_sec"] == pytest.approx(0.45)
    assert isinstance(attrs["p99_latency_sec"], float)
    assert attrs["from_cache"] is True
    assert isinstance(attrs["from_cache"], bool)
    # Falsy int is kept (not dropped).
    assert attrs["result_count"] == 0


def test_make_attrs_drops_none_values():
    """None values are dropped — OTel rejects them and would raise TypeError."""
    attrs = _make_attrs(
        {
            "event": "tool.end",
            "tool_name": "time",
            "error_kind": None,
        }
    )
    assert "tool_name" in attrs
    assert "error_kind" not in attrs


def test_make_attrs_skips_internal_and_underscore_keys():
    """``event``, ``timestamp``, ``_record``, etc. are filtered out."""
    attrs = _make_attrs(
        {
            "event": "tool.end",
            "timestamp": "2025-01-01T00:00:00Z",
            "level": "info",
            "logger": "x",
            "_record": object(),
            "_exc_info": object(),
            "tool_name": "time",
        }
    )
    assert attrs == {"tool_name": "time"}


def test_make_attrs_redacts_repr_for_complex_values():
    """Complex objects (dicts, custom classes) are repr'd, redacted, and capped."""

    class _Box:
        def __repr__(self):
            return "Box(token=Bearer abcdef0123456789zzzz)"

    attrs = _make_attrs(
        {
            "event": "tool.error",
            "context": _Box(),
        }
    )
    assert "Bearer abcdef0123456789zzzz" not in attrs["context"]
    assert "Bearer ***REDACTED***" in attrs["context"]


def test_make_attrs_caps_long_repr():
    """Very long repr fallbacks are capped at 256 chars."""

    long_string = "x" * 1000

    class _LongBox:
        def __repr__(self):
            return long_string

    attrs = _make_attrs({"event": "x", "blob": _LongBox()})
    assert len(attrs["blob"]) <= 256


def test_make_attrs_redacts_bytes():
    """``bytes`` decoded via UTF-8 are redacted as strings."""
    attrs = _make_attrs(
        {
            "event": "tool.end",
            "raw": b"Bearer abcdef0123456789zzzz appended",
        }
    )
    assert "Bearer abcdef0123456789zzzz" not in attrs["raw"]
    assert "Bearer ***REDACTED***" in attrs["raw"]


# ---------------------------------------------------------------------------
# Phase 8 RD Fix 1 — set_status(description=...) + record_exception attrs
# are also redacted (they go through a different code path than _make_attrs).
# ---------------------------------------------------------------------------


def _reset_otel_tracer_provider() -> None:
    """Reset OTel's tracer-provider singleton so each end-to-end test can
    install its own ``InMemorySpanExporter``-backed provider.

    OTel logs ``"Overriding of current TracerProvider is not allowed"`` and
    silently keeps the old provider when ``set_tracer_provider`` is called
    twice. Without this reset, the second + third end-to-end tests in this
    module would silently lose their exporter and fail with "no spans".
    """
    import opentelemetry.trace as _otel_trace

    _otel_trace._TRACER_PROVIDER = None
    once = getattr(_otel_trace, "_TRACER_PROVIDER_SET_ONCE", None)
    if once is not None and hasattr(once, "_done"):
        once._done = False


def test_otel_span_status_description_is_redacted_end_to_end():
    """``span.set_status(StatusCode.ERROR, description=...)`` MUST run
    ``error_msg`` through :func:`redact_text` before reaching the exporter.

    Phase 8 RD Fix 1 (gpt-5.5 BLOCKER): span status descriptions are NOT
    covered by ``_make_attrs`` — they go through ``span.set_status``, which
    Phase 7's ``_make_attrs`` redaction did not touch. Without explicit
    redaction here, a Bearer token in an ``error_message`` event field
    would leak verbatim into the OTel collector under the span's status
    description.
    """
    pytest.importorskip(
        "opentelemetry.sdk.trace.export.in_memory_span_exporter",
        reason="pip install tether[otel]",
    )

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    import structlog

    from tether.core.logging import (
        configure_logging,
        reset_logging_for_tests,
    )
    from tether.observability.otel_adapter import (
        reset_otel_adapter_for_tests,
    )

    reset_logging_for_tests()
    reset_otel_adapter_for_tests()
    _reset_otel_tracer_provider()

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    from tether.config.settings import load_settings

    settings = load_settings(
        env={
            "TETHER__OBSERVABILITY__OTEL__ENABLED": "true",
            "TETHER__OBSERVABILITY__OTEL__EXPERIMENTAL_ACKNOWLEDGED": "true",
            "TETHER__OBSERVABILITY__OTEL__EXPORTER": "console",
        }
    )

    from unittest.mock import patch

    with patch("opentelemetry.trace.set_tracer_provider"):
        configure_logging(settings)

    log = structlog.get_logger("test_status_description_redaction")
    secret_msg = "auth failed: Bearer abcdef0123456789zzzz invalid"
    log.error("tool.error", error_message=secret_msg, tool_name="web_search")

    spans = exporter.get_finished_spans()
    assert spans, "Expected an OTel span from the tool.error event"

    span = spans[0]
    description = span.status.description or ""
    assert "Bearer abcdef0123456789zzzz" not in description, (
        f"Phase 8 RD Fix 1 regressed: span status description leaked the "
        f"Bearer token verbatim:\n  description={description!r}"
    )
    assert "Bearer ***REDACTED***" in description, (
        f"Expected 'Bearer ***REDACTED***' in span status description, "
        f"got: {description!r}"
    )

    reset_logging_for_tests()
    reset_otel_adapter_for_tests()


def test_otel_record_exception_message_is_redacted_end_to_end():
    """When ``_exc_info`` is present on a ``*.error`` event, the resulting
    span exception event's ``exception.message`` MUST be redacted.

    Phase 8 RD Fix 1: the OTel SDK's default ``record_exception`` builds
    its event attributes from ``str(exc)`` and ``type(exc).__name__``.
    A Bearer token in ``str(exc)`` would otherwise appear verbatim in the
    span's exception event. The fix overrides ``exception.message`` with
    a redacted value via the ``attributes`` kwarg.

    Note: ``exception.stacktrace`` is NOT redacted by this fix —
    see TODO(fu-otel-traceback-redaction) in the adapter.
    """
    pytest.importorskip(
        "opentelemetry.sdk.trace.export.in_memory_span_exporter",
        reason="pip install tether[otel]",
    )

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    import structlog

    from tether.core.logging import (
        configure_logging,
        reset_logging_for_tests,
    )
    from tether.observability.otel_adapter import (
        reset_otel_adapter_for_tests,
    )

    reset_logging_for_tests()
    reset_otel_adapter_for_tests()
    _reset_otel_tracer_provider()

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    from tether.config.settings import load_settings

    settings = load_settings(
        env={
            "TETHER__OBSERVABILITY__OTEL__ENABLED": "true",
            "TETHER__OBSERVABILITY__OTEL__EXPERIMENTAL_ACKNOWLEDGED": "true",
            "TETHER__OBSERVABILITY__OTEL__EXPORTER": "console",
        }
    )

    from unittest.mock import patch

    with patch("opentelemetry.trace.set_tracer_provider"):
        configure_logging(settings)

    # Construct an exception whose __str__ leaks a Bearer token.
    exc = RuntimeError("upstream rejected: Bearer abcdef0123456789zzzz")
    log = structlog.get_logger("test_record_exception_redaction")
    log.error(
        "tool.error",
        error_message="boom",
        _exc_info=exc,
        tool_name="web_search",
    )

    spans = exporter.get_finished_spans()
    assert spans, "Expected an OTel span from the tool.error event"

    span = spans[0]
    # The exception is recorded as a span event named "exception".
    exception_events = [e for e in span.events if e.name == "exception"]
    assert exception_events, (
        "Expected a span 'exception' event from record_exception()"
    )
    attrs = dict(exception_events[0].attributes or {})
    message = attrs.get("exception.message", "")
    assert "Bearer abcdef0123456789zzzz" not in message, (
        f"Phase 8 RD Fix 1 regressed: record_exception leaked the Bearer "
        f"token verbatim in exception.message:\n  message={message!r}"
    )
    assert "Bearer ***REDACTED***" in message, (
        f"Expected 'Bearer ***REDACTED***' in exception.message, "
        f"got: {message!r}"
    )
    # Type is unchanged (never sensitive).
    assert attrs.get("exception.type") == "RuntimeError"

    reset_logging_for_tests()
    reset_otel_adapter_for_tests()


# ---------------------------------------------------------------------------
# End-to-end test via InMemorySpanExporter (skipped without OTel SDK).
# ---------------------------------------------------------------------------


def test_otel_span_attribute_is_redacted_end_to_end():
    """Install adapter, emit a structlog event with a secret, capture the
    resulting span via :class:`InMemorySpanExporter`, and assert the secret
    is NOT in the span attributes.

    This is the end-to-end proof that FIX 2 actually flows redaction through
    to the exporter (not just the helper).
    """
    pytest.importorskip(
        "opentelemetry.sdk.trace.export.in_memory_span_exporter",
        reason="pip install tether[otel]",
    )

    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    import structlog

    from tether.core.logging import (
        configure_logging,
        reset_logging_for_tests,
    )
    from tether.observability.otel_adapter import (
        reset_otel_adapter_for_tests,
    )

    # Fresh OTel setup
    reset_logging_for_tests()
    reset_otel_adapter_for_tests()
    _reset_otel_tracer_provider()

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # We're going to call the adapter's processor directly — install the
    # adapter to register it inside structlog's processor chain.
    from tether.config.settings import load_settings

    settings = load_settings(
        env={
            "TETHER__OBSERVABILITY__OTEL__ENABLED": "true",
            "TETHER__OBSERVABILITY__OTEL__EXPERIMENTAL_ACKNOWLEDGED": "true",
            "TETHER__OBSERVABILITY__OTEL__EXPORTER": "console",
        }
    )

    # configure_logging will install the OTel adapter, which prepends an OTel
    # processor into structlog. But it also calls
    # ``trace.set_tracer_provider`` with its OWN provider — overwriting ours.
    # To keep our InMemorySpanExporter active, install_otel_adapter is patched
    # so set_tracer_provider is a no-op.
    from unittest.mock import patch

    with patch("opentelemetry.trace.set_tracer_provider"):
        configure_logging(settings)

    # Now emit a structlog event with a secret. The OTel processor should
    # redact attributes via _make_attrs before creating the span.
    log = structlog.get_logger("test_otel_redaction")
    log.error(
        "tool.error",
        error_message="upstream auth failed: Bearer abcdef0123456789zzzz",
        tool_name="web_search",
    )

    spans = exporter.get_finished_spans()
    assert spans, "Expected an OTel span from the tool.error event"

    attrs = dict(spans[0].attributes or {})
    # Secret scrubbed.
    em = attrs.get("error_message", "")
    assert "Bearer abcdef0123456789zzzz" not in em, (
        f"Secret leaked into OTel span attribute: {em!r}"
    )
    assert "Bearer ***REDACTED***" in em
    # Non-secret kept.
    assert attrs.get("tool_name") == "web_search"

    # Cleanup
    reset_logging_for_tests()
    reset_otel_adapter_for_tests()
