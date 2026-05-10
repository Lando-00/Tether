"""OpenTelemetry adapter for Tether observability.

Phase 7 step 76. Synthesis §3 (observability), B3 step 10.

This module is **lazy-imported only when Settings.observability.otel.enabled=True**.
It translates structlog events into OpenTelemetry spans so production deployments
that already have an OTel collector can correlate Tether traces with the rest of
their stack.

When ``enabled=False`` (default), this module is never imported and OTel is not a
runtime dependency. Install with ``pip install tether[otel]`` to use.

## MVP span model (Phase 7)

Each ``*.end`` or ``*.error`` structlog event emitted by the orchestrator creates
a *closed* OTel span carrying the event's attributes (including ``duration_ms``
when present). This is a **fire-and-forget** model: no parent/child relationship
is tracked across events because structlog processors are stateless.

A future iteration can add full span-lifetime tracking (start span on
``provider.stream.start``, close on ``provider.stream.end``) by storing the active
span in an OTel context var and passing it through the async call stack.

Known event names translated to spans:
  - ``provider.stream.end``   → span "provider.stream"
  - ``provider.stream.error`` → span "provider.stream" (status=ERROR)
  - ``tool.start``            → span "tool.run" (child of provider.stream if resolvable)
  - ``tool.end``              → span "tool.run"
  - ``tool.error``            → span "tool.run" (status=ERROR)
  - ``generation.cancelled``  → span event on "provider.stream" span
  - All other events          → ignored (pass-through)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tether_service.config.settings import Settings


_initialized = False

# Structlog events that map to OTel spans (closed immediately — MVP model).
_SPAN_EVENTS: dict[str, tuple[str, bool]] = {
    # event_name -> (span_name, is_error)
    "provider.stream.end": ("provider.stream", False),
    "provider.stream.error": ("provider.stream", True),
    "tool.end": ("tool.run", False),
    "tool.error": ("tool.run", True),
}

# Events translated to OTel span *events* rather than full spans.
_SPAN_EVENT_NAMES = frozenset({"generation.cancelled", "tool.start"})

# Keys excluded from OTel span attributes (internal / non-serialisable).
_SKIP_ATTRS = frozenset({"event", "timestamp", "level", "logger", "_record"})


def _make_attrs(event_dict: dict[str, Any]) -> dict[str, str]:
    """Extract span attributes from a structlog event dict.

    OTel attribute values must be primitives; we coerce everything to ``str``
    so callers don't have to worry about complex values.
    """
    return {
        k: str(v)
        for k, v in event_dict.items()
        if k not in _SKIP_ATTRS and not k.startswith("_")
    }


def install_otel_adapter(settings: "Settings") -> None:
    """Idempotent install.  Wires structlog → OTel span events.

    Raises:
        ImportError: If ``enabled=True`` but OTel packages are not installed.
        ValueError: If ``exporter`` requires ``exporter_endpoint`` but it is
            ``None``.

    Safe to call multiple times — subsequent calls are no-ops once the adapter
    has been initialized.
    """
    global _initialized
    if _initialized:
        return

    if not settings.observability.otel.enabled:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
        from opentelemetry.trace import SpanKind, StatusCode
    except ImportError as exc:
        raise ImportError(
            "OpenTelemetry adapter is enabled in settings but the OTel "
            "packages are not installed. Run: pip install tether[otel]"
        ) from exc

    cfg = settings.observability.otel
    resource = Resource.create({"service.name": cfg.service_name})

    sampler = TraceIdRatioBased(cfg.sample_rate) if cfg.sample_rate < 1.0 else None
    provider_kwargs: dict[str, Any] = {"resource": resource}
    if sampler is not None:
        provider_kwargs["sampler"] = sampler

    otel_provider = TracerProvider(**provider_kwargs)

    if cfg.exporter == "console":
        otel_provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )
    elif cfg.exporter in ("otlp_http", "otlp_grpc"):
        if not cfg.exporter_endpoint:
            raise ValueError(
                f"OTel exporter='{cfg.exporter}' requires exporter_endpoint to be set."
            )
        if cfg.exporter == "otlp_http":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        else:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        otel_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=cfg.exporter_endpoint))
        )

    trace.set_tracer_provider(otel_provider)

    tracer = trace.get_tracer("tether")

    def _otel_structlog_processor(
        logger: Any,
        method_name: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        """Structlog processor: map Tether events to closed OTel spans.

        Runs *after* merge_contextvars / add_log_level so that all bound
        context vars (session_id, request_id, …) are present in event_dict
        and will be emitted as span attributes.

        This is the MVP fire-and-forget model: each end/error event creates
        and immediately closes a span.  No parent/child linking is attempted
        in this phase.
        """
        event = event_dict.get("event", "")

        if event in _SPAN_EVENTS:
            span_name, is_error = _SPAN_EVENTS[event]
            attrs = _make_attrs(event_dict)
            with tracer.start_as_current_span(
                name=span_name,
                kind=SpanKind.INTERNAL,
                attributes=attrs,
                record_exception=False,
            ) as span:
                if is_error:
                    error_msg = event_dict.get("error") or event_dict.get("exc_info") or ""
                    span.set_status(StatusCode.ERROR, description=str(error_msg))
                    exc = event_dict.get("_exc_info")
                    if isinstance(exc, BaseException):
                        span.record_exception(exc)

        elif event in _SPAN_EVENT_NAMES:
            # Lightweight: add a span event to whatever span is currently active.
            current = trace.get_current_span()
            if current and current.is_recording():
                attrs = _make_attrs(event_dict)
                current.add_event(name=event, attributes=attrs)

        return event_dict

    # Insert the OTel processor into structlog's processor chain.
    # We patch structlog's configuration by prepending to the existing
    # processor list so that ordering (merge_contextvars → otel → renderer)
    # is respected.
    import structlog

    current_cfg = structlog.get_config()
    existing_processors = list(current_cfg.get("processors", []))

    # Insert after merge_contextvars (index 0) but before the renderer
    # (last item). If we can't identify the insertion point, append before
    # the final renderer.
    insert_at = 1  # default: right after merge_contextvars
    for i, proc in enumerate(existing_processors):
        name = getattr(proc, "__name__", None) or getattr(
            getattr(proc, "__class__", None), "__name__", ""
        )
        if name in ("merge_contextvars",):
            insert_at = i + 1
            break

    new_processors = (
        existing_processors[:insert_at]
        + [_otel_structlog_processor]
        + existing_processors[insert_at:]
    )

    structlog.configure(processors=new_processors)

    _initialized = True


def reset_otel_adapter_for_tests() -> None:
    """Reset the initialization flag. **Test-only helper.**

    Mirrors ``reset_logging_for_tests()`` in ``core.logging``. Not part of
    the public library API.
    """
    global _initialized
    _initialized = False
