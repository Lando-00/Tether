"""Tether's typed exception taxonomy.

These exceptions replace the substring-grep + bare RuntimeError patterns that
the legacy code uses for fatality classification. Phase 3 step 34
(p3-mlc-as-hwlifecycle) migrates raise sites; Phase 3 step 36
(p3-loop-recovery) migrates orchestrator's exception handling.

This module is intentionally additive in this PR — nothing else in
``tether_service`` imports it yet. p3-mlc-as-hwlifecycle / p3-loop-recovery /
p3-watchdog-supervisor will start raising and catching these types.

Synthesis §4 Phase 3 step 33; A5 step 2.
"""
from __future__ import annotations


class TetherError(Exception):
    """Base for all Tether-specific exceptions.

    Catching this catches every other Tether error class
    (``FatalProviderError``, ``ToolError``, ...) but not the bare
    ``RuntimeError`` / ``ValueError`` / ``TVMError`` instances that the
    codebase still raises today. Phase 3 will progressively migrate those
    raise sites onto this hierarchy.
    """


class FatalProviderError(TetherError):
    """A model-provider error from which recovery requires teardown of the
    provider's hardware resources (OpenCL/TVM driver state corrupted, model
    graph in unrecoverable state).

    Examples:
        - TVM error during inference (CL_OUT_OF_RESOURCES, CLML segfault)
        - MLC engine reports a fatal context-window violation that the
          driver cannot recover from
        - GPU driver returns an error that requires re-initialization

    The orchestrator catches this and asks ``HardwareWatchdog`` (Phase 3) to
    perform an ``hw_reset(model_name)``. That tears down + reloads the model.

    Phase 3 step 34 (p3-mlc-as-hwlifecycle) will rewrite ``MLCProvider``'s
    ``stream()`` exception handler to raise this instead of bare
    ``RuntimeError`` when the substring check fires (synthesis §6 row 13 / R21).
    """


class TransientProviderError(TetherError):
    """A model-provider error that does NOT require hardware reset.

    The caller may retry, or surface the error to the user.

    Examples:
        - Network timeout to an external provider
        - Transient model-loading error that resolves on retry
        - Tool-call argument parsing failure on the provider side

    The orchestrator does NOT trigger watchdog reset for this class; it
    surfaces an error event and exits the loop normally.
    """


class ToolError(TetherError):
    """A tool-execution failure.

    Distinct from ``FatalProviderError`` (no hardware reset needed) and from
    ``TransientProviderError`` (this is the tool layer, not the provider
    layer).

    The orchestrator catches this and feeds it back to the model
    (``FEED_BACK_TO_MODEL`` policy per synthesis §3 + Phase 0C).

    Note: ``ToolRunner`` currently raises ``ValueError`` when a tool isn't
    found and ``asyncio.TimeoutError`` when a tool exceeds timeout. Phase 3+
    may migrate these to ``ToolError`` or keep them separate; for now this
    class exists as a slot in the taxonomy.
    """


class CancelledByClient(TetherError):
    """The orchestrator's ``cancel_event`` was set (Phase 2 step 20 plumbing).

    Distinct from ``asyncio.CancelledError`` so the ``finally`` block in
    ``orchestrate()`` can distinguish "user disconnected mid-stream" from
    "Python is tearing down the event loop".

    Currently NOT raised by anything in the codebase; Phase 5 step 56
    (CancelToken introduction) will start raising it. Defining it now keeps
    the taxonomy complete for code review.
    """


class LoopLimitReached(TetherError):
    """The orchestrator hit ``max_tool_loops`` before the model produced a
    final answer.

    Phase 5 step 56 (``LoopLimitPolicy.EMIT_LIMIT_EVENT``) converts this to
    a wire event; for now the loop just exits silently (current behavior).
    Defining it keeps the taxonomy complete.
    """


class ConnectorNotConfiguredError(TetherError):
    """A connector tool was invoked while the connector is in
    ``ConnectorState.UNCONFIGURED`` or ``ConnectorState.LOGGED_OUT``.

    Tools registered by a not-yet-authenticated connector are still
    visible to the model (so it can describe them in turn-1 responses),
    but invoking one before login completes raises this error so the
    orchestrator can surface a clear "please log in via begin_login"
    message via the ``FEED_BACK_TO_MODEL`` policy.

    Per connector spec §3.1 (tool methods MUST raise this in
    UNCONFIGURED / LOGGED_OUT states).
    """


class ConnectorAuthError(TetherError):
    """A connector login flow (``begin_login`` / ``complete_login``)
    failed.

    Distinct from ``ConnectorNotConfiguredError`` (which fires when tools
    are called before login completes) — this fires when login itself
    fails (bad QR scan, OAuth code rejected, MFA mismatch, expired
    challenge).

    Per connector spec §3.1.
    """


class ConfigError(TetherError):
    """A configuration-level error detected during Engine boot.

    Raised when typed settings parse cleanly but their *combination* is
    inconsistent — e.g., the legacy ``providers.session_store.args.dsn``
    disagrees with ``storage.sqlite.dsn`` (Phase-9 P0-A: ``storage.sqlite.dsn``
    is the single source of truth per ADR-0009).
    """


class UnknownProviderError(KeyError):
    """``provider_id`` is not a key of ``Engine.providers`` and not a known
    failed registry entry.

    Maps to HTTP 422 at the chat router boundary (ADR-0021 contract §8).
    Subclasses :class:`KeyError` so existing call sites that catch
    ``KeyError`` for missing-provider conditions keep working.
    """

    def __init__(self, provider_id: str):
        super().__init__(provider_id)
        self.provider_id = provider_id


class ProviderUnhealthyError(RuntimeError):
    """``provider_id`` is a known registry entry but its construction or
    warm-up failed; the engine cannot route requests to it.

    Maps to HTTP 503 at the chat router boundary (ADR-0021 contract §8).
    """

    def __init__(self, provider_id: str, message: str):
        super().__init__(f"Provider {provider_id!r} unhealthy: {message}")
        self.provider_id = provider_id
        self.message = message


__all__ = [
    "TetherError",
    "FatalProviderError",
    "TransientProviderError",
    "ToolError",
    "CancelledByClient",
    "LoopLimitReached",
    "ConnectorNotConfiguredError",
    "ConnectorAuthError",
    "ConfigError",
    "UnknownProviderError",
    "ProviderUnhealthyError",
]
