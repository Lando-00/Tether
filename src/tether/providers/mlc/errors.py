"""Canonical MLC error classification.

Replaces 3 duplicated substring-grep sites in the legacy code:

    - ``app/http/api.py:82`` (handle_thread_exception's is_fatal check)
    - ``protocol/orchestration/orchestrator.py:202-205`` (stream-error is_fatal)
    - ``providers/mlc/provider.py:469`` (MLCProvider.stream's is_fatal)

This is the single source of truth. ``MLCProvider.hw_classify`` delegates
to it (synthesis §4 Phase 3 step 34); ``p3-loop-recovery`` (Phase 3 step
36) migrates the orchestrator to call ``provider.hw_classify`` (which lands
here); ``p3-lifespan-slim`` migrates ``api.py`` via ``signal_supervisor``
relocation.

Phase 3 follow-up (rubber-duck consensus): also honors the typed Tether
taxonomy (``FatalProviderError`` / ``TransientProviderError``) via
``isinstance`` and walks the ``__cause__`` chain so the wrap pattern in
``MLCProvider.stream`` (``raise FatalProviderError(...) from e``) doesn't
lose the original ``TVMError`` class-name signal.

This module is intentionally tiny — only stdlib (``asyncio``) plus
``HwErrorClass`` from ``providers.hw`` and the typed taxonomy from
``core.errors``. No ``mlc_llm`` import, so importing this module does NOT
pull the heavy MLC runtime; the lazy-imports invariant (synthesis §4
Phase 2 step 22 / R8) is preserved.

Synthesis §4 Phase 3 step 34; A5 step 2; §6 row 13.
"""
from __future__ import annotations

import asyncio

from tether.core.errors import FatalProviderError, TransientProviderError
from tether.providers.hw import HwErrorClass


def _classify_raw(exc: BaseException) -> HwErrorClass:
    """Classify a RAW (unwrapped) exception by type name + message substring.

    Internal helper for :func:`classify_mlc_error`; preserves the legacy
    substring patterns for the historical sites that still raise raw
    exceptions (and for inspecting each level of the ``__cause__`` chain).
    """
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return HwErrorClass.BENIGN

    error_type = type(exc).__name__
    error_msg = str(exc)
    is_fatal = (
        "TVMError" in error_type
        or "CLML" in error_msg
        or "CL_" in error_msg
    )
    return HwErrorClass.FATAL_RECOVERABLE if is_fatal else HwErrorClass.TRANSIENT


def classify_mlc_error(exc: BaseException) -> HwErrorClass:
    """Classify an exception originating from the MLC engine.

    Honors the typed Tether taxonomy from synthesis §4 Phase 3 step 33:

        - ``FatalProviderError`` instances classify as ``FATAL_RECOVERABLE``
        - ``TransientProviderError`` instances classify as ``TRANSIENT``

    For raw exceptions (the legacy substring-grep sites), inspects the
    type name + message AND walks the ``__cause__`` chain to recover the
    original signal that may have been lost when ``MLCProvider.stream``
    wrapped the exception in a ``FatalProviderError``.

    Patterns recognized at each level of the chain (mirrors the legacy
    substring-grep that lived at api.py:82, orchestrator.py:202-205,
    provider.py:469):

        - ``asyncio.CancelledError`` / ``GeneratorExit``
            → ``BENIGN`` (caller cancelled; not a hardware error)
        - ``"TVMError"`` in exception class name (Apache TVM emits this) OR
          ``"CLML"`` in message (Qualcomm CLML errors) OR
          ``"CL_"`` in message (OpenCL error codes, e.g. CL_OUT_OF_RESOURCES)
            → ``FATAL_RECOVERABLE`` (corrupt OpenCL/TVM state; needs
              ``hw_reset`` — teardown + eager reload)
        - Any other exception
            → ``TRANSIENT`` (caller may retry; no ``hw_reset`` required)

    Note: this function does NOT distinguish ``FATAL_UNRECOVERABLE`` from
    ``FATAL_RECOVERABLE`` — the legacy substring grep can't tell the
    difference, and we have no observed ``FATAL_UNRECOVERABLE`` patterns
    from the Adreno/Snapdragon hardware.

    The walk only follows ``__cause__`` (the ``raise X from e`` chain),
    not ``__context__`` (implicit chaining inside an ``except``). The
    wrap-loses-type-name case we care about (``MLCProvider.stream``)
    always uses ``raise ... from e``, so ``__cause__`` is sufficient. If
    a real failure mode emerges where only ``__context__`` carries the
    signal, extend the walk; for now keep it surgical.

    Args:
        exc: The exception raised by the MLC engine (``AsyncMLCEngine``)
            or anything caught in ``MLCProvider.stream``'s except clauses.

    Returns:
        ``HwErrorClass`` — see ``providers/hw.py`` for the four values.

    Synthesis §4 Phase 3 step 34; A5 step 2; §6 row 13. Phase 3 follow-up:
    walk ``__cause__`` + ``isinstance`` check fix the wrap-loses-type-name
    bug flagged by the rubber-duck consensus (xhigh BLOCKING-1, gpt-5.5
    BLOCKING).
    """
    # Honor the typed taxonomy first — Phase 3 step 33 formalized this.
    # ``FatalProviderError`` / ``TransientProviderError`` carry the
    # classification by construction; we don't need to re-grep their
    # messages.
    if isinstance(exc, FatalProviderError):
        return HwErrorClass.FATAL_RECOVERABLE
    if isinstance(exc, TransientProviderError):
        return HwErrorClass.TRANSIENT

    # Raw exception (or unknown subclass) — apply substring patterns to
    # ``exc`` itself AND each level of the ``__cause__`` chain. The
    # original signal (TVMError class name) may have been wrapped before
    # reaching us; recover it.
    current: BaseException | None = exc
    visited: set[int] = set()  # cycle guard (rare but possible)
    while current is not None:
        cid = id(current)
        if cid in visited:
            break
        visited.add(cid)
        result = _classify_raw(current)
        # Short-circuit on a definitive classification: BENIGN means the
        # caller was cancelling at this level (don't override with a
        # deeper FATAL); FATAL_RECOVERABLE is the signal we're trying to
        # recover after wrapping. Only TRANSIENT lets us look deeper.
        if result == HwErrorClass.FATAL_RECOVERABLE:
            return result
        if result == HwErrorClass.BENIGN:
            return result
        current = current.__cause__

    # No FATAL_RECOVERABLE / BENIGN found at any level → TRANSIENT.
    return HwErrorClass.TRANSIENT


__all__ = ["classify_mlc_error"]
