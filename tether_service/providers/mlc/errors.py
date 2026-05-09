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

This module is intentionally tiny — only stdlib (``asyncio``) plus
``HwErrorClass`` from ``providers.hw``. No ``mlc_llm`` import, so importing
this module does NOT pull the heavy MLC runtime; the lazy-imports invariant
(synthesis §4 Phase 2 step 22 / R8) is preserved.

Synthesis §4 Phase 3 step 34; A5 step 2; §6 row 13.
"""
from __future__ import annotations

import asyncio

from tether_service.providers.hw import HwErrorClass


def classify_mlc_error(exc: BaseException) -> HwErrorClass:
    """Classify an exception originating from the MLC engine.

    Patterns derived from the legacy substring-grep code (api.py:82,
    orchestrator.py:202-205, provider.py:469):

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
    from the Adreno/Snapdragon hardware. Future iteration may add patterns;
    this PR preserves the classification surface as-it-was.

    Args:
        exc: The exception raised by the MLC engine (``AsyncMLCEngine``)
            or anything caught in ``MLCProvider.stream``'s except clauses.

    Returns:
        ``HwErrorClass`` — see ``providers/hw.py`` for the four values.
    """
    # Cancellation is not a hardware error — the caller / event loop is
    # tearing things down on purpose.
    if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
        return HwErrorClass.BENIGN

    # The two patterns the legacy code recognizes as fatal:
    #   - "TVMError" in the exception class name (Apache TVM emits this)
    #   - "CLML" or "CL_" in the message (Qualcomm CLML / OpenCL error codes)
    # These mean the OpenCL/TVM driver state is corrupted and we need an
    # ``hw_reset`` (teardown + eager reload) to recover.
    error_type = type(exc).__name__
    error_msg = str(exc)

    is_fatal = (
        "TVMError" in error_type
        or "CLML" in error_msg
        or "CL_" in error_msg
    )

    if is_fatal:
        return HwErrorClass.FATAL_RECOVERABLE

    return HwErrorClass.TRANSIENT


__all__ = ["classify_mlc_error"]
