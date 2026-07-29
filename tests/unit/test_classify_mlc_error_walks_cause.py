"""Phase 3 follow-up tests for ``classify_mlc_error``.

Covers the wrap-loses-type-name regression of bug #13 flagged by the
rubber-duck consensus (xhigh BLOCKING-1; gpt-5.5 BLOCKING):

    - ``classify_mlc_error(TVMError())`` → FATAL_RECOVERABLE ✓
    - ``classify_mlc_error(FatalProviderError(...wrap...))`` USED to
      return TRANSIENT because the wrapper class name "FatalProviderError"
      doesn't contain "TVMError". Now ``isinstance`` short-circuits to
      FATAL_RECOVERABLE; the ``__cause__`` walk also recovers the signal
      for raw-wrap patterns.

Existing tests in ``test_mlc_classify.py`` continue to cover the
unwrapped legacy patterns; tests here cover the typed-taxonomy branch
and the ``__cause__`` walk.

Synthesis §4 Phase 3 step 34 + Phase 3 follow-up.
"""
from __future__ import annotations

import asyncio

import pytest

from tether.core.errors import (
    FatalProviderError,
    TransientProviderError,
)
from tether.providers.hw import HwErrorClass
from tether.providers.mlc.errors import classify_mlc_error

# ---------------------------------------------------------------------------
# Stand-in for Apache TVM's TVMError (real TVM not required for tests).
# The legacy substring matcher checks for "TVMError" in the class name.
# ---------------------------------------------------------------------------


_TVMError = type("TVMError", (RuntimeError,), {})


# ---------------------------------------------------------------------------
# Typed-taxonomy short-circuit — the BLOCKER fix.
# ---------------------------------------------------------------------------


def test_classify_fatal_provider_error_direct():
    """A direct ``FatalProviderError`` (no cause) classifies as
    FATAL_RECOVERABLE via the ``isinstance`` check, regardless of
    message content. This is the typed-taxonomy contract from
    synthesis Phase 3 step 33."""
    assert classify_mlc_error(FatalProviderError("any message")) is (
        HwErrorClass.FATAL_RECOVERABLE
    )


def test_classify_transient_provider_error_direct():
    """A direct ``TransientProviderError`` classifies as TRANSIENT via
    the ``isinstance`` check. Even if the message happens to contain
    'TVMError' (e.g., as a description), the typed class wins."""
    assert classify_mlc_error(TransientProviderError("TVMError in description")) is (
        HwErrorClass.TRANSIENT
    )


# ---------------------------------------------------------------------------
# __cause__ walk — recovers the wrapped signal.
# ---------------------------------------------------------------------------


def test_classify_wrapped_tvm_error_recovers_via_cause():
    """The ``MLCProvider.stream`` wrap pattern: a TVMError caught and
    re-raised as ``raise FatalProviderError(...) from e``.

    The ``isinstance`` check on ``FatalProviderError`` short-circuits to
    FATAL_RECOVERABLE before the cause walk even runs. Either way, the
    classification is correct.

    This is the empirical reproduction of the bug from the rubber-duck
    consensus: pre-fix, the wrapper's class name "FatalProviderError"
    has no 'TVMError' substring, so the legacy classifier returned
    TRANSIENT for a clearly fatal error.
    """
    original = _TVMError("inference failure")
    wrapped = FatalProviderError(
        "Model streaming failed for 'X': TVMError - inference failure"
    )
    wrapped.__cause__ = original

    assert classify_mlc_error(wrapped) is HwErrorClass.FATAL_RECOVERABLE


def test_classify_unwrapped_tvm_error_still_works():
    """Unwrapped TVMError continues to classify as FATAL_RECOVERABLE via
    the legacy substring path (preserves ``test_mlc_classify.py`` behavior).
    """
    assert classify_mlc_error(_TVMError("inference failure")) is (
        HwErrorClass.FATAL_RECOVERABLE
    )


def test_classify_walks_multi_level_cause():
    """A 3-deep chain where only the deepest level is TVMError-class.

    Chain shape:
        outer (RuntimeError, plain)
        └── middle (RuntimeError, plain)
            └── deepest (TVMError)

    The classifier must walk past the two transient-looking levels and
    return FATAL_RECOVERABLE based on the deepest cause.
    """
    deepest = _TVMError("native crash")
    middle = RuntimeError("middle layer error")
    middle.__cause__ = deepest
    outer = RuntimeError("outer layer error")
    outer.__cause__ = middle

    assert classify_mlc_error(outer) is HwErrorClass.FATAL_RECOVERABLE


def test_classify_cycle_guard():
    """A pathological ``__cause__`` cycle (a → b, b → a) must not cause
    an infinite loop; the visited-set guard breaks out and falls back
    to TRANSIENT (no fatal pattern was found before the cycle).
    """
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a

    # Should terminate (the test framework would hang otherwise).
    result = classify_mlc_error(a)
    # Neither a nor b matches any fatal pattern, so result is TRANSIENT.
    assert result is HwErrorClass.TRANSIENT


# ---------------------------------------------------------------------------
# Negative cases: walking the chain doesn't false-positive transients.
# ---------------------------------------------------------------------------


def test_classify_returns_transient_for_truly_transient():
    """A wrapper around a wrapper with NO fatal pattern at any level must
    classify as TRANSIENT — the cause walk MUST NOT spuriously upgrade
    benign chains to FATAL."""
    inner = ValueError("bad arg")
    outer = RuntimeError("transient failure")
    outer.__cause__ = inner

    assert classify_mlc_error(outer) is HwErrorClass.TRANSIENT


# ---------------------------------------------------------------------------
# BENIGN short-circuit — cancellation at any level wins over a deeper FATAL.
# ---------------------------------------------------------------------------


def test_classify_benign_short_circuits():
    """If the *outermost* exception is BENIGN (cancellation), we do NOT
    wander deeper looking for a FATAL — cancellation is the dominant
    signal: the caller is tearing down, watchdog must NOT fire ``hw_reset``.

    Construction: an ``asyncio.CancelledError`` whose ``__cause__`` happens
    to be a TVMError (rare, but possible if a CancelledError was raised
    inside a TVM-error handler).
    """
    fatal_inner = _TVMError("would otherwise be fatal")
    cancelled = asyncio.CancelledError()
    # NOTE: CancelledError.__cause__ assignment is allowed in CPython 3.8+.
    cancelled.__cause__ = fatal_inner

    assert classify_mlc_error(cancelled) is HwErrorClass.BENIGN


def test_classify_benign_generator_exit_short_circuits():
    """Same contract for ``GeneratorExit`` — client-disconnect dominant."""
    fatal_inner = _TVMError("inner fatal")
    gx = GeneratorExit()
    gx.__cause__ = fatal_inner

    assert classify_mlc_error(gx) is HwErrorClass.BENIGN


# ---------------------------------------------------------------------------
# Return-type contract preserved across the new branches.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        FatalProviderError("x"),
        TransientProviderError("x"),
        _TVMError("x"),
        RuntimeError("plain"),
    ],
)
def test_classify_returns_hw_error_class(exc):
    """Every return value is a :class:`HwErrorClass` enum member, never a
    string or int. Guards against accidental ``return "fatal_recoverable"``
    regression."""
    result = classify_mlc_error(exc)
    assert isinstance(result, HwErrorClass)
