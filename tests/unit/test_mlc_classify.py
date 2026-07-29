"""Tests for the canonical ``classify_mlc_error``.

This is the single source of truth that replaces 3 duplicated substring-grep
sites (api.py:82, orchestrator.py:202-205, provider.py:469). Tests here
cover the same patterns the legacy code recognized, plus the
cancellation-is-benign rule.

Synthesis §4 Phase 3 step 34; A5 step 2; §6 row 13.
"""
from __future__ import annotations

import asyncio

import pytest

from tether.providers.hw import HwErrorClass
from tether.providers.mlc.errors import classify_mlc_error

# ---------------------------------------------------------------------------
# FATAL_RECOVERABLE patterns — the three legacy substring-grep matchers
# ---------------------------------------------------------------------------


class _TVMError(RuntimeError):
    """Stand-in for Apache TVM's TVMError class. The legacy classifier
    matches by class name (substring), so a subclass with this exact name
    triggers the fatal path even without the real TVM dependency installed."""


def test_classify_tvm_error_fatal():
    """An exception whose class name contains ``TVMError`` is FATAL — TVM
    has no recovery contract beyond a full engine reload."""
    exc = _TVMError("inference graph aborted")
    assert classify_mlc_error(exc) is HwErrorClass.FATAL_RECOVERABLE


def test_classify_clml_message_fatal():
    """A RuntimeError whose message contains ``CLML`` (Qualcomm's CLML
    backend error string) is FATAL — the OpenCL state is corrupted."""
    exc = RuntimeError("CLML segfault during prefill foo")
    assert classify_mlc_error(exc) is HwErrorClass.FATAL_RECOVERABLE


def test_classify_cl_message_fatal():
    """A RuntimeError whose message contains ``CL_`` (OpenCL error codes
    like CL_OUT_OF_RESOURCES) is FATAL — the GPU command queue is in an
    unrecoverable state without a driver re-init."""
    exc = RuntimeError("CL_OUT_OF_RESOURCES detail")
    assert classify_mlc_error(exc) is HwErrorClass.FATAL_RECOVERABLE


# ---------------------------------------------------------------------------
# BENIGN — cancellation / generator close
# ---------------------------------------------------------------------------


def test_classify_cancelled_benign():
    """``asyncio.CancelledError`` is the caller cancelling — not a
    hardware fault. Watchdog must NOT trigger ``hw_reset`` on cancel."""
    assert classify_mlc_error(asyncio.CancelledError()) is HwErrorClass.BENIGN


def test_classify_generator_exit_benign():
    """``GeneratorExit`` fires when the consumer of an async generator
    closes it (typically: client disconnect mid-stream). Not a fault."""
    assert classify_mlc_error(GeneratorExit()) is HwErrorClass.BENIGN


# ---------------------------------------------------------------------------
# TRANSIENT — anything else
# ---------------------------------------------------------------------------


def test_classify_value_error_transient():
    """A plain ``ValueError`` (e.g. bad model_name argument) is transient —
    caller can retry / surface to user; no hardware reset needed."""
    assert classify_mlc_error(ValueError("bad arg")) is HwErrorClass.TRANSIENT


def test_classify_runtime_error_transient():
    """A generic ``RuntimeError`` without TVM/CLML/CL_ patterns is
    transient — fail open rather than reset hardware on every error."""
    assert classify_mlc_error(RuntimeError("generic failure")) is HwErrorClass.TRANSIENT


# ---------------------------------------------------------------------------
# Return-type contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        _TVMError("x"),
        RuntimeError("CLML boom"),
        RuntimeError("CL_FAIL"),
        asyncio.CancelledError(),
        GeneratorExit(),
        ValueError("x"),
        RuntimeError("plain"),
    ],
)
def test_classify_returns_hw_error_class(exc):
    """Every return value is an ``HwErrorClass`` enum member, never a
    string or int. Downstream code (``hw_classify``, watchdog) does
    ``== HwErrorClass.FATAL_RECOVERABLE`` identity comparisons that would
    silently fail if we returned the raw string ``"fatal_recoverable"``."""
    result = classify_mlc_error(exc)
    assert isinstance(result, HwErrorClass)
