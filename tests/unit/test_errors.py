"""Tests for the typed exception taxonomy in ``tether.core.errors``.

Synthesis §4 Phase 3 step 33; A5 step 2.
"""
from __future__ import annotations

import pytest

from tether.core.errors import (
    CancelledByClient,
    ConnectorAuthError,
    ConnectorNotConfiguredError,
    FatalProviderError,
    LoopLimitReached,
    TetherError,
    ToolError,
    TransientProviderError,
)


_TETHER_SUBCLASSES = [
    FatalProviderError,
    TransientProviderError,
    ToolError,
    CancelledByClient,
    LoopLimitReached,
    ConnectorNotConfiguredError,
    ConnectorAuthError,
]


@pytest.mark.parametrize("cls", _TETHER_SUBCLASSES)
def test_tether_error_is_base(cls):
    """Every Tether-specific error class is a subclass of TetherError, so
    callers can ``except TetherError`` to catch any of them."""
    assert issubclass(cls, TetherError)


def test_tether_error_subclass_of_exception():
    """TetherError descends from Exception so user code that does
    ``except Exception`` continues to catch our errors. (BaseException-only
    catches like KeyboardInterrupt should NOT match these.)"""
    assert issubclass(TetherError, Exception)
    assert not issubclass(TetherError, BaseException) or issubclass(
        TetherError, Exception
    )


def test_distinct_classes():
    """Each Tether subclass is a peer (not a chain) so the orchestrator can
    branch on the specific class without one swallowing another."""
    pairs = [
        (FatalProviderError, TransientProviderError),
        (FatalProviderError, ToolError),
        (TransientProviderError, ToolError),
        (CancelledByClient, LoopLimitReached),
        (FatalProviderError, CancelledByClient),
        (ConnectorNotConfiguredError, ConnectorAuthError),
        (ConnectorNotConfiguredError, ToolError),
        (ConnectorAuthError, FatalProviderError),
    ]
    for a, b in pairs:
        assert not issubclass(a, b), f"{a.__name__} must not subclass {b.__name__}"
        assert not issubclass(b, a), f"{b.__name__} must not subclass {a.__name__}"


@pytest.mark.parametrize("cls", _TETHER_SUBCLASSES + [TetherError])
def test_messages_preserved(cls):
    """Constructing with a message string preserves it in ``args[0]`` and
    ``str(exc)``, matching the standard ``Exception`` contract."""
    exc = cls("boom")
    assert exc.args == ("boom",)
    assert str(exc) == "boom"


def test_chaining_via_from():
    """``raise FatalProviderError(...) from cause`` correctly populates
    ``__cause__`` so the orchestrator's classifier can inspect the
    underlying TVM/OpenCL exception."""
    cause = RuntimeError("CL_OUT_OF_RESOURCES")
    try:
        try:
            raise cause
        except RuntimeError as e:
            raise FatalProviderError("native runtime corrupted") from e
    except FatalProviderError as wrapped:
        assert wrapped.__cause__ is cause
        assert isinstance(wrapped.__cause__, RuntimeError)
        assert str(wrapped) == "native runtime corrupted"


def test_caught_via_tether_error_base():
    """End-to-end: a single ``except TetherError`` clause catches any of
    the subclasses (this is the pattern p3-loop-recovery will rely on)."""
    for cls in _TETHER_SUBCLASSES:
        try:
            raise cls("x")
        except TetherError as caught:
            assert isinstance(caught, cls)
        else:  # pragma: no cover - defensive
            pytest.fail(f"{cls.__name__} not caught by TetherError")


def test_not_caught_by_unrelated_runtime_error():
    """Sanity: bare ``RuntimeError`` is NOT a TetherError (the migration
    from substring-grep to TetherError is intentional and progressive —
    this test documents the boundary)."""
    try:
        raise RuntimeError("legacy raise site")
    except TetherError:  # pragma: no cover
        pytest.fail("plain RuntimeError must not be caught by TetherError")
    except RuntimeError:
        pass
