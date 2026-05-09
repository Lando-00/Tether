"""Tests for the HardwareLifecycle Protocol + HwErrorClass + HwHealth.

Synthesis §2.3 + §4 Phase 3 step 32; B6 step 4; A3.
"""
from __future__ import annotations

import dataclasses

import pytest

from tether_service.providers.hw import (
    HardwareLifecycle,
    HwErrorClass,
    HwHealth,
)


# ---------------------------------------------------------------------------
# HwErrorClass enum
# ---------------------------------------------------------------------------


def test_hw_error_class_values():
    """The 4 documented members exist with stable string values. The string
    values are part of the wire/log surface, so changing them is a breaking
    change."""
    assert HwErrorClass.BENIGN.value == "benign"
    assert HwErrorClass.TRANSIENT.value == "transient"
    assert HwErrorClass.FATAL_RECOVERABLE.value == "fatal_recoverable"
    assert HwErrorClass.FATAL_UNRECOVERABLE.value == "fatal_unrecoverable"
    # And no extra members snuck in.
    assert {m.name for m in HwErrorClass} == {
        "BENIGN",
        "TRANSIENT",
        "FATAL_RECOVERABLE",
        "FATAL_UNRECOVERABLE",
    }


def test_hw_error_class_distinct():
    assert len({m.value for m in HwErrorClass}) == 4


# ---------------------------------------------------------------------------
# HwHealth dataclass
# ---------------------------------------------------------------------------


def test_hw_health_dataclass_basic_construction():
    """Positional + kwargs construction works; ``details`` defaults to an
    empty dict (independent per instance, not a shared mutable default)."""
    h1 = HwHealth("healthy")
    assert h1.status == "healthy"
    assert h1.details == {}

    h2 = HwHealth(status="degraded", details={"engine_count": 1})
    assert h2.status == "degraded"
    assert h2.details == {"engine_count": 1}

    # Each instance gets its own dict (the default_factory contract).
    assert HwHealth("healthy").details is not HwHealth("healthy").details


def test_hw_health_is_frozen():
    """``frozen=True`` so a snapshot can't be mutated by mistake — callers
    should construct a new one if state changes."""
    h = HwHealth("healthy")
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.status = "error"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.details = {"x": 1}  # type: ignore[misc]


def test_hw_health_status_literal_not_runtime_enforced():
    """OBSERVATION: ``Literal`` is a typing hint, and stdlib dataclasses
    don't validate it at runtime. This test documents the gap — future
    mypy gating (or a Pydantic StrictModel wrapper) will catch invalid
    values; for now construction with an arbitrary string SUCCEEDS.
    Synthesis §4 Phase 3 step 32 (pure-types phase, no validation)."""
    bogus = HwHealth(status="invalid")  # type: ignore[arg-type]
    assert bogus.status == "invalid"


# ---------------------------------------------------------------------------
# HardwareLifecycle Protocol
# ---------------------------------------------------------------------------


class _FullImpl:
    """Minimal valid implementation of HardwareLifecycle for runtime checks."""

    def hw_classify(self, exc: BaseException) -> HwErrorClass:
        return HwErrorClass.BENIGN

    async def hw_reset(self, model_name: str) -> None:
        return None

    async def hw_health(self) -> HwHealth:
        return HwHealth("healthy")

    @property
    def hw_shutdown_budget_sec(self) -> float:
        return 3.0

    @property
    def hw_per_engine_terminate_sec(self) -> float:
        return 0.75


class _MissingMethodImpl:
    """Lacks ``hw_health`` — must NOT pass the runtime check."""

    def hw_classify(self, exc: BaseException) -> HwErrorClass:
        return HwErrorClass.BENIGN

    async def hw_reset(self, model_name: str) -> None:
        return None

    @property
    def hw_shutdown_budget_sec(self) -> float:
        return 3.0

    @property
    def hw_per_engine_terminate_sec(self) -> float:
        return 0.75


class _MissingPropertyImpl:
    """Lacks ``hw_per_engine_terminate_sec`` — must NOT pass the runtime check."""

    def hw_classify(self, exc: BaseException) -> HwErrorClass:
        return HwErrorClass.BENIGN

    async def hw_reset(self, model_name: str) -> None:
        return None

    async def hw_health(self) -> HwHealth:
        return HwHealth("healthy")

    @property
    def hw_shutdown_budget_sec(self) -> float:
        return 3.0


def test_protocol_runtime_checkable_full_impl():
    """A class with all 5 methods + 2 properties IS detected by
    ``isinstance``. This is the contract HardwareWatchdog (Phase 3 step 30)
    will rely on to filter providers."""
    obj = _FullImpl()
    assert isinstance(obj, HardwareLifecycle)


def test_protocol_partial_no_match_missing_method():
    """A class missing ``hw_health`` is NOT a HardwareLifecycle. The
    watchdog must skip it rather than crash later when calling the missing
    member."""
    obj = _MissingMethodImpl()
    assert not isinstance(obj, HardwareLifecycle)


def test_protocol_partial_no_match_missing_property():
    """A class missing the per-engine terminate property is NOT a
    HardwareLifecycle. Note: ``runtime_checkable`` Protocols only inspect
    method/attribute *presence*, not signatures — that's a known stdlib
    limitation (PEP 544). For Tether's seam, presence is sufficient."""
    obj = _MissingPropertyImpl()
    assert not isinstance(obj, HardwareLifecycle)


def test_dummy_class_implements_protocol_uses_methods():
    """End-to-end: the protocol-conformant dummy can be used through the
    Protocol surface (smoke test for type-checker confidence)."""
    impl: HardwareLifecycle = _FullImpl()  # type: ignore[assignment]
    cls = impl.hw_classify(RuntimeError("boom"))
    assert cls is HwErrorClass.BENIGN
    assert impl.hw_shutdown_budget_sec == 3.0
    assert impl.hw_per_engine_terminate_sec == 0.75


def test_unrelated_class_not_a_hardware_lifecycle():
    class _Unrelated:
        pass

    assert not isinstance(_Unrelated(), HardwareLifecycle)
