"""Tests for ``MLCProvider``'s HardwareLifecycle Protocol implementation
and typed-error raise sites in ``stream()``.

Synthesis §4 Phase 3 step 34; B6 step 5; A3 step 6; A5 step 2.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from tether.core.errors import FatalProviderError, TransientProviderError
from tether.providers.dummy.provider import DummyProvider
from tether.providers.hw import HardwareLifecycle, HwErrorClass, HwHealth
from tether.providers.mlc.provider import MLCProvider

# ---------------------------------------------------------------------------
# Test fixtures: build an MLCProvider without touching real MLC engines.
# ---------------------------------------------------------------------------


def _make_provider(tmp_path: Path) -> MLCProvider:
    """Construct an MLCProvider rooted at tmp_path.

    ``MLCProvider.__init__`` doesn't load any engines — it only stores
    paths + initializes empty caches/locks. So construction is cheap and
    requires no real MLC libs as long as we never trigger
    ``_ensure_engine`` / ``_get_engine``.

    Mirrors the helper in ``tests/security/test_path_traversal.py`` but
    uses the public ``__init__`` rather than ``object.__new__`` since
    we need real attributes for ``_engine_cache`` and ``_cache_lock``.
    """
    models_root = tmp_path / "models"
    models_root.mkdir()
    (models_root / "libs").mkdir()
    return MLCProvider(models_root=str(models_root), device="auto", max_tokens=1024)


# ---------------------------------------------------------------------------
# isinstance contract: MLCProvider IS, DummyProvider IS NOT.
# ---------------------------------------------------------------------------


def test_mlc_provider_isinstance_hardware_lifecycle(tmp_path: Path):
    """The watchdog detects hardware-owning providers via
    ``isinstance(provider, HardwareLifecycle)``. MLCProvider must satisfy
    this check after Phase 3 step 34."""
    provider = _make_provider(tmp_path)
    assert isinstance(provider, HardwareLifecycle)


def test_dummy_provider_not_hardware_lifecycle():
    """DummyProvider deliberately does NOT implement HardwareLifecycle
    so the watchdog skips it (no native handles to release). This is the
    other half of the Protocol-is-opt-in contract."""
    provider = DummyProvider()
    assert not isinstance(provider, HardwareLifecycle)


# ---------------------------------------------------------------------------
# hw_classify: delegates to canonical classifier
# ---------------------------------------------------------------------------


def test_hw_classify_delegates_to_canonical(tmp_path: Path):
    """``hw_classify`` is a thin wrapper around ``classify_mlc_error``;
    behaviour is fully covered in test_mlc_classify.py. Here we just
    verify the wiring (a canonical fatal pattern returns FATAL_RECOVERABLE
    via the wrapper)."""
    provider = _make_provider(tmp_path)
    cls = provider.hw_classify(RuntimeError("CLML segfault foo"))
    assert cls is HwErrorClass.FATAL_RECOVERABLE


def test_hw_classify_benign_for_cancel(tmp_path: Path):
    """Cancellation must classify as BENIGN through the wrapper too —
    otherwise the watchdog would fire ``hw_reset`` on every disconnect."""
    provider = _make_provider(tmp_path)
    assert provider.hw_classify(asyncio.CancelledError()) is HwErrorClass.BENIGN


# ---------------------------------------------------------------------------
# hw_health: cache-state snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hw_health_no_engines(tmp_path: Path):
    """Empty engine cache → ``degraded`` with ``loaded_models=0``. Not an
    error — readyz consumers decide whether cold == not-ready."""
    provider = _make_provider(tmp_path)
    health = await provider.hw_health()
    assert isinstance(health, HwHealth)
    assert health.status == "degraded"
    assert health.details["loaded_models"] == 0
    assert "cold cache" in health.details.get("note", "")


@pytest.mark.asyncio
async def test_hw_health_with_engines(tmp_path: Path):
    """Pre-populate the cache with a sentinel → ``healthy`` with
    ``loaded_models=1``. We don't construct a real engine; any object will
    do because hw_health only counts entries."""
    provider = _make_provider(tmp_path)
    provider._engine_cache["fake-model:auto:fake.dll"] = object()  # type: ignore[assignment]

    health = await provider.hw_health()
    assert health.status == "healthy"
    assert health.details["loaded_models"] == 1


# ---------------------------------------------------------------------------
# hw_shutdown_budget_sec / hw_per_engine_terminate_sec: stable defaults
# ---------------------------------------------------------------------------


def test_hw_shutdown_budget_sec_default(tmp_path: Path):
    """Default 3.0s matches the legacy ``api.py:189`` constant. Changing
    this is a deployment-impacting change — the watchdog uses this as the
    daemon thread's wait budget."""
    provider = _make_provider(tmp_path)
    assert provider.hw_shutdown_budget_sec == 3.0


def test_hw_per_engine_terminate_sec_default(tmp_path: Path):
    """Default ``budget / 4 = 0.75`` matches the existing
    ``shutdown_all(per_engine_timeout=0.75)`` and
    ``_terminate_bounded(timeout=0.75)`` constants in provider.py."""
    provider = _make_provider(tmp_path)
    assert provider.hw_per_engine_terminate_sec == 0.75


# ---------------------------------------------------------------------------
# hw_reset: validates name → unloads → eager-reloads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hw_reset_calls_unload_then_ensure(tmp_path: Path, monkeypatch):
    """``hw_reset`` is the recovery contract: tear down + eager reload.
    We mock both halves to verify ordering and arguments without touching
    real OpenCL/TVM."""
    provider = _make_provider(tmp_path)

    call_order: List[str] = []

    def fake_unload(model_name: str) -> bool:
        call_order.append(f"unload:{model_name}")
        return True

    async def fake_ensure(model_name: str):
        call_order.append(f"ensure:{model_name}")
        return object()

    monkeypatch.setattr(provider, "unload_model", fake_unload)
    monkeypatch.setattr(provider, "_ensure_engine", fake_ensure)

    await provider.hw_reset("test-model")

    assert call_order == ["unload:test-model", "ensure:test-model"]


@pytest.mark.asyncio
async def test_hw_reset_validates_model_name(tmp_path: Path):
    """Path-traversal guard from Phase 0A still applies in the recovery
    path. A FatalProviderError surfaced from anywhere with an attacker-
    influenced model_name would otherwise let hw_reset wander outside
    models_root."""
    provider = _make_provider(tmp_path)
    with pytest.raises(ValueError):
        await provider.hw_reset("../../../etc/passwd")


@pytest.mark.asyncio
async def test_hw_reset_rejects_path_separator_windows(tmp_path: Path):
    """Cross-OS coverage of the path-traversal guard. The validator
    rejects backslashes regardless of platform."""
    provider = _make_provider(tmp_path)
    with pytest.raises(ValueError):
        await provider.hw_reset("..\\windows\\system32")


# ---------------------------------------------------------------------------
# stream() error path: typed exceptions instead of bare RuntimeError
# ---------------------------------------------------------------------------


class _FakeCLMLError(RuntimeError):
    """Simulated TVM/CLML runtime error. The legacy classifier matches
    on substring, so any RuntimeError whose message contains ``CLML`` is
    classified FATAL_RECOVERABLE."""


def _build_fake_engine_that_raises(exc: BaseException) -> MagicMock:
    """Construct an AsyncMLCEngine substitute whose
    ``chat.completions.create`` raises ``exc`` synchronously when awaited.

    The shape mirrors the call site in ``MLCProvider.stream``:
        ``await engine.chat.completions.create(...)``.
    """
    fake = MagicMock(name="fake_engine")
    fake.chat = MagicMock()
    fake.chat.completions = MagicMock()
    fake.chat.completions.create = AsyncMock(side_effect=exc)
    fake._abort = MagicMock()
    return fake


@pytest.mark.asyncio
async def test_stream_error_raises_fatal_provider_error(tmp_path: Path, monkeypatch):
    """When the underlying engine raises a fatal (CLML) error, stream()
    must raise ``FatalProviderError`` (NOT bare RuntimeError) and chain
    the original via ``__cause__`` so callers can introspect the root."""
    provider = _make_provider(tmp_path)
    cause = _FakeCLMLError("CLML kernel panic during prefill")

    async def fake_ensure(model_name: str):
        return _build_fake_engine_that_raises(cause)

    monkeypatch.setattr(provider, "_ensure_engine", fake_ensure)

    gen = provider.stream("any-model", messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(FatalProviderError) as excinfo:
        await gen.__anext__()

    assert excinfo.value.__cause__ is cause


@pytest.mark.asyncio
async def test_stream_error_raises_transient_for_generic(tmp_path: Path, monkeypatch):
    """A generic RuntimeError (no TVM/CLML/CL_ pattern) must surface as
    ``TransientProviderError`` — the orchestrator does NOT trigger
    watchdog reset for transient errors."""
    provider = _make_provider(tmp_path)
    cause = RuntimeError("generic non-fatal failure")

    async def fake_ensure(model_name: str):
        return _build_fake_engine_that_raises(cause)

    monkeypatch.setattr(provider, "_ensure_engine", fake_ensure)

    gen = provider.stream("any-model", messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(TransientProviderError) as excinfo:
        await gen.__anext__()

    assert excinfo.value.__cause__ is cause


@pytest.mark.asyncio
async def test_stream_error_does_not_raise_runtime_error(tmp_path: Path, monkeypatch):
    """Belt-and-suspenders: bare RuntimeError must NOT surface at the
    seam any more. If a future regression bypasses the typed taxonomy,
    this test catches it (TetherError subclasses are NOT RuntimeError)."""
    provider = _make_provider(tmp_path)
    cause = RuntimeError("something broke")

    async def fake_ensure(model_name: str):
        return _build_fake_engine_that_raises(cause)

    monkeypatch.setattr(provider, "_ensure_engine", fake_ensure)

    gen = provider.stream("any-model", messages=[{"role": "user", "content": "hi"}])
    raised: Optional[BaseException] = None
    try:
        await gen.__anext__()
    except BaseException as e:
        raised = e

    assert raised is not None
    # A typed TransientProviderError, NOT a bare RuntimeError pollution.
    # (TransientProviderError is a TetherError → Exception, not RuntimeError.)
    assert isinstance(raised, TransientProviderError)
    assert not isinstance(raised, RuntimeError)
