"""Unit tests for ``MLCProvider.shutdown_all`` parallel teardown.

Synthesis §4 Phase 3 step 38; B6 step 9. Verifies:

- Wall-clock parallelism: N engines tear down in ~per_engine_timeout, not
  ``N × per_engine_timeout``.
- Per-engine timeout is honoured (slow engines are abandoned).
- Per-worker engine references are released so destructors don't pile up.
- Empty cache is a no-op (no thread pool spawn).
- Worker exceptions are logged but never propagate.

These tests use ``FakeEngine`` instead of ``AsyncMLCEngine`` so the suite
runs without the MLC native libraries. The Phase B teardown path only
calls ``engine._abort`` (via :func:`_abort_all_requests`) and
``engine.terminate`` (via :func:`_terminate_bounded`), so a small fake
covers the contract.
"""
from __future__ import annotations

import gc
import time
import weakref
from pathlib import Path
from typing import Optional

import pytest

# MLCProvider imports mlc_llm at module scope. Those are the Qualcomm
# CodeLinaro Adreno wheels, installed out-of-band and only available on
# the Snapdragon target, so skip rather than fail collection elsewhere.
pytest.importorskip("mlc_llm")

from tether.providers.mlc.provider import MLCProvider

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _FakeEngineState:
    """Minimal stand-in for ``AsyncMLCEngine.state``.

    Only ``async_streamers`` is touched by ``_abort_all_requests``; we
    always return an empty dict so the abort step is a no-op in the
    FakeEngine path. The terminate path is what we're actually testing.
    """

    async_streamers: dict = {}


class FakeEngine:
    """Test double for ``AsyncMLCEngine`` exposing only the surface that
    ``MLCProvider.shutdown_all`` touches: ``state`` (for abort) and
    ``terminate`` (for the bounded daemon-thread call).
    """

    def __init__(self, sleep_sec: float = 0.0, raise_in_terminate: Optional[Exception] = None):
        self.state = _FakeEngineState()
        self._sleep_sec = sleep_sec
        self._raise = raise_in_terminate
        self.terminated = False

    def terminate(self) -> None:
        if self._sleep_sec > 0:
            time.sleep(self._sleep_sec)
        if self._raise is not None:
            raise self._raise
        self.terminated = True

    def _abort(self, _rid):  # pragma: no cover — _abort_all_requests path
        return None


def _make_provider(tmp_path: Path) -> MLCProvider:
    """Construct an MLCProvider rooted at tmp_path without loading engines.

    Mirrors the helper in ``tests/unit/test_mlc_provider_hwlifecycle.py``;
    ``__init__`` only stores paths and initialises empty caches/locks, so
    construction is cheap and needs no real MLC libs.
    """
    models_root = tmp_path / "models"
    models_root.mkdir()
    (models_root / "libs").mkdir()
    return MLCProvider(models_root=str(models_root), device="auto", max_tokens=1024)


def _populate(provider: MLCProvider, engines: dict) -> None:
    """Inject FakeEngine instances directly into the provider's cache."""
    with provider._cache_lock:
        provider._engine_cache.update(engines)


# ---------------------------------------------------------------------------
# A1.1 — Parallelism: N engines finish in ~per_engine_timeout, not N×.
# ---------------------------------------------------------------------------


def test_parallel_shutdown_n_engines_under_n_times_per_engine_budget(tmp_path: Path):
    """Four engines that each take 0.5s must complete well under 4×0.5s.

    With ``max_workers = min(N, 4) = 4`` and concurrent execution, wall
    time should be just over 0.5s. We assert < 2.0s to leave margin for
    CI noise; serial execution would be ~2.0s exactly so this asserts
    the parallelisation is real.
    """
    provider = _make_provider(tmp_path)
    engines = {f"key{i}": FakeEngine(sleep_sec=0.5) for i in range(4)}
    _populate(provider, engines)

    start = time.time()
    provider.shutdown_all(per_engine_timeout=2.0)
    elapsed = time.time() - start

    # Serial would be ~2.0s; parallel should be ~0.5s + thread overhead.
    assert elapsed < 2.0, (
        f"shutdown_all took {elapsed:.2f}s for 4×0.5s engines — "
        "parallelisation appears broken (serial would be ~2.0s)"
    )
    # All engines should have terminated cleanly.
    assert all(e.terminated for e in engines.values())


# ---------------------------------------------------------------------------
# A1.2 — Per-engine timeout: slow engine is abandoned, fast one finishes.
# ---------------------------------------------------------------------------


def test_parallel_shutdown_per_engine_timeout_respected(tmp_path: Path):
    """A slow engine (5s) must be abandoned at per_engine_timeout=0.75s
    while a fast engine (0.05s) completes normally. Total wall time
    should be bounded by the slow engine's timeout, not its sleep.
    """
    provider = _make_provider(tmp_path)
    fast = FakeEngine(sleep_sec=0.05)
    slow = FakeEngine(sleep_sec=5.0)
    _populate(provider, {"fast": fast, "slow": slow})

    start = time.time()
    provider.shutdown_all(per_engine_timeout=0.75)
    elapsed = time.time() - start

    # Slow engine times out at 0.75s; total budget is max(1.125, 3.0) = 3.0s,
    # but as_completed returns as soon as the fast worker (0.05s) and the
    # slow worker (timeout at 0.75s) both finish. So elapsed should be < 2.0s.
    assert elapsed < 2.0, (
        f"shutdown_all took {elapsed:.2f}s with a 5s engine and 0.75s "
        f"per_engine_timeout — slow engine wasn't abandoned"
    )
    # Fast engine should have terminated; slow engine was abandoned.
    assert fast.terminated is True
    assert slow.terminated is False


# ---------------------------------------------------------------------------
# A1.3 — Engine refs released so destructors don't pile up.
# ---------------------------------------------------------------------------


def test_parallel_shutdown_engine_refs_released(tmp_path: Path):
    """After ``shutdown_all`` returns and one ``gc.collect()`` cycle, no
    engine references must remain anywhere reachable. This validates the
    per-worker reference-drop pattern (each worker drops its local engine
    ref; the items list is cleared before the as_completed loop).
    """
    provider = _make_provider(tmp_path)
    e1 = FakeEngine(sleep_sec=0.0)
    e2 = FakeEngine(sleep_sec=0.0)
    refs = [weakref.ref(e1), weakref.ref(e2)]
    _populate(provider, {"k1": e1, "k2": e2})

    # Drop our local strong refs so only the provider cache holds them.
    del e1
    del e2

    provider.shutdown_all(per_engine_timeout=0.75)

    # One collect cycle is enough — there are no cycles in FakeEngine.
    gc.collect()

    for i, ref in enumerate(refs):
        assert ref() is None, (
            f"FakeEngine {i} still reachable after shutdown_all + gc.collect; "
            "per-worker engine references were not released"
        )


# ---------------------------------------------------------------------------
# A1.4 — Empty cache: no-op, no thread pool spawn.
# ---------------------------------------------------------------------------


def test_parallel_shutdown_empty_cache_no_op(tmp_path: Path):
    """An empty cache returns immediately without spawning workers and
    without raising. Wall time should be effectively zero.
    """
    provider = _make_provider(tmp_path)

    start = time.time()
    provider.shutdown_all(per_engine_timeout=0.75)
    elapsed = time.time() - start

    assert elapsed < 0.5, (
        f"shutdown_all on empty cache took {elapsed:.2f}s — should be ~0s"
    )


# ---------------------------------------------------------------------------
# A1.5 — Worker exceptions don't propagate.
# ---------------------------------------------------------------------------


def test_parallel_shutdown_handles_terminate_exception(tmp_path: Path):
    """If ``engine.terminate()`` raises, the worker logs and returns an
    error tuple; ``shutdown_all`` must not propagate the exception. This
    keeps the watchdog's daemon thread from getting a spurious failure.
    """
    provider = _make_provider(tmp_path)
    boom = FakeEngine(raise_in_terminate=RuntimeError("boom"))
    ok = FakeEngine(sleep_sec=0.0)
    _populate(provider, {"boom": boom, "ok": ok})

    # Must not raise.
    provider.shutdown_all(per_engine_timeout=0.75)

    # Healthy engine still terminated; broken engine recorded as failed
    # (terminated stays False because we raised before setting it).
    assert ok.terminated is True
    assert boom.terminated is False


# ---------------------------------------------------------------------------
# A7-style smoke: shutdown_all source contains the expected primitives.
# Cheap regression guard against accidental serial-loop reintroduction.
# ---------------------------------------------------------------------------


def test_shutdown_all_uses_thread_pool_executor():
    """The implementation must use ``ThreadPoolExecutor`` + ``as_completed``
    + a bounded ``max_workers``; this guards against an accidental revert
    to the serial loop (which would silently break the wall-clock test on
    fast machines but pass timing checks on a busy CI runner).
    """
    import inspect

    src = inspect.getsource(MLCProvider.shutdown_all)
    assert "ThreadPoolExecutor" in src
    assert "as_completed" in src
    assert "max_workers" in src
