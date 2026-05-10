"""Regression test: ``gc.disable()`` inside ``daemon_thread_call`` must hold
for the duration of the wrapped callable when ``gc_disable=True``.

Phase 8 RD Fix 4 (xhigh COVERAGE GAP). Synthesis §3.3 + ADR-0003: the
GC-disable invariant inside the MLC shutdown daemon thread is
**load-bearing**. Re-enabling GC would re-introduce the Qwen2.5-7B
Ctrl+C hang traced to OpenCL/TVM destructors interacting with the
cyclic GC. This test pins the invariant so a future careless edit can't
silently regress it.

The existing ``tests/unit/runtime/test_daemon_call.py`` covers the
``gc_disable`` flag in passing; this file complements it with assertions
phrased explicitly as "the GC-disable invariant" so a code reviewer
sees the load-bearing context next to the test name.
"""
from __future__ import annotations

import gc

import pytest

from tether.runtime.daemon_call import daemon_thread_call


@pytest.fixture(autouse=True)
def _restore_gc():
    """Restore the parent-thread GC state after each test.

    ``daemon_thread_call(gc_disable=True)`` deliberately disables GC for
    the rest of the process when used in real shutdown paths; in tests we
    don't want one test's GC-disable to bleed into the next.
    """
    was_enabled = gc.isenabled()
    yield
    if was_enabled and not gc.isenabled():
        gc.enable()
    elif not was_enabled and gc.isenabled():
        gc.disable()


def test_daemon_call_disables_gc_when_flag_set() -> None:
    """``gc_disable=True`` MUST disable GC inside the daemon thread.

    Pins ADR-0003: this is the invariant that fixed the Qwen2.5-7B
    shutdown hang. Re-enabling GC inside the daemon thread (or skipping
    the disable) re-introduces the regression.
    """
    if not gc.isenabled():
        gc.enable()  # ensure starting state so the disable is observable

    result = daemon_thread_call(
        lambda: gc.isenabled(),
        timeout=1.0,
        gc_disable=True,
        label="gc-invariant",
    )
    assert result is False, (
        "gc.disable() invariant violated: daemon_thread_call(gc_disable=True) "
        "must run callable under gc.disable() (load-bearing per ADR-0003)."
    )


def test_daemon_call_with_flag_off_leaves_gc_alone() -> None:
    """``gc_disable=False`` MUST NOT touch GC state.

    The default is intentionally ``False`` (Phase 3 follow-up — Phase 4.5
    connectors that use ``daemon_thread_call`` during process lifetime
    must NOT silently disable GC for the rest of the run). This test
    pins that the off-path is genuinely a no-op.
    """
    if not gc.isenabled():
        gc.enable()

    result = daemon_thread_call(
        lambda: gc.isenabled(),
        timeout=1.0,
        gc_disable=False,
        label="gc-noop",
    )
    assert result is True
    assert gc.isenabled() is True


def test_daemon_call_never_re_enables_gc_after_completion() -> None:
    """The daemon helper MUST NEVER call ``gc.enable()`` itself.

    Synthesis §3.3 + ADR-0003: re-enabling GC in the daemon thread
    re-introduces the shutdown hang. This test pins the contract by
    asserting that after a successful ``gc_disable=True`` call, GC stays
    disabled (the helper does not re-enable it on the way out).
    """
    if not gc.isenabled():
        gc.enable()

    daemon_thread_call(
        lambda: None,
        timeout=1.0,
        gc_disable=True,
        label="never-re-enable",
    )
    assert gc.isenabled() is False, (
        "daemon_thread_call(gc_disable=True) re-enabled GC after the wrapped "
        "callable returned — this would re-introduce the Qwen2.5-7B shutdown "
        "hang (ADR-0003). The helper must NEVER call gc.enable() itself."
    )
