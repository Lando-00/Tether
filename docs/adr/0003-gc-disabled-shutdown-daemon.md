# ADR-0003: GC-disabled daemon-thread shutdown for OpenCL

- **Status**: Accepted (Phase 3 of refactor)
- **Date**: 2026-05 (Phase 3)
- **Synthesis citation**: §3.3, §4 Phase 3 (steps 28–38), §11 R11

## Context

On Snapdragon X Elite Adreno GPU, MLC-LLM's OpenCL backend exhibits a model-dependent
shutdown hang. Empirically, `Qwen2.5-7B` (`prefill_chunk_size=256`) hangs on Ctrl+C while
`Qwen3-4B` (`prefill_chunk_size=2048`) exits cleanly. Root cause: when Python's GC runs
destructors during interpreter shutdown, the OpenCL driver state on the smaller-chunk model
hangs. The pre-refactor mitigation lived as ~150 LOC of OpenCL/GC/daemon-thread code
spread across `app/http/api.py`. See `docs/runbooks/shutdown-hang-fix-summary.md` and
`docs/runbooks/model-dependent-shutdown-fix.md` for the original investigation.

## Decision

Extract all hardware-shutdown gymnastics into a **`HardwareWatchdog` runtime module**
(`src/tether/runtime/hw_watchdog.py`, ~280 LOC) with a strict invariant set:

- **GC is disabled inside the shutdown daemon thread**. `gc.disable()` is called before
  invoking any provider `terminate()`. The thread is a daemon so the OS reclaims it on
  process exit.
- **Per-engine `terminate()` calls are bounded** by `hw_per_engine_terminate_sec` (default
  `budget/4 = 0.75s`). Engines that don't exit are abandoned to GC-via-process-exit.
- **MLC teardown NEVER uses `asyncio.get_running_loop().run_in_executor(None, ...)`** — the
  default `ThreadPoolExecutor`'s threads are non-daemon and Python waits for them on exit.
  All MLC `terminate()` calls go through dedicated daemon threads via a
  `daemon_thread_call(fn, timeout)` helper. Codified in §11 R11; regression-tested in
  `tests/hardware/test_no_executor_leak.py`.
- **Three watchdog modes** via `WatchdogMode` enum: `SERVER` (signal handlers + `os._exit`
  on stuck shutdown after `force_after_sec=5.0` default) / `LIBRARY` (no signal handlers;
  raise `TimeoutError` on stuck shutdown). `SUPERVISED` was dropped for v1 (§11 R14).
- **`signal_supervisor`** is a separate module installed only by the SERVER entrypoint
  (`src/tether/server/main.py`), never by `Engine` directly.
- The watchdog implements `HardwareLifecycle` Protocol against MLC-bound providers only;
  Ollama/Dummy providers do not implement it.

## Consequences

### Positive
- `app/http/api.py` shrinks to ~50 LOC; lifespan is <30 lines.
- Library mode (`Engine.aclose()`) raises `TimeoutError` instead of hanging — predictable for
  embedders.
- Server mode reliably exits within 5s even on stuck OpenCL state, without requiring a
  second Ctrl+C.
- The "never re-enable GC" invariant is documented in one place + tested.

### Negative
- Stuck `terminate()` calls leak GPU resources for the lifetime of the daemon thread (the
  process is exiting anyway, so the OS reclaims).
- MLC contributors writing new shutdown code must remember the daemon-thread + no-default-
  executor rules. This is mitigated by the central helper and a regression test.

### Trade-offs accepted
- Resource cleanup on shutdown is best-effort, bounded in time. The alternative (clean
  cleanup) is unachievable without an upstream OpenCL fix.

## Alternatives considered

- **Subprocess isolation** — keep MLC in a child process and SIGKILL on parent exit. Designed
  in B6 §3.10 as a future seam; not built. Adds IPC cost and complicates streaming.
- **Wait indefinitely for clean shutdown** — rejected: empirically hangs; user must kill -9.
- **Re-enable GC during shutdown** — rejected: this is precisely the bug. Codified as an
  invariant.

## References

- `files/investigations/_synthesis.md` §3.3, §4 Phase 3, §6 bugs #14, #16, §11 R11, §12.1
- `docs/runbooks/shutdown-hang-fix-summary.md`, `docs/runbooks/model-dependent-shutdown-fix.md`
- `src/tether/runtime/hw_watchdog.py`, `src/tether/runtime/signal_supervisor.py`
- `src/tether/providers/hw.py` (`HardwareLifecycle` Protocol)
