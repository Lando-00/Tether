"""HardwareWatchdog: encapsulates GC-disabled daemon-thread shutdown for
hardware-backed providers (currently MLC; future Nexa / on-device GPU).

Phase 2 (already shipped):
    - :class:`tether_service.runtime.watchdog_mode.WatchdogMode` enum
    - ``Engine.aclose()`` placeholder that delegates to ``provider.shutdown_all()``

This phase (``p3-watchdog-supervisor``):
    - :class:`HardwareWatchdog` lives here
    - :meth:`HardwareWatchdog.shutdown_all` uses
      :func:`tether_service.runtime.daemon_call.daemon_thread_call` (M1)
    - ``Engine.aclose`` body is NOT YET REPLACED — that's
      ``p3-lifespan-slim`` (Phase 3 step 35)
    - FastAPI lifespan is NOT YET REWIRED — that's ``p3-lifespan-slim``

After ``p3-lifespan-slim``, ``app/http/api.py`` will shrink: this class
takes over ``shutdown_provider_with_timeout`` + holds the watchdog state.

Synthesis §4 Phase 3 steps 29-30; B6 step 1-2.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from tether_service.providers.hw import (
    HardwareLifecycle,
    HwErrorClass,
)
from tether_service.runtime.daemon_call import daemon_thread_call
from tether_service.runtime.watchdog_mode import WatchdogMode

logger = logging.getLogger(__name__)

# Default budget matches the historical ``app/http/api.py:187`` hardcoded
# value (``shutdown_provider_with_timeout(provider, timeout_sec=3.0)``).
DEFAULT_SHUTDOWN_BUDGET_SEC = 3.0


class HardwareWatchdog:
    """Owns the bounded teardown for any provider implementing
    :class:`HardwareLifecycle`.

    Construction is via ``HardwareWatchdog(providers, mode=...)`` where:

    * ``providers``: a list of any objects (NOT all need to implement
      ``HardwareLifecycle``); we filter at construction time so callers
      can pass the engine's full provider set.
    * ``mode``: :attr:`WatchdogMode.LIBRARY` (caller manages signals) or
      :attr:`WatchdogMode.SERVER` (signal handlers installed externally
      by :class:`tether_service.runtime.signal_supervisor.SignalSupervisor`;
      this class only does shutdown).

    The class itself is intentionally small — most concurrency lives in
    :func:`daemon_thread_call` (M1). ``HardwareWatchdog`` adds:

    * Health aggregation across providers (for ``/readyz``).
    * Recovery handover (``hw_reset`` on ``FATAL_RECOVERABLE``;
      Phase 3 step 36 calls this from the orchestrator).
    * Shutdown orchestration with bounded total budget.

    Synthesis §4 Phase 3 step 30; B6 step 2.
    """

    def __init__(
        self,
        providers: List[Any],
        *,
        mode: WatchdogMode = WatchdogMode.LIBRARY,
        shutdown_budget_sec: Optional[float] = None,
    ) -> None:
        # Filter to only providers that implement HardwareLifecycle.
        # DummyProvider, future stateless HTTP providers, etc. don't need
        # this watchdog and are silently skipped.
        self._hw_providers: List[HardwareLifecycle] = [
            p for p in providers if isinstance(p, HardwareLifecycle)
        ]
        self._mode = mode
        self._budget_sec = (
            shutdown_budget_sec
            if shutdown_budget_sec is not None
            else self._compute_budget()
        )
        logger.info(
            "HardwareWatchdog: %d HW provider(s), mode=%s, budget=%.2fs",
            len(self._hw_providers),
            mode.value,
            self._budget_sec,
        )

    @property
    def mode(self) -> WatchdogMode:
        return self._mode

    @property
    def shutdown_budget_sec(self) -> float:
        return self._budget_sec

    @property
    def hw_provider_count(self) -> int:
        """Number of providers that pass the HardwareLifecycle isinstance
        check. Useful for tests + the eventual /readyz route."""
        return len(self._hw_providers)

    def _compute_budget(self) -> float:
        """Total budget = max of provider budgets, default if no HW providers."""
        if not self._hw_providers:
            return DEFAULT_SHUTDOWN_BUDGET_SEC
        return max(p.hw_shutdown_budget_sec for p in self._hw_providers)

    async def health_summary(self) -> Dict[str, Any]:
        """Aggregate health across all HW providers. Used by ``/readyz``
        (Phase 3 step 37 — ``p3-lifespan-slim`` plugs this into the route).

        Returns:
            ``{"providers": [...], "overall": "healthy" | "degraded" | "error"}``
            where each provider entry has ``{"status": ..., "details": {...}}``.

        With no HW providers (e.g., DummyProvider only), returns
        ``{"providers": [], "overall": "healthy"}``.
        """
        if not self._hw_providers:
            return {"providers": [], "overall": "healthy"}

        results: List[Dict[str, Any]] = []
        statuses: List[str] = []
        for p in self._hw_providers:
            try:
                h = await p.hw_health()
                results.append({"status": h.status, "details": h.details})
                statuses.append(h.status)
            except Exception as e:
                logger.exception(
                    "HardwareWatchdog.health_summary: provider %s health check failed: %s",
                    type(p).__name__,
                    e,
                )
                results.append({"status": "error", "details": {"error": str(e)}})
                statuses.append("error")

        # Overall = worst of all
        if "error" in statuses:
            overall = "error"
        elif "degraded" in statuses:
            overall = "degraded"
        else:
            overall = "healthy"

        return {"providers": results, "overall": overall}

    async def reset_after(
        self, exc: BaseException, *, model_name: str
    ) -> bool:
        """Per Phase 3 step 36: classify the exception against each HW
        provider; if any classifies it as ``FATAL_RECOVERABLE``, call the
        provider's ``hw_reset(model_name)``.

        Used by the orchestrator's exception handler — replaces the
        substring-grep ``is_fatal`` pattern at orchestrator.py:202-205. The
        orchestrator update itself is ``p3-loop-recovery``'s job, not this
        PR's.

        Returns:
            ``True`` if a reset was performed (provider was corrupted,
            recovered); ``False`` if no provider claimed the exception or
            all classified it as TRANSIENT/BENIGN/FATAL_UNRECOVERABLE.

        Synthesis §4 Phase 3 step 36; B6 step 7.
        """
        any_reset = False
        for p in self._hw_providers:
            cls = p.hw_classify(exc)
            if cls == HwErrorClass.FATAL_RECOVERABLE:
                logger.warning(
                    "HardwareWatchdog.reset_after: provider %s classified %s "
                    "as FATAL_RECOVERABLE; running hw_reset(%s)",
                    type(p).__name__,
                    type(exc).__name__,
                    model_name,
                )
                try:
                    await p.hw_reset(model_name)
                    any_reset = True
                except Exception as reset_exc:
                    logger.exception(
                        "HardwareWatchdog.reset_after: hw_reset failed: %s",
                        reset_exc,
                    )
        return any_reset

    def shutdown_all(self) -> None:
        """Synchronous bounded shutdown of every HW provider. Uses
        :func:`daemon_thread_call` (M1) — GC disabled, abandoned on timeout.

        Each provider may have its own ``hw_shutdown_budget_sec``; total
        shutdown is bounded by :attr:`shutdown_budget_sec` (the max of
        provider budgets at construction time).

        For Phase 3 step 30: simple sequential teardown. Phase 3 step 38
        (``p3-parallel-shutdown``) will parallelize this with bounded
        per-engine timeouts via ``ThreadPoolExecutor`` + ``as_completed``.
        """
        if not self._hw_providers:
            logger.debug("HardwareWatchdog.shutdown_all: no HW providers")
            return

        logger.info(
            "HardwareWatchdog.shutdown_all: %d HW provider(s), budget=%.2fs total",
            len(self._hw_providers),
            self._budget_sec,
        )

        for p in self._hw_providers:
            provider_budget = p.hw_shutdown_budget_sec
            # Snapshot the callable at this scope to avoid late-binding bugs;
            # if the provider lacks ``shutdown_all`` (rare — most do), the
            # daemon thread runs a no-op.
            shutdown_callable = (
                p.shutdown_all
                if hasattr(p, "shutdown_all")
                else (lambda: None)
            )
            try:
                daemon_thread_call(
                    shutdown_callable,
                    timeout=provider_budget,
                    gc_disable=True,
                    label=f"shutdown-{type(p).__name__}",
                )
            except TimeoutError:
                logger.warning(
                    "HardwareWatchdog.shutdown_all: provider %s shutdown "
                    "timed out (budget=%.2fs); abandoning",
                    type(p).__name__,
                    provider_budget,
                )
            except Exception as e:
                logger.exception(
                    "HardwareWatchdog.shutdown_all: provider %s shutdown failed: %s",
                    type(p).__name__,
                    e,
                )


__all__ = ["HardwareWatchdog", "DEFAULT_SHUTDOWN_BUDGET_SEC"]
