"""HardwareWatchdog: encapsulates GC-disabled daemon-thread shutdown for
hardware-backed providers (currently MLC; future Nexa / on-device GPU).

Phase 2 (already shipped):
    - :class:`tether.runtime.watchdog_mode.WatchdogMode` enum
    - ``Engine.aclose()`` placeholder that delegates to ``provider.shutdown_all()``

This phase (``p3-watchdog-supervisor``):
    - :class:`HardwareWatchdog` lives here
    - :meth:`HardwareWatchdog.shutdown_all` uses
      :func:`tether.runtime.daemon_call.daemon_thread_call` (M1)
    - ``Engine.aclose`` body is NOT YET REPLACED — that's
      ``p3-lifespan-slim`` (Phase 3 step 35)
    - FastAPI lifespan is NOT YET REWIRED — that's ``p3-lifespan-slim``

After ``p3-lifespan-slim``, ``app/http/api.py`` will shrink: this class
takes over ``shutdown_provider_with_timeout`` + holds the watchdog state.

Synthesis §4 Phase 3 steps 29-30; B6 step 1-2.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from tether.providers.hw import (
    HardwareLifecycle,
    HwErrorClass,
)
from tether.runtime.daemon_call import daemon_thread_call
from tether.runtime.watchdog_mode import WatchdogMode

logger = logging.getLogger(__name__)

# Default budget matches the historical ``app/http/api.py:187`` hardcoded
# value (``shutdown_provider_with_timeout(provider, timeout_sec=3.0)``).
DEFAULT_SHUTDOWN_BUDGET_SEC = 3.0


class HardwareWatchdog:
    """Owns the bounded teardown for any provider implementing
    :class:`HardwareLifecycle`.

    Construction is via ``HardwareWatchdog(providers, mode=...)`` where:

    * ``providers``: a provider-ID mapping or a list of any objects (NOT all
      need to implement ``HardwareLifecycle``); we filter at construction
      time so callers can pass the engine's full provider set. Engine uses a
      mapping, which lets health and recovery remain scoped to one provider.
    * ``mode``: :attr:`WatchdogMode.LIBRARY` (caller manages signals) or
      :attr:`WatchdogMode.SERVER` (signal handlers installed externally
      by :class:`tether.runtime.signal_supervisor.SignalSupervisor`;
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
        providers: Mapping[str, Any] | List[Any],
        *,
        mode: WatchdogMode = WatchdogMode.LIBRARY,
        shutdown_budget_sec: Optional[float] = None,
    ) -> None:
        # Preserve a list for existing callers/tests while also retaining the
        # Engine's provider IDs. A raw list remains supported for direct,
        # single-provider use and intentionally has no reset routing key.
        if isinstance(providers, Mapping):
            entries = list(providers.items())
            self._provider_ids_are_available = True
        else:
            entries = [(None, provider) for provider in providers]
            self._provider_ids_are_available = False
        self._hw_provider_entries: list[tuple[Optional[str], HardwareLifecycle]] = [
            (provider_id, provider) for provider_id, provider in entries if isinstance(provider, HardwareLifecycle)
        ]
        self._hw_providers: List[HardwareLifecycle] = [provider for _, provider in self._hw_provider_entries]
        self._hw_providers_by_id: Dict[str, HardwareLifecycle] = {
            provider_id: provider for provider_id, provider in self._hw_provider_entries if provider_id is not None
        }
        self._mode = mode
        self._budget_sec = shutdown_budget_sec if shutdown_budget_sec is not None else self._compute_budget()
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
        for provider_id, p in self._hw_provider_entries:
            try:
                h = await p.hw_health()
                result: Dict[str, Any] = {
                    "status": h.status,
                    "details": h.details,
                }
                statuses.append(h.status)
            except Exception as e:
                logger.exception(
                    "HardwareWatchdog.health_summary: provider_id=%s provider %s health check failed: %s",
                    provider_id,
                    type(p).__name__,
                    e,
                )
                result = {"status": "error", "details": {"error": str(e)}}
                statuses.append("error")
            if provider_id is not None:
                result["provider_id"] = provider_id
            results.append(result)

        # Overall = worst of all
        if "error" in statuses:
            overall = "error"
        elif "degraded" in statuses:
            overall = "degraded"
        else:
            overall = "healthy"

        return {"providers": results, "overall": overall}

    async def reset_after(
        self,
        exc: BaseException,
        *,
        model_name: str,
        provider_id: Optional[str] = None,
    ) -> bool:
        """Classify and recover only the hardware provider that served a turn.

        Used by the orchestrator's exception handler — replaces the
        substring-grep ``is_fatal`` pattern at orchestrator.py:202-205. The
        orchestrator update itself is ``p3-loop-recovery``'s job, not this
        PR's.

        ``provider_id`` is required for an Engine's keyed provider mapping.
        Raw-list construction retains the historical behavior of classifying
        every hardware provider because it has no identity information to
        safely scope recovery.

        Returns:
            ``True`` if a reset was performed (provider was corrupted,
            recovered); ``False`` if no provider claimed the exception or
            all classified it as TRANSIENT/BENIGN/FATAL_UNRECOVERABLE.

        Synthesis §4 Phase 3 step 36; B6 step 7.
        """
        if provider_id is not None:
            provider = self._hw_providers_by_id.get(provider_id)
            if provider is None:
                return False
            candidates = [provider]
        elif not self._provider_ids_are_available:
            candidates = self._hw_providers
        else:
            logger.warning(
                "HardwareWatchdog.reset_after: refusing unscoped reset across %d hardware providers",
                len(self._hw_providers),
            )
            return False

        any_reset = False
        for provider in candidates:
            cls = provider.hw_classify(exc)
            if cls != HwErrorClass.FATAL_RECOVERABLE:
                continue
            logger.warning(
                "HardwareWatchdog.reset_after: provider_id=%s provider=%s "
                "classified %s as FATAL_RECOVERABLE; running hw_reset(%s)",
                provider_id,
                type(provider).__name__,
                type(exc).__name__,
                model_name,
            )
            try:
                await provider.hw_reset(model_name)
                any_reset = True
            except Exception as reset_exc:
                logger.exception(
                    "HardwareWatchdog.reset_after: provider_id=%s reset failed: %s",
                    provider_id,
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
            shutdown_callable = p.shutdown_all if hasattr(p, "shutdown_all") else (lambda: None)
            try:
                daemon_thread_call(
                    shutdown_callable,
                    timeout=provider_budget,
                    gc_disable=True,
                    label=f"shutdown-{type(p).__name__}",
                )
            except TimeoutError:
                logger.warning(
                    "HardwareWatchdog.shutdown_all: provider %s shutdown timed out (budget=%.2fs); abandoning",
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
