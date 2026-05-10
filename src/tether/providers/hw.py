"""HardwareLifecycle Protocol + supporting types.

The Protocol defines the seam between Engine/Orchestrator and any provider
that owns native hardware resources (MLC's OpenCL+TVM, future Nexa NPU,
future on-device GPU providers). Implementations of this Protocol are
plugged into ``HardwareWatchdog`` (Phase 3 step 30) which provides:

    - GC-disabled daemon-thread shutdown (R5)
    - Per-engine bounded teardown
    - Health probing for ``/readyz``
    - Recovery handover (``hw_reset`` on ``FatalProviderError``)

This module is intentionally additive in this PR — nothing else in
``tether_service`` imports it yet. ``MLCProvider`` will start implementing
it in p3-mlc-as-hwlifecycle.

Synthesis §2.3 + §4 Phase 3 step 32; B6 step 4; A3 (HwErrorClass + HwHealth).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Literal, Protocol, runtime_checkable


class HwErrorClass(Enum):
    """Classification of a provider-side exception in terms of what
    recovery action the watchdog should take.

    Members:
        BENIGN              — non-error / informational; no action.
        TRANSIENT           — retryable; do NOT reset; surface to user.
        FATAL_RECOVERABLE   — provider native state corrupted but
                              recoverable via ``hw_reset(model_name)``
                              (teardown + eager reload).
        FATAL_UNRECOVERABLE — entire process should restart (driver crash,
                              out-of-VRAM that won't release). Watchdog logs
                              and the engine refuses further requests.
    """

    BENIGN = "benign"
    TRANSIENT = "transient"
    FATAL_RECOVERABLE = "fatal_recoverable"
    FATAL_UNRECOVERABLE = "fatal_unrecoverable"


@dataclass(frozen=True)
class HwHealth:
    """Snapshot of a hardware-backed provider's health for ``/readyz``.

    Attributes:
        status: high-level signal — ``healthy`` means ready to accept
            requests; ``degraded`` means partial capability; ``error``
            means unusable. Note that ``Literal`` is a static-typing hint
            and is NOT enforced at runtime by stdlib dataclasses; future
            mypy gating will catch invalid values.
        details: free-form diagnostic dict (model count, last error, etc.)
            for debugging. NOT shown in ``/readyz`` public output by
            default — see Phase 7 (observability) for the redaction policy.

    Frozen so callers can't mutate the snapshot; build a new one each call.
    """

    status: Literal["healthy", "degraded", "error"]
    details: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class HardwareLifecycle(Protocol):
    """A provider that owns native hardware resources (e.g.,
    ``MLCProvider``'s OpenCL+TVM engine cache, future ``NexaProvider``'s
    NPU runtime).

    Implementations are detected by ``HardwareWatchdog`` at startup via
    ``isinstance(provider, HardwareLifecycle)``. Providers that do NOT own
    native resources (e.g., ``DummyProvider``, future ``OllamaProvider``
    going over HTTP) do NOT need to implement this Protocol — the watchdog
    will skip them.

    Synthesis §2.3 + §4 Phase 3 step 32.
    """

    def hw_classify(self, exc: BaseException) -> HwErrorClass:
        """Classify an exception for recovery action.

        Called by the orchestrator's exception handler (Phase 3 step 36
        replaces the substring-grep). Implementations inspect the
        exception type/message/cause and return the appropriate class.
        Synthesis §6 row 13.
        """
        ...

    async def hw_reset(self, model_name: str) -> None:
        """Tear down + EAGER reload the named model.

        Called when ``hw_classify`` returns ``FATAL_RECOVERABLE``.
        Implementations should:

            1. Abort any in-flight request on the engine for ``model_name``.
            2. Tear down the engine (release OpenCL/TVM handles).
            3. EAGERLY reload — do NOT defer to next request, because the
               next request may immediately hit the corrupted state again.

        Synthesis §4 Phase 3 step 36; A3 step 6.
        """
        ...

    async def hw_health(self) -> HwHealth:
        """Cheap status check.

        MUST NOT touch the model graph or run inference. Used by the
        ``/readyz`` handler (Phase 3 step 37 replaces the ``list_models()``
        probe with this).

        Implementations should report:

            healthy   — engines initialized, no recent errors
            degraded  — one engine errored recently but others healthy
            error     — all engines errored or watchdog reset failed

        Synthesis §4 Phase 3 step 37.
        """
        ...

    @property
    def hw_shutdown_budget_sec(self) -> float:
        """Total time budget for ``aclose`` of all engines this provider
        owns.

        The watchdog's daemon thread waits this long before abandoning
        cleanup. Default for most providers: 3.0s (matches the current
        ``api.py:189`` hardcoded value).

        Synthesis §4 Phase 3 step 30.
        """
        ...

    @property
    def hw_per_engine_terminate_sec(self) -> float:
        """Per-engine cap when shutting down multiple engines in parallel
        (Phase 3 step 38).

        Default: ``hw_shutdown_budget_sec / 4``.

        Synthesis §4 Phase 3 step 38.
        """
        ...


__all__ = ["HwErrorClass", "HwHealth", "HardwareLifecycle"]
