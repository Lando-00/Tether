"""Phase 9.6 I-4 (W2-B): NotebookPhaseProgress heartbeats during long phases.

The motivating symptom was a 43 s live silence during the first provider call
(``_plan`` -> first ``__anext__``). The fix wraps the planner / extractor /
synthesizer provider streams in a single-consumer
``asyncio.wait({pending}, timeout=_HEARTBEAT_INTERVAL_SEC)`` loop. Whenever the
timeout elapses with no chunk in hand, the orchestrator emits a
:class:`NotebookPhaseProgress` event so consumers (UI, CLI) see liveness.

These tests monkeypatch the module-level ``_HEARTBEAT_INTERVAL_SEC`` constant
to a small value (~10–50 ms) so the suite runs fast while still exercising the
real wait/timeout path (we do not stub ``asyncio.wait`` itself).

Coverage:

* Fast planner -> no heartbeat (negative control, must not pass against the
  pre-diff code by accident).
* Slow planner -> heartbeat with ``phase="plan"``, ``iteration=0``, ordered
  before ``NotebookQueryAdded``.
* Slow extract -> heartbeat with ``phase="extract"`` and the correct
  ``iteration`` (1-indexed, matching ``NotebookPhaseStart`` convention).
* Slow synth -> heartbeat with ``phase="synthesize"``, ``iteration=0``, ordered
  before the final ``MessageStop``.
* External cancel during a heartbeat idle wait -> the in-flight ``__anext__()``
  task is cancelled within ``_TOOL_CANCEL_GRACE_SEC`` and no orphan task
  survives.
"""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest

from tether.config.settings import ResearchSettings
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration import notebook as notebook_module
from tether.protocol.orchestration.chatty import _TOOL_CANCEL_GRACE_SEC
from tether.protocol.orchestration.notebook import NotebookOrchestrator
from tether.protocol.parsers.sliding import SlidingParser
from tether.protocol.wire.events import (
    MessageStop,
    NotebookPhaseProgress,
    NotebookPhaseStart,
    NotebookQueryAdded,
    TextDelta,
)
from tests.fixtures.fake_research_provider import FakeResearchProvider


# Cancellation tests use the same slack budget as the Phase 9.5 synth-cancel
# suite (test_notebook_synth_cancel_grace.py): the grace bound is 250 ms; we
# allow 500 ms for scheduling jitter on top of that.
GRACE_SLACK_SEC = 0.5


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeStore:
    pass


class _FakeToolRegistry:
    pass


class _StubToolRunner:
    """Returns a fixed web_search payload so the explore phase makes
    progress without hitting the network. Tests override ``run`` if they
    need different semantics."""

    async def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        del name, args
        return {
            "results": [
                {
                    "rank": 1,
                    "title": "stub",
                    "url": "https://example.com/stub",
                    "snippet": "stub snippet body",
                }
            ]
        }


class _SlowPhaseProvider(FakeResearchProvider):
    """Inject an async sleep into a chosen phase's ``stream()``.

    The sleep happens *before* the first chunk is yielded — it sits in
    front of the canned ``super().stream()`` async generator — so the
    consumer's ``__anext__()`` await is what blocks. That is exactly the
    cold-load shape ``_HEARTBEAT_INTERVAL_SEC`` is meant to catch.
    """

    def __init__(self, *, slow_phase: str, sleep_sec: float) -> None:
        super().__init__()
        self._slow_phase = slow_phase
        self._sleep_sec = sleep_sec

    async def stream(  # type: ignore[override]
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        phase = self._detect_phase(messages)
        if phase == self._slow_phase:
            await asyncio.sleep(self._sleep_sec)
        async for chunk in super().stream(
            model_name, messages, tools=tools, request_id=request_id
        ):
            yield chunk


class _HangingPhaseProvider(FakeResearchProvider):
    """A provider whose chosen phase blocks forever inside the first
    ``__anext__()`` (until cancelled). Used by the external-cancel test."""

    def __init__(self, *, hung_phase: str) -> None:
        super().__init__()
        self._hung_phase = hung_phase
        self.entered = asyncio.Event()

    async def stream(  # type: ignore[override]
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        phase = self._detect_phase(messages)
        if phase == self._hung_phase:
            self.entered.set()
            # Cancellable sleep. Sized so the test's grace assertion is
            # meaningful (we *must* cancel before this returns).
            await asyncio.sleep(60.0)
        async for chunk in super().stream(
            model_name, messages, tools=tools, request_id=request_id
        ):
            yield chunk


def _orch(
    provider: FakeResearchProvider,
    *,
    max_iterations: int = 1,
    max_facts: int = 5,
) -> NotebookOrchestrator:
    # Canned plan/extract/synth responses so non-slow phases finish fast.
    provider.set_planner_response({"key_elements": ["query about X"]})
    provider.set_extractor_responses(
        [
            {
                "facts": [{"text": "fact about X", "confidence": "high"}],
                "follow_up_queries": [],
            }
        ]
    )
    provider.set_synthesizer_response("Final answer about X.")
    return NotebookOrchestrator(
        provider=provider,
        store=_FakeStore(),
        tool_registry=_FakeToolRegistry(),
        tool_runner=_StubToolRunner(),  # type: ignore[arg-type]
        parser=SlidingParser(),
        config=OrchestratorConfig(
            max_tool_loops=3,
            auto_reload_on_fatal_error=False,
            save_thinking=False,
            include_thinking_in_history=False,
        ),
        research_settings=ResearchSettings(
            max_facts=max_facts,
            max_iterations=max_iterations,
            max_facts_per_extract=3,
        ),
        clock=lambda: date(2026, 6, 1),
    )


async def _drain(
    orch: NotebookOrchestrator,
    *,
    session_id: str,
    prompt: str = "What is X?",
    model_name: str = "dummy",
) -> list[object]:
    events: list[object] = []
    async for event in orch.run(
        session_id=session_id, prompt=prompt, model_name=model_name
    ):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# 1. Fast planner: no heartbeat
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_fast_planner_emits_no_progress(monkeypatch):
    """When all phases complete well within the heartbeat interval, no
    :class:`NotebookPhaseProgress` event must be emitted.

    Sets the interval to 1.0 s — well above what any fake-provider phase
    actually takes (microseconds). If a stray heartbeat leaks anyway, the
    assertion fires and we know either (a) the bound is not enforced or
    (b) scheduling jitter is large enough to violate the contract.
    """
    monkeypatch.setattr(notebook_module, "_HEARTBEAT_INTERVAL_SEC", 1.0)

    orch = _orch(FakeResearchProvider())
    events = await _drain(orch, session_id="s-fast")

    progress = [e for e in events if isinstance(e, NotebookPhaseProgress)]
    assert progress == [], (
        "fast planner/extract/synth must NOT emit NotebookPhaseProgress; "
        f"got {progress!r}"
    )

    # Sanity: the turn actually completed end-to-end.
    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1
    assert stops[0].stop_reason == "complete"


# ---------------------------------------------------------------------------
# 2. Slow planner: at least one heartbeat, ordered before query_added
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_slow_planner_emits_heartbeat_before_query_added(monkeypatch):
    """Slow planner stream -> at least one
    ``NotebookPhaseProgress(phase="plan", iteration=0)`` arrives BEFORE
    any ``NotebookQueryAdded`` event."""
    interval_sec = 0.02
    monkeypatch.setattr(
        notebook_module, "_HEARTBEAT_INTERVAL_SEC", interval_sec
    )

    # Sleep ~5× the interval so we comfortably see at least one heartbeat
    # despite jitter, while still completing in <200 ms.
    provider = _SlowPhaseProvider(slow_phase="planner", sleep_sec=interval_sec * 5)
    orch = _orch(provider)

    events = await _drain(orch, session_id="s-slow-plan")

    plan_progress = [
        e
        for e in events
        if isinstance(e, NotebookPhaseProgress) and e.phase == "plan"
    ]
    assert plan_progress, (
        "slow planner must emit at least one NotebookPhaseProgress(phase=plan)"
    )
    for hb in plan_progress:
        assert hb.iteration == 0, (
            f"plan-phase heartbeat iteration must be 0; got {hb.iteration}"
        )
        # elapsed_ms must reflect real wait time. The minimum is the
        # heartbeat interval (the wait we just woke from). Use 50% of the
        # interval as the lower bound to absorb scheduling jitter on slow
        # CI runners; the orchestrator's wall clock is monotonic, so
        # this still proves the timer ran.
        assert hb.elapsed_ms >= int(interval_sec * 1000 * 0.5), (
            f"heartbeat elapsed_ms {hb.elapsed_ms} below interval half-bound "
            f"({int(interval_sec * 1000 * 0.5)})"
        )

    # Order: every plan-phase heartbeat must precede every query_added.
    plan_progress_seqs = [e.seq for e in plan_progress]
    query_added_seqs = [
        e.seq for e in events if isinstance(e, NotebookQueryAdded)
    ]
    assert query_added_seqs, "fixture must emit at least one query_added"
    assert max(plan_progress_seqs) < min(query_added_seqs), (
        f"plan-phase heartbeats {plan_progress_seqs} must all precede the "
        f"first query_added {min(query_added_seqs)}"
    )

    # And the plan-phase heartbeats sit AFTER NotebookPhaseStart(phase=plan)
    # (the phase_start must already be on the wire before progress events).
    plan_start = next(
        e
        for e in events
        if isinstance(e, NotebookPhaseStart) and e.phase == "plan"
    )
    assert plan_start.seq < min(plan_progress_seqs)


# ---------------------------------------------------------------------------
# 3. Slow extract: heartbeat with phase=extract, iteration=1
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_slow_extract_emits_heartbeat_with_iteration(monkeypatch):
    """Slow extractor stream -> at least one
    ``NotebookPhaseProgress(phase="extract", iteration=1)`` arrives during
    the first explore/extract loop iteration."""
    interval_sec = 0.02
    monkeypatch.setattr(
        notebook_module, "_HEARTBEAT_INTERVAL_SEC", interval_sec
    )

    provider = _SlowPhaseProvider(slow_phase="extractor", sleep_sec=interval_sec * 5)
    orch = _orch(provider)

    events = await _drain(orch, session_id="s-slow-extract")

    extract_progress = [
        e
        for e in events
        if isinstance(e, NotebookPhaseProgress) and e.phase == "extract"
    ]
    assert extract_progress, (
        "slow extractor must emit at least one NotebookPhaseProgress(phase=extract)"
    )
    for hb in extract_progress:
        # Iteration follows the NotebookPhaseStart(phase=extract) convention:
        # 1-indexed because notebook_state.iteration is incremented BEFORE
        # the explore/extract emit (run() lines around iteration += 1).
        assert hb.iteration == 1, (
            f"extract-phase heartbeat iteration must be 1; got {hb.iteration}"
        )
        assert hb.elapsed_ms >= int(interval_sec * 1000 * 0.5)

    # Order: extract heartbeats sit after NotebookPhaseStart(phase=extract)
    # and before MessageStop.
    extract_start = next(
        e
        for e in events
        if isinstance(e, NotebookPhaseStart) and e.phase == "extract"
    )
    stop = next(e for e in events if isinstance(e, MessageStop))
    for hb in extract_progress:
        assert extract_start.seq < hb.seq < stop.seq


# ---------------------------------------------------------------------------
# 4. Slow synth: heartbeat with phase=synthesize, iteration=0, before MessageStop
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_slow_synth_emits_heartbeat_before_message_stop(monkeypatch):
    """Slow synthesizer stream -> at least one
    ``NotebookPhaseProgress(phase="synthesize", iteration=0)`` arrives
    BEFORE the final ``MessageStop``."""
    interval_sec = 0.02
    monkeypatch.setattr(
        notebook_module, "_HEARTBEAT_INTERVAL_SEC", interval_sec
    )

    provider = _SlowPhaseProvider(slow_phase="synthesizer", sleep_sec=interval_sec * 5)
    orch = _orch(provider)

    events = await _drain(orch, session_id="s-slow-synth")

    synth_progress = [
        e
        for e in events
        if isinstance(e, NotebookPhaseProgress) and e.phase == "synthesize"
    ]
    assert synth_progress, (
        "slow synthesizer must emit at least one NotebookPhaseProgress(phase=synthesize)"
    )
    for hb in synth_progress:
        assert hb.iteration == 0, (
            f"synth-phase heartbeat iteration must be 0; got {hb.iteration}"
        )
        assert hb.elapsed_ms >= int(interval_sec * 1000 * 0.5)

    # MessageStop must follow every synth heartbeat.
    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1
    assert stops[0].stop_reason == "complete"
    assert max(hb.seq for hb in synth_progress) < stops[0].seq

    # Sanity: the synth final answer made it through despite the
    # heartbeat-driving slowdown.
    text_events = [e for e in events if isinstance(e, TextDelta)]
    combined = "".join(e.text for e in text_events)
    assert combined == "Final answer about X."


# ---------------------------------------------------------------------------
# 5. External cancel during heartbeat idle wait
# ---------------------------------------------------------------------------


async def _wait_done(task: asyncio.Task[Any]) -> None:
    try:
        await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


@pytest.mark.anyio
async def test_external_cancel_during_heartbeat_idle_wait(monkeypatch):
    """External cancel arriving while the orchestrator is parked in the
    heartbeat ``asyncio.wait`` must:

    1. Propagate through the orchestrator within ``_TOOL_CANCEL_GRACE_SEC + slack``
       (the bounded cleanup contract from Phase 9.5).
    2. Cancel the in-flight ``__anext__()`` task — no orphan survives.
    3. Still emit a ``MessageStop(stop_reason="cancelled")`` (the
       cancellation contract from Phase 9.5).
    """
    interval_sec = 0.02
    monkeypatch.setattr(
        notebook_module, "_HEARTBEAT_INTERVAL_SEC", interval_sec
    )

    # Snapshot live tasks BEFORE the orchestrator starts so we can detect
    # orphans (uses the same idiom as test_notebook_external_cancel.py).
    baseline = {t for t in asyncio.all_tasks() if not t.done()}

    provider = _HangingPhaseProvider(hung_phase="planner")
    orch = _orch(provider)

    events: list[object] = []
    first_heartbeat = asyncio.Event()

    async def _consume() -> None:
        async for event in orch.run(
            session_id="s-cancel-hb",
            prompt="What is X?",
            model_name="dummy",
        ):
            events.append(event)
            if isinstance(event, NotebookPhaseProgress) and event.phase == "plan":
                first_heartbeat.set()

    consumer = asyncio.create_task(_consume())

    # Wait until the orchestrator is provably parked in the heartbeat wait
    # loop (one heartbeat has fired) AND the provider's hang point has
    # been entered. Bound the wait so a logic regression doesn't hang the
    # test forever.
    try:
        await asyncio.wait_for(provider.entered.wait(), timeout=2.0)
        await asyncio.wait_for(first_heartbeat.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        consumer.cancel()
        raise

    loop = asyncio.get_running_loop()
    cancel_started = loop.time()
    consumer.cancel()

    # The consumer must complete (cleanup unwind) within the grace bound.
    try:
        await asyncio.wait_for(
            asyncio.shield(_wait_done(consumer)),
            timeout=_TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC,
        )
    except asyncio.TimeoutError:
        if not consumer.done():
            consumer.cancel()
        raise

    elapsed = loop.time() - cancel_started
    assert elapsed <= _TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC, (
        f"external cancel during heartbeat idle wait took {elapsed:.3f}s; "
        f"finally block must bound pending __anext__ + aclose within "
        f"{_TOOL_CANCEL_GRACE_SEC:.3f}s"
    )

    # Phase 9.5 cancellation contract: MessageStop(cancelled) must still emit.
    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1, f"expected 1 MessageStop, got {stops!r}"
    assert stops[0].stop_reason == "cancelled"

    # Yield a couple of ticks so any in-flight cancelled tasks finish
    # unwinding before we sample asyncio.all_tasks().
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    live = {t for t in asyncio.all_tasks() if not t.done()} - baseline
    live.discard(asyncio.current_task())
    assert not live, (
        f"orphan __anext__ task survived external cancel: {live!r}"
    )
