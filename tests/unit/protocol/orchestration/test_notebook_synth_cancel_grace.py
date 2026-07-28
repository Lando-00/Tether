"""Phase 9.5 fu-research-synth-cancel-grace + Phase 9.7 W4
(nho-fu-w4-synth-abandon / fu-research-synth-cancel-child-task).

Verifies the synthesize-phase async iterator's ``aclose()`` is bounded by
``_TOOL_CANCEL_GRACE_SEC`` even when the provider's cleanup is
uncooperative (catches ``CancelledError`` and re-awaits a shielded inner).

Phase 9.5 wrapped ``astream.aclose()`` in ``wait_for(..., GRACE)``. That
bound holds only for cancellable cleanup — a provider whose ``finally:``
catches ``CancelledError`` and re-awaits the inner with no shield will
make ``wait_for(aclose(), GRACE)`` block for the full inner duration.

Phase 9.7 W4 closes that gap: ``astream.aclose()`` is wrapped in a Task,
``wait_for(shield(aclose_task), GRACE)`` lets the timeout cancel only
the outer waiter (not the task), and on timeout the orchestrator
abandons the aclose task into a bounded process-level tracker
(``tether.runtime.abandoned_tasks``). MessageStop emits within the
grace contract; the abandoned cleanup task continues in the background
and drains when the provider's natural cleanup completes.

Design note on what's exercised: the bounded ``aclose()`` matters when
the synth stream is blocked inside ``__anext__`` (i.e., waiting for the
next chunk from the provider) and an EXTERNAL ``asyncio.CancelledError``
propagates through the orchestrator. The cooperative ``cancel_token``
path can only break the loop between chunks, so it's covered by the
existing ``test_cancel_inside_synthesize_stream`` test in
``test_notebook_orchestrator_cancellation.py``.
"""
from __future__ import annotations

import asyncio
import warnings
from datetime import date
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest

from tether.config.settings import ResearchSettings
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration import notebook as notebook_module
from tether.protocol.orchestration.chatty import _TOOL_CANCEL_GRACE_SEC
from tether.protocol.orchestration.notebook import NotebookOrchestrator
from tether.protocol.parsers.sliding import SlidingParser
from tether.protocol.wire.events import MessageStop, TextDelta
from tether.runtime.abandoned_tasks import get_notebook_abandoned_task_tracker
from tests.fixtures.fake_research_provider import FakeResearchProvider


GRACE_SLACK_SEC = 0.5


def _cleanup_tracker():
    """The bounded process tracker that now owns abandoned cleanup tasks."""
    return get_notebook_abandoned_task_tracker()


@pytest.fixture(autouse=True)
def _isolate_cleanup_tracker():
    """Keep tracker counts and latched health state per-test.

    Strong references to still-pending cleanup tasks are preserved across
    the reset: dropping them would let CPython finalize a pending task and
    emit "Task was destroyed but it is pending!". Only counters/latches
    are cleared.
    """
    _retain_pending_and_reset()
    yield
    _retain_pending_and_reset()


def _retain_pending_and_reset() -> None:
    tracker = _cleanup_tracker()
    pending = [task for task in tracker._tracked_tasks_for_tests() if not task.done()]
    tracker._reset_for_tests()
    for task in pending:
        tracker.track(task, kind="retained_by_test")


from tests.fixtures.recording_research_store import RecordingResearchStore


class _FakeStore(RecordingResearchStore):
    pass


class _FakeToolRegistry:
    pass


class _StubToolRunner:
    async def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "results": [
                {
                    "rank": 1,
                    "title": "stub",
                    "url": "https://example.com/stub",
                    "snippet": "stub",
                }
            ]
        }


class _SlowSynthProvider(FakeResearchProvider):
    """Synth-phase generator yields one chunk then sleeps far past the grace.

    The ``await asyncio.sleep(10)`` is cancellable, so when the orchestrator
    calls ``astream.aclose()`` the sent ``GeneratorExit`` interrupts the
    sleep cleanly and aclose returns quickly. This is the CANCELLABLE-cleanup
    path — the only path the current implementation can bound.

    For the UNCANCELLABLE-cleanup path (real MLC native engine teardown,
    blocked socket flush), see :class:`_UncancellableSynthProvider` and the
    skipped ``test_synth_cancel_with_uncancellable_cleanup_KNOWN_GAP`` test;
    closing that gap requires the architectural change tracked as
    ``fu-research-synth-cancel-child-task``.
    """

    async def stream(  # type: ignore[override]
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        phase = self._detect_phase(messages)
        if phase != "synthesizer":
            async for chunk in super().stream(
                model_name, messages, tools=tools, request_id=request_id
            ):
                yield chunk
            return
        yield "first chunk "
        # Block far past the cancellation grace; sleep is cancellable.
        await asyncio.sleep(10)
        yield "should never arrive"


class _UncancellableSynthProvider(FakeResearchProvider):
    """Synth-phase generator whose ``finally:`` re-suspends on a shielded
    ``sleep``, simulating an MLC engine teardown that holds an opaque
    handle past the cancellation grace.

    Pre-Phase-9.5: ``astream.aclose()`` was unbounded and the consumer
    would wait the full shielded duration. Phase 9.5 bounded
    ``wait_for(aclose(), GRACE)``; in this fixture that bound DOES hold
    because the shielded ``sleep()`` is cancellable at the shield outer
    (wait_for's cancel propagates to the shield, which raises
    ``CancelledError`` without cancelling the inner sleep — the inner
    becomes an orphan asyncio Task). Phase 9.7 W4 reroutes the orphan
    through ``_abandoned_cleanup_tasks`` for clean tracking and adds
    abandonment for the harder cancel-swallowing aclose case (see
    :class:`_CancelSwallowingAcloseProvider` below).
    """

    async def stream(  # type: ignore[override]
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        phase = self._detect_phase(messages)
        if phase != "synthesizer":
            async for chunk in super().stream(
                model_name, messages, tools=tools, request_id=request_id
            ):
                yield chunk
            return
        try:
            yield "first chunk "
            await asyncio.sleep(10)
            yield "should never arrive"
        finally:
            await asyncio.shield(asyncio.sleep(_TOOL_CANCEL_GRACE_SEC * 4))


class _AcloseRaisingSynthProvider(FakeResearchProvider):
    """Synth-phase generator whose finally (triggered by aclose) raises.

    Kept for documentation purposes — see test 2 below for the monkeypatched
    alternative we ended up using. This provider isn't currently exercised
    because the inner provider gen isn't automatically closed when the
    wrapper's ``async for`` cleanup runs; only the wrapper itself sees
    GeneratorExit. Forcing the wrapper to raise required monkeypatching.
    """

    async def stream(  # type: ignore[override]
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        phase = self._detect_phase(messages)
        if phase != "synthesizer":
            async for chunk in super().stream(
                model_name, messages, tools=tools, request_id=request_id
            ):
                yield chunk
            return
        try:
            yield "first chunk "
            await asyncio.sleep(10)
        except (GeneratorExit, asyncio.CancelledError):
            raise RuntimeError("synth provider aclose blew up")


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _orch(provider: FakeResearchProvider) -> NotebookOrchestrator:
    provider.set_planner_response({"key_elements": ["query"]})
    provider.set_extractor_responses(
        [
            {
                "facts": [{"text": "fact", "confidence": "high"}],
                "follow_up_queries": [],
            }
        ]
    )
    # Synth chunks come from the overridden stream(); no canned queue needed.
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
            max_facts=5,
            max_iterations=1,
            max_facts_per_extract=3,
        ),
        clock=lambda: date(2026, 5, 16),
    )


async def _wait_for_done(task: asyncio.Task[Any]) -> None:
    try:
        await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


@pytest.mark.anyio
async def test_synth_cancel_within_grace():
    """External cancel during a hung synth stream must finish within grace."""
    orch = _orch(_SlowSynthProvider())

    events: list[object] = []
    first_chunk_seen = asyncio.Event()

    async def _consume() -> None:
        async for event in orch.run(
            session_id="s-synth-cancel",
            prompt="What is X?",
            model_name="dummy",
        ):
            events.append(event)
            if isinstance(event, TextDelta):
                first_chunk_seen.set()

    consumer = asyncio.create_task(_consume())

    # Wait until the first synth chunk arrives so the orchestrator is
    # parked inside astream.__anext__ (awaiting sleep(10) for the next).
    try:
        await asyncio.wait_for(first_chunk_seen.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        consumer.cancel()
        raise

    loop = asyncio.get_running_loop()
    cancel_started = loop.time()
    consumer.cancel()

    # Orchestrator's outer try/except CancelledError must yield
    # MessageStop(cancelled) and re-raise within
    # ``_TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC``.
    try:
        await asyncio.wait_for(
            asyncio.shield(_wait_for_done(consumer)),
            timeout=_TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC,
        )
    except asyncio.TimeoutError:
        if not consumer.done():
            consumer.cancel()
        raise

    elapsed = loop.time() - cancel_started
    assert elapsed <= _TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC, (
        f"cancelled synth took {elapsed:.3f}s; aclose must be bounded at "
        f"{_TOOL_CANCEL_GRACE_SEC:.3f}s"
    )

    # The orchestrator's outer except-CancelledError block emits one
    # MessageStop(cancelled). That's the cancellation contract.
    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1, f"expected 1 MessageStop, got {stops!r}"
    assert stops[0].stop_reason == "cancelled"


@pytest.mark.anyio
async def test_synth_aclose_exception_does_not_escape(monkeypatch):
    """If the synth iterator's cleanup raises, the exception must not escape orch.run.

    Uses monkeypatching to replace ``orch._synthesize_stream`` with an async
    generator whose ``GeneratorExit`` handler raises. This isolates the
    orchestrator's ``except Exception`` in the aclose finally from the
    real provider's machinery (which has its own cleanup semantics that
    don't typically propagate exceptions through an outer async-for wrapper).
    """
    orch = _orch(FakeResearchProvider())

    async def raising_astream(**_kwargs: Any) -> AsyncGenerator[str, None]:
        try:
            yield "first "
            yield "second "
            await asyncio.sleep(10)
        except GeneratorExit:
            # Triggered by aclose() on the next ``__anext__`` after the
            # orchestrator breaks out of the loop. Raise a non-cancel
            # exception to verify the orchestrator's ``except Exception``
            # catches it (logs ``notebook.synth_aclose_error``) instead
            # of letting it escape.
            raise RuntimeError("synth provider aclose blew up")

    monkeypatch.setattr(orch, "_synthesize_stream", raising_astream)

    cancel_token = _CooperativeCancelToken()
    events: list[object] = []

    async def _consume() -> None:
        async for event in orch.run(
            session_id="s-aclose-raise",
            prompt="What is X?",
            model_name="dummy",
            cancel_token=cancel_token,
        ):
            events.append(event)
            if isinstance(event, TextDelta):
                cancel_token.cancel()

    # The aclose-raises path must NOT escape — the orchestrator's
    # aclose-except-Exception block logs ``notebook.synth_aclose_error``
    # and swallows the RuntimeError. Bound the wait to keep the test
    # honest if a regression lets the exception out.
    try:
        await asyncio.wait_for(
            _consume(),
            timeout=_TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC + 1.0,
        )
    except RuntimeError as exc:
        pytest.fail(
            f"aclose RuntimeError escaped orchestrator.run: {exc!r}; "
            f"finally block must catch and log via notebook.synth_aclose_error"
        )

    # MessageStop(cancelled) must still emit on the cooperative-cancel
    # path even when aclose blew up.
    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1, f"expected 1 MessageStop, got {stops!r}"
    assert stops[0].stop_reason == "cancelled"


async def _drain_bag(
    bag: "Any", *, timeout: float = 2.0
) -> None:
    """Drain abandoned cleanup tasks at end-of-test so they don't leak
    into the test session and trigger orphan assertions in sibling tests.

    Accepts either the tracker itself or an iterable of tasks. Snapshots
    first because the done-callback discards entries as tasks complete.
    """
    if hasattr(bag, "_tracked_tasks_for_tests"):
        snapshot = bag._tracked_tasks_for_tests()
    else:
        snapshot = list(bag)
    if not snapshot:
        return
    try:
        await asyncio.wait_for(
            asyncio.gather(*snapshot, return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        # Last-ditch: cancel everything still pending so the loop can
        # shut down cleanly even if a misbehaving fixture's cleanup
        # never finishes naturally.
        for task in snapshot:
            if not task.done():
                task.cancel()
        await asyncio.gather(*snapshot, return_exceptions=True)


async def _drain_orphans(
    baseline: "set[asyncio.Task[Any]]", *, timeout: float = 2.0
) -> None:
    """Drain any new asyncio tasks that appeared since ``baseline``.

    Catches ``asyncio.shield``'s internal task wrapping the inner
    coroutine (the source of the orphan in the
    ``_UncancellableSynthProvider`` fixture), plus any other untracked
    cleanup task. This keeps loop teardown from emitting "Task was
    destroyed but it is pending!" warnings.
    """
    current = {t for t in asyncio.all_tasks() if not t.done()}
    current.discard(asyncio.current_task())
    new = list(current - baseline)
    if not new:
        return
    try:
        await asyncio.wait_for(
            asyncio.gather(*new, return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        for task in new:
            if not task.done():
                task.cancel()
        await asyncio.gather(*new, return_exceptions=True)


# ---------------------------------------------------------------------------
# Phase 9.7 W4 — abandon uncooperative cleanup after grace
#
# Replaces the (skipped) ``test_synth_cancel_with_uncancellable_cleanup_KNOWN_GAP``
# placeholder. The pre-W4 docstring claimed the bound was violated for the
# shielded-sleep finally, but probing (W0-A §2.1) showed it actually held;
# the residual symptom was an orphan asyncio Task from
# ``asyncio.shield(asyncio.sleep)``. W4 reroutes that orphan through the
# module-level ``_abandoned_cleanup_tasks`` bag so it's discoverable and
# drainable, and adds a real abandonment path for the genuinely-
# cancel-swallowing cleanup case (next test below).
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_synth_cancel_with_uncancellable_cleanup():
    """Shielded-sleep cleanup: MessageStop within grace; no orphan warning.

    The ``_UncancellableSynthProvider`` finally awaits
    ``asyncio.shield(asyncio.sleep(GRACE*4))``. Cancellation propagates to
    the outer shield (raising ``CancelledError`` without cancelling the
    inner sleep), so the gen's frame exits within grace. The inner sleep
    coroutine — wrapped in an internal Task by ``asyncio.shield`` — is
    the orphan.

    Post-W4 behavior we verify:

    1. ``MessageStop(cancelled)`` emits within
       ``_TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC``.
    2. ``_abandoned_cleanup_tasks`` length is bounded (``<= 1``) at the
       moment of MessageStop emission — proves the abandon-bag path is
       wired even when this specific fixture does not exercise it.
    3. No "Task was destroyed but it is pending!" warning surfaces — the
       orchestrator's bounded cleanup keeps the orphan tracked via the
       bag (when its own pending/aclose task is abandoned) or lets the
       inner sleep drain naturally before the test exits.
    4. The bag drains cleanly at end-of-test so this test does not leak
       into sibling tests' ``asyncio.all_tasks()`` assertions.
    """
    orch = _orch(_UncancellableSynthProvider())

    # Snapshot live tasks BEFORE the orchestrator runs so we can detect
    # and drain orphan asyncio tasks created during cleanup (notably
    # ``asyncio.shield``'s internal Task wrapping the inner sleep in
    # the fixture's gen finally — that orphan is NOT tracked by
    # ``_abandoned_cleanup_tasks`` and would surface as a
    # "Task was destroyed but it is pending!" warning at loop teardown
    # if we did not drain it explicitly).
    baseline = {t for t in asyncio.all_tasks() if not t.done()}

    events: list[object] = []
    first_chunk_seen = asyncio.Event()

    async def _consume() -> None:
        async for event in orch.run(
            session_id="s-synth-cancel-shielded",
            prompt="What is X?",
            model_name="dummy",
        ):
            events.append(event)
            if isinstance(event, TextDelta):
                first_chunk_seen.set()

    consumer = asyncio.create_task(_consume())
    try:
        await asyncio.wait_for(first_chunk_seen.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        consumer.cancel()
        raise

    loop = asyncio.get_running_loop()
    cancel_started = loop.time()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        consumer.cancel()

        try:
            await asyncio.wait_for(
                asyncio.shield(_wait_for_done(consumer)),
                timeout=_TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC,
            )
        except asyncio.TimeoutError:
            if not consumer.done():
                consumer.cancel()
            raise

        elapsed = loop.time() - cancel_started
        # Snapshot the bag at the moment MessageStop emission has
        # already happened (consumer task has finished unwinding).
        bag_size_after_stop = _cleanup_tracker().snapshot().count

    assert elapsed <= _TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC, (
        f"cancelled synth took {elapsed:.3f}s; abandon-on-timeout must "
        f"keep MessageStop within {_TOOL_CANCEL_GRACE_SEC:.3f}s"
    )
    assert bag_size_after_stop <= 1, (
        f"abandoned-cleanup bag overflowed (size={bag_size_after_stop}); "
        f"orchestrator should append at most one entry per cancelled request"
    )

    # Cancellation contract: MessageStop(cancelled) must still emit.
    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1, f"expected 1 MessageStop, got {stops!r}"
    assert stops[0].stop_reason == "cancelled"

    # Drain the abandoned-cleanup bag (orchestrator-tracked) and any
    # orphan asyncio tasks (e.g. asyncio.shield's internal task) so
    # loop teardown does not emit "Task was destroyed but it is pending!"
    # warnings. The longest live coroutine in this fixture is the
    # ``asyncio.sleep(GRACE * 4)`` wrapped by ``asyncio.shield``.
    await _drain_bag(
        _cleanup_tracker(),
        timeout=_TOOL_CANCEL_GRACE_SEC * 8,
    )
    await _drain_orphans(baseline, timeout=_TOOL_CANCEL_GRACE_SEC * 8)

    # After draining, no "Task was destroyed but it is pending!" warning
    # should have fired from the orchestrator's cleanup path. (Warnings
    # captured during the cancel+drain window only; loop-teardown
    # warnings fire after the test exits, but the orphan drain above
    # prevents them.)
    destroyed_pending = [
        w for w in caught
        if "Task was destroyed but it is pending" in str(w.message)
    ]
    assert not destroyed_pending, (
        f"orchestrator leaked a pending Task that was GC'd: {destroyed_pending!r}"
    )


# ---------------------------------------------------------------------------
# Phase 9.7 W4 — cancel-swallowing aclose() defeats the pre-W4 bound
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_synth_cancel_with_cancel_swallowing_aclose(monkeypatch):
    """``aclose()`` whose finally catches ``CancelledError`` and re-awaits
    the inner with no shield blocks the pre-W4 ``wait_for(aclose, GRACE)``
    for the full inner duration (~GRACE*4 — empirically 1000 ms vs the
    250 ms bound).

    Post-W4: ``_bounded_aclose`` wraps ``aclose()`` in a Task and uses
    ``asyncio.shield`` so ``wait_for`` cancels only the outer waiter.
    On timeout the orchestrator abandons the task; MessageStop emits
    within ``_TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC`` and the bag
    contains the in-flight aclose task. The task drains naturally
    once the provider's swallowed cancellation chain unwinds.

    Implementation: monkeypatch ``orch._synthesize_stream`` so the
    cancel-swallowing finally lives in the OUTER asyncgen — the one
    the orch directly drives via ``__anext__``/``aclose``. Without the
    monkeypatch, the wrapper's ``async for`` would not propagate
    ``GeneratorExit`` into the inner provider stream during aclose
    unwind, so the inner gen's finally would only fire at GC time.
    """
    orch = _orch(FakeResearchProvider())

    async def cancel_swallowing_synth(**_kwargs: Any) -> AsyncGenerator[str, None]:
        try:
            yield "first chunk "
            yield "second chunk"
        finally:
            # P2 from W0-A §2.1: shield inner, then on CancelledError
            # await the inner with NO shield — defeats the pre-W4
            # ``wait_for(aclose(), GRACE)`` because wait_for blocks
            # awaiting the swallowed-cancel coroutine to actually
            # complete.
            inner = asyncio.create_task(
                asyncio.sleep(_TOOL_CANCEL_GRACE_SEC * 4)
            )
            try:
                await asyncio.shield(inner)
            except asyncio.CancelledError:
                await inner
                raise

    monkeypatch.setattr(orch, "_synthesize_stream", cancel_swallowing_synth)

    # Snapshot live tasks BEFORE the orchestrator runs so we can drain
    # any orphan tasks (e.g. ``asyncio.shield``'s internal task
    # wrapping the inner sleep) at end-of-test — keeps loop teardown
    # warning-free.
    baseline = {t for t in asyncio.all_tasks() if not t.done()}

    cancel_token = _CooperativeCancelToken()
    events: list[object] = []

    async def _consume() -> None:
        async for event in orch.run(
            session_id="s-synth-cancel-swallow",
            prompt="What is X?",
            model_name="dummy",
            cancel_token=cancel_token,
        ):
            events.append(event)
            if isinstance(event, TextDelta):
                cancel_token.cancel()

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    try:
        await asyncio.wait_for(
            _consume(),
            timeout=_TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC,
        )
    except asyncio.TimeoutError:
        pytest.fail(
            f"orch.run did not return within "
            f"{_TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC:.3f}s under "
            f"cancel-swallowing aclose; abandon-on-timeout is broken"
        )

    elapsed = loop.time() - t0
    assert elapsed <= _TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC, (
        f"cancel-swallowing aclose pinned the request for {elapsed:.3f}s; "
        f"_bounded_aclose must abandon the aclose task at "
        f"{_TOOL_CANCEL_GRACE_SEC:.3f}s"
    )

    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1, f"expected 1 MessageStop, got {stops!r}"
    assert stops[0].stop_reason == "cancelled"

    # The orchestrator should have abandoned the aclose task — bag
    # length is >= 1 at this point because the swallowed cancellation
    # is still unwinding inner. Drain to keep the test honest.
    bag_size = _cleanup_tracker().snapshot().count
    assert bag_size >= 1, (
        "expected abandon-bag to contain at least the aclose task after "
        f"MessageStop; got size={bag_size}"
    )
    await _drain_bag(
        _cleanup_tracker(),
        timeout=_TOOL_CANCEL_GRACE_SEC * 8,
    )
    assert _cleanup_tracker().snapshot().count == 0, (
        "bag did not drain — orchestrator may be leaking strong refs"
    )
    # Drain any orphan asyncio tasks left over from the fixture's
    # cleanup chain (e.g. asyncio.shield's inner sleep task), so loop
    # teardown does not emit "Task was destroyed but it is pending!".
    await _drain_orphans(baseline, timeout=_TOOL_CANCEL_GRACE_SEC * 8)


# ---------------------------------------------------------------------------
# Phase 9.7 W4 — ordering guard: never call aclose on a running asyncgen
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_bounded_aclose_skips_aclose_when_pending_chunk_not_done():
    """The ordering guard: if ``pending_chunk`` is still running after its
    grace, ``_bounded_aclose`` must NOT call ``astream.aclose()`` — that
    would raise ``RuntimeError("aclose(): asynchronous generator is
    already running")`` because the asyncgen is currently being advanced.

    Unit-level test of the helper directly: pass a not-done sentinel
    task as ``pending_chunk`` and assert that the astream's ``aclose``
    is never called (the gen remains open). The pending task is added
    to the abandon bag instead.
    """
    aclose_called = False

    async def make_open_gen() -> AsyncGenerator[int, None]:
        try:
            yield 1
            yield 2
        finally:
            # Sentinel so we can detect if aclose unwound the gen.
            nonlocal aclose_called
            aclose_called = True

    gen = make_open_gen()
    # Drive once so the gen is suspended at the first yield (not at
    # entry — calling aclose on a never-started gen is a no-op and
    # wouldn't exercise the guard).
    first = await gen.__anext__()
    assert first == 1

    # A not-done sentinel task standing in for an uncooperative
    # in-flight ``__anext__`` task.
    sentinel = asyncio.create_task(asyncio.sleep(10), name="sentinel-pending")
    try:
        await notebook_module._bounded_aclose(
            gen, pending_chunk=sentinel, kind="test_guard"
        )

        # The helper must have abandoned the sentinel (added to the
        # bag) and NOT called gen.aclose() — proven by the gen's
        # finally not having fired.
        assert sentinel in _cleanup_tracker()._tracked_tasks_for_tests(), (
            "sentinel pending task was not added to abandon bag"
        )
        assert not aclose_called, (
            "_bounded_aclose called aclose() while pending_chunk was "
            "still running — would have raised RuntimeError(\"aclose(): "
            "asynchronous generator is already running\")"
        )
        # The gen frame must still be open (cleanup did not run).
        assert gen.ag_frame is not None, (
            "gen was unexpectedly closed by _bounded_aclose"
        )
    finally:
        sentinel.cancel()
        try:
            await sentinel
        except (asyncio.CancelledError, BaseException):
            pass
        # Now safe to close the gen properly so the test doesn't leak it.
        try:
            await gen.aclose()
        except Exception:
            pass
        # Drain bag (sentinel is the only entry).
        await _drain_bag(_cleanup_tracker())


class _CooperativeCancelToken:
    """Minimal CancelToken implementation for cooperative-cancel tests."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def cancelled(self) -> bool:
        return self._cancelled
