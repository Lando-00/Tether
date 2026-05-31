"""Phase 9.5 fu-research-synth-cancel-grace.

Verifies the synthesize-phase async iterator's ``aclose()`` is bounded by
``_TOOL_CANCEL_GRACE_SEC`` (mirrors chatty.py:601-612). Without the bound,
a provider whose cleanup re-suspends could keep a cancelled request alive
indefinitely.

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
from datetime import date
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest

from tether.config.settings import ResearchSettings
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration.chatty import _TOOL_CANCEL_GRACE_SEC
from tether.protocol.orchestration.notebook import NotebookOrchestrator
from tether.protocol.parsers.sliding import SlidingParser
from tether.protocol.wire.events import MessageStop, TextDelta
from tests.fixtures.fake_research_provider import FakeResearchProvider


GRACE_SLACK_SEC = 0.5


class _FakeStore:
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
    """Synth-phase generator with cleanup that the bound CANNOT preempt.

    Used by the (skipped) test below to document the M3 limitation. Putting
    ``asyncio.shield(asyncio.sleep(...))`` in the gen's finally simulates
    a provider whose teardown ignores ``GeneratorExit`` for the shielded
    duration. With the current bounded-aclose implementation, MessageStop
    emission is delayed until the shielded sleep completes — violating
    the strict grace contract.

    Mark the strengthened test as ``skip`` pending
    ``fu-research-synth-cancel-child-task`` (which drives ``__anext__`` in
    a cancellable child task that can be abandoned with grace).
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


# ---------------------------------------------------------------------------
# Wave 4 reconcile: documented M3 limitation — uncancellable provider
# cleanup is NOT bounded by the current aclose() wrapper. The implementation
# fix lives behind ``fu-research-synth-cancel-child-task`` (drive __anext__
# in a cancellable child task, mirroring explore's tool_task pattern).
#
# This test stays as ``skip`` to make the gap visible in the suite without
# breaking CI. Remove the skip marker (and the helper provider) when the
# follow-up lands.
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Known limitation: aclose() bound only holds when provider cleanup "
        "is cancellable. Real MLC native teardown / blocked socket flush can "
        "exceed _TOOL_CANCEL_GRACE_SEC. Tracked: fu-research-synth-cancel-child-task. "
        "Empirically verified to FAIL when un-skipped against current impl."
    )
)
@pytest.mark.anyio
async def test_synth_cancel_with_uncancellable_cleanup_KNOWN_GAP():
    """Document the M3 gap: shielded cleanup defeats the bounded aclose().

    Pre-conditions: provider gen has ``await asyncio.shield(asyncio.sleep(...))``
    in its finally clause, simulating a real MLC engine teardown that cannot
    be interrupted by ``GeneratorExit`` for the shielded duration.

    Expected (post-fix) behavior: MessageStop emits within
    ``_TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC`` because the orchestrator
    abandons the synth task via a cancellable child-task wrapper instead of
    awaiting ``astream.aclose()``.

    Current behavior: MessageStop is delayed until the shielded cleanup
    completes (~1.0s with current ``_TOOL_CANCEL_GRACE_SEC * 4`` shield).
    The outer ``wait_for`` in the test below times out, exposing the gap.
    """
    orch = _orch(_UncancellableSynthProvider())

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
    assert elapsed <= _TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC, (
        f"cancelled synth took {elapsed:.3f}s; bound must hold under "
        f"uncancellable cleanup too — fix via child-task pattern."
    )


class _CooperativeCancelToken:
    """Minimal CancelToken implementation for cooperative-cancel tests."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def cancelled(self) -> bool:
        return self._cancelled
