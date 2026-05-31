"""Phase 9.5 fu-research-external-cancel-pattern.

Verifies the explore-phase ``try/finally`` cancels the in-flight
``tool_task`` within the cancellation grace when an EXTERNAL
``asyncio.CancelledError`` arrives — mirrors the chatty.py F3 pattern
at lines 1292-1322.

Without the finally, ``CancelledError`` propagating from the inner
``asyncio.wait({tool_task}, timeout=0.01)`` would unwind the loop
without cancelling the tool task, leaking an in-flight web_search.
"""
from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

import pytest

from tether.config.settings import ResearchSettings
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration.chatty import _TOOL_CANCEL_GRACE_SEC
from tether.protocol.orchestration.notebook import NotebookOrchestrator
from tether.protocol.parsers.sliding import SlidingParser
from tether.protocol.wire.events import NotebookPhaseStart
from tests.fixtures.fake_research_provider import FakeResearchProvider


# Bound the assertion window: tool task must terminate within
# ``_TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC`` after the outer cancel.
GRACE_SLACK_SEC = 0.5


class _FakeStore:
    pass


class _FakeToolRegistry:
    pass


class _SlowToolRunner:
    """Tool runner whose web_search blocks until cancelled (or N seconds)."""

    def __init__(self, *, block_seconds: float = 5.0, ignore_cancel_for: float = 0.0) -> None:
        self.block_seconds = block_seconds
        self.ignore_cancel_for = ignore_cancel_for
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tool_completed = asyncio.Event()
        self.tool_started = asyncio.Event()
        self.tool_task_was_cancelled = False

    async def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, args))
        self.tool_started.set()
        try:
            await asyncio.sleep(self.block_seconds)
            return {"results": []}
        except asyncio.CancelledError:
            self.tool_task_was_cancelled = True
            if self.ignore_cancel_for > 0:
                # Shielded sleep: pretend the tool body has cleanup work
                # that ignores cancellation for ``ignore_cancel_for`` sec.
                try:
                    await asyncio.shield(asyncio.sleep(self.ignore_cancel_for))
                except asyncio.CancelledError:
                    pass
            raise
        finally:
            self.tool_completed.set()


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _provider() -> FakeResearchProvider:
    provider = FakeResearchProvider()
    provider.set_planner_response({"key_elements": ["query under cancel"]})
    provider.set_extractor_responses(
        [
            {
                "facts": [{"text": "unused", "confidence": "high"}],
                "follow_up_queries": [],
            }
        ]
    )
    provider.set_synthesizer_response("unused")
    return provider


def _orch(tool_runner: _SlowToolRunner) -> NotebookOrchestrator:
    return NotebookOrchestrator(
        provider=_provider(),
        store=_FakeStore(),
        tool_registry=_FakeToolRegistry(),
        tool_runner=tool_runner,  # type: ignore[arg-type]
        parser=SlidingParser(),
        config=OrchestratorConfig(
            max_tool_loops=3,
            auto_reload_on_fatal_error=False,
            save_thinking=False,
            include_thinking_in_history=False,
        ),
        research_settings=ResearchSettings(
            max_facts=5,
            max_iterations=2,
            max_facts_per_extract=3,
        ),
        clock=lambda: date(2026, 5, 16),
    )


async def _run_until_explore_then_cancel(
    orch: NotebookOrchestrator, runner: _SlowToolRunner
) -> tuple[list[object], float]:
    """Start the orchestrator in a task, wait for explore-phase start, then cancel.

    Returns (collected_events, elapsed_seconds_from_cancel_to_task_done).
    """
    events: list[object] = []
    explore_seen = asyncio.Event()

    async def _consume() -> None:
        async for event in orch.run(
            session_id="s-extcancel",
            prompt="What is X?",
            model_name="dummy",
        ):
            events.append(event)
            if isinstance(event, NotebookPhaseStart) and event.phase == "explore":
                explore_seen.set()

    consumer = asyncio.create_task(_consume())
    # Wait until the orchestrator has entered the explore phase AND the
    # slow tool has actually started running. Bound this to avoid hanging
    # the test on a logic regression.
    try:
        await asyncio.wait_for(explore_seen.wait(), timeout=2.0)
        await asyncio.wait_for(runner.tool_started.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        consumer.cancel()
        raise

    # External cancel of the OUTER task — this is the F3 scenario:
    # CancelledError propagates from inside `asyncio.wait(... timeout=0.01)`.
    cancel_started = asyncio.get_running_loop().time()
    consumer.cancel()

    # The consumer must finish within the cancellation grace + slack.
    try:
        await asyncio.wait_for(
            asyncio.shield(_await_done(consumer)),
            timeout=_TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC,
        )
    except asyncio.TimeoutError:
        # Last-ditch: force-kill the consumer so the test doesn't hang.
        if not consumer.done():
            consumer.cancel()
        raise

    elapsed = asyncio.get_running_loop().time() - cancel_started
    return events, elapsed


async def _await_done(task: asyncio.Task[None]) -> None:
    try:
        await task
    except (asyncio.CancelledError, Exception):  # noqa: BLE001
        pass


@pytest.mark.anyio
async def test_external_cancel_cancels_tool_task_within_grace():
    # Snapshot live tasks before exercising the orchestrator so we can
    # assert no orphan task survives.
    baseline = {t for t in asyncio.all_tasks() if not t.done()}

    runner = _SlowToolRunner(block_seconds=10.0, ignore_cancel_for=0.0)
    orch = _orch(runner)

    _events, elapsed = await _run_until_explore_then_cancel(orch, runner)

    # The tool task either honored the cancel or was bounded by grace.
    assert runner.tool_task_was_cancelled, (
        "external cancel must have propagated to the tool task; "
        "the finally block is missing the cancel call"
    )
    # Cleanup latency is bounded by the cancellation grace + a small slack.
    assert elapsed <= _TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC, (
        f"cancellation took {elapsed:.3f}s > {_TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC:.3f}s"
    )

    # Yield once so any cancelled tasks finish unwinding.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # No orphaned tasks beyond the baseline.
    live = {t for t in asyncio.all_tasks() if not t.done()} - baseline
    # Exclude the current task itself (the test coroutine).
    live.discard(asyncio.current_task())
    assert not live, f"orphaned tasks survived external cancel: {live!r}"


@pytest.mark.anyio
async def test_external_cancel_with_tool_ignoring_cancel_is_bounded():
    # Even if the tool's CancelledError handler does cleanup work that
    # ignores cancellation for ~1s, the orchestrator's grace bound must
    # still hold: total cleanup latency <= _TOOL_CANCEL_GRACE_SEC + slack.
    baseline = {t for t in asyncio.all_tasks() if not t.done()}

    runner = _SlowToolRunner(block_seconds=10.0, ignore_cancel_for=1.0)
    orch = _orch(runner)

    _events, elapsed = await _run_until_explore_then_cancel(orch, runner)

    assert runner.tool_task_was_cancelled
    # Grace-bounded: orchestrator gave up on the tool after _TOOL_CANCEL_GRACE_SEC.
    assert elapsed <= _TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC, (
        f"cancellation took {elapsed:.3f}s; orchestrator must bound cleanup "
        f"at {_TOOL_CANCEL_GRACE_SEC + GRACE_SLACK_SEC:.3f}s even when tool ignores cancel"
    )

    # Wait a touch longer so the tool's shielded sleep can exit naturally;
    # without this its stuck task would show up as orphaned.
    await asyncio.sleep(runner.ignore_cancel_for + 0.2)

    live = {t for t in asyncio.all_tasks() if not t.done()} - baseline
    live.discard(asyncio.current_task())
    assert not live, f"orphaned tasks survived bounded external cancel: {live!r}"
