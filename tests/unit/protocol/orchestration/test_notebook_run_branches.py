"""NotebookOrchestrator run-loop branch coverage (Phase 9.5, Wave 3).

These tests exercise branches of :meth:`NotebookOrchestrator.run` that were
not previously covered by the smoke / integration suite:

* Explore tool error → loop continues silently (warning log only).
* ``max_iterations`` exhausted → :class:`NotebookLimitReached` then
  synthesize still runs.
* ``max_facts`` hit mid-iteration → loop short-circuits before refine and
  synthesize still runs on the partial Notebook.
* Refine produces empty queue → clean termination, no limit event, no
  refine phase event.
* :meth:`NotebookState.try_add_fact` higher-confidence replacement → second
  :class:`NotebookFactAdded` IS emitted (observed semantics, documented
  below).

Citations: ADR-0020 §D5 (bounds), §D6 (event ordering); post-Phase-9
testing review (run-loop branch coverage gap).
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pytest
import structlog

from tests.fixtures.fake_research_provider import FakeResearchProvider
from tests.fixtures.recording_research_store import RecordingResearchStore
from tether.config.settings import ResearchSettings
from tether.core.logging import reset_logging_for_tests
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration.notebook import NotebookOrchestrator
from tether.protocol.parsers.sliding import SlidingParser
from tether.protocol.wire.events import (
    MessageStop,
    NotebookFactAdded,
    NotebookLimitReached,
    NotebookPhaseStart,
)


class _FakeStore(RecordingResearchStore):
    pass


class _FakeToolRegistry:
    pass


class _ScriptedToolRunner:
    """Tool runner whose responses are a scripted list (FIFO).

    Each scripted entry is either a ``dict`` (returned as the tool result)
    or an :class:`Exception` instance (raised when consumed).
    """

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, args))
        if not self._script:
            raise AssertionError(
                f"ScriptedToolRunner exhausted; unexpected call to {name}({args})"
            )
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _ok_result(query: str) -> dict[str, Any]:
    return {
        "results": [
            {
                "rank": 1,
                "title": f"Result for {query}",
                "url": "https://example.com/r",
                "snippet": f"Snippet about {query}.",
            }
        ],
        "meta": {"query": query},
    }


def _build_orch(
    *,
    provider: FakeResearchProvider,
    tool_runner: _ScriptedToolRunner,
    research_settings: ResearchSettings,
) -> NotebookOrchestrator:
    return NotebookOrchestrator(
        provider=provider,
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
        research_settings=research_settings,
        clock=lambda: date(2026, 5, 16),
    )


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def _reset_logging_for_capture():
    """Restore a clean structlog config before each test.

    See ``test_notebook_query_redaction.py`` for the full rationale —
    Phase 9.7 W2 moved the cached-proxy invalidation into
    ``tether.core.logging.reset_logging_for_tests`` so notebook tests
    no longer need a bespoke fixture.

    Tracked: ``fu-notebook-tests-structlog-isolation``.
    """
    reset_logging_for_tests()
    yield
    reset_logging_for_tests()


# ---------------------------------------------------------------------------
# Test 1 — explore tool error continues the loop
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_explore_tool_error_continues_loop():
    """Iteration 1 tool raises → ``notebook.explore_tool_error`` logged and
    the loop continues into iteration 2 which succeeds and yields a fact.
    """
    provider = FakeResearchProvider()
    provider.set_planner_response({"key_elements": ["q1", "q2"]})
    # Only one extract is reached (iter 1 short-circuits before extract).
    provider.set_extractor_responses(
        [
            {
                "facts": [{"text": "fact from q2", "confidence": "high"}],
                "follow_up_queries": [],
            }
        ]
    )
    provider.set_synthesizer_response("Synth.")

    tool_runner = _ScriptedToolRunner(
        [RuntimeError("simulated brave 500"), _ok_result("q2")]
    )
    orch = _build_orch(
        provider=provider,
        tool_runner=tool_runner,
        research_settings=ResearchSettings(
            max_facts=10, max_iterations=10, max_facts_per_extract=5
        ),
    )

    with structlog.testing.capture_logs() as captured:
        events = [
            e
            async for e in orch.run(
                session_id="s1",
                prompt="Tell me about q.",
                model_name="dummy",
                cancel_token=None,
            )
        ]

    fact_events = [e for e in events if isinstance(e, NotebookFactAdded)]
    assert len(fact_events) >= 1
    assert fact_events[0].fact_text == "fact from q2"

    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1 and stops[0].stop_reason == "complete"

    # No NotebookLimitReached — loop exited naturally (queue drained).
    assert not [e for e in events if isinstance(e, NotebookLimitReached)]

    # The warning was emitted on iteration 1.
    explore_errors = [
        rec
        for rec in captured
        if rec.get("event") == "notebook.explore_tool_error"
        and rec.get("iteration") == 1
    ]
    assert explore_errors, f"missing explore_tool_error log; got: {captured}"
    assert explore_errors[0].get("error_type") == "RuntimeError"

    # Both tool calls were attempted.
    assert [c[1]["query"] for c in tool_runner.calls] == ["q1", "q2"]


# ---------------------------------------------------------------------------
# Test 2 — max_iterations exhausted
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_max_iterations_exhausted_emits_limit():
    """With ``max_iterations=2`` and a self-replenishing extractor, exactly
    two iterations run; a NotebookLimitReached(max_iterations, count=2) is
    emitted before synthesize, and stop_reason is ``complete``.
    """
    provider = FakeResearchProvider()
    # Two seeds so that after iter 2 pops one, the queue still has a
    # leftover item — NotebookLimitReached only fires when the queue is
    # non-empty at the saturation point (notebook.py:331).
    provider.set_planner_response({"key_elements": ["seed-a", "seed-b"]})
    provider.set_extractor_responses(
        [
            {
                "facts": [{"text": "fact 1", "confidence": "high"}],
                "follow_up_queries": ["follow up 1"],
            },
            {
                "facts": [{"text": "fact 2", "confidence": "high"}],
                "follow_up_queries": [],
            },
        ]
    )
    provider.set_synthesizer_response("Synth.")

    tool_runner = _ScriptedToolRunner(
        [_ok_result("seed-a"), _ok_result("seed-b")]
    )
    orch = _build_orch(
        provider=provider,
        tool_runner=tool_runner,
        research_settings=ResearchSettings(
            max_facts=100, max_iterations=2, max_facts_per_extract=5
        ),
    )

    events = [
        e
        async for e in orch.run(
            session_id="s2",
            prompt="?",
            model_name="dummy",
            cancel_token=None,
        )
    ]

    limit_events = [e for e in events if isinstance(e, NotebookLimitReached)]
    assert len(limit_events) == 1
    assert limit_events[0].limit_kind == "max_iterations"
    assert limit_events[0].count == 2

    # Exactly 2 explore-phase iterations.
    explore_phases = [
        e
        for e in events
        if isinstance(e, NotebookPhaseStart) and e.phase == "explore"
    ]
    assert len(explore_phases) == 2
    assert [p.iteration for p in explore_phases] == [1, 2]

    # Synthesize runs AFTER the limit event.
    limit_index = events.index(limit_events[0])
    synth_phases = [
        i
        for i, e in enumerate(events)
        if isinstance(e, NotebookPhaseStart) and e.phase == "synthesize"
    ]
    assert len(synth_phases) == 1
    assert synth_phases[0] > limit_index

    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1 and stops[0].stop_reason == "complete"


# ---------------------------------------------------------------------------
# Test 3 — max_facts hit mid-iteration
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_max_facts_hit_mid_iteration_emits_limit():
    """With ``max_facts=3`` and an extractor returning 5 facts in iteration
    1, only 3 NotebookFactAdded events are emitted; loop breaks via
    ``limit_kind == 'max_facts'`` and synthesize still runs.

    Two seed queries are used so that the queue is non-empty when the loop
    breaks (NotebookLimitReached is only emitted when queue is non-empty —
    see notebook.py lines 329-341).
    """
    provider = FakeResearchProvider()
    provider.set_planner_response({"key_elements": ["seed-a", "seed-b"]})
    provider.set_extractor_responses(
        [
            {
                "facts": [
                    {"text": f"fact {i}", "confidence": "high"} for i in range(5)
                ],
                "follow_up_queries": [],
            }
        ]
    )
    provider.set_synthesizer_response("Synth.")

    tool_runner = _ScriptedToolRunner([_ok_result("seed-a")])
    orch = _build_orch(
        provider=provider,
        tool_runner=tool_runner,
        research_settings=ResearchSettings(
            max_facts=3, max_iterations=10, max_facts_per_extract=5
        ),
    )

    events = [
        e
        async for e in orch.run(
            session_id="s3",
            prompt="?",
            model_name="dummy",
            cancel_token=None,
        )
    ]

    fact_events = [e for e in events if isinstance(e, NotebookFactAdded)]
    assert len(fact_events) == 3
    assert [f.fact_text for f in fact_events] == ["fact 0", "fact 1", "fact 2"]
    # total_facts is monotonically increasing 1..3
    assert [f.total_facts for f in fact_events] == [1, 2, 3]

    limit_events = [e for e in events if isinstance(e, NotebookLimitReached)]
    assert len(limit_events) == 1
    assert limit_events[0].limit_kind == "max_facts"
    assert limit_events[0].count == 3

    synth_phases = [
        e
        for e in events
        if isinstance(e, NotebookPhaseStart) and e.phase == "synthesize"
    ]
    assert len(synth_phases) == 1

    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1 and stops[0].stop_reason == "complete"


# ---------------------------------------------------------------------------
# Test 4 — refine empty queue terminates cleanly
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_refine_empty_queue_terminates_cleanly():
    """Single seed query, extractor returns 2 facts and 0 follow-ups: no
    refine phase event, no NotebookLimitReached, clean ``complete`` stop.
    """
    provider = FakeResearchProvider()
    provider.set_planner_response({"key_elements": ["only-q"]})
    provider.set_extractor_responses(
        [
            {
                "facts": [
                    {"text": "fact A", "confidence": "high"},
                    {"text": "fact B", "confidence": "medium"},
                ],
                "follow_up_queries": [],
            }
        ]
    )
    provider.set_synthesizer_response("Synth.")

    tool_runner = _ScriptedToolRunner([_ok_result("only-q")])
    orch = _build_orch(
        provider=provider,
        tool_runner=tool_runner,
        research_settings=ResearchSettings(
            max_facts=10, max_iterations=10, max_facts_per_extract=5
        ),
    )

    events = [
        e
        async for e in orch.run(
            session_id="s4",
            prompt="?",
            model_name="dummy",
            cancel_token=None,
        )
    ]

    assert not [e for e in events if isinstance(e, NotebookLimitReached)]
    refine_phases = [
        e
        for e in events
        if isinstance(e, NotebookPhaseStart) and e.phase == "refine"
    ]
    assert refine_phases == []

    synth_phases = [
        e
        for e in events
        if isinstance(e, NotebookPhaseStart) and e.phase == "synthesize"
    ]
    assert len(synth_phases) == 1

    fact_events = [e for e in events if isinstance(e, NotebookFactAdded)]
    assert len(fact_events) == 2

    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1 and stops[0].stop_reason == "complete"


# ---------------------------------------------------------------------------
# Test 5 — higher-confidence replacement via try_add_fact
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_replacement_fact_via_higher_confidence():
    """Same fact text inserted twice, second time with higher confidence.

    Observed semantics of :meth:`NotebookState.try_add_fact`
    (``notebook_state.py`` lines 95-110): returns ``True`` on a
    higher-confidence replacement (overwriting in place). The orchestrator
    treats that True the same as a fresh insert and emits a second
    :class:`NotebookFactAdded` event (``notebook.py`` lines 280-289).

    Therefore: 2 NotebookFactAdded events are emitted, but the final
    notebook contains exactly 1 fact (text="x", confidence="high").
    """
    provider = FakeResearchProvider()
    provider.set_planner_response({"key_elements": ["q1"]})
    provider.set_extractor_responses(
        [
            {
                "facts": [{"text": "x", "confidence": "low"}],
                "follow_up_queries": ["q2"],
            },
            {
                "facts": [{"text": "x", "confidence": "high"}],
                "follow_up_queries": [],
            },
        ]
    )
    provider.set_synthesizer_response("Synth.")

    tool_runner = _ScriptedToolRunner([_ok_result("q1"), _ok_result("q2")])
    orch = _build_orch(
        provider=provider,
        tool_runner=tool_runner,
        research_settings=ResearchSettings(
            max_facts=10, max_iterations=10, max_facts_per_extract=5
        ),
    )

    events = [
        e
        async for e in orch.run(
            session_id="s5",
            prompt="?",
            model_name="dummy",
            cancel_token=None,
        )
    ]

    fact_events = [e for e in events if isinstance(e, NotebookFactAdded)]
    # Two emissions — the replacement returns True from try_add_fact.
    assert len(fact_events) == 2
    assert all(f.fact_text == "x" for f in fact_events)
    # total_facts stays at 1 across both emissions (dedup keeps length=1).
    assert [f.total_facts for f in fact_events] == [1, 1]

    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1 and stops[0].stop_reason == "complete"
