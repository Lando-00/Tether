"""NotebookNoFacts emission tests (Phase 9.7 W3-B).

Covers ``fu-research-empty-notebook-signal`` / ``nho-fu-w3b-empty-signal``:
when the Notebook loop exits with zero facts, the orchestrator emits a
dedicated :class:`NotebookNoFacts` event BEFORE the synthesize phase.
Synthesis still runs and the turn still terminates with
``MessageStop(stop_reason="complete")`` — this is a signal, not an Error
and not a NotebookLimitReached.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from tether.config.settings import ResearchSettings
from tether.core.logging import reset_logging_for_tests
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration.notebook import NotebookOrchestrator
from tether.protocol.parsers.sliding import SlidingParser
from tether.protocol.wire.events import (
    MessageStop,
    NotebookFactAdded,
    NotebookLimitReached,
    NotebookNoFacts,
    NotebookPhaseStart,
    NotebookQueryAdded,
)
from tests.fixtures.fake_research_provider import FakeResearchProvider


from tests.fixtures.recording_research_store import RecordingResearchStore


class _FakeStore(RecordingResearchStore):
    pass


class _FakeToolRegistry:
    pass


class _ScriptedToolRunner:
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
def _reset_logging():
    reset_logging_for_tests()
    yield
    reset_logging_for_tests()


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


# ---------------------------------------------------------------------------
# Test 1 — all tool calls return rate_limited error → NotebookNoFacts emitted
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_all_queries_rate_limited_emits_no_facts():
    """Planner produces 2 queries; both tool calls return
    ``{'error': 'rate_limited'}`` so the loop exits with zero facts.

    Expected: exactly one :class:`NotebookNoFacts` is emitted before the
    synthesize phase. Synthesis still runs and ``MessageStop`` is
    ``complete``. No :class:`NotebookLimitReached` and no
    :class:`NotebookFactAdded`.
    """
    provider = FakeResearchProvider()
    provider.set_planner_response({"key_elements": ["q1", "q2"]})
    # No extractor responses queued — extractor must never be invoked
    # because both explore tool calls error before extract.
    provider.set_extractor_responses([])
    provider.set_synthesizer_response("No info found.")

    tool_runner = _ScriptedToolRunner(
        [{"error": "rate_limited"}, {"error": "rate_limited"}]
    )
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
            session_id="s-empty-1",
            prompt="Tell me about q.",
            model_name="dummy",
            cancel_token=None,
        )
    ]

    no_facts = [e for e in events if isinstance(e, NotebookNoFacts)]
    assert len(no_facts) == 1
    assert no_facts[0].queries_attempted == 2
    assert no_facts[0].iterations == 2
    # When at least one query was attempted, no "empty plan" hint.
    assert no_facts[0].note is None

    # No facts and no limit reached.
    assert not [e for e in events if isinstance(e, NotebookFactAdded)]
    assert not [e for e in events if isinstance(e, NotebookLimitReached)]

    # NotebookNoFacts comes strictly BEFORE the synthesize phase.
    no_facts_idx = events.index(no_facts[0])
    synth_phases = [
        i
        for i, e in enumerate(events)
        if isinstance(e, NotebookPhaseStart) and e.phase == "synthesize"
    ]
    assert len(synth_phases) == 1
    assert synth_phases[0] > no_facts_idx

    # Synthesis still ran to completion.
    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1 and stops[0].stop_reason == "complete"


# ---------------------------------------------------------------------------
# Test 2 — empty plan falls back to original prompt as one query
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_empty_plan_emits_no_facts_with_zero_counters():
    """Planner returns an empty ``key_elements`` list, so the loop never
    enters an iteration.

    Phase 9.7 CLI smoke found this path in the wild: the planner returned no
    queries for a typo-heavy multi-part prompt, causing immediate empty
    synthesis. The orchestrator now falls back to the original user prompt as
    a broad single search query. If that search also fails, NotebookNoFacts is
    still emitted — but with one attempted query / iteration rather than an
    ``empty plan`` note.
    """
    provider = FakeResearchProvider()
    provider.set_planner_response({"key_elements": []})
    provider.set_extractor_responses([])
    provider.set_synthesizer_response("Nothing to summarize.")

    tool_runner = _ScriptedToolRunner([{"error": "planner_empty_fallback_failed"}])
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
            session_id="s-empty-2",
            prompt="Tell me about nothing.",
            model_name="dummy",
            cancel_token=None,
        )
    ]

    assert [call[1]["query"] for call in tool_runner.calls] == ["Tell me about nothing."]

    fallback_queries = [e for e in events if isinstance(e, NotebookQueryAdded)]
    assert len(fallback_queries) == 1
    assert fallback_queries[0].query == "Tell me about nothing."

    no_facts = [e for e in events if isinstance(e, NotebookNoFacts)]
    assert len(no_facts) == 1
    assert no_facts[0].queries_attempted == 1
    assert no_facts[0].iterations == 1
    assert no_facts[0].note is None

    # Synthesize still runs and turn completes.
    synth_phases = [
        e
        for e in events
        if isinstance(e, NotebookPhaseStart) and e.phase == "synthesize"
    ]
    assert len(synth_phases) == 1
    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1 and stops[0].stop_reason == "complete"


# ---------------------------------------------------------------------------
# Test 3 — at least one fact gathered → no NotebookNoFacts
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_one_fact_gathered_does_not_emit_no_facts():
    """When extract yields at least one fact, NotebookNoFacts must NOT
    be emitted."""
    provider = FakeResearchProvider()
    provider.set_planner_response({"key_elements": ["q1"]})
    provider.set_extractor_responses(
        [
            {
                "facts": [{"text": "a real fact", "confidence": "high"}],
                "follow_up_queries": [],
            }
        ]
    )
    provider.set_synthesizer_response("Here is the answer.")

    tool_runner = _ScriptedToolRunner([_ok_result("q1")])
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
            session_id="s-has-facts",
            prompt="Tell me about q1.",
            model_name="dummy",
            cancel_token=None,
        )
    ]

    assert [e for e in events if isinstance(e, NotebookFactAdded)]
    assert not [e for e in events if isinstance(e, NotebookNoFacts)]

    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1 and stops[0].stop_reason == "complete"
