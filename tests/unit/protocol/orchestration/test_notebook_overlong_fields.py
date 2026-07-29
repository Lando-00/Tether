"""NotebookOrchestrator overlong-field truncation (Phase 9.5, Wave 4 reconcile).

Verifies the H1 reconcile fix: real LLMs that emit a fact text or sub-query
exceeding the wire-event Pydantic ``max_length`` caps must not crash the
orchestrator. The orchestrator truncates at the yield site (using
``_MAX_FACT_LENGTH`` / ``_MAX_QUERY_LENGTH``) so the event carries truncated
text and the stream completes cleanly with a ``MessageStop``.

Without this fix, a 5000-char fact would raise ``pydantic.ValidationError``
at the ``yield NotebookFactAdded(fact_text=fact.text, ...)`` site; the outer
``except asyncio.CancelledError`` does NOT catch it, so the consumer hangs
without a ``MessageStop`` — violating ADR-0020 D7 pairing invariant.

Citations: nho-rev-RECONCILE.md §A1; nho-rev-xhigh §E + Findings H1;
nho-rev-gpt55 §2 HIGH + §5 counter-proposal.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from tests.fixtures.fake_research_provider import FakeResearchProvider
from tests.fixtures.recording_research_store import RecordingResearchStore
from tether.config.settings import ResearchSettings
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration.notebook import (
    _MAX_FACT_LENGTH,
    _MAX_QUERY_LENGTH,
    NotebookOrchestrator,
)
from tether.protocol.parsers.sliding import SlidingParser
from tether.protocol.wire.events import (
    MessageStop,
    NotebookFactAdded,
    NotebookQueryAdded,
)


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
        clock=lambda: date(2026, 5, 31),
    )


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_overlong_fact_and_query_truncated_not_aborted():
    """Real LLM emits a 6000-char fact and an unsafe 700-char follow-up.

    Pre-reconcile behavior: ValidationError at ``yield NotebookFactAdded(...)``
    propagates past the outer ``except CancelledError``; stream dies WITHOUT
    a ``MessageStop`` → consumer hangs forever.

    The fact reaches the existing wire truncation safeguard. The query is
    rejected by the parser's search-query sanitizer before it can be queued.
    """
    overlong_fact = "X" * 6000  # > _MAX_FACT_LENGTH (4096)
    overlong_followup = "Y" * 700  # > _MAX_QUERY_LENGTH (512)
    seed_query = "seed q"

    provider = FakeResearchProvider()
    provider.set_planner_response({"key_elements": [seed_query]})
    provider.set_extractor_responses(
        [
            {
                "facts": [{"text": overlong_fact, "confidence": "high"}],
                "follow_up_queries": [overlong_followup],
            },
        ]
    )
    provider.set_synthesizer_response("Synth complete.")

    tool_runner = _ScriptedToolRunner([_ok_result(seed_query)])
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
            session_id="s1",
            prompt="Question.",
            model_name="dummy",
            cancel_token=None,
        )
    ]

    # The orchestrator must NOT have aborted via ValidationError → it must
    # have emitted a MessageStop with complete (not cancelled, not error).
    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1, (
        f"expected exactly one MessageStop; got {len(stops)} — orchestrator "
        f"likely died on Pydantic ValidationError pre-reconcile."
    )
    assert stops[0].stop_reason == "complete"

    # Fact event carries truncated text (exactly _MAX_FACT_LENGTH chars).
    fact_events = [e for e in events if isinstance(e, NotebookFactAdded)]
    assert len(fact_events) == 1
    assert len(fact_events[0].fact_text) == _MAX_FACT_LENGTH
    # Truncated to the cap, but still preserves the source character.
    assert fact_events[0].fact_text == overlong_fact[:_MAX_FACT_LENGTH]
    assert fact_events[0].fact_text.endswith("X")

    # The unsafe follow-up never reaches the wire queue.
    query_events = [e for e in events if isinstance(e, NotebookQueryAdded)]
    assert [event.query for event in query_events] == [seed_query]


@pytest.mark.anyio
async def test_at_limit_fact_is_not_truncated_and_unsafe_query_is_not_accepted():
    """Wire caps do not make an unsafe search query acceptable.

    Catches off-by-one regressions in the truncation slicing.
    """
    at_limit_fact = "F" * _MAX_FACT_LENGTH  # exactly 4096
    at_limit_query = "Q" * _MAX_QUERY_LENGTH  # exactly 512

    provider = FakeResearchProvider()
    provider.set_planner_response({"key_elements": [at_limit_query]})
    provider.set_extractor_responses(
        [
            {
                "facts": [{"text": at_limit_fact, "confidence": "high"}],
                "follow_up_queries": [],
            }
        ]
    )
    provider.set_synthesizer_response("ok.")

    tool_runner = _ScriptedToolRunner([_ok_result("q")])
    orch = _build_orch(
        provider=provider,
        tool_runner=tool_runner,
        research_settings=ResearchSettings(
            max_facts=5, max_iterations=5, max_facts_per_extract=5
        ),
    )

    events = [
        e
        async for e in orch.run(
            session_id="s1",
            prompt="q",
            model_name="dummy",
            cancel_token=None,
        )
    ]

    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1 and stops[0].stop_reason == "complete"

    fact_events = [e for e in events if isinstance(e, NotebookFactAdded)]
    assert len(fact_events) == 1
    assert len(fact_events[0].fact_text) == _MAX_FACT_LENGTH

    query_events = [e for e in events if isinstance(e, NotebookQueryAdded)]
    assert [event.query for event in query_events] == ["q"]
