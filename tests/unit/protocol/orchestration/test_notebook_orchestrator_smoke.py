from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from tests.fixtures.fake_research_provider import FakeResearchProvider
from tests.fixtures.recording_research_store import RecordingResearchStore
from tether.config.settings import ResearchSettings
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration.notebook import NotebookOrchestrator
from tether.protocol.parsers.sliding import SlidingParser
from tether.protocol.wire.events import (
    MessageStart,
    MessageStop,
    NotebookFactAdded,
    NotebookPhaseStart,
    NotebookQueryAdded,
    TextDelta,
)


class _FakeStore(RecordingResearchStore):
    pass


class _FakeToolRegistry:
    pass


class _StubToolRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, args))
        return {
            "results": [
                {
                    "rank": 1,
                    "title": "X overview",
                    "url": "https://example.com/x",
                    "snippet": "X was launched in 2026.",
                }
            ],
            "meta": {"query": args["query"]},
        }


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _orch(
    provider: FakeResearchProvider, tool_runner: _StubToolRunner
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
        research_settings=ResearchSettings(
            max_facts=5,
            max_iterations=1,
            max_facts_per_extract=3,
        ),
        clock=lambda: date(2026, 5, 16),
    )


@pytest.mark.anyio
async def test_notebook_orchestrator_hanov_smoke_event_sequence():
    provider = FakeResearchProvider()
    provider.set_planner_response({"key_elements": ["X launch details"]})
    provider.set_extractor_responses(
        [
            {
                "facts": [
                    {"text": "X was launched in 2026", "confidence": "high"},
                    {"text": "X launch details were public", "confidence": "medium"},
                ],
                "follow_up_queries": [],
            }
        ]
    )
    provider.set_synthesizer_response(["X launched ", "in 2026 [1]."])
    tool_runner = _StubToolRunner()
    orch = _orch(provider, tool_runner)

    events = [
        e
        async for e in orch.run(
            session_id="s",
            prompt="What is X?",
            model_name="dummy",
        )
    ]

    assert [type(e) for e in events] == [
        NotebookPhaseStart,
        NotebookQueryAdded,
        NotebookPhaseStart,
        NotebookPhaseStart,
        NotebookFactAdded,
        NotebookFactAdded,
        NotebookPhaseStart,
        MessageStart,
        # Phase 9.6 I-1: ``_ThinkStripper`` adds an OVERLAP-sized hold
        # buffer in front of TextDelta emission, so two synth chunks
        # (no markers) flush across three deltas: one per chunk after
        # the first, plus a finalize() tail.
        TextDelta,
        TextDelta,
        TextDelta,
        MessageStop,
    ]
    assert [(n.phase, n.iteration) for n in events if isinstance(n, NotebookPhaseStart)] == [
        ("plan", 0),
        ("explore", 1),
        ("extract", 1),
        ("synthesize", 0),
    ]
    # Chunking is an implementation detail of the stripper; assert the
    # concatenated synth text exactly matches what the provider streamed.
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == (
        "X launched in 2026 [1]."
    )
    assert [e.seq for e in events] == list(range(len(events)))
    assert events[-1].stop_reason == "complete"  # type: ignore[attr-defined]
    assert tool_runner.calls == [("web_search", {"query": "X launch details", "count": 5})]
