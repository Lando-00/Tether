"""AutoOrchestrator direct-answer path (design: fact-based-orchestration-default).

The fact-based loop is only safe as the *default* orchestrator if turns that
need no external evidence never reach the search backend. These tests pin that:
a direct turn must issue **zero** tool calls and emit **no** notebook events,
while an evidence-seeking turn still runs the full research loop.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from tests.fixtures.fake_research_provider import FakeResearchProvider
from tests.fixtures.recording_research_store import RecordingResearchStore
from tether.config.settings import ResearchSettings
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration.notebook import (
    AutoOrchestrator,
    NotebookOrchestrator,
)
from tether.protocol.parsers.sliding import SlidingParser
from tether.protocol.wire.events import (
    MessageStop,
    NotebookPhaseStart,
    NotebookQueryAdded,
    TextDelta,
)


class _RecordingToolRunner:
    """Records every tool call and reports a benign failure."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, name: str, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        # ``**kwargs`` absorbs the chat orchestrator's ``context`` /
        # ``tool_call_id``, which the notebook loop does not pass.
        self.calls.append((name, args))
        return {"error": "no search backend in tests"}


def _build(
    cls: type[NotebookOrchestrator],
    provider: FakeResearchProvider,
    tool_runner: _RecordingToolRunner,
) -> NotebookOrchestrator:
    return cls(
        provider=provider,
        store=RecordingResearchStore(),
        tool_registry=object(),  # type: ignore[arg-type]
        tool_runner=tool_runner,  # type: ignore[arg-type]
        parser=SlidingParser(),
        config=OrchestratorConfig(
            max_tool_loops=3,
            auto_reload_on_fatal_error=False,
            save_thinking=False,
            include_thinking_in_history=False,
        ),
        research_settings=ResearchSettings(
            max_facts=10, max_iterations=10, max_facts_per_extract=5
        ),
        clock=lambda: date(2026, 5, 16),
    )


async def _run(orch: NotebookOrchestrator, prompt: str) -> list[Any]:
    return [
        event
        async for event in orch.run(
            session_id="s-auto",
            prompt=prompt,
            model_name="dummy",
            cancel_token=None,
        )
    ]


async def test_small_talk_skips_planning_and_search() -> None:
    provider = FakeResearchProvider()
    provider.set_chat_response("Hello! How can I help?")
    runner = _RecordingToolRunner()
    orch = _build(AutoOrchestrator, provider, runner)

    events = await _run(orch, "hello")

    # No planning, no explore, no notebook bookkeeping, and crucially no search.
    assert not [e for e in events if isinstance(e, NotebookPhaseStart)]
    assert not [e for e in events if isinstance(e, NotebookQueryAdded)]
    assert runner.calls == []
    # The answer still streams and the turn completes normally.
    assert "".join(e.text for e in events if isinstance(e, TextDelta))
    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1
    assert stops[0].stop_reason == "complete"


async def test_creative_request_skips_search() -> None:
    provider = FakeResearchProvider()
    provider.set_chat_response("Waves fold on grey stone")
    runner = _RecordingToolRunner()
    orch = _build(AutoOrchestrator, provider, runner)

    events = await _run(orch, "Write a haiku about the sea")

    assert not [e for e in events if isinstance(e, NotebookPhaseStart)]
    assert runner.calls == []
    assert "".join(e.text for e in events if isinstance(e, TextDelta))


async def test_direct_turns_keep_the_tool_loop() -> None:
    """A DIRECT turn must still be able to call local tools.

    Regression guard: the direct path used to answer inline with *no* tools, so
    "what time is it?" was pushed into web research and failed. DIRECT turns are
    delegated to the chat orchestrator precisely so the tool loop stays live.
    """
    provider = FakeResearchProvider()
    provider.set_chat_response(
        '<<function_call>> {"name":"time","arguments":{"timezone":"UTC"}}'
    )
    runner = _RecordingToolRunner()
    orch = _build(AutoOrchestrator, provider, runner)

    events = await _run(orch, "hello")

    assert not [e for e in events if isinstance(e, NotebookPhaseStart)]
    assert [name for name, _ in runner.calls] == ["time"]


async def test_evidence_question_still_researches() -> None:
    """Triage must not disable research for questions that need evidence."""
    provider = FakeResearchProvider()
    provider.set_planner_response({"key_elements": ["latest python version"]})
    provider.set_synthesizer_response("...")
    runner = _RecordingToolRunner()
    orch = _build(AutoOrchestrator, provider, runner)

    events = await _run(orch, "What is the latest version of Python?")

    assert [e for e in events if isinstance(e, NotebookPhaseStart)]
    assert [name for name, _ in runner.calls] == ["web_search"]


async def test_explicit_research_orchestrator_ignores_triage() -> None:
    """``mode="research"`` must never be downgraded to a direct answer."""
    provider = FakeResearchProvider()
    provider.set_planner_response({"key_elements": ["hello"]})
    provider.set_synthesizer_response("...")
    runner = _RecordingToolRunner()
    orch = _build(NotebookOrchestrator, provider, runner)

    events = await _run(orch, "hello")

    assert [e for e in events if isinstance(e, NotebookPhaseStart)]
    assert runner.calls, "NotebookOrchestrator should research even small talk"
