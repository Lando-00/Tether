from __future__ import annotations

from datetime import date
from typing import Any

import pytest

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
from tests.fixtures.fake_research_provider import FakeResearchProvider


class MockCancelToken:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def cancelled(self) -> bool:
        return self._cancelled


from tests.fixtures.recording_research_store import RecordingResearchStore


class _FakeStore(RecordingResearchStore):
    pass


class _FakeToolRegistry:
    pass


class _StubToolRunner:
    def __init__(self, cancel_token: MockCancelToken | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.cancel_token = cancel_token

    async def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, args))
        if self.cancel_token is not None:
            self.cancel_token.cancel()
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


class _CancelAfterExtractorProvider(FakeResearchProvider):
    def __init__(self, cancel_token: MockCancelToken) -> None:
        super().__init__()
        self.cancel_token = cancel_token

    async def stream(self, model_name, messages, tools=None, *, request_id=None):  # type: ignore[no-untyped-def]
        phase = self._detect_phase(messages)
        async for chunk in super().stream(
            model_name, messages, tools=tools, request_id=request_id
        ):
            yield chunk
        if phase == "extractor":
            self.cancel_token.cancel()


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _provider(synth_chunks: list[str] | None = None) -> FakeResearchProvider:
    provider = FakeResearchProvider()
    provider.set_planner_response({"key_elements": ["X launch details"]})
    provider.set_extractor_responses(
        [
            {
                "facts": [{"text": "X was launched in 2026", "confidence": "high"}],
                "follow_up_queries": ["X launch follow-up"],
            }
        ]
    )
    provider.set_synthesizer_response(synth_chunks or ["X launched ", "in 2026 [1]."])
    return provider


def _orch(provider: FakeResearchProvider, tool_runner: _StubToolRunner) -> NotebookOrchestrator:
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
            max_iterations=2,
            max_facts_per_extract=3,
        ),
        clock=lambda: date(2026, 5, 16),
    )


async def _collect(
    orch: NotebookOrchestrator, cancel_token: MockCancelToken
) -> list[object]:
    return await __import__("asyncio").wait_for(
        _collect_unbounded(orch, cancel_token),
        timeout=0.5,
    )


async def _collect_unbounded(
    orch: NotebookOrchestrator, cancel_token: MockCancelToken
) -> list[object]:
    return [
        event
        async for event in orch.run(
            session_id="s",
            prompt="What is X?",
            model_name="dummy",
            cancel_token=cancel_token,
        )
    ]


@pytest.mark.anyio
async def test_cancel_before_plan_phase():
    cancel_token = MockCancelToken()
    cancel_token.cancel()
    orch = _orch(_provider(), _StubToolRunner())

    events = await _collect(orch, cancel_token)

    assert [type(event) for event in events] == [MessageStart, MessageStop]
    assert events[-1].stop_reason == "cancelled"  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_cancel_after_plan_phase():
    cancel_token = MockCancelToken()
    orch = _orch(_provider(), _StubToolRunner())

    events = []
    async for event in orch.run(
        session_id="s",
        prompt="What is X?",
        model_name="dummy",
        cancel_token=cancel_token,
    ):
        events.append(event)
        if isinstance(event, NotebookQueryAdded):
            cancel_token.cancel()

    assert [type(event) for event in events] == [
        NotebookPhaseStart,
        NotebookQueryAdded,
        MessageStart,
        MessageStop,
    ]
    assert events[-1].stop_reason == "cancelled"  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_cancel_during_explore_phase():
    cancel_token = MockCancelToken()
    orch = _orch(_provider(), _StubToolRunner(cancel_token))

    events = await _collect(orch, cancel_token)

    assert any(
        isinstance(event, NotebookPhaseStart) and event.phase == "explore"
        for event in events
    )
    assert not any(isinstance(event, NotebookFactAdded) for event in events)
    assert events[-1].stop_reason == "cancelled"  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_cancel_during_extract_phase():
    cancel_token = MockCancelToken()
    provider = _CancelAfterExtractorProvider(cancel_token)
    provider.set_planner_response({"key_elements": ["X launch details"]})
    provider.set_extractor_responses(
        [
            {
                "facts": [{"text": "X was launched in 2026", "confidence": "high"}],
                "follow_up_queries": ["X launch follow-up"],
            }
        ]
    )
    provider.set_synthesizer_response("unused")
    orch = _orch(provider, _StubToolRunner())

    events = await _collect(orch, cancel_token)

    assert any(
        isinstance(event, NotebookPhaseStart) and event.phase == "extract"
        for event in events
    )
    assert not any(
        isinstance(event, NotebookPhaseStart) and event.phase == "refine"
        for event in events
    )
    assert not any(isinstance(event, NotebookFactAdded) for event in events)
    assert events[-1].stop_reason == "cancelled"  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_cancel_inside_synthesize_stream():
    cancel_token = MockCancelToken()
    orch = _orch(_provider(["first ", "second ", "third"]), _StubToolRunner())

    events = []
    async for event in orch.run(
        session_id="s",
        prompt="What is X?",
        model_name="dummy",
        cancel_token=cancel_token,
    ):
        events.append(event)
        if isinstance(event, TextDelta):
            cancel_token.cancel()

    text_events = [event for event in events if isinstance(event, TextDelta)]
    assert [event.text for event in text_events] == ["first "]
    assert events[-1].stop_reason == "cancelled"  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_cancel_emits_message_stop_exactly_once():
    cancel_token = MockCancelToken()
    orch = _orch(_provider(), _StubToolRunner(cancel_token))

    events = await _collect(orch, cancel_token)

    assert sum(isinstance(event, MessageStop) for event in events) == 1
