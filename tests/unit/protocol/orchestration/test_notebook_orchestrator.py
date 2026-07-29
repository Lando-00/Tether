"""Tests for the NotebookOrchestrator implementation."""
from __future__ import annotations

import pytest

from tests.fixtures.recording_research_store import RecordingResearchStore
from tether.config.settings import ResearchSettings
from tether.protocol.orchestration.notebook import NotebookOrchestrator


class _FakeProvider:
    pass


class _FakeParser:
    pass


class _FakeStore(RecordingResearchStore):
    pass


class _FakeRunner:
    pass


class _FakeToolRegistry:
    pass


def _build_stub() -> NotebookOrchestrator:
    """Build a NotebookOrchestrator with stub deps. Constructor doesn't
    actually use them; run() raises before touching anything."""
    return NotebookOrchestrator(
        provider=_FakeProvider(),
        store=_FakeStore(),
        tool_registry=_FakeToolRegistry(),
        tool_runner=_FakeRunner(),
        parser=_FakeParser(),
        config=object(),
        research_settings=ResearchSettings(),
    )


def test_construct_does_not_raise():
    """NotebookOrchestrator instantiation works."""
    _ = _build_stub()


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_research_mode_is_implemented():
    """HTTP router uses this flag to allow research-mode requests."""
    assert NotebookOrchestrator.is_implemented is True


def test_constructor_args_stored():
    """Constructor args are stored on self for future use."""
    orch = _build_stub()
    assert orch.provider is not None
    assert orch.parser is not None
    assert orch.store is not None
    assert orch.tool_registry is not None
    assert orch.config is not None
    assert orch.tool_runner is not None
    assert isinstance(orch.research_settings, ResearchSettings)
    assert orch.clock() is not None
