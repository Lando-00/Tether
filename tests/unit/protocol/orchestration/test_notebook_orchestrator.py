"""Tests for the NotebookOrchestrator stub.

Briefing §2 Seam B item 4: stub raises NotImplementedError; full impl
tracked in docs/research/06_context_strategies.md.
"""
from __future__ import annotations

import pytest

from tether.protocol.orchestration.notebook import NotebookOrchestrator


class _FakeProvider:
    pass


class _FakeParser:
    pass


class _FakeStore:
    pass


class _FakeRunner:
    pass


def _build_stub() -> NotebookOrchestrator:
    """Build a NotebookOrchestrator with stub deps. Constructor doesn't
    actually use them; run() raises before touching anything."""
    return NotebookOrchestrator(
        provider=_FakeProvider(),
        parser=_FakeParser(),
        store=_FakeStore(),
        tools={},
        system_prompt="",
        config=object(),
        tool_runner=_FakeRunner(),
    )


def test_construct_does_not_raise():
    """NotebookOrchestrator instantiation works (only run() raises)."""
    _ = _build_stub()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_run_raises_not_implemented():
    """NotebookOrchestrator.run() raises NotImplementedError with the
    expected docs reference."""
    orch = _build_stub()

    with pytest.raises(NotImplementedError) as excinfo:
        async for _ in orch.run(
            session_id="s",
            prompt="p",
            model_name="m",
        ):
            pass

    assert "docs/research/06_context_strategies.md" in str(excinfo.value)


def test_constructor_args_stored():
    """Constructor args are stored on self for future use."""
    orch = _build_stub()
    assert orch.provider is not None
    assert orch.parser is not None
    assert orch.store is not None
    assert orch.tools == {}
    assert orch.system_prompt == ""
    assert orch.config is not None
    assert orch.tool_runner is not None
    assert orch.hw_watchdog is None
