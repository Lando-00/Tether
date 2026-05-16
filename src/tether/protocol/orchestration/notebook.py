"""NotebookOrchestrator — research-mode strategy stub.

A future strategy for "notebook" or research-style interaction
patterns where the orchestrator manages multi-turn context retrieval
and structured exploration. Tracked in
docs/research/06_context_strategies.md.

Currently a stub: run() raises NotImplementedError. The class exists
so the Orchestrator ABC has at least two concrete impls (anti-
overengineering rule R6 satisfied: not a single-impl abstraction).

p5-mode-strategy-routing will add the per-request mode field and the
Engine wiring that picks an Orchestrator impl based on the requested
mode.

Briefing §2 Seam B item 4.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import AsyncIterator, ClassVar, Optional, TYPE_CHECKING

from tether.core.interfaces import (
    ModelProvider,
    Orchestrator,
    SessionStore,
    StreamParser,
)

if TYPE_CHECKING:
    from tether.config.settings import ResearchSettings
    from tether.core.tool_registry import ToolRegistry
    from tether.core.types import OrchestratorConfig as ChatSettings
    from tether.protocol.orchestration.cancel import CancelToken
    from tether.protocol.orchestration.tool_runner import ToolRunner
    from tether.protocol.wire.events import WireEvent


class NotebookOrchestrator(Orchestrator):
    """Stub for research-mode orchestration.

    Constructor is pinned by ADR-0020 §D5 so Engine.chat() can thread
    research settings via inspect.signature. run() raises NotImplementedError
    until the research-mode work in docs/research/06_context_strategies.md
    is implemented.
    """

    # Stub — not yet implemented. The HTTP router checks is_implemented
    # before streaming to return 501 early. Briefing §2 Seam B item 4.
    is_implemented: ClassVar[bool] = False

    def __init__(
        self,
        *,
        # Inherited ABC kwargs (engine.py threads via inspect.signature):
        provider: "ModelProvider",
        store: "SessionStore",
        tool_registry: "ToolRegistry",
        tool_runner: "ToolRunner",
        parser: "StreamParser",
        config: "ChatSettings",
        # Notebook-specific (engine.py adds these when mode="research"):
        research_settings: "ResearchSettings",
        clock: Callable[[], date] = lambda: date.today(),
    ) -> None:
        # Save constructor args so a future impl can use them. No
        # processing happens until run() is called (and run() raises).
        self.provider = provider
        self.store = store
        self.tool_registry = tool_registry
        self.tool_runner = tool_runner
        self.parser = parser
        self.config = config
        self.research_settings = research_settings
        self.clock = clock

    async def run(
        self,
        *,
        session_id: str,
        prompt: str,
        model_name: str,
        cancel_token: Optional["CancelToken"] = None,
    ) -> AsyncIterator["WireEvent"]:
        """Not yet implemented; tracked in
        docs/research/06_context_strategies.md."""
        raise NotImplementedError(
            "research mode tracked in docs/research/06_context_strategies.md"
        )
        # Unreachable — needed for type-checker to recognize this as an
        # async generator. (When the real impl lands, this `yield` line
        # comes alive.)
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]


__all__ = ["NotebookOrchestrator"]
