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

from typing import AsyncIterator, Dict, Optional, TYPE_CHECKING

from tether.core.interfaces import (
    ModelProvider,
    Orchestrator,
    SessionStore,
    StreamParser,
    Tool,
)

if TYPE_CHECKING:
    from tether.core.types import OrchestratorConfig
    from tether.protocol.orchestration.cancel import CancelToken
    from tether.protocol.orchestration.tool_runner import ToolRunner
    from tether.protocol.wire.events import WireEvent
    from tether.runtime.hw_watchdog import HardwareWatchdog


class NotebookOrchestrator(Orchestrator):
    """Stub for research-mode orchestration.

    Constructor matches ChattyAgentOrchestrator's so future wiring can
    swap impls without changing the call site. run() raises
    NotImplementedError until the research-mode work in
    docs/research/06_context_strategies.md is implemented.
    """

    # Stub — not yet implemented. The HTTP router checks is_implemented
    # before streaming to return 501 early. Briefing §2 Seam B item 4.
    is_implemented: bool = False

    def __init__(
        self,
        *,
        provider: "ModelProvider",
        parser: "StreamParser",
        store: "SessionStore",
        tools: Dict[str, "Tool"],
        system_prompt: str,
        config: "OrchestratorConfig",
        tool_runner: "ToolRunner",
        hw_watchdog: Optional["HardwareWatchdog"] = None,
    ):
        # Save constructor args so a future impl can use them. No
        # processing happens until run() is called (and run() raises).
        self.provider = provider
        self.parser = parser
        self.store = store
        self.tools = tools
        self.system_prompt = system_prompt
        self.config = config
        self.tool_runner = tool_runner
        self.hw_watchdog = hw_watchdog

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
