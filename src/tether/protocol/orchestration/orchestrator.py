"""Core orchestration loop — thin shim around :class:`ChattyAgentOrchestrator`.

``orchestrate()`` is the legacy function-shape entry point. It now
delegates to :class:`tether.protocol.orchestration.chatty.ChattyAgentOrchestrator`
and serializes each yielded :class:`WireEvent` into legacy v0 NDJSON
bytes via :func:`v0_compat_serialize`.

External surface (kept stable for back-compat with patch-based tests
in ``tests/unit/test_engine_cleanup.py``):

  ``async def orchestrate(*, session_id, prompt, model_name, provider,
                          parser, store, tools, system_prompt, config,
                          tool_runner, cancel_event=None,
                          hw_watchdog=None) -> AsyncGenerator[bytes, None]``

Engine.stream still calls this; Engine.chat skips it and yields
typed :class:`WireEvent` directly. Synthesis §3.4 + §3.5; §11.3 R1 / R7.
"""
from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Dict, Optional, TYPE_CHECKING

from tether.core.interfaces import (
    ModelProvider,
    SessionStore,
    StreamParser,
    Tool,
)
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration.cancel import AsyncEventCancelToken
from tether.protocol.orchestration.emitter import v0_compat_serialize
from tether.protocol.orchestration.chatty import ChattyAgentOrchestrator
from tether.protocol.orchestration.tool_runner import ToolRunner

if TYPE_CHECKING:
    from tether.runtime.hw_watchdog import HardwareWatchdog


async def orchestrate(
    *,
    session_id: str,
    prompt: str,
    model_name: str,
    provider: ModelProvider,
    parser: StreamParser,
    store: SessionStore,
    tools: Dict[str, Tool],
    system_prompt: str,
    config: OrchestratorConfig,
    tool_runner: ToolRunner,
    cancel_event: Optional[asyncio.Event] = None,
    hw_watchdog: Optional["HardwareWatchdog"] = None,
) -> AsyncGenerator[bytes, None]:
    """Drive one turn of model→parser→tool execution; yield v0 NDJSON bytes.

    Legacy function-shape kept for back-compat with :meth:`Engine.stream`
    and patch-based tests. Internally constructs an :class:`Orchestrator`
    and serializes each :class:`WireEvent` via :func:`v0_compat_serialize`,
    so the bytes wire vocabulary is preserved (``text``, ``think``,
    ``tool_started``, ``tool_completed``, ``tool_error``, ``error``,
    ``info``, ``loop_limit_reached``, ``done``).

    Synthesis §4 Phase 5 step 52 (orchestrator class), §11.3 R7
    (CancelToken).
    """
    orch = ChattyAgentOrchestrator(
        provider=provider,
        parser=parser,
        store=store,
        tools=tools,
        system_prompt=system_prompt,
        config=config,
        tool_runner=tool_runner,
        hw_watchdog=hw_watchdog,
    )
    cancel_token = AsyncEventCancelToken(cancel_event) if cancel_event else None

    async for wire_event in orch.run(
        session_id=session_id,
        prompt=prompt,
        model_name=model_name,
        cancel_token=cancel_token,
    ):
        bytes_out = v0_compat_serialize(wire_event)
        if bytes_out:
            yield bytes_out
