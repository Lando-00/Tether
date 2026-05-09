"""The public face of Tether — library-first composition root.

Construct via ``Engine.from_settings(settings)``. The async-context-manager
semantics make it safe for short-lived embedding (CLI request batches) and
long-lived servers (FastAPI lifespan + signal handling).

Per _synthesis.md §2.3 + §4 Phase 2 (steps 21, 22). Importing
``tether_service`` (which re-exports ``Engine``) MUST NOT pull in FastAPI,
MLC, or Brave — every concrete provider/parser/store/tool is lazy-imported
inside ``Engine.from_settings`` (see R8 lazy-import rule).

Phase 3 step 35 will replace the ``aclose()`` body with
``await self.hw_watchdog.shutdown_all()`` once ``HardwareWatchdog`` is in
place (per §11.3 R22 dependency-graph clarification).
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

from tether_service.config.settings import Settings
from tether_service.core.interfaces import (
    ModelProvider,
    SessionStore,
    StreamParser,
    Tool,
)
from tether_service.core.types import OrchestratorConfig
from tether_service.protocol.orchestration.tool_runner import ToolRunner
from tether_service.runtime.watchdog_mode import WatchdogMode


class Engine:
    """Library-first composition root for Tether."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        parser: StreamParser,
        session_store: SessionStore,
        tools: Dict[str, Tool],
        system_prompt: str,
        watchdog_mode: WatchdogMode = WatchdogMode.LIBRARY,
        orchestrator_config: Optional[OrchestratorConfig] = None,
        tool_runner: Optional[ToolRunner] = None,
    ):
        """Build an Engine from already-constructed components.

        ``orchestrator_config`` and ``tool_runner`` may be omitted; sensible
        defaults are produced (``OrchestratorConfig`` with library defaults,
        ``ToolRunner`` with the default 15s timeout) so direct constructors
        and the deprecated ``GenerationService`` alias keep working. Tests
        and advanced callers may pass them explicitly. Per p2-cleanup
        (synthesis §4 Phase 2 step 23).
        """
        self.provider = provider
        self.parser = parser
        self.store = session_store
        self.tools = tools
        self.system_prompt = system_prompt
        self.watchdog_mode = watchdog_mode
        self.orchestrator_config = orchestrator_config or OrchestratorConfig(
            max_tool_loops=5,
            auto_reload_on_fatal_error=True,
            save_thinking=True,
            include_thinking_in_history=False,
        )
        self.tool_runner = tool_runner or ToolRunner(tools)
        self._closed = False

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        watchdog_mode: WatchdogMode = WatchdogMode.LIBRARY,
    ) -> "Engine":
        """Build an Engine from a typed Settings object.

        Lazy-imports concrete provider/parser/store classes so that
        ``import tether_service`` does not pull in MLC, FastAPI, or Brave.
        Per R8 lazy-import rule (synthesis).

        Raises ValueError / RuntimeError if any required impl path doesn't
        resolve (delegated to ``load`` and ``ToolRegistry``).
        """
        from tether_service.core.factory import load
        from tether_service.core.tool_registry import ToolRegistry

        model_spec = settings.providers.model
        provider = load(model_spec.impl, **model_spec.args)

        parser_spec = settings.providers.parser
        parser = load(parser_spec.impl, **parser_spec.args)

        store_spec = settings.providers.session_store
        store = load(store_spec.impl, **store_spec.args)

        tools_settings = settings.tools
        registry_cfg = [
            {"name": t.name, "impl": t.impl, "args": t.args}
            for t in tools_settings.registry
        ]
        registry = ToolRegistry(registry_cfg, list(tools_settings.enabled))
        tools = registry.all()

        orchestrator_config = OrchestratorConfig.from_settings(settings)
        tool_runner = ToolRunner(tools, timeout_sec=settings.limits.tool_timeout_sec)

        return cls(
            provider=provider,
            parser=parser,
            session_store=store,
            tools=tools,
            system_prompt=settings.system.prompt,
            watchdog_mode=watchdog_mode,
            orchestrator_config=orchestrator_config,
            tool_runner=tool_runner,
        )

    # --- Streaming chat (the core API) ---

    async def stream(
        self,
        *,
        session_id: str,
        prompt: str,
        model_name: str,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Drive the core orchestration to stream NDJSON bytes."""
        from tether_service.protocol.orchestration.orchestrator import orchestrate

        async for chunk in orchestrate(
            session_id=session_id,
            prompt=prompt,
            model_name=model_name,
            provider=self.provider,
            parser=self.parser,
            store=self.store,
            tools=self.tools,
            system_prompt=self.system_prompt,
            config=self.orchestrator_config,
            tool_runner=self.tool_runner,
            cancel_event=cancel_event,
        ):
            yield chunk

    # --- Session / model CRUD pass-throughs (no business logic) ---

    async def create_session(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        import time
        import uuid
        from datetime import datetime

        session_id = str(uuid.uuid4())
        created_at = int(time.time())
        await self.store.create_session(session_id, created_at)
        created_at_iso = datetime.fromtimestamp(created_at).isoformat()
        return {"session_id": session_id, "created_at": created_at_iso}

    async def list_sessions(self) -> List[Dict[str, Any]]:
        return await self.store.list_sessions()

    async def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        return await self.store.get_history(session_id)

    async def delete_session(self, session_id: str) -> bool:
        return await self.store.delete_session(session_id)

    async def delete_all_sessions(self) -> int:
        return await self.store.delete_all_sessions()

    def list_models(self) -> List[str]:
        return self.provider.list_models()

    def unload_model(self, model_name: str) -> bool:
        return self.provider.unload_model(model_name)

    # --- Lifecycle ---

    async def __aenter__(self) -> "Engine":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Phase 2 placeholder — delegates to ``provider.shutdown_all()``.

        Phase 3 step 35 will replace this body with
        ``await self.hw_watchdog.shutdown_all()`` once HardwareWatchdog is in
        place. Per _synthesis.md §11.3 R22.
        """
        if self._closed:
            return
        self._closed = True
        if hasattr(self.provider, "shutdown_all"):
            self.provider.shutdown_all()
