"""The public face of Tether — library-first composition root.

Construct via ``Engine.from_settings(settings)``. The async-context-manager
semantics make it safe for short-lived embedding (CLI request batches) and
long-lived servers (FastAPI lifespan + signal handling).

Per _synthesis.md §2.3 + §4 Phase 2 (steps 21, 22). Importing
``tether_service`` (which re-exports ``Engine``) MUST NOT pull in FastAPI,
MLC, or Brave — every concrete provider/parser/store/tool is lazy-imported
inside ``Engine.from_settings`` (see R8 lazy-import rule).

Phase 3 step 35 (this PR): ``aclose()`` routes through
:class:`HardwareWatchdog` when available, replacing the Phase 2 placeholder
body and the legacy ``shutdown_provider_with_timeout`` path that lived in
``app/http/api.py``. Per §11.3 R22.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, TYPE_CHECKING

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

if TYPE_CHECKING:
    from tether_service.core.connector_registry import ConnectorRegistry
    from tether_service.runtime.hw_watchdog import HardwareWatchdog


logger = logging.getLogger(__name__)


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
        hw_watchdog: Optional["HardwareWatchdog"] = None,
        connector_registry: Optional["ConnectorRegistry"] = None,
    ):
        """Build an Engine from already-constructed components.

        ``orchestrator_config`` and ``tool_runner`` may be omitted; sensible
        defaults are produced (``OrchestratorConfig`` with library defaults,
        ``ToolRunner`` with the default 15s timeout) so direct constructors
        and the deprecated ``GenerationService`` alias keep working. Tests
        and advanced callers may pass them explicitly. Per p2-cleanup
        (synthesis §4 Phase 2 step 23).

        ``hw_watchdog`` is optional. ``from_settings`` always builds one
        and passes it; direct constructors (used by tests + the legacy
        ``GenerationService`` alias) may pass ``None``, in which case
        :meth:`aclose` falls back to ``provider.shutdown_all()`` if present.
        Synthesis §4 Phase 3 step 35.

        ``connector_registry`` is optional; ``from_settings`` always
        builds one (empty when ``settings.connectors.registry`` is empty)
        and passes it. Connector tools are merged into ``tools`` BEFORE
        construction by ``from_settings`` so the orchestrator/ToolRunner
        sees a single flat dict regardless of source. Per Phase 4.5
        steps 47d-47e (synthesis §4) and connector spec §3.3.
        """
        self.provider = provider
        self.parser = parser
        self.store = session_store
        self.tools = tools
        self.system_prompt = system_prompt
        self.watchdog_mode = watchdog_mode
        self.hw_watchdog = hw_watchdog
        self.connector_registry = connector_registry
        self.orchestrator_config = orchestrator_config or OrchestratorConfig(
            max_tool_loops=5,
            auto_reload_on_fatal_error=True,
            save_thinking=True,
            include_thinking_in_history=False,
        )
        self.tool_runner = tool_runner or ToolRunner(tools)
        self._closed = False
        # Phase 4.5 step 47d: __aenter__ schedules start_connector(id) for
        # each READY connector; __aexit__/aclose cancels any still-pending
        # tasks before invoking stop_all so we never tear down a connector
        # mid-start.
        self._connector_start_tasks: List[asyncio.Task] = []

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

        Constructs a :class:`HardwareWatchdog` around the provider list.
        Providers that don't implement :class:`HardwareLifecycle` (e.g.,
        ``DummyProvider``) are silently filtered out by the watchdog, so
        an engine with a non-HW provider gets a watchdog with zero entries
        — ``aclose`` becomes a no-op, ``/readyz`` reports the empty list.
        Synthesis §4 Phase 3 step 35.

        Builds a :class:`ConnectorRegistry` from
        ``settings.connectors.registry``. Each enabled entry is
        instantiated via ``load(spec.impl, **spec.args)``. Connector
        tools are aggregated into the ``tools`` dict (single flat
        registry) BEFORE the ``ToolRunner`` is built, so the orchestrator
        sees one merged tool dict regardless of source. M5 prefix +
        collision validation runs at registry construction (connector
        spec §3.3); ``tool_names = set(tools.keys())`` prevents a
        connector tool from shadowing an in-tree tool. Synthesis §4
        Phase 4.5 step 47d.

        Raises ValueError / RuntimeError if any required impl path doesn't
        resolve (delegated to ``load`` and ``ToolRegistry``).
        """
        from tether_service.core.connector_registry import ConnectorRegistry
        from tether_service.core.factory import load
        from tether_service.core.tool_registry import ToolRegistry
        from tether_service.runtime.hw_watchdog import HardwareWatchdog

        model_spec = settings.providers.model
        provider = load(model_spec.impl, **model_spec.args)

        parser_spec = settings.providers.parser
        parser = load(parser_spec.impl, **parser_spec.args)

        store_spec = settings.providers.session_store
        store = load(store_spec.impl, **store_spec.args)

        tools_settings = settings.tools
        registry = ToolRegistry.from_settings(tools_settings)
        tools = registry.all()

        # Phase 4.5 step 47d: build ConnectorRegistry from typed settings.
        # Pass the in-tree tool names as the M5 forbidden set so connector
        # tools cannot shadow them (connector spec §3.3).
        connectors: List[Any] = []
        for cid, spec in settings.connectors.registry.items():
            if not spec.enabled:
                continue
            conn = load(spec.impl, **spec.args)
            connectors.append(conn)
        connector_registry = ConnectorRegistry(
            connectors, tool_names=set(tools.keys())
        )

        # Merge connector tools into the flat dict the orchestrator sees.
        # Safe to merge naively — registry construction proved no
        # cross-connector or in-tree collisions exist.
        tools.update(connector_registry.aggregate_tools())

        orchestrator_config = OrchestratorConfig.from_settings(settings)
        tool_runner = ToolRunner(tools, timeout_sec=settings.limits.tool_timeout_sec)

        hw_watchdog = HardwareWatchdog([provider], mode=watchdog_mode)

        return cls(
            provider=provider,
            parser=parser,
            session_store=store,
            tools=tools,
            system_prompt=settings.system.prompt,
            watchdog_mode=watchdog_mode,
            orchestrator_config=orchestrator_config,
            tool_runner=tool_runner,
            hw_watchdog=hw_watchdog,
            connector_registry=connector_registry,
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
            hw_watchdog=self.hw_watchdog,
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
        """Phase 4 step 41: run startup() on every tool concurrently.

        Required-tool failures propagate as :class:`RuntimeError` (the
        engine refuses to start). Optional-tool failures are logged and
        the failing tool is dropped from both ``self.tools`` and
        ``self.tool_runner.tools`` so the registry stays consistent
        with what the model can actually call.

        If a required tool fails, ``aclose`` runs explicitly here (not
        via ``__aexit__`` — Python's async-context protocol does NOT
        call ``__aexit__`` when ``__aenter__`` raises). This guarantees
        any tools that DID start get a ``shutdown()`` call before the
        ``RuntimeError`` surfaces.

        See :func:`tether_service.tools.lifecycle.startup_all` for the
        gather semantics (synthesis §13.2 R5).
        """
        if self.tools:
            from tether_service.tools.lifecycle import startup_all

            try:
                failures = await startup_all(
                    self.tools, fail_fast_required=True
                )
            except Exception:
                # Required failure surfaced after gather completed.
                # Clean up tools that successfully started before re-raising.
                await self.aclose()
                raise
            for name in failures:
                # Required failures already raised inside startup_all; only
                # optional failures reach here. Drop them from BOTH the engine's
                # tool dict and the tool_runner's so the orchestrator can't
                # accidentally invoke a half-initialized tool.
                self.tools.pop(name, None)
                self.tool_runner.tools.pop(name, None)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Phase 3: bounded shutdown via :class:`HardwareWatchdog` when present.

        Phase 4 (step 41): tool ``shutdown()`` runs FIRST, before the
        hardware-watchdog teardown. Tools may legitimately depend on
        the provider during their own cleanup (e.g., a tool that wraps
        a provider call to flush state); reversing the order would
        risk hitting an already-closed provider. Tool shutdown is
        best-effort — failures are logged but never raised.

        Routes through ``self.hw_watchdog.shutdown_all()`` for any
        provider implementing :class:`HardwareLifecycle` — that path
        uses :func:`tether_service.runtime.daemon_call.daemon_thread_call`
        (M1) so GC stays disabled in the daemon thread (R5) and native
        cleanup is bounded by per-provider budgets. Production /
        SERVER-mode usage always reaches this branch because
        ``from_settings`` always builds a watchdog.

        Falls back to ``provider.shutdown_all()`` when ``hw_watchdog``
        is ``None`` — only direct constructor / test paths produce
        that.

        Synthesis §4 Phase 3 step 35 + §4 Phase 4 step 41; supersedes
        the §11.3 R22 placeholder.
        """
        if self._closed:
            return
        self._closed = True

        # Tools first (they may still need the provider during shutdown).
        if self.tools:
            from tether_service.tools.lifecycle import shutdown_all

            await shutdown_all(self.tools)

        # Then provider via watchdog.
        if self.hw_watchdog is not None:
            self.hw_watchdog.shutdown_all()
        elif hasattr(self.provider, "shutdown_all"):
            self.provider.shutdown_all()
