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
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional

from tether.config.settings import Settings
from tether.core.interfaces import (
    ModelProvider,
    SessionStore,
    StreamParser,
    Tool,
)
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration.tool_runner import ToolRunner
from tether.runtime.watchdog_mode import WatchdogMode

if TYPE_CHECKING:
    from tether.context.inbox_store import InboundInbox
    from tether.core.connector_registry import ConnectorRegistry
    from tether.protocol.intent.classifier import ConfirmIntentClassifier
    from tether.protocol.orchestration.cancel import CancelToken
    from tether.protocol.wire.events import WireEvent
    from tether.runtime.hw_watchdog import HardwareWatchdog


logger = logging.getLogger(__name__)


class Engine:
    """Library-first composition root for Tether."""

    def __init__(
        self,
        *,
        # ADR-0021 P2.A: dict-based provider registry. The legacy
        # ``provider=`` (singular) kwarg is preserved as a one-cycle
        # back-compat shim — when it is passed, we synthesise
        # ``providers={"default": provider}`` with
        # ``default_provider_id="default"``. The 1660-test suite predates
        # the dict form and relies on this shim; remove together with
        # ``providers.model`` (legacy singular settings field).
        providers: Optional[Dict[str, ModelProvider]] = None,
        default_provider_id: Optional[str] = None,
        provider_start_failures: Optional[Dict[str, str]] = None,
        provider: Optional[ModelProvider] = None,
        parser: Optional[StreamParser] = None,
        parser_factory: Optional[Callable[[], StreamParser]] = None,
        session_store: SessionStore,
        tools: Dict[str, Tool],
        system_prompt: str,
        watchdog_mode: WatchdogMode = WatchdogMode.LIBRARY,
        orchestrator_config: Optional[OrchestratorConfig] = None,
        tool_runner: Optional[ToolRunner] = None,
        hw_watchdog: Optional["HardwareWatchdog"] = None,
        connector_registry: Optional["ConnectorRegistry"] = None,
        inbox: Optional["InboundInbox"] = None,
        orchestrator_registry: Optional[Dict[str, str]] = None,
        orchestrator_default_mode: str = "chat",
    ):
        """Build an Engine from already-constructed components.

        ``orchestrator_config`` and ``tool_runner`` may be omitted; sensible
        defaults are produced (``OrchestratorConfig`` with library defaults,
        ``ToolRunner`` with the default 15s timeout) so direct constructors
        and the deprecated ``GenerationService`` alias keep working. Tests
        and advanced callers may pass them explicitly. Per p2-cleanup
        (synthesis §4 Phase 2 step 23).

        Either ``parser`` (a single instance) or ``parser_factory`` (a
        callable producing a fresh parser per turn) MUST be provided.
        ``parser_factory`` is the recommended path for any Engine that
        serves concurrent requests — a single shared parser instance
        race-conditions on its internal buffer (gpt-5.5 / Phase 5 review
        F1). ``Engine.from_settings`` uses ``parser_factory`` so that
        production deployments construct a fresh ``SlidingParser`` per
        turn. If only ``parser`` is given, ``parser_factory`` defaults to
        a closure returning the same instance — back-compat for tests
        and library callers that explicitly intend single-threaded use.

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

        ``orchestrator_registry`` and ``orchestrator_default_mode`` control
        which Orchestrator class is instantiated per-request based on the
        ``mode`` parameter to :meth:`chat`. Mirrors the tools.registry
        pattern. Briefing §2 Seam B item 4; synthesis §3.5.
        """
        if parser is None and parser_factory is None:
            raise ValueError(
                "Engine requires either parser= or parser_factory="
            )
        if parser is not None and parser_factory is not None:
            raise ValueError(
                "Engine takes parser= OR parser_factory=, not both"
            )

        if parser_factory is None:
            # Back-compat: callers passing a single instance get a closure
            # that returns the same instance. Single-threaded use only.
            _shared = parser
            self._parser_factory: Callable[[], StreamParser] = (
                lambda: _shared  # type: ignore[return-value]
            )
        else:
            self._parser_factory = parser_factory

        # ADR-0021 P2.A: resolve provider registry from either the new
        # ``providers=`` dict + ``default_provider_id`` OR the legacy
        # singular ``provider=`` kwarg (one-cycle shim for the 1660-test
        # suite; remove together with ``providers.model`` settings).
        if providers is not None:
            if not providers:
                raise ValueError(
                    "Engine: providers dict cannot be empty"
                )
            if default_provider_id is None:
                raise ValueError(
                    "Engine: default_provider_id required when "
                    "providers= is passed"
                )
            # ADR-0021: default_provider_id MUST be a healthy provider.
            # Construction-time validation is strict because direct
            # constructors (tests, library callers) MUST pre-resolve the
            # fallback themselves. ``from_settings`` already falls back to
            # the first healthy id before reaching this constructor, so
            # this branch only catches programmer errors (e.g. tests that
            # forget the dependency between providers + default_provider_id).
            if default_provider_id not in providers:
                raise ValueError(
                    f"Engine: default_provider_id "
                    f"{default_provider_id!r} not in providers "
                    f"(known: {sorted(providers)}). The default must be a "
                    "healthy provider; if its construction failed, fall "
                    "back to a healthy id before constructing the Engine."
                )
            self.providers: Dict[str, ModelProvider] = dict(providers)
            self.default_provider_id: str = default_provider_id
            self._provider_start_failures: Dict[str, str] = dict(
                provider_start_failures or {}
            )
            # Legacy back-compat alias (synthesise the singular attr).
            self.provider: ModelProvider = self.providers[
                self.default_provider_id
            ]
        elif provider is not None:
            # Legacy singular-provider construction path. Synthesise the
            # one-entry registry so the rest of Engine has the same
            # invariants regardless of how it was built.
            self.providers = {"default": provider}
            self.default_provider_id = "default"
            self._provider_start_failures = dict(provider_start_failures or {})
            self.provider = provider
        else:
            raise ValueError(
                "Engine requires either providers= (dict) or "
                "provider= (legacy singular)"
            )
        # ``self.parser`` is retained for back-compat reads (tests +
        # introspection). It's the parser produced by the factory at
        # construction time; new code should call ``self._parser_factory()``
        # to get a fresh instance per turn (see :meth:`chat`).
        self.parser = parser if parser is not None else self._parser_factory()
        self.store = session_store
        self.tools = tools
        self.system_prompt = system_prompt
        self.watchdog_mode = watchdog_mode
        self.hw_watchdog = hw_watchdog
        self.connector_registry = connector_registry
        # Phase 6.5 step 66e (synthesis §4 + ADR-0009): the inbox is
        # held on Engine so :meth:`__aenter__` / :meth:`aclose` can
        # sequence its connect / aclose alongside the session store.
        # The ConnectorRegistry has its own non-owning reference for
        # drain-task wiring; Engine owns the lifecycle here.
        self.inbox = inbox
        self.orchestrator_config = orchestrator_config or OrchestratorConfig(
            max_tool_loops=5,
            auto_reload_on_fatal_error=True,
            save_thinking=True,
            include_thinking_in_history=False,
        )
        self.tool_runner = tool_runner or ToolRunner(tools)
        self._closed = False
        self._orchestrator_registry: Dict[str, str] = orchestrator_registry or {
            "chat": "tether.protocol.orchestration.chatty.ChattyAgentOrchestrator",
            "research": "tether.protocol.orchestration.notebook.NotebookOrchestrator",
        }
        self._orchestrator_default_mode = orchestrator_default_mode
        # Phase 7 step 74: whether to store raw args_json in tool_audit.
        # Default False (privacy-preserving). Set by Engine.from_settings
        # from settings.security.audit_log.store_args. Synthesis B5 step 7.
        self._audit_store_args: bool = False
        self._confirm_intent_classifier: Optional["ConfirmIntentClassifier"] = None
        # Phase 4.5 step 47d: __aenter__ schedules start_connector(id) for
        # each READY connector; __aexit__/aclose cancels any still-pending
        # tasks before invoking stop_all so we never tear down a connector
        # mid-start.
        self._connector_start_tasks: List[asyncio.Task] = []
        # P0-F / Tribunal P0-07 (A2-F2): connectors whose start() raised
        # during __aenter__. Exposed via /readyz so process supervisors
        # can take action; populated by the awaited start loop in
        # __aenter__ and consumed by app.http.routers.health.
        self._connector_start_failures: List[str] = []

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        watchdog_mode: WatchdogMode = WatchdogMode.LIBRARY,
    ) -> "Engine":
        """Build an Engine from a typed Settings object.

        Lazy-imports concrete provider/parser/store classes so that
        ``import tether`` does not pull in MLC, FastAPI, or Brave.
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
        from tether.core.connector_registry import ConnectorRegistry
        from tether.core.factory import load

        # Configure logging FIRST so any subsequent setup logs flow through
        # the structured pipeline. Idempotent — safe to call multiple times.
        # Citations: _synthesis.md §3 (observability), §4 Phase 7 step 67.
        from tether.core.logging import configure_logging
        from tether.core.tool_registry import ToolRegistry
        from tether.runtime.hw_watchdog import HardwareWatchdog
        configure_logging(settings)

        model_registry = settings.providers.model_registry
        default_pid_requested = settings.providers.default_model_provider
        assert default_pid_requested is not None, (
            "ProvidersSettings validator must have resolved default_model_provider"
        )

        providers: Dict[str, ModelProvider] = {}
        provider_failures: Dict[str, str] = {}
        for pid, spec in model_registry.items():
            try:
                providers[pid] = load(spec.impl, **spec.args)
                logger.info(
                    "provider.loaded provider_id=%s impl=%s", pid, spec.impl
                )
            except Exception as exc:  # noqa: BLE001 - degraded-mode start
                from tether.core.errors import ConfigError as _ConfigErrLocal

                logger.error(
                    "provider.load_failed provider_id=%s impl=%s "
                    "error_class=%s error_message=%s",
                    pid,
                    spec.impl,
                    type(exc).__name__,
                    str(exc),
                )
                provider_failures[pid] = (
                    f"{type(exc).__name__}: {exc}"
                )
                # Defensive: never let ConfigError leak as a degraded
                # failure — it's always a fail-fast condition.
                if isinstance(exc, _ConfigErrLocal):
                    raise

        if not providers:
            from tether.core.errors import ConfigError as _CE

            raise _CE(
                f"All {len(provider_failures)} providers failed to construct: "
                f"{provider_failures!r}"
            )

        default_pid = default_pid_requested
        if default_pid not in providers:
            # Default provider itself failed. Fall back to first healthy
            # id in declaration order. Loud warning — operator must fix
            # config.
            original = default_pid
            default_pid = next(iter(providers))
            logger.warning(
                "provider.default_unhealthy_fallback requested=%s fallback=%s",
                original,
                default_pid,
            )

        parser_spec = settings.providers.parser

        # Per-request parser FACTORY rather than a single shared instance.
        # The Phase 5 rubber-duck review (gpt-5.5, F1) showed that a shared
        # SlidingParser cross-contaminates concurrent requests because its
        # internal buffer / state is mutable (``self.buf``, ``self._json_depth``,
        # ``self._tool_started``, ...). Per-turn construction is the fix.
        # ``importlib.import_module`` caches modules, so re-loading per turn
        # is essentially a constructor call.
        def _new_parser() -> StreamParser:
            return load(parser_spec.impl, **parser_spec.args)

        store_spec = settings.providers.session_store
        # Phase 6 step 59: apply pending schema migrations BEFORE constructing
        # the store so the DB is always at the latest schema version when the
        # store opens its connection. Migration is idempotent — calling it on
        # an already-current DB is a no-op. Synthesis §3.6, B1 step 2.
        # Phase 6 step 60: DSN resolved from settings.storage.sqlite.dsn
        # (platformdirs default when unset) instead of the legacy CWD-relative
        # literal. Synthesis §3.6, §4 Phase 6 step 60.
        _store_dsn: str = settings.storage.resolved_dsn()
        # Phase 9 P0-A (Tribunal §3 P0-01 / A3-F1, A12-F1):
        # ``storage.sqlite.dsn`` is the single source of truth for the SQLite
        # database file used by both the session store and the inbox
        # (ADR-0009). The legacy ``providers.session_store.args.dsn`` is now
        # forbidden as a divergent override — if present it must match the
        # resolved storage DSN exactly. This prevents the silent split that
        # routed sessions and inbox events into two different files.
        _legacy_store_dsn = store_spec.args.get("dsn")
        if _legacy_store_dsn is not None and _legacy_store_dsn != _store_dsn:
            from tether.core.errors import ConfigError

            raise ConfigError(
                f"providers.session_store.args.dsn={_legacy_store_dsn!r} "
                f"disagrees with storage.resolved_dsn()={_store_dsn!r}. "
                "Remove the legacy session-store DSN; storage.sqlite.dsn is "
                "the single source of truth (ADR-0009)."
            )
        _store_args = {**store_spec.args, "dsn": _store_dsn}
        from tether.context.migration_runner import apply_pending_migrations
        try:
            apply_pending_migrations(_store_dsn)
        except Exception as _mig_exc:
            logger.exception(
                "Schema migration failed at Engine startup: %s", _mig_exc
            )
            raise
        store = load(store_spec.impl, **_store_args)

        # Phase 6.5 step 66e (synthesis §4 + ADR-0009): SqliteInbox
        # shares the session-store DSN — one DB file, one connection
        # lifecycle, one migration track. The inbox is constructed here
        # rather than in ConnectorRegistry so Engine.aclose can sequence
        # its lifecycle alongside the session store.
        inbox: Optional["InboundInbox"] = None
        if settings.inbox.enabled:
            from tether.context.inbox_store import SqliteInbox

            inbox = SqliteInbox(
                _store_dsn,
                max_payload_bytes=settings.inbox.max_payload_bytes,
                max_summary_chars=settings.inbox.max_summary_chars,
            )

        registry = ToolRegistry.from_settings(settings)
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
            connectors, tool_names=set(tools.keys()), inbox=inbox
        )

        # Merge connector tools into the flat dict the orchestrator sees.
        # Safe to merge naively — registry construction proved no
        # cross-connector or in-tree collisions exist.
        tools.update(connector_registry.aggregate_tools())

        orchestrator_config = OrchestratorConfig.from_settings(settings)
        tool_runner = ToolRunner(tools, timeout_sec=settings.limits.tool_timeout_sec,
                                 result_max_bytes=settings.security.tool_result_max_bytes)

        hw_watchdog = HardwareWatchdog(list(providers.values()), mode=watchdog_mode)

        # Phase 5 followups F6 (rubber-duck review): validate the orchestrator
        # registry at boot rather than letting a typo surface as a 500
        # mid-handler. ``importlib.import_module`` caches modules so this is
        # cheap. The default mode is checked first (most-likely typo target),
        # followed by every other registered mode.
        registry_dict = dict(settings.orchestrator.registry)
        from tether.protocol.orchestration.registry import (
            UnknownOrchestratorMode,
            resolve_orchestrator_class,
        )
        for mode in registry_dict:
            try:
                resolve_orchestrator_class(mode, registry_dict)
            except UnknownOrchestratorMode:
                # Cannot happen — we just iterated the registry's keys.
                raise
            except (ImportError, AttributeError, TypeError) as e:
                raise ValueError(
                    f"Engine.from_settings: orchestrator registry entry "
                    f"{mode!r} -> {registry_dict[mode]!r} is invalid: {e}"
                ) from e

        # Phase 2b (ADR-0019): construct the confirm-intent classifier from
        # settings. ChattyAgentOrchestrator uses it to flip
        # ToolExecutionContext.user_confirmed_send.
        confirm_intent_classifier = load(settings.intent.classifier_impl)

        engine = cls(
            providers=providers,
            default_provider_id=default_pid,
            provider_start_failures=provider_failures,
            parser_factory=_new_parser,
            session_store=store,
            tools=tools,
            system_prompt=settings.system.prompt,
            watchdog_mode=watchdog_mode,
            orchestrator_config=orchestrator_config,
            tool_runner=tool_runner,
            hw_watchdog=hw_watchdog,
            connector_registry=connector_registry,
            inbox=inbox,
            orchestrator_registry=registry_dict,
            orchestrator_default_mode=settings.orchestrator.default,
        )
        # Phase 7 step 74: wire audit_log.store_args so the orchestrator
        # knows whether to populate args_json in tool_audit rows.
        # Synthesis B5 step 7.
        engine._audit_store_args = settings.security.audit_log.store_args
        engine._confirm_intent_classifier = confirm_intent_classifier
        return engine

    # --- Streaming chat (the core API) ---

    async def stream(
        self,
        *,
        session_id: str,
        prompt: str,
        model_name: str,
        mode: Optional[str] = None,
        cancel_event: Optional[asyncio.Event] = None,
        reasoning_effort: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Drive the core orchestration to stream NDJSON bytes (v0 vocabulary).

        Routes through :meth:`chat` (which performs mode-based orchestrator
        selection via the registry) and serializes each :class:`WireEvent`
        through the v0 compatibility shim. Mode dispatch happens at the
        orchestrator-resolution boundary; bytes wire format is independent.

        Replaces the former ``orchestrate()`` shim call so that
        ``mode`` is actually honored: callers passing ``mode="research"``
        get :class:`NotebookOrchestrator` (which raises
        :class:`NotImplementedError`) rather than silently falling back to
        ``ChattyAgentOrchestrator``. Briefing §2 Seam B item 4.

        ``reasoning_effort`` (when non-``None``) is forwarded to the
        orchestrator's ``run()`` method, which forwards it to the provider
        for models that advertise reasoning support. Validation that the
        chosen model accepts the requested value happens at the HTTP
        boundary (``app/http/routers/chat.py``).
        """
        from tether.protocol.orchestration.cancel import AsyncEventCancelToken
        from tether.protocol.orchestration.emitter import v0_compat_serialize

        cancel_token = AsyncEventCancelToken(cancel_event) if cancel_event else None

        async for wire_event in self.chat(
            session_id=session_id,
            prompt=prompt,
            model_name=model_name,
            mode=mode,
            cancel_token=cancel_token,
            reasoning_effort=reasoning_effort,
            provider_id=provider_id,
        ):
            bytes_out = v0_compat_serialize(wire_event)
            if bytes_out:  # MessageStart returns b"" — skip
                yield bytes_out

    async def chat(
        self,
        *,
        session_id: str,
        prompt: str,
        model_name: str,
        mode: Optional[str] = None,
        cancel_token: Optional["CancelToken"] = None,
        reasoning_effort: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> AsyncGenerator["WireEvent", None]:
        """Library-mode typed event stream (synthesis §3.4).

        Yields typed :class:`WireEvent` Python objects. HTTP transports
        use :meth:`stream` (which serializes to v0 NDJSON bytes via the
        back-compat shim). Library consumers and the future SSE / dual-
        emit transport (``p5-cutover-a-dual-emit``) iterate
        :class:`WireEvent` directly.

        ``mode`` selects the Orchestrator class from the registry. Defaults
        to ``self._orchestrator_default_mode`` ("chat") when None.
        Briefing §2 Seam B item 4; synthesis §3.5.

        ``reasoning_effort`` is forwarded to ``orch.run(...)`` only when the
        orchestrator's ``run`` signature accepts it (introspected via
        ``inspect.signature``), mirroring the audit_store_args pattern. The
        HTTP boundary validates the value against the chosen model's
        :class:`ModelDetails` before this is reached.
        """
        from tether.protocol.orchestration.registry import (
            resolve_orchestrator_class,
        )

        # ADR-0021 P2.A: resolve provider routing FIRST so unknown /
        # unhealthy ids surface as typed errors before any orchestrator
        # work. HTTP layer (Phase 2.B) maps these to 422 / 503.
        from tether.core.errors import (
            ProviderUnhealthyError,
            UnknownProviderError,
        )

        pid = provider_id or self.default_provider_id
        if pid not in self.providers:
            if pid in self._provider_start_failures:
                raise ProviderUnhealthyError(
                    pid, self._provider_start_failures[pid]
                )
            raise UnknownProviderError(pid)
        provider = self.providers[pid]

        effective_mode = mode if mode is not None else self._orchestrator_default_mode
        orchestrator_cls = resolve_orchestrator_class(
            effective_mode, self._orchestrator_registry
        )

        # Per-turn parser instance — Phase 5 followups F1. A shared
        # SlidingParser would race-condition on its mutable buffer when
        # two requests interleave (rubber-duck review by gpt-5.5).
        per_turn_parser = self._parser_factory()

        # Phase 7 step 74: pass audit_store_args to orchestrators that
        # support it. inspect.signature check avoids breaking stub
        # orchestrators (e.g., NotebookOrchestrator) that don't yet
        # have the kwarg in their __init__. Synthesis B5 step 7.
        import inspect as _inspect
        _orch_sig = _inspect.signature(orchestrator_cls.__init__)
        _orch_kwargs: Dict[str, Any] = dict(
            provider=provider,
            parser=per_turn_parser,
            store=self.store,
            tools=self.tools,
            system_prompt=self.system_prompt,
            config=self.orchestrator_config,
            tool_runner=self.tool_runner,
            hw_watchdog=self.hw_watchdog,
        )
        if "audit_store_args" in _orch_sig.parameters:
            _orch_kwargs["audit_store_args"] = self._audit_store_args
        if "confirm_intent_classifier" in _orch_sig.parameters:
            _orch_kwargs["confirm_intent_classifier"] = self._confirm_intent_classifier

        orch = orchestrator_cls(**_orch_kwargs)

        # Per-turn run kwargs: only forward reasoning_effort to orchestrator
        # impls that opt in via their ``run()`` signature. Stub
        # orchestrators (NotebookOrchestrator) and existing impls without
        # the kwarg keep working unchanged.
        _run_kwargs: Dict[str, Any] = dict(
            session_id=session_id,
            prompt=prompt,
            model_name=model_name,
            cancel_token=cancel_token,
        )
        if reasoning_effort is not None:
            _run_sig = _inspect.signature(orchestrator_cls.run)
            if "reasoning_effort" in _run_sig.parameters:
                _run_kwargs["reasoning_effort"] = reasoning_effort

        async for wire_event in orch.run(**_run_kwargs):
            yield wire_event

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
        """Merged ``list[str]`` across healthy providers.

        ADR-0021 P2.A contract §2 / contract §4 ``/api/v1/models``:
        union across healthy providers in declaration order; duplicate
        model_names disambiguated as ``"<provider_id>/<model_name>"``;
        non-duplicates remain bare model_names (back-compat with
        single-provider deployments).
        """
        # First pass: build per-provider model lists in declaration order.
        per_pid: List[tuple] = [
            (pid, list(self.providers[pid].list_models()))
            for pid in self.providers
        ]
        # Count occurrences to know which names collide.
        from collections import Counter

        counts: Counter = Counter(
            name for _, models in per_pid for name in models
        )
        out: List[str] = []
        for pid, models in per_pid:
            for name in models:
                out.append(f"{pid}/{name}" if counts[name] > 1 else name)
        return out

    def list_model_info(self):
        """Return :class:`ModelDetails` merged across healthy providers.

        ADR-0021 P2.A contract §2 / contract §5: each provider's
        ``list_model_info()`` result is wrapped via
        ``info.model_copy(update={"provider_id": pid})`` so callers see
        the engine's stable registry id rather than the sentinel
        ``"_unwrapped_"`` default. No de-dup at this layer.
        """
        out = []
        for pid, prov in self.providers.items():
            for info in prov.list_model_info():
                # ``model_copy`` is Pydantic v2 — every ModelDetails is
                # a frozen BaseModel so we cannot mutate; copy with the
                # update applied.
                out.append(info.model_copy(update={"provider_id": pid}))
        return out

    def unload_model(
        self, model_name: str, *, provider_id: Optional[str] = None
    ) -> bool:
        """Unload ``model_name`` from one or more providers.

        ADR-0021 P2.A contract §2:
        - ``provider_id`` given → dispatch directly to that provider
          (rejects unknown / unhealthy ids).
        - ``provider_id`` omitted → fan out to every healthy provider
          that lists ``model_name``; return True if ANY returned True.
        """
        if provider_id is not None:
            from tether.core.errors import (
                ProviderUnhealthyError,
                UnknownProviderError,
            )
            if provider_id not in self.providers:
                if provider_id in self._provider_start_failures:
                    raise ProviderUnhealthyError(
                        provider_id,
                        self._provider_start_failures[provider_id],
                    )
                raise UnknownProviderError(provider_id)
            return self.providers[provider_id].unload_model(model_name)
        any_unloaded = False
        for prov in self.providers.values():
            try:
                models = prov.list_models()
            except Exception:  # noqa: BLE001 - defensive
                models = []
            if model_name in models:
                if prov.unload_model(model_name):
                    any_unloaded = True
        return any_unloaded

    def list_provider_health(self) -> Dict[str, Dict[str, Any]]:
        """Per-provider snapshot for ``/readyz``.

        Cheap, sync, no network. Includes entries for both healthy
        providers (from ``self.providers``) AND failed registry entries
        (from ``self._provider_start_failures``). ADR-0021 contract §2.
        """
        out: Dict[str, Dict[str, Any]] = {}
        for pid, prov in self.providers.items():
            try:
                kind = prov.kind
            except Exception:  # noqa: BLE001 - defensive
                kind = "unknown"
            try:
                source = prov.source
            except Exception:  # noqa: BLE001 - defensive
                source = "unknown"
            if source not in ("local", "remote"):
                source = "unknown"
            out[pid] = {
                "healthy": True,
                "kind": kind,
                "source": source,
                "error": None,
            }
        for pid, msg in self._provider_start_failures.items():
            # Don't overwrite a healthy entry — degraded warm_up failures
            # may have moved the provider into _provider_start_failures
            # after construction succeeded; in that case the healthy
            # branch above wouldn't have added it because it's no longer
            # in self.providers.
            out[pid] = {
                "healthy": False,
                "kind": "unknown",
                "source": "unknown",
                "error": msg,
            }
        return out

    # --- Lifecycle ---

    async def _warm_up_providers_degraded(self) -> None:
        """Run ``warm_up`` for every provider opting in via capabilities.

        ADR-0021 P2.A: failures are non-fatal. The failing provider is
        demoted (moved from ``self.providers`` to
        ``self._provider_start_failures``) so routing rejects it with
        ProviderUnhealthyError (HTTP 503). If the demoted id was the
        default, fall back to the first remaining healthy id with a
        warning. If every provider fails, do NOT abort here — engine
        startup may still serve introspection endpoints; chat requests
        will surface 503 cleanly.
        """
        if not self.providers:
            return
        # Snapshot to avoid mutating while iterating.
        for pid in list(self.providers.keys()):
            prov = self.providers[pid]
            caps = getattr(prov, "capabilities", None)
            if caps is None or not getattr(caps, "warm_up_required", False):
                continue
            model = None
            try:
                model = prov.default_model()
            except Exception:  # noqa: BLE001 - defensive
                model = None
            if not model:
                try:
                    models = prov.list_models()
                except Exception:  # noqa: BLE001 - defensive
                    models = []
                model = models[0] if models else None
            if not model:
                # Nothing to warm up against; skip silently.
                continue
            try:
                await prov.warm_up(model)
            except Exception as exc:  # noqa: BLE001 - degraded warm-up
                logger.warning(
                    "provider.warm_up_failed provider_id=%s model=%s "
                    "error_class=%s error_message=%s",
                    pid,
                    model,
                    type(exc).__name__,
                    str(exc),
                )
                self._provider_start_failures[pid] = (
                    f"warm_up: {type(exc).__name__}: {exc}"
                )
                self.providers.pop(pid, None)
                if pid == self.default_provider_id and self.providers:
                    new_default = next(iter(self.providers))
                    logger.warning(
                        "provider.default_demoted_fallback "
                        "previous=%s fallback=%s",
                        pid,
                        new_default,
                    )
                    self.default_provider_id = new_default
                    self.provider = self.providers[new_default]

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

        Phase 4.5 step 47d: after tool startup, schedule
        ``start_connector(id)`` for every connector whose
        ``auth_status()`` reports :class:`ConnectorState.READY`.
        Connectors in any other state (UNCONFIGURED, LOGGED_OUT, ERROR)
        stay stopped — their tools are still registered but raise
        ``ConnectorNotConfiguredError`` until the user runs the login
        flow via the HTTP routes (connector spec §3.3 step 4 + §3.8).
        Started concurrently as fire-and-forget tasks tracked on
        ``self._connector_start_tasks`` so a slow ``start()`` does not
        block ``__aenter__``; ``aclose`` cancels any still-pending
        tasks before invoking ``stop_all`` (connector spec §3.3 step 6).

        See :func:`tether.tools.lifecycle.startup_all` for the
        gather semantics (synthesis §13.2 R5).
        """
        # Phase 6 step 63: open the SqliteSessionStore's aiosqlite
        # connection eagerly inside __aenter__. ``from_settings`` only
        # constructs the store (sync); the async connect() must happen
        # under an event loop. ``isinstance(self.store, SessionStore)``
        # excludes raw ``MagicMock()`` test fakes (which auto-create
        # any attribute) so direct-constructor unit tests using mocks
        # are unaffected. MemoryStore inherits a no-op connect() from
        # the ABC; SqliteSessionStore overrides it. Synthesis §3.6.
        # ADR-0021 P2.A: degraded provider warm-up BEFORE store / inbox
        # / tools / connectors. Providers whose ``capabilities``
        # advertise ``warm_up_required=True`` get a ``warm_up`` call;
        # failures demote the provider into ``_provider_start_failures``
        # and remove it from ``self.providers`` so subsequent routing
        # rejects them cleanly with ProviderUnhealthyError (503).
        # NOTE: ADR-0021 ambiguity — the ADR text says "demotes the
        # provider but does not abort engine startup". We interpret
        # "demote" as removal from the live registry so that any
        # request routed to that pid hits the typed 503 path rather
        # than racing against a half-initialised provider.
        await self._warm_up_providers_degraded()

        if isinstance(self.store, SessionStore):
            try:
                await self.store.connect()
            except Exception:
                # If the store can't open we cannot proceed; don't bother
                # starting tools or connectors. aclose handles partial
                # state via its idempotency guards.
                await self.aclose()
                raise

        # Phase 6.5 step 66e: open the SqliteInbox alongside the store
        # so the connector drain tasks (spawned below in
        # ``start_connector``) have a live aiosqlite connection to
        # write into. Independently lifecycled — a missing inbox is
        # not fatal; the connector registry simply skips drain wiring.
        if self.inbox is not None and hasattr(self.inbox, "connect"):
            try:
                await self.inbox.connect()
            except Exception:
                await self.aclose()
                raise

        if self.tools:
            from tether.tools.lifecycle import startup_all

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

        if self.connector_registry is not None:
            # Lazy-imported because connectors module triggers no eager
            # imports at the package level (R8 lazy-import rule).
            from tether.connectors.types import ConnectorState

            for conn in self.connector_registry.all():
                try:
                    status = await conn.auth_status()
                except Exception as exc:  # noqa: BLE001 - defensive
                    logger.exception(
                        "auth_status() raised for connector %s during "
                        "startup; skipping. (%s)",
                        conn.id,
                        exc,
                    )
                    continue
                if status.state is ConnectorState.READY:
                    task = asyncio.create_task(
                        self.connector_registry.start_connector(conn.id),
                        name=f"start_connector:{conn.id}",
                    )
                    self._connector_start_tasks.append(task)

            # P0-F / Tribunal P0-07 / A2-F2: await connector startup BEFORE
            # returning control to the FastAPI lifespan. Otherwise the first
            # chat() can arrive at a half-initialized connector. Failures are
            # logged and the connector is removed from the registry so
            # subsequent tool dispatch sees a deterministic
            # ConnectorNotConfiguredError rather than a phantom object.
            if self._connector_start_tasks:
                results = await asyncio.gather(
                    *self._connector_start_tasks, return_exceptions=True
                )
                for task, outcome in zip(self._connector_start_tasks, results):
                    if isinstance(outcome, BaseException):
                        task_name = task.get_name() or ""
                        cid_from_name = (
                            task_name.split(":", 1)[-1]
                            if ":" in task_name
                            else "<unknown>"
                        )
                        logger.exception(
                            "connector.start_failed cid=%s error_class=%s "
                            "error_message=%s",
                            cid_from_name,
                            type(outcome).__name__,
                            str(outcome),
                            exc_info=outcome,
                        )
                        self._connector_start_failures.append(cid_from_name)
                        # Remove the failing connector so it cannot serve
                        # traffic. ``pop(..., None)`` is a no-op if the
                        # registry already evicted it.
                        self.connector_registry._connectors.pop(
                            cid_from_name, None
                        )
                # Tasks are no longer pending; aclose() doesn't need to
                # cancel them.
                self._connector_start_tasks = []
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

        Phase 4.5 step 47d: connectors are stopped BEFORE tools. The
        order is: cancel any still-pending ``_connector_start_tasks``
        → ``connector_registry.stop_all(timeout_sec=2.0)`` → tool
        ``shutdown_all`` → watchdog/provider shutdown. Connectors must
        stop before tools because connector tools dispatch through the
        connector instance (e.g. an ``echo_send`` tool that references
        the connector's open session); tearing down the connector
        first guarantees no in-flight tool ``invoke()`` outlives its
        owner. The 2 s cooperative budget is the connector spec §3.3
        step 6 contract — connectors with potentially blocking native
        cleanup are responsible for their own daemon-thread + force-exit
        pattern (see ``tether.connectors.base.Connector``).

        Routes through ``self.hw_watchdog.shutdown_all()`` for any
        provider implementing :class:`HardwareLifecycle` — that path
        uses :func:`tether.runtime.daemon_call.daemon_thread_call`
        (M1) so GC stays disabled in the daemon thread (R5) and native
        cleanup is bounded by per-provider budgets. Production /
        SERVER-mode usage always reaches this branch because
        ``from_settings`` always builds a watchdog.

        Falls back to ``provider.shutdown_all()`` when ``hw_watchdog``
        is ``None`` — only direct constructor / test paths produce
        that.

        Synthesis §4 Phase 3 step 35 + §4 Phase 4 step 41 + §4 Phase
        4.5 step 47d; supersedes the §11.3 R22 placeholder.
        """
        if self._closed:
            return
        self._closed = True

        # Phase 4.5: cancel any pending start_connector tasks first so we
        # don't tear down a connector mid-start. ``return_exceptions=True``
        # swallows the ``CancelledError`` we just induced.
        if self._connector_start_tasks:
            for task in self._connector_start_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                *self._connector_start_tasks, return_exceptions=True
            )
            self._connector_start_tasks.clear()

        # Phase 4.5: stop connectors before tools (see docstring).
        if self.connector_registry is not None:
            await self.connector_registry.stop_all(timeout_sec=2.0)

        # Tools next (they may still need the provider during shutdown).
        if self.tools:
            from tether.tools.lifecycle import shutdown_all

            await shutdown_all(self.tools)

        # Then provider via watchdog.
        # ADR-0021 P2.A: aclose() fan-out for non-HardwareLifecycle
        # providers BEFORE the watchdog shutdown (which only handles
        # HardwareLifecycle providers). The watchdog's bounded daemon-
        # thread path is for HW providers only (ADR-0003 invariant);
        # stateless / HTTP providers (Copilot, Dummy) get a plain async
        # aclose() here. Each call is individually shielded — one
        # provider's cleanup failure cannot block the others, and a
        # MagicMock-style fake whose ``aclose()`` returns a non-awaitable
        # is tolerated (test paths use MagicMock providers).
        import inspect as _inspect
        from tether.providers.hw import HardwareLifecycle as _HwLC
        non_hw_providers = [
            p for p in self.providers.values()
            if not isinstance(p, _HwLC)
        ]
        for _p in non_hw_providers:
            _close = getattr(_p, "aclose", None)
            if _close is None:
                continue
            try:
                _maybe = _close()
            except Exception as _exc:  # noqa: BLE001 - defensive
                logger.exception(
                    "Engine.aclose: provider aclose() raised: %s", _exc
                )
                continue
            if _inspect.isawaitable(_maybe):
                try:
                    await _maybe
                except Exception as _exc:  # noqa: BLE001 - defensive
                    logger.exception(
                        "Engine.aclose: provider aclose() awaited "
                        "raise: %s",
                        _exc,
                    )
        if self.hw_watchdog is not None:
            self.hw_watchdog.shutdown_all()
        elif hasattr(self.provider, "shutdown_all"):
            self.provider.shutdown_all()

        # Phase 6 step 63: close the SqliteSessionStore's aiosqlite
        # connection. Last in the teardown order so the provider /
        # tools / connectors finish their cleanup against a still-open
        # store if they depend on it. ``isinstance`` excludes raw
        # ``MagicMock()`` test fakes; MemoryStore inherits a no-op
        # aclose() from the ABC. Failure is logged but never raised —
        # aclose must always complete. Synthesis §3.6.
        if isinstance(self.store, SessionStore):
            try:
                await self.store.aclose()
            except Exception as exc:  # noqa: BLE001 - defensive
                logger.exception(
                    "SessionStore.aclose() raised during Engine teardown: %s",
                    exc,
                )

        # Phase 6.5 step 66e: close the SqliteInbox connection AFTER
        # the connectors stopped (which already cancelled their drain
        # tasks above) and AFTER the session store. The inbox shares
        # the same DB file as the session store but holds an
        # independent aiosqlite worker thread; closing both is
        # required to release the WAL lock cleanly. Failure is logged
        # but never raised — aclose must always complete.
        if self.inbox is not None and hasattr(self.inbox, "aclose"):
            try:
                await self.inbox.aclose()
            except Exception as exc:  # noqa: BLE001 - defensive
                logger.exception(
                    "SqliteInbox.aclose() raised during Engine teardown: %s",
                    exc,
                )
