from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Dict, List, Optional

from tether.protocol.parsers.events import ParserEvent
from tether.providers.types import ModelDetails, ProviderCapabilities, ProviderEvent

if TYPE_CHECKING:
    from tether.core.types import ToolExecutionContext
    from tether.protocol.orchestration.cancel import CancelToken
    from tether.protocol.wire.events import WireEvent


class ModelProvider(ABC):
    # ------------------------------------------------------------------
    # LEGACY contract (UNCHANGED; orchestrator still uses these).
    # Phase 5 step 52 will migrate the orchestrator to ``stream_typed``;
    # at that point ``stream()`` may become a shim.
    # ------------------------------------------------------------------

    @abstractmethod
    def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[str | List[Dict[str, Any]], None]:
        """Stream raw text chunks for a given model, history, and tools.

        Legacy: yields ``str`` for text deltas and ``List[Dict[str, Any]]``
        for native tool_calls (MLC-style). DEPRECATED in favor of
        :meth:`stream_typed` which returns typed :class:`ProviderEvent`
        values; kept for one cycle until Phase 5 step 52 migrates the
        orchestrator. See _synthesis.md §4 Phase 3 step 39, §6 bug #12.

        ``request_id`` is the caller's correlation ID (bound to structlog
        contextvars by RequestIdMiddleware). Providers accept it so any
        internal log calls they make can include it for cross-layer
        correlation. Phase 7 step 72.

        ``reasoning_effort`` is the per-request reasoning effort hint for
        providers/models that support it (e.g. Copilot SDK ``gpt-5``).
        Providers that ignore this field MUST still accept the kwarg.
        The orchestrator only forwards a non-``None`` value when the
        caller explicitly sets it and the model advertises support via
        :class:`tether.providers.types.ModelDetails`. Synthesis §3.4.
        """
        ...

    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models."""
        ...

    @abstractmethod
    def unload_model(self, model_name: str) -> bool:
        """Unload a model."""
        ...

    @abstractmethod
    def get_context_window(self, model_name: str) -> int:
        """
        Get the context window size for a specific model.
        
        Args:
            model_name: Name of the model (e.g., "Qwen3-4B-q4f16_0-MLC")
        
        Returns:
            Context window size in tokens
        """
        ...

    # ------------------------------------------------------------------
    # v2 typed contract (Phase 3 step 39).
    # Defaults so existing concrete classes still construct without
    # immediate overrides; concrete classes SHOULD override these to opt
    # into the v2 path. Synthesis §4 Phase 3 step 39, §11.3 R21.
    # ------------------------------------------------------------------

    @property
    def kind(self) -> str:
        """Provider kind identifier (e.g., ``"mlc"``, ``"dummy"``,
        ``"ollama"``).

        Concrete classes MUST override. Default raises
        :class:`NotImplementedError` so missing overrides fail loudly at
        introspection time. Synthesis §4 Phase 3 step 39.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override `kind` property"
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Static introspection of provider capabilities.

        Default returns all-False capabilities (fail closed); concrete
        classes should override with the right flags. Synthesis §4 Phase
        3 step 39.
        """
        return ProviderCapabilities()

    async def warm_up(self, model_name: str) -> None:
        """Pre-load a model so first inference is fast.

        Default no-op for providers that don't need eager warm-up
        (HTTP-based providers, dummy providers). Hardware-owning providers
        (MLC) override to call their cold-start path. Synthesis §4 Phase 3
        step 39.
        """
        return None

    async def aclose(self) -> None:
        """Provider-level shutdown.

        Default no-op for stateless providers (Dummy, future HTTP-based
        providers).

        Hardware-lifecycle providers (MLC) typically override to call
        their ``shutdown_all()``. Note: this is the PROVIDER's ``aclose``;
        ``Engine.aclose`` routes through :class:`HardwareWatchdog` for
        providers that implement :class:`HardwareLifecycle`. For non-HW
        providers, ``Engine.aclose`` calls this directly.
        """
        return None

    async def stream_typed(
        self,
        *,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        cancel_token: Optional[Any] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncIterator[ProviderEvent]:
        """v2 stream method yielding typed :class:`ProviderEvent` values.

        Default implementation raises :class:`NotImplementedError` so
        concrete classes must explicitly opt into the new contract.

        Phase 5 step 52 will migrate the orchestrator to consume this;
        until then the orchestrator uses the legacy :meth:`stream`.

        ``reasoning_effort`` mirrors the legacy :meth:`stream` kwarg;
        providers that don't support it MUST still accept the parameter.

        Synthesis §4 Phase 3 step 39, §6 bug #12 (native MLC tool_calls
        are emitted as :class:`ProviderToolCall` here, not silently
        dropped as the legacy ``stream()`` shape allowed).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override `stream_typed` for v2 contract"
        )
        # Unreachable, but makes the function an async generator so callers
        # can ``async for`` over the result without TypeError.
        if False:
            yield  # type: ignore[unreachable]

    # ------------------------------------------------------------------
    # Model metadata (HTTP /models/details endpoint).
    # ``list_models`` returns ``list[str]`` for back-compat with the
    # legacy ``GET /api/v1/models`` endpoint. ``list_model_info`` adds
    # the richer per-model capability metadata used by clients to
    # render selection UIs (CLI ``\models`` table, etc.).
    # ------------------------------------------------------------------

    @property
    def source(self) -> str:
        """Provider source class: ``"local"`` or ``"remote"``.

        ``"local"`` is the default for on-device providers (MLC, dummy).
        Hosted providers (Copilot SDK, future Anthropic, OpenAI, …)
        override to ``"remote"``. Used by clients to render network/cost
        hints without inspecting :attr:`kind` strings directly.
        """
        return "local"

    def default_model(self) -> Optional[str]:
        """Return the configured default model name, or ``None``.

        Concrete providers may override to point clients at the
        canonical first-choice model (e.g. the value the user supplied
        in ``providers.model.args.model``). When ``None``, callers
        should not mark any model as default. The ABC's default
        :meth:`list_model_info` implementation flags the matching
        ``ModelDetails.is_default``.
        """
        return None

    def list_model_info(self) -> List[ModelDetails]:
        """Return :class:`ModelDetails` metadata for each available model.

        Default implementation synthesises one :class:`ModelDetails` per
        entry returned by :meth:`list_models`, using :meth:`kind`,
        :attr:`capabilities`, :meth:`get_context_window`, :attr:`source`,
        and :meth:`default_model`. ``supports_reasoning_effort`` defaults
        to ``False`` — providers that accept :class:`reasoning_effort`
        in :meth:`stream` MUST override this method to advertise the
        accepted values.

        Serialised on ``GET /api/v1/models/details``; the back-compat
        ``GET /api/v1/models`` endpoint continues to return
        ``list[str]``. Synthesis §3.4.
        """
        default = self.default_model()
        models = self.list_models()
        caps = self.capabilities
        source = self.source
        if source not in ("local", "remote"):
            source = "local"
        kind = self.kind
        return [
            ModelDetails(
                id=name,
                # Sentinel: Engine wraps each provider's list_model_info()
                # output via model_copy(update={"provider_id": pid}) before
                # serving it. Rows reaching clients with this sentinel
                # bypassed the engine wrap (e.g. direct provider tests).
                provider_id="_unwrapped_",
                provider_kind=kind,
                source=source,  # type: ignore[arg-type]
                context_window=self.get_context_window(name),
                supports_thinking=caps.thinking_channel,
                supports_reasoning_effort=False,
                reasoning_efforts=None,
                is_default=(default is not None and default == name),
            )
            for i, name in enumerate(models)
        ]


class StreamParser(ABC):
    @abstractmethod
    def feed(self, chunk: str) -> List[ParserEvent]:
        """Ingest a raw model chunk and return zero or more typed parser events.

        Phase 5 ``p5-parser-typed-events``: returns
        :class:`tether.protocol.parsers.events.ParserEvent` values
        (frozen dataclasses), not dicts. Synthesis §4 Phase 5 step 51.
        """
        ...

    @abstractmethod
    def finalize(self) -> List[ParserEvent]:
        """Flush any residual state and return final typed parser events.

        Phase 5 ``p5-parser-typed-events``: returns
        :class:`tether.protocol.parsers.events.ParserEvent` values.
        Synthesis §4 Phase 5 step 51.
        """
        ...


class SessionStore(ABC):
    @abstractmethod
    async def create_session(self, session_id: str, created_at: int) -> None:
        """Create a new session."""
        ...

    @abstractmethod
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions."""
        ...

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID. Returns True if deleted, False if not found."""
        ...

    @abstractmethod
    async def delete_all_sessions(self) -> int:
        """Delete all sessions and return the count of deleted sessions."""
        ...

    @abstractmethod
    async def add_user(
        self,
        session_id: str,
        text: str,
        *,
        turn_id: Optional[str] = None,
        seq_start: Optional[int] = None,
    ) -> None:
        ...

    @abstractmethod
    async def add_assistant_text(
        self,
        session_id: str,
        text: str,
        thinking_text: Optional[str] = None,
        save_thinking: bool = True,
        *,
        turn_id: Optional[str] = None,
        seq_start: Optional[int] = None,
    ) -> None:
        """Persist an assistant turn, optionally including thinking content.

        Note: starting with Phase 6 step 65, ``thinking_text`` is persisted as
        a separate ``role='thinking'`` row in the messages table, not as a column
        on the assistant row. ``get_history(include_thinking=True)`` merges
        consecutive thinking + assistant rows into the canonical output shape
        ``{"role": "assistant", "content": <thinking><content>}``.
        Implementations MUST preserve this output shape. Synthesis §3.6.
        """
        ...

    @abstractmethod
    async def add_assistant_toolcall(
        self,
        session_id: str,
        tool_name: str,
        args: Dict[str, Any],
        *,
        turn_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        seq_start: Optional[int] = None,
    ) -> None:
        ...

    @abstractmethod
    async def add_tool_result(
        self,
        session_id: str,
        tool_name: str,
        result: Any,
        *,
        turn_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        seq_start: Optional[int] = None,
        status: str = "ok",
        error: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        ...

    @abstractmethod
    async def start_turn(
        self,
        session_id: str,
        turn_id: str,
        *,
        model_name: Optional[str] = None,
    ) -> None:
        """Insert a turns row and mark it running.

        Must be called before any add_* calls that pass turn_id.
        Synthesis §3.6 + b1-persistence.md v2 table design.
        """
        ...

    @abstractmethod
    async def complete_turn(
        self,
        turn_id: str,
        *,
        status: str = "completed",
        stop_reason: Optional[str] = None,
        error_json: Optional[str] = None,
    ) -> None:
        """Update the turns row with completion status and timestamp.

        status must be one of: completed, failed, cancelled.
        Synthesis §3.6 + b1-persistence.md v2 table design.
        """
        ...

    @abstractmethod
    async def record_raw_event(
        self,
        session_id: str,
        turn_id: str,
        seq: int,
        event_type: str,
        payload: Dict[str, Any],
        *,
        tool_call_id: Optional[str] = None,
    ) -> None:
        """Persist a single raw_events row for the replay/debug timeline.

        Duplicate (turn_id, seq) is silently skipped — UNIQUE constraint
        violation is logged at WARNING and swallowed; the event log can
        tolerate sparse gaps. Synthesis §3.6.
        """
        ...

    @abstractmethod
    async def get_history(
        self, session_id: str, include_thinking: bool = False
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    async def ensure_system_prompt(self, session_id: str, prompt: str) -> None:
        ...

    # ------------------------------------------------------------------
    # Audit log (default no-op; SqliteSessionStore overrides).
    # Phase 7 step 74 (synthesis §3.6 + B5 step 7): every tool dispatch
    # writes ONE row to tool_audit. MemoryStore inherits the no-op so
    # in-memory state stays simple. SqliteSessionStore overrides with a
    # real INSERT.
    # ------------------------------------------------------------------

    async def audit_tool_call(
        self,
        *,
        correlation_id: str,
        session_id: str,
        turn_id: str,
        tool_call_id: Optional[str],
        tool_name: str,
        args_sha256: str,
        args_json: Optional[str],
        status: str,
        error_kind: Optional[str],
        duration_ms: Optional[int],
    ) -> None:
        """Append-only audit log entry for a tool call. Default no-op."""
        return None

    # ------------------------------------------------------------------
    # Lifecycle (default no-op; SqliteSessionStore overrides).
    # Phase 6 step 63 (synthesis §3.6): aiosqlite needs an explicit
    # async open/close pair. Keeping the contract on the ABC means
    # ``Engine`` can call ``await store.connect()`` unconditionally
    # against any concrete SessionStore — MemoryStore inherits the
    # no-op, SqliteSessionStore opens/closes its aiosqlite connection.
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the underlying connection (if any). No-op for in-memory."""
        return None

    async def aclose(self) -> None:
        """Close the underlying connection (if any). No-op for in-memory."""
        return None


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def invoke(
        self,
        args: Dict[str, Any],
        *,
        context: Optional["ToolExecutionContext"] = None,
    ) -> Any:
        """Invoke the tool with a dict of arguments and an optional execution
        context.

        The registry-facing API: what the orchestrator and ToolRunner call.
        The author-facing API is :meth:`BaseTool.run` (``**kwargs``);
        ``BaseTool.invoke`` is the shim that unpacks the dict and
        dispatches ``context`` only to ``run`` signatures that opt in.

        ``context`` is keyword-only and defaults to ``None`` so existing
        tools (TimeTool, WeatherTool, …) keep working unchanged. Connector
        tools shipping in Phase 4.5+ consume
        ``context.user_confirmed_send`` for the draft+confirm send-safety
        pattern (connector spec §4 footer).

        Synthesis §6 row 4 / A2 step 1 + §4 Phase 4 step 41a.
        """
        ...


class Orchestrator(ABC):
    """Drives one turn of model → parser → tool-execution.

    Two impls today:
      - ChattyAgentOrchestrator (chatty.py): the standard tool-loop
        agent that processes a user prompt through model + tools and
        yields typed WireEvent objects.
      - NotebookOrchestrator (notebook.py): research-mode strategy,
        currently a stub. Tracked in docs/research/06_context_strategies.md.

    Briefing §2 Seam B (1-4): the ABC was deferred until both impls
    were concrete enough to justify the abstraction. Anti-overengineering
    rule (R6) satisfied: not a single-impl ABC.

    All implementations yield AsyncIterator[WireEvent]. The bytes
    transport (NDJSON / SSE) lives in protocol/wire/transport_*.py.
    """

    # Set to False in stub/unimplemented subclasses so the HTTP router
    # can return 501 before starting to stream. Briefing §2 Seam B item 4;
    # synthesis §3.5 (Orchestrator strategy seam).
    is_implemented: bool = True

    @abstractmethod
    async def run(
        self,
        *,
        session_id: str,
        prompt: str,
        model_name: str,
        cancel_token: Optional["CancelToken"] = None,
    ) -> AsyncIterator["WireEvent"]:
        """Run one turn. Yields typed WireEvent objects.

        Implementations MUST:
          - emit MessageStart first
          - emit MessageStop last (exactly one)
          - honor cancel_token at chunk boundaries (granularity is
            implementation choice)
        """
        ...
        # Unreachable, but makes abstract async generators type-check
        # correctly for callers that do ``async for e in orch.run(...)``.
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]
