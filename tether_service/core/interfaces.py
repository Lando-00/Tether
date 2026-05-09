from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional

from tether_service.providers.types import ProviderCapabilities, ProviderEvent


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
    ) -> AsyncGenerator[str | List[Dict[str, Any]], None]:
        """Stream raw text chunks for a given model, history, and tools.

        Legacy: yields ``str`` for text deltas and ``List[Dict[str, Any]]``
        for native tool_calls (MLC-style). DEPRECATED in favor of
        :meth:`stream_typed` which returns typed :class:`ProviderEvent`
        values; kept for one cycle until Phase 5 step 52 migrates the
        orchestrator. See _synthesis.md §4 Phase 3 step 39, §6 bug #12.
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
    ) -> AsyncIterator[ProviderEvent]:
        """v2 stream method yielding typed :class:`ProviderEvent` values.

        Default implementation raises :class:`NotImplementedError` so
        concrete classes must explicitly opt into the new contract.

        Phase 5 step 52 will migrate the orchestrator to consume this;
        until then the orchestrator uses the legacy :meth:`stream`.

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


class StreamParser(ABC):
    @abstractmethod
    def feed(self, chunk: str | List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ingest a raw model chunk and return zero or more protocol events"""
        ...

    @abstractmethod
    def finalize(self) -> List[Dict[str, Any]]:
        """Flush any residual state and return final protocol events"""
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
    async def add_user(self, session_id: str, text: str) -> None:
        ...

    @abstractmethod
    async def add_assistant_text(
        self,
        session_id: str,
        text: str,
        thinking_text: Optional[str] = None,
        save_thinking: bool = True,
    ) -> None:
        ...

    @abstractmethod
    async def add_assistant_toolcall(self, session_id: str, tool_name: str, args: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    async def add_tool_result(self, session_id: str, tool_name: str, result: Any) -> None:
        ...

    @abstractmethod
    async def get_history(
        self, session_id: str, include_thinking: bool = False
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    async def ensure_system_prompt(self, session_id: str, prompt: str) -> None:
        ...


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
    async def invoke(self, args: Dict[str, Any]) -> Any:
        """Invoke the tool with a dict of arguments.

        The registry-facing API: what the orchestrator and ToolRunner call.
        The author-facing API is BaseTool.run(**kwargs); BaseTool.invoke is
        a shim that unpacks the dict. Synthesis §6 row 4 / A2 step 1.
        """
        ...
