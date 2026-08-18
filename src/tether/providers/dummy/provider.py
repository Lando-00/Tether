import asyncio
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

import structlog

from tether.core.interfaces import ModelProvider

if TYPE_CHECKING:
    from tether.providers.types import ModelDetails

_log = structlog.get_logger(__name__)

class DummyProvider(ModelProvider):
    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Simulate streaming text chunks based on last user message.

        ``request_id`` accepted for interface parity (Phase 7 step 72)
        but not used — DummyProvider has no internal logger.
        """
        prompt = messages[-1].get('content', '') if messages else ''
        for i in range(3):
            await asyncio.sleep(0.1)
            yield f"{prompt}-{i}"

    def list_models(self) -> List[str]:
        """Return a fixed list of dummy models."""
        return ["dummy-model-1", "dummy-model-2"]

    def list_model_info(self) -> List["ModelDetails"]:
        """Reference implementation of the model-catalog contract.

        Every model is enumerated (not just the default), the list is built
        without any I/O that could raise, and ``dummy-model-1`` is flagged as
        the default so selection UIs have something to pre-select.
        """
        from tether.providers.types import ModelDetails

        return [
            ModelDetails(
                id=name,
                provider_kind=self.kind,
                source="local",
                context_window=self.get_context_window(name),
                supports_thinking=False,
                supports_reasoning_effort=False,
                is_default=(name == "dummy-model-1"),
            )
            for name in self.list_models()
        ]

    def unload_model(self, model_name: str) -> bool:
        """Simulate unloading a model."""
        _log.debug("dummy.engine.unloaded", model_name=model_name)
        return True

    def get_context_window(self, model_name: str) -> int:
        """
        Get the context window size for a dummy model.

        Args:
            model_name: Name of the model (e.g., "dummy-model-1")

        Returns:
            Context window size in tokens (hardcoded to 2048 for testing)
        """
        return 2048

    # ------------------------------------------------------------------
    # Provider v2 contract (Phase 3 step 39).
    # warm_up + aclose use the ABC defaults (no-op) — DummyProvider has
    # no native handles to release. Synthesis §4 Phase 3 step 39.
    # ------------------------------------------------------------------

    @property
    def kind(self) -> str:
        return "dummy"

    @property
    def capabilities(self):
        # Lazy-import to keep ``providers/types`` off the import-time path.
        from tether.providers.types import ProviderCapabilities
        return ProviderCapabilities(
            streaming=True,
            tools_native=False,        # DummyProvider just echoes prompt.
            tools_marker=False,        # Nor markers.
            thinking_channel=False,
            cancel_inflight=False,
            multi_model=True,
            warm_up_required=False,
        )

    async def stream_typed(
        self,
        *,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        cancel_token: Optional[Any] = None,
    ):
        """v2 typed stream — yields :class:`ProviderText` events derived
        from the legacy :meth:`stream` output. DummyProvider never emits
        native tool_calls, so the union is effectively single-variant
        here. Synthesis §4 Phase 3 step 39."""
        from tether.providers.types import ProviderText

        async for chunk in self.stream(
            model_name=model_name, messages=messages, tools=tools
        ):
            if isinstance(chunk, str):
                yield ProviderText(text=chunk)
            else:
                yield ProviderText(text=str(chunk))
