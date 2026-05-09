import asyncio
import datetime

from tether_service.core.interfaces import ModelProvider
from tether_service.core.types import Event


from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio

class DummyProvider(ModelProvider):
    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Simulate streaming text chunks based on last user message"""
        prompt = messages[-1].get('content', '') if messages else ''
        for i in range(3):
            await asyncio.sleep(0.1)
            yield f"{prompt}-{i}"

    def list_models(self) -> List[str]:
        """Return a fixed list of dummy models."""
        return ["dummy-model-1", "dummy-model-2"]

    def unload_model(self, model_name: str) -> bool:
        """Simulate unloading a model."""
        print(f"Unloaded dummy model: {model_name}")
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
        from tether_service.providers.types import ProviderCapabilities
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
        from tether_service.providers.types import ProviderText

        async for chunk in self.stream(
            model_name=model_name, messages=messages, tools=tools
        ):
            if isinstance(chunk, str):
                yield ProviderText(text=chunk)
            else:
                yield ProviderText(text=str(chunk))
