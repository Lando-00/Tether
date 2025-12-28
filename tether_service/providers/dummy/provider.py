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
