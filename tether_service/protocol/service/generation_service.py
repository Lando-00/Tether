from datetime import datetime
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List

from tether_service.core.interfaces import (
    ModelProvider,
    SessionStore,
    StreamParser,
    Tool,
)

class GenerationService:
    def __init__(
        self,
        provider: ModelProvider,
        parser: StreamParser,
        session_store: SessionStore,
        tools: Dict[str, Tool],
        system_prompt: str,
    ):
        """Initialize with provider, parser, session store, and tools registry"""
        self.provider = provider
        self.parser = parser
        self.store = session_store
        self.tools = tools
        self.system_prompt = system_prompt

    async def stream(
        self, session_id: str, prompt: str, model_name: str
    ) -> AsyncGenerator[bytes, None]:
        """Drive the core orchestration to stream NDJSON bytes"""
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
        ):
            yield chunk

    # --- Session Management ---

    async def create_session(self, model_name: str | None = None) -> Dict[str, Any]:
        """Creates a new session and returns its details."""
        session_id = str(uuid.uuid4())
        created_at = int(time.time())
        # The model_name is not stored in the session in this architecture
        await self.store.create_session(session_id, created_at)
        
        # Convert timestamp to ISO 8601 string format to ensure consistency
        created_at_iso = datetime.fromtimestamp(created_at).isoformat()
        
        return {"session_id": session_id, "created_at": created_at_iso}

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """Lists all sessions."""
        return await self.store.list_sessions()

    async def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Gets messages for a session."""
        return await self.store.get_history(session_id)

    async def delete_session(self, session_id: str) -> bool:
        """Deletes a session by its ID."""
        return await self.store.delete_session(session_id)

    async def delete_all_sessions(self) -> int:
        """Deletes all sessions."""
        return await self.store.delete_all_sessions()

    # --- Model Management ---

    def list_models(self) -> List[str]:
        """Lists available models from the provider."""
        return self.provider.list_models()

    def unload_model(self, model_name: str) -> bool:
        """Unloads a model via the provider."""
        return self.provider.unload_model(model_name)
