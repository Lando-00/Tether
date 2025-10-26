from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any, Optional


class ModelProvider(ABC):
    @abstractmethod
    def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str | List[Dict[str, Any]], None]:
        """Stream raw text chunks for a given model, history, and tools"""
        ...

    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models."""
        ...

    @abstractmethod
    def unload_model(self, model_name: str) -> bool:
        """Unload a model."""
        ...


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
    async def add_assistant_text(self, session_id: str, text: str) -> None:
        ...

    @abstractmethod
    async def add_assistant_toolcall(self, session_id: str, tool_name: str, args: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    async def add_tool_result(self, session_id: str, tool_name: str, result: Any) -> None:
        ...

    @abstractmethod
    async def get_history(self, session_id: str) -> List[Dict[str, Any]]:
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
    async def run(self, args: Dict[str, Any]) -> Any:
        ...
