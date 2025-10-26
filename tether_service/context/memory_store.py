"""Async in-memory session store implementing SessionStore"""
from typing import Dict, Any
from tether_service.core.interfaces import SessionStore

class MemoryStore(SessionStore):
    def __init__(self):
        self.sessions: Dict[str, Any] = {}

    async def add_user(self, session_id: str, text: str) -> None:
        history = self.sessions.setdefault(session_id, [])
        history.append({"role": "user", "content": text})

    async def add_assistant_text(self, session_id: str, text: str) -> None:
        history = self.sessions.setdefault(session_id, [])
        history.append({"role": "assistant", "content": text})

    async def add_assistant_toolcall(self, session_id: str, tool_name: str, args: Dict[str, Any]) -> None:
        history = self.sessions.setdefault(session_id, [])
        history.append({"role": "tool", "tool": tool_name, "args": args})

    async def add_tool_result(self, session_id: str, tool_name: str, result: Any) -> None:
        history = self.sessions.setdefault(session_id, [])
        history.append({"role": "tool_result", "tool": tool_name, "result": result})

    async def get_history(self, session_id: str) -> list:
        return self.sessions.get(session_id, [])

    async def ensure_system_prompt(self, session_id: str, prompt: str) -> None:
        # if first message, insert system prompt
        if session_id not in self.sessions or not self.sessions[session_id]:
            self.sessions.setdefault(session_id, []).insert(0, {"role": "system", "content": prompt})