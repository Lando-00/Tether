"""Async in-memory session store implementing SessionStore"""
from typing import Dict, Any, Optional
from tether_service.core.interfaces import SessionStore

class MemoryStore(SessionStore):
    def __init__(self):
        self.sessions: Dict[str, Any] = {}

    async def add_user(self, session_id: str, text: str) -> None:
        history = self.sessions.setdefault(session_id, [])
        history.append({"role": "user", "content": text})

    async def add_assistant_text(
        self,
        session_id: str,
        text: str,
        thinking_text: Optional[str] = None,
        save_thinking: bool = True,
    ) -> None:
        history = self.sessions.setdefault(session_id, [])
        entry = {"role": "assistant", "content": text}
        if save_thinking and thinking_text:
            entry["thinking_text"] = thinking_text
        history.append(entry)

    async def add_assistant_toolcall(self, session_id: str, tool_name: str, args: Dict[str, Any]) -> None:
        history = self.sessions.setdefault(session_id, [])
        history.append({"role": "tool", "tool": tool_name, "args": args})

    async def add_tool_result(self, session_id: str, tool_name: str, result: Any) -> None:
        history = self.sessions.setdefault(session_id, [])
        history.append({"role": "tool_result", "tool": tool_name, "result": result})

    async def get_history(self, session_id: str, include_thinking: bool = False) -> list:
        history = []
        for message in self.sessions.get(session_id, []):
            if message.get("role") == "assistant" and not include_thinking:
                msg = {"role": "assistant", "content": message.get("content", "")}
            elif message.get("role") == "assistant" and include_thinking:
                thinking = message.get("thinking_text")
                content = message.get("content", "")
                if thinking:
                    content = f"{thinking}{content}"
                msg = {"role": "assistant", "content": content}
            else:
                msg = message
            history.append(msg)
        return history

    async def ensure_system_prompt(self, session_id: str, prompt: str) -> None:
        # if first message, insert system prompt
        if session_id not in self.sessions or not self.sessions[session_id]:
            self.sessions.setdefault(session_id, []).insert(0, {"role": "system", "content": prompt})