"""Async in-memory session store implementing SessionStore.

get_history() output MUST match SqliteSessionStore.get_history() exactly.
Contract locked by tests/contract/test_session_store_history_contract.py
(Phase 5 → Phase 6 gate, synthesis §11.3 R19).
"""
import json
from typing import Any, Dict, List, Optional

from tether_service.core.interfaces import SessionStore


class MemoryStore(SessionStore):
    def __init__(self):
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Session lifecycle (required by SessionStore ABC)
    # ------------------------------------------------------------------

    async def create_session(self, session_id: str, created_at: int) -> None:
        self.sessions.setdefault(session_id, [])

    async def list_sessions(self) -> List[Dict[str, Any]]:
        return [{"session_id": sid} for sid in self.sessions]

    async def delete_session(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    async def delete_all_sessions(self) -> int:
        count = len(self.sessions)
        self.sessions.clear()
        return count

    # ------------------------------------------------------------------
    # Message writes
    # ------------------------------------------------------------------

    async def add_user(self, session_id: str, text: str) -> None:
        self.sessions.setdefault(session_id, []).append(
            {"role": "user", "content": text}
        )

    async def add_assistant_text(
        self,
        session_id: str,
        text: str,
        thinking_text: Optional[str] = None,
        save_thinking: bool = True,
    ) -> None:
        entry: Dict[str, Any] = {"role": "assistant", "content": text}
        if save_thinking and thinking_text:
            entry["thinking_text"] = thinking_text
        self.sessions.setdefault(session_id, []).append(entry)

    async def add_assistant_toolcall(
        self, session_id: str, tool_name: str, args: Dict[str, Any]
    ) -> None:
        self.sessions.setdefault(session_id, []).append(
            {"role": "tool", "tool": tool_name, "args": args}
        )

    async def add_tool_result(
        self, session_id: str, tool_name: str, result: Any
    ) -> None:
        self.sessions.setdefault(session_id, []).append(
            {"role": "tool_result", "tool": tool_name, "result": result}
        )

    # ------------------------------------------------------------------
    # History reconstruction — canonical model-facing shape.
    #
    # Output MUST be identical to SqliteSessionStore.get_history().
    # Contract: tests/contract/test_session_store_history_contract.py.
    # Synthesis §11.3 R19: Phase 5 → Phase 6 gate.
    # ------------------------------------------------------------------

    async def get_history(
        self, session_id: str, include_thinking: bool = False
    ) -> List[Dict[str, Any]]:
        """Reconstruct the canonical model-facing history.

        Output shape MUST match SqliteSessionStore.get_history() — see
        tests/contract/test_session_store_history_contract.py.
        """
        history: List[Dict[str, Any]] = []
        for message in self.sessions.get(session_id, []):
            role = message.get("role")
            if role == "user":
                history.append({"role": "user", "content": message.get("content", "")})
            elif role == "assistant":
                content = message.get("content", "")
                if include_thinking:
                    thinking = message.get("thinking_text") or ""
                    if thinking:
                        content = f"{thinking}{content}"
                history.append({"role": "assistant", "content": content})
            elif role == "system":
                history.append({"role": "system", "content": message.get("content", "")})
            elif role == "tool":
                tool_name = message.get("tool")
                args = message.get("args") or {}
                tool_call_json = json.dumps({"name": tool_name, "arguments": args})
                history.append({
                    "role": "assistant",
                    "content": f"<<function_call>> {tool_call_json}",
                })
            elif role == "tool_result":
                tool_name = message.get("tool")
                result = message.get("result") if message.get("result") is not None else {}
                result_text = json.dumps(result, indent=2)
                history.append({
                    "role": "user",
                    "content": f"Tool '{tool_name}' returned:\n{result_text}",
                })
        return history

    async def ensure_system_prompt(self, session_id: str, prompt: str) -> None:
        if session_id not in self.sessions or not self.sessions[session_id]:
            self.sessions.setdefault(session_id, []).insert(
                0, {"role": "system", "content": prompt}
            )