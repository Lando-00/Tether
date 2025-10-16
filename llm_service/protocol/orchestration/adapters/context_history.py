import json
from typing import Optional, Dict, Any

from llm_service.protocol.core.interfaces import HistoryWriter, Logger
from llm_service.protocol.core.loggers import NoOpLogger


class ContextHistoryWriter(HistoryWriter):
    """
    Handles session context and system prompt management for chat history.
    """
    def __init__(self, ctx, dynamic_system_prompt_fn, logger: Optional[Logger] = None):
        self.ctx = ctx
        self._dyn = dynamic_system_prompt_fn
        self._logger = logger or NoOpLogger()

    def add_user(self, session_id: str, content: str) -> None:
        """Add a user message to the session history."""
        try:
            self.ctx.add_message(session_id, "user", content)
            self._logger.debug(
                "Added user message to session %s: %d chars", session_id, len(content) if content else 0
            )
        except Exception as e:
            self._logger.exception(f"Error adding user message: {e}")

    def add_assistant_text(self, session_id: str, content: str) -> None:
        """Add an assistant message to the session history."""
        try:
            self.ctx.add_message(session_id, "assistant", content)
            self._logger.debug(
                "Added assistant message to session %s: %d chars", session_id, len(content) if content else 0
            )
        except Exception as e:
            self._logger.exception(f"Error adding assistant message: {e}")

    def add_assistant_toolcall(self, session_id: str, history: list, tool_id: str, pub_name: str, args: Dict[str, Any]) -> None:
        """Add a tool call event to the assistant's history."""
        try:
            tool_call_event = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_id,
                    "type": "function",
                    "function": {"name": pub_name, "arguments": json.dumps(args)},
                }],
            }
            history.append(tool_call_event)
            self._logger.debug("Added tool call to history: %s with args %s", pub_name, args)
            # Persist the tool call event to the session DB
            self.ctx.add_message(session_id, "assistant", json.dumps(tool_call_event, ensure_ascii=False))
        except Exception as e:
            self._logger.exception(f"Error adding tool call to history: {e}")
            # Fallback to simpler format to avoid breaking the conversation
            fallback_event = {
                "role": "assistant",
                "content": f"[Tool call: {pub_name}]",
            }
            history.append(fallback_event)
            self.ctx.add_message(session_id, "assistant", json.dumps(fallback_event, ensure_ascii=False))

    def add_tool_result(self, session_id: str, history: list, tool_id: str, pub_name: str, result: Any) -> None:
        """Add a tool result event to the history and persist it."""
        try:
            def _to_text(v: Any) -> str:
                s = str(v)
                return s if len(s) <= 10000 else (s[:9997] + "...")

            result_obj = {
                "name": pub_name,
                "tool_call_id": tool_id,
                "result": _to_text(result),
                "status": "success"
            }
            result_json = json.dumps(result_obj, ensure_ascii=False)
            history.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result_json
            })
            self._logger.debug("Added tool result to history: %d chars", len(result_json))
            self.ctx.add_message(session_id, "tool", result_json)
        except Exception as e:
            self._logger.exception(f"Error adding tool result: {e}")
            # Fallback to structured error so model can react
            fallback_json = json.dumps({
                "name": pub_name,
                "tool_call_id": tool_id,
                "error": str(e),
                "status": "error"
            }, ensure_ascii=False)
            history.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": fallback_json
            })
            self.ctx.add_message(session_id, "tool", fallback_json)

    def get_history(self, session_id: str) -> list:
        """Retrieve the chat history for a session."""
        try:
            history = self.ctx.get_conversation_history(session_id, format="chat")
            self._logger.debug(
                "Retrieved history for session %s: %d messages", session_id, len(history)
            )
            return history
        except Exception as e:
            self._logger.exception(f"Error getting conversation history: {e}")
            return [{"role": "system", "content": "Error retrieving conversation history."}]

    def ensure_system_prompt(self, history: list, tools: list[dict]) -> None:
        """Ensure a system prompt is present in the history if tools are provided."""
        try:
            if tools and not any(m.get("role") == "system" for m in history):
                system_message = self._dyn(tools)
                history.insert(0, system_message)
                self._logger.debug("Added system prompt with %d tools", len(tools))
        except Exception as e:
            self._logger.exception(f"Error adding system prompt: {e}")
            if not any(m.get("role") == "system" for m in history):
                history.insert(0, {
                    "role": "system",
                    "content": "You are a helpful assistant."
                })
