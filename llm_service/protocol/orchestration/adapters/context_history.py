import json
from typing import Optional, Dict, Any, Callable

from llm_service.protocol.core.interfaces import HistoryWriter, Logger
from llm_service.protocol.core.loggers import NoOpLogger


class ContextHistoryWriter(HistoryWriter):
    """
    Wraps your self.context & system prompt utility.
    """
    def __init__(self, ctx, dynamic_system_prompt_fn, logger: Optional[Logger] = None):
        self.ctx = ctx
        self._dyn = dynamic_system_prompt_fn
        self._logger = logger or NoOpLogger()

    def add_user(self, session_id: str, content: str) -> None:
        try:
            self.ctx.add_message(session_id, "user", content)
            self._logger.debug("Added user message to session %s: %d chars", 
                             session_id, len(content) if content else 0)
        except Exception as e:
            self._logger.exception("Error adding user message: %s", str(e))

    def add_assistant_text(self, session_id: str, content: str) -> None:
        try:
            self.ctx.add_message(session_id, "assistant", content)
            self._logger.debug("Added assistant message to session %s: %d chars", 
                             session_id, len(content) if content else 0)
        except Exception as e:
            self._logger.exception("Error adding assistant message: %s", str(e))

    def add_assistant_toolcall(self, history: list, tool_id: str, pub_name: str, args: Dict[str, Any]) -> None:
        try:
            history.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_id,
                    "type": "function",
                    "function": {"name": pub_name, "arguments": json.dumps(args)},
                }],
            })
            self._logger.debug("Added tool call to history: %s with args %s", pub_name, args)
        except Exception as e:
            self._logger.exception("Error adding tool call to history: %s", str(e))
            # Fallback to simpler format to avoid breaking the conversation
            history.append({
                "role": "assistant",
                "content": f"[Tool call: {pub_name}]",
            })

    def add_tool_result(self, tool_id: str, result: Any, history: list) -> None:
        try:
            # Convert result to string and truncate if too long
            result_str = str(result)
            if len(result_str) > 10000:  # Prevent excessively large results
                result_str = result_str[:9997] + "..."
                self._logger.warning("Tool result truncated from %d to 10000 chars", len(str(result)))
                
            history.append({
                "role": "tool", 
                "tool_call_id": tool_id,
                "content": result_str
            })
            self._logger.debug("Added tool result to history: result length: %d", len(result_str))
        except Exception as e:
            self._logger.exception("Error adding tool result: %s", str(e))
            # Fallback to simpler format
            history.append({
                "role": "system",
                "content": f"[Tool result error: {str(e)}]",
            })

    def get_history(self, session_id: str) -> list:
        try:
            history = self.ctx.get_conversation_history(session_id, format="chat")
            self._logger.debug("Retrieved history for session %s: %d messages", 
                             session_id, len(history))
            return history
        except Exception as e:
            self._logger.exception("Error getting conversation history: %s", str(e))
            # Return minimal history to avoid breaking the app
            return [{"role": "system", "content": "Error retrieving conversation history."}]

    def ensure_system_prompt(self, history: list, tools: list[dict]) -> None:
        try:
            if tools and not any(m.get("role") == "system" for m in history):
                system_message = self._dyn(tools)
                history.insert(0, system_message)
                self._logger.debug("Added system prompt with %d tools", len(tools))
        except Exception as e:
            self._logger.exception("Error adding system prompt: %s", str(e))
            # Add a minimal system prompt
            if not any(m.get("role") == "system" for m in history):
                history.insert(0, {
                    "role": "system",
                    "content": "You are a helpful assistant."
                })