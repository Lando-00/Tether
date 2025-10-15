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

    def add_tool_result(self, session_id: str, history: list, tool_id: str, pub_name: str, result: Any) -> None:
        try:
            # Prepare a JSON-safe result payload
            # 1) Convert non-serializable types safely
            # 2) Truncate ONLY the textual leaf, not the whole JSON (to keep JSON valid)
            def _to_text(v: Any) -> str:
                s = str(v)
                return s if len(s) <= 10000 else (s[:9997] + "...")
            
            result_obj = {
                "name": pub_name,
                "tool_call_id": tool_id,          # helps when multiple calls are in-flight
                "result": _to_text(result),
                "status": "success"
            }
            result_json = json.dumps(result_obj, ensure_ascii=False)
            # in-memory history (STRUCTURED tool turn)
            history.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result_json
            })
            self._logger.debug("Added tool result to history: %d chars", len(result_json))
            self.ctx.add_message(session_id, "tool", result_json)
            
            # # Convert result to string and truncate if too long
            # result_str = str(result)
            # if len(result_str) > 10000:  # Prevent excessively large results
            #     result_str = result_str[:9997] + "..."
            #     self._logger.warning("Tool result truncated from %d to 10000 chars", len(str(result)))
                
            # # in-memory history
            # history.append({
            #     "role": "tool", 
            #     "tool_call_id": tool_id,
            #     "content": result_str
            # })
            # self._logger.debug("Added tool result to history: result length: %d", len(result_str))

            # # persist into SQLite
            # self.ctx.add_message(session_id, "tool", result_str)
        except Exception as e:
            # self._logger.exception("Error adding tool result: %s", str(e))
            # # Fallback to simpler format
            # history.append({
            #     "role": "system",
            #     "content": f"[Tool result error: {str(e)}]",
            # })
            self._logger.exception("Error adding tool result: %s", str(e))
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