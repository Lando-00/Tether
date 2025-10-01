import json
from typing import Any, Dict, List, Optional, Union, Generator, AsyncGenerator

from llm_service.protocol.core.interfaces import EventEmitter, Logger
from llm_service.protocol.core.loggers import NoOpLogger


class NdjsonEventEmitter(EventEmitter):
    """
    Emits events in NDJSON (Newline-Delimited JSON) format.
    Used for streaming API responses.
    """
    def __init__(self, logger: Optional[Logger] = None):
        self._logger = logger or NoOpLogger()

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> bytes:
        """
        Formats and emits a single event as NDJSON.
        
        Args:
            event_type: The type of event (e.g., "thinking", "tool_call", "token")
            data: The event payload
            
        Returns:
            A bytes containing the NDJSON event
        """
        event = {
            "type": event_type,
            **data
        }
        
        try:
            return (json.dumps(event) + "\n").encode("utf-8")
        except Exception as e:
            self._logger.error(f"Error serializing event: {e}")
            # Fallback for serialization errors
            return (json.dumps({
                "type": "error",
                "error": f"Failed to serialize {event_type} event: {str(e)}"
            }) + "\n").encode("utf-8")
    
    def token(self, text: str) -> bytes:
        """
        Emits a token event.
        """ as bytes
        """
        return self._emit_event("token", {
            "content": text
        })
    
    def hidden_thought(self, text: str, phase: str) -> bytes:
        """
        Emits a hidden thought event.
        
        Args:
            text: The hidden thought content
            phase: The phase of the hidden thought (e.g., "pre_tool", "post_tool")
            
        Returns:
            Formatted NDJSON event as bytes
        """
        return self._emit_event("hidden_thought", {
            "content": text,
            "phase": phase
        })
    
    def tool_start(self, tc_id: str, published_name: str) -> bytes:
        """
        Emits a tool start event.
        
        Args:
            tc_id: The tool call ID
            published_name: The published name of the tool
            
        Returns:
            Formatted NDJSON event as bytes
        """
        return self._emit_event("tool_start", {
            "id": tc_id,
            "name": published_name
        })
    
    def tool_end(self, tc_id: str, published_name: str, result: Any) -> bytes:
        """
        Emits a tool end event.
        
        Args:
            tc_id: The tool call ID
            published_name: The published name of the tool
            result: The result of the tool call
            
        Returns:
            Formatted NDJSON event as bytes
        """
        return self._emit_event("tool_end", {
            "id": tc_id,
            "name": published_name,
            "result": result
        })
    
    def done(self) -> bytes:
        """
        Emits a done event.
        
        Returns:
            Formatted NDJSON event as bytes
        """
        return self._emit_event("done", {})
            Formatted NDJSON event
        """
        return self.emit_event("tool_call", {
            "name": name,
            "args": args
        })
    
    def emit_tool_result(self, name: str, result: Any, success: bool = True) -> str:
        """
        Emits a tool result event.
        
        Args:
            name: Name of the tool that was called
            result: The result returned by the tool
            success: Whether the tool execution was successful
            
        Returns:
            Formatted NDJSON event
        """
        return self.emit_event("tool_result", {
            "name": name,
            "result": result,
            "success": success
        })
    
    def emit_thinking(self, content: str) -> str:
        """
        Emits a thinking event.
        
        Args:
            content: The thinking content
            
        Returns:
            Formatted NDJSON event
        """
        return self.emit_event("thinking", {
            "content": content
        })
    
    def emit_token(self, token: str) -> str:
        """
        Emits a token event.
        
        Args:
            token: A text token from the model
            
        Returns:
            Formatted NDJSON event
        """
        return self.emit_event("token", {
            "content": token
        })
    
    def emit_done(self) -> str:
        """
        Emits a done event.
        
        Returns:
            Formatted NDJSON event
        """
        return self.emit_event("done", {})
    
    def emit_error(self, message: str, code: Optional[str] = None) -> str:
        """
        Emits an error event.
        
        Args:
            message: The error message
            code: Optional error code
            
        Returns:
            Formatted NDJSON event
        """
        error_data = {"message": message}
        if code:
            error_data["code"] = code
            
        return self.emit_event("error", error_data)
