import json
from typing import Any, Dict, List, Optional, Union, Generator, AsyncGenerator

from llm_service.protocol.core.interfaces import EventEmitter, Logger
from llm_service.protocol.core.loggers import NoOpLogger
from llm_service.protocol.orchestration.parsers import StreamEvent


class NdjsonEventEmitter(EventEmitter):
    """
    Emits events in NDJSON (Newline-Delimited JSON) format.
    Used for streaming API responses.
    """
    def __init__(self, logger: Optional[Logger] = None):
        self._logger = logger or NoOpLogger()

    def _emit_event(self, 
                    event_type: str, 
                    data: Dict[str, Any], 
                    se: Optional[StreamEvent] = None) -> bytes:
        """
        Formats and emits a single event as NDJSON.
        
        Args:
            event_type: The type of event (e.g., "thinking", "tool_call", "token")
            data: The event payload as a dictionary
            se: Optional StreamEvent enum value for additional context
            
        Returns:
            A bytes containing the NDJSON event
        """
        event = {
            "type": event_type,
            **data,
            "stream_event": str(se) if se else None
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
        
        Args:
            text: The token text
            
        Returns:
            Formatted NDJSON event as bytes
        """
        return self._emit_event("token", {
            "content": text
        },
        se=StreamEvent.TEXT)
    
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
        },
        se=StreamEvent.THINK_STREAM)
    
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
        },
        se=StreamEvent.TOOL_STARTED)
    
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
        },
        se=StreamEvent.TOOL_COMPLETE)
    
    def done(self) -> bytes:
        """
        Emits a done event.
        
        Returns:
            Formatted NDJSON event as bytes
        """
        return self._emit_event("done", {}, se=StreamEvent.DONE)
    
    def tool_progress(self) -> bytes:
        """
        Emits a tool progress event.
        Returns:
            Formatted NDJSON event as bytes
        """
        return self._emit_event("tool_progress", {}, se=StreamEvent.TOOL_PROGRESS)

    def error(self, message: str, code: Optional[str] = None) -> bytes:
        """
        Emits an error event as NDJSON bytes.

        Args:
            message: Human-readable error message.
            code: Optional machine-readable error code.

        Returns:
            Formatted NDJSON event as bytes.
        """
        payload = {"message": message}
        if code is not None:
            payload["code"] = code
        return self._emit_event("error", payload, se=StreamEvent.ERROR)

    def cancelled(self) -> bytes:
        """
        Emits a cancellation event as NDJSON bytes.

        Returns:
            Formatted NDJSON event as bytes.
        """
        return self._emit_event("cancelled", {}, se=StreamEvent.CANCELLED)
