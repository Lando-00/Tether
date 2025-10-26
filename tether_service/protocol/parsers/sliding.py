import json
from typing import Any, Dict, List

from tether_service.core.interfaces import StreamParser
from tether_service.core.types import StreamEvent
from tether_service.core.logging import logger


class SlidingParser(StreamParser):
    """
    Stateful streaming parser adapted from llm_service.
    - Streams <think>...</think> as THINK events
    - Detects literal '<<function_call>>' across chunk boundaries
    - After marker, captures a balanced JSON payload
    - Keeps a small overlap buffer to recognize split tags/markers
    - Emits TOOL_STARTED when marker is detected, TOOL_COMPLETE when JSON is captured
    """

    MARKER = "<<function_call>>"
    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"
    OVERLAP = max(len(MARKER), len(THINK_OPEN), len(THINK_CLOSE)) - 1

    def __init__(self, max_tool_chars: int = 32768):
        self.mode = "text"  # "text" | "think" | "await_payload" | "json"
        self.buf = ""  # pending data and overlap
        self.max_tool_chars = max_tool_chars
        # state for JSON capture
        self._json_depth = 0
        self._in_str = False
        self._esc = False
        self._tool_started = False  # Track if we've emitted TOOL_STARTED
        self._tick_bytes = 0
        self._tick_threshold = 128

    def feed(self, chunk: str | List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if isinstance(chunk, list):
            # This parser expects raw text, but if it receives pre-parsed events,
            # it should pass them through.
            return chunk

        events: List[Dict[str, Any]] = []
        if not chunk:
            return events

        self.buf += chunk
        logger.debug(f"Parser feed: mode={self.mode}, buf_len={len(self.buf)}, chunk_len={len(chunk)}")

        while True:
            if self.mode == "think":
                idx = self.buf.lower().find(self.THINK_CLOSE)
                if idx != -1:
                    think_text = self.buf[:idx]
                    if think_text:
                        events.append({"type": StreamEvent.THINK, "data": {"delta": think_text}})
                    self.buf = self.buf[idx + len(self.THINK_CLOSE) :]
                    self.mode = "text"
                    continue
                else:
                    if len(self.buf) > self.OVERLAP:
                        emit = self.buf[: -self.OVERLAP]
                        events.append({"type": StreamEvent.THINK, "data": {"delta": emit}})
                        self.buf = self.buf[-self.OVERLAP :]
                    break

            if self.mode == "json":
                start = self.buf.find("{")
                if start == -1:
                    # No '{' found yet, check if buffer is too large
                    if len(self.buf) > self.max_tool_chars:
                        raw = self.buf
                        self.buf = ""
                        self.mode = "text"
                        self._tool_started = False
                        logger.error(f"Parser: tool payload truncated (no JSON start), size={len(raw)}")
                        events.append({"type": StreamEvent.ERROR, "data": {"message": "tool_payload_truncated", "raw": raw[:200]}})
                    break

                # Log any non-whitespace prefix before '{'
                if start > 0:
                    prefix = self.buf[:start]
                    if prefix.strip():
                        logger.warning(f"Parser: non-whitespace before JSON: {repr(prefix[:50])}")
                    self.buf = self.buf[start:]
                    start = 0

                self._json_depth = 0
                self._in_str = False
                self._esc = False
                i = start
                while i < len(self.buf):
                    ch = self.buf[i]
                    if self._in_str:
                        if self._esc:
                            self._esc = False
                        elif ch == "\\":
                            self._esc = True
                        elif ch == '"':
                            self._in_str = False
                    else:
                        if ch == '"':
                            self._in_str = True
                        elif ch == "{":
                            self._json_depth += 1
                        elif ch == "}":
                            self._json_depth -= 1
                            if self._json_depth == 0:
                                raw = self.buf[start : i + 1]
                                self.buf = self.buf[i + 1 :]
                                self.mode = "text"
                                self._tool_started = False
                                
                                logger.info(f"Parser: JSON capture complete, length={len(raw)}")
                                
                                try:
                                    obj = json.loads(raw)
                                    tool_name = obj.get("name")
                                    tool_args = obj.get("arguments", {})
                                    logger.info(f"Parser: tool call parsed: name={tool_name}, args={tool_args}")
                                    evt = {
                                        "type": StreamEvent.TOOL_COMPLETE,
                                        "data": {
                                            "tool_name": tool_name,
                                            "tool_args": tool_args,
                                            "raw": raw,
                                        },
                                    }
                                except Exception as e:
                                    logger.error(f"Parser: JSON parse error: {e}, raw={raw[:100]}")
                                    evt = {"type": StreamEvent.ERROR, "data": {"message": "json_parse_error", "raw": raw, "error": str(e)}}
                                
                                events.append(evt)
                                break  # Exit inner loop, continue outer
                    i += 1
                    self._tick_bytes += 1
                    if self._tick_bytes >= self._tick_threshold:
                        logger.debug(f"Parser: JSON parsing progress, depth={self._json_depth}, pos={i}")
                        self._tick_bytes = 0
                
                if self.mode == "json": # If we're still in JSON mode
                    if len(self.buf) > self.max_tool_chars:
                        raw = self.buf
                        self.buf = ""
                        self.mode = "text"
                        self._tool_started = False
                        logger.error(f"Parser: tool payload truncated, size={len(raw)}")
                        events.append({"type": StreamEvent.ERROR, "data": {"message": "tool_payload_truncated", "raw": raw[:200]}})
                    break # Exit outer loop, wait for more chunks
                else:
                    continue # Continue outer loop

            if self.mode == "await_payload":
                stripped = self.buf.lstrip(" \t\r\n")
                if not stripped:
                    break  # Need more data
                
                logger.debug(f"Parser: await_payload mode, buf starts with: {repr(stripped[:20])}")
                self.buf = stripped
                if self.buf.startswith("{"):
                    logger.info("Parser: detected JSON payload start after marker")
                    self.mode = "json"
                    continue
                else:
                    # Invalid start, revert to text mode and treat as text
                    logger.warning(f"Parser: expected JSON after marker, got: {repr(self.buf[:50])}")
                    self.mode = "text"
                    self._tool_started = False
                    # Fall through to text processing

            # --- Text Mode ---
            idx_open = self.buf.lower().find(self.THINK_OPEN)
            idx_mark = self.buf.find(self.MARKER)

            # Find the first occurring marker
            first_marker_pos = -1
            marker_type = None

            if idx_open != -1:
                first_marker_pos = idx_open
                marker_type = "think"

            if idx_mark != -1 and (first_marker_pos == -1 or idx_mark < first_marker_pos):
                first_marker_pos = idx_mark
                marker_type = "tool"

            if marker_type:
                # Emit text before the marker
                if first_marker_pos > 0:
                    text_before = self.buf[:first_marker_pos]
                    logger.debug(f"Parser: emitting text before {marker_type} marker: {repr(text_before[:50])}")
                    events.append({"type": StreamEvent.TEXT, "data": {"delta": text_before}})
                
                # Transition to the new mode
                if marker_type == "think":
                    self.buf = self.buf[first_marker_pos + len(self.THINK_OPEN) :]
                    self.mode = "think"
                    logger.debug("Parser: entering think mode")
                else: # tool
                    logger.info("Parser: detected <<function_call>> marker, emitting TOOL_STARTED")
                    self.buf = self.buf[first_marker_pos + len(self.MARKER) :]
                    self.mode = "await_payload"
                    # Emit TOOL_STARTED event
                    if not self._tool_started:
                        events.append({"type": StreamEvent.TOOL_STARTED, "data": {}})
                        self._tool_started = True
                continue
            else:
                # No markers found, emit text but keep overlap
                if len(self.buf) > self.OVERLAP:
                    emit = self.buf[: -self.OVERLAP]
                    if emit:
                        logger.debug(f"Parser: emitting text chunk: {repr(emit[:50])}")
                    events.append({"type": StreamEvent.TEXT, "data": {"delta": emit}})
                    self.buf = self.buf[-self.OVERLAP :]
                break

        return events

    def finalize(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if self.mode == "await_payload":
            logger.warning("Parser finalize: tool marker found but no payload received")
            events.append({"type": StreamEvent.ERROR, "data": {"message": "tool_payload_missing"}})
        elif self.mode in ("json", "py") and self.buf.strip():
            logger.warning(f"Parser finalize: incomplete tool payload: {repr(self.buf[:100])}")
            events.append({"type": StreamEvent.ERROR, "data": {"message": "tool_payload_incomplete", "raw": self.buf}})
        elif self.buf:
            if self.mode == "think":
                events.append({"type": StreamEvent.THINK, "data": {"delta": self.buf}})
            else:
                events.append({"type": StreamEvent.TEXT, "data": {"delta": self.buf}})
        
        logger.info("Parser finalize: resetting state")
        self.buf = ""
        self.mode = "text"
        self._json_depth = 0
        self._in_str = False
        self._esc = False
        self._tool_started = False
        
        events.append({"type": StreamEvent.DONE})
        return events
