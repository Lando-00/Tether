# simple_sliding_tool_parser.py
from typing import List, Dict, Any, Optional
import json

class StreamEvent:
    TEXT = "text"
    THINK_STREAM = "think_stream"
    TOOL_STARTED = "tool_started"
    TOOL_PROGRESS = "tool_progress"
    TOOL_COMPLETE = "tool_complete"

class SimpleSlidingToolStreamParser:
    """
    Minimal streaming parser:
    - Streams <think>...</think> verbatim (as THINK_STREAM events)
    - Detects literal '<<function_call>>' across chunk boundaries
    - After marker:
        * If next non-space is '{' => capture balanced JSON (string-aware)
        * Else => capture single-line Python call until '\n'
    - Keeps only a small overlap to recognize split tags/markers across chunks
    """

    # Recognize tool invocation markers from model output
    MARKER = "<<function_call>>"
    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"
    OVERLAP = max(len(MARKER), len(THINK_OPEN), len(THINK_CLOSE)) - 1

    def __init__(self, max_tool_chars: int = 32768):
        self.mode = "text"               # "text" | "think" | "await_payload" | "json" | "py"
        self.buf = ""                    # pending data and overlap
        self.max_tool_chars = max_tool_chars
        # state for JSON capture
        self._json_depth = 0
        self._in_str = False
        self._esc = False
        # progress heartbeat state
        self._tool_active = False
        self._tick_bytes = 0
        self._tick_threshold = 128

    @staticmethod
    def _lstrip_ws(s: str) -> tuple[str, int]:
        i = 0
        while i < len(s) and s[i] in " \t\r\n":
            i += 1
        return s[i:], i

    def feed(self, chunk: str) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if not chunk:
            return events

        self.buf += chunk

        while True:
            if self.mode == "think":
                idx = self.buf.lower().find(self.THINK_CLOSE)
                if idx != -1:
                    # emit think content up to start of closing tag
                    think_text = self.buf[:idx]
                    if think_text:
                        events.append({"type": StreamEvent.THINK_STREAM, "text": think_text})
                    # consume closing tag
                    self.buf = self.buf[idx + len(self.THINK_CLOSE):]
                    self.mode = "text"
                    continue
                else:
                    # emit all but overlap for partial think
                    if len(self.buf) > self.OVERLAP:
                        emit = self.buf[:-self.OVERLAP]
                        events.append({"type": StreamEvent.THINK_STREAM, "text": emit})
                        self.buf = self.buf[-self.OVERLAP:]
                    break

            if self.mode == "json":
                start = self.buf.find("{")
                if start == -1:
                    break
                # beginning of new JSON payload: reset depth and flags
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
                        i += 1
                        # heartbeat tick
                        self._tick_bytes += 1
                        if self._tick_bytes >= self._tick_threshold:
                            events.append({"type": StreamEvent.TOOL_PROGRESS})
                            self._tick_bytes = 0
                        continue
                    else:
                        if ch == '"':
                            self._in_str = True
                        elif ch == '{':
                            self._json_depth += 1
                        elif ch == '}':
                            self._json_depth -= 1
                            if self._json_depth == 0:
                                raw = self.buf[start:i+1]
                                self.buf = self.buf[i+1:]
                                self.mode = "text"
                                self._json_depth = 0
                                self._in_str = False
                                self._esc = False
                                obj = None
                                try:
                                    obj = json.loads(raw)
                                except Exception:
                                    pass
                                evt = {"type": StreamEvent.TOOL_COMPLETE, "style": "json", "raw": raw}
                                if obj is not None:
                                    evt["obj"] = obj
                                events.append(evt)
                                # complete payload: stop tool activity
                                self._tool_active = False
                                self._tick_bytes = 0
                                break
                        i += 1
                if self.mode == "json":
                    if len(self.buf) > self.max_tool_chars:
                        raw = self.buf[start:]
                        self.buf = ""
                        self.mode = "text"
                        self._json_depth = 0
                        self._in_str = False
                        self._esc = False
                        events.append({"type": StreamEvent.TOOL_COMPLETE, "style": "json", "raw": raw})
                        # complete payload: stop tool activity
                        self._tool_active = False
                        self._tick_bytes = 0
                    break
                else:
                    continue

            if self.mode == "py":
                # Capture a Python-style tool call with balanced parentheses
                # Trim leading whitespace
                buf = self.buf.lstrip()
                leading = len(self.buf) - len(buf)
                # Find first '(' indicating args start
                paren_idx = buf.find('(')
                if paren_idx != -1:
                    depth = 0
                    i = paren_idx
                    # iterate to find matching closing ')'
                    while i < len(buf):
                        ch = buf[i]
                        if ch == '(':
                            depth += 1
                        elif ch == ')':
                            depth -= 1
                            if depth == 0:
                                # full call captured
                                call_raw = buf[:i+1].strip()
                                # consume from original buffer
                                end = leading + i + 1
                                self.buf = self.buf[end:]
                                self.mode = "text"
                                events.append({"type": StreamEvent.TOOL_COMPLETE, "style": "python", "raw": call_raw})
                                # complete payload: stop tool activity
                                self._tool_active = False
                                self._tick_bytes = 0
                                break
                        i += 1
                    # if not closed yet, wait for more data
                    if self.mode == "py":
                        break
                else:
                    # no parentheses, fallback to newline or max length
                    nl = self.buf.find("\n")
                    if nl == -1:
                        if len(self.buf) > self.max_tool_chars:
                            raw = self.buf.strip()
                            self.buf = ""
                            self.mode = "text"
                            events.append({"type": StreamEvent.TOOL_COMPLETE, "style": "python", "raw": raw})
                            self._tool_active = False
                            self._tick_bytes = 0
                        break
                    raw = self.buf[:nl].strip()
                    self.buf = self.buf[nl+1:]
                    self.mode = "text"
                    events.append({"type": StreamEvent.TOOL_COMPLETE, "style": "python", "raw": raw})
                    self._tool_active = False
                    self._tick_bytes = 0
                    continue

            if self.mode == "await_payload":
                stripped, skipped = self._lstrip_ws(self.buf)
                if skipped > 0:
                    self.buf = stripped
                if not self.buf:
                    break
                # start tool payload if not already
                if not self._tool_active:
                    events.append({"type": StreamEvent.TOOL_STARTED})
                    self._tool_active = True
                    self._tick_bytes = 0
                if self.buf[0] == '{':
                    # enter JSON mode; reset parser state
                    self.mode = "json"
                    self._json_depth = 0
                    self._in_str = False
                    self._esc = False
                else:
                    self.mode = "py"
                continue

            # TEXT MODE
            idx_open = self.buf.lower().find(self.THINK_OPEN)
            idx_mark = self._find_marker(self.buf)
            idx = None
            kind = None
            if idx_open != -1 and (idx_mark == -1 or idx_open < idx_mark):
                idx = idx_open
                kind = "think"
            elif idx_mark != -1:
                idx = idx_mark
                kind = "marker"

            if idx is not None:
                if idx > 0:
                    events.append({"type": StreamEvent.TEXT, "text": self.buf[:idx]})
                if kind == "think":
                    self.buf = self.buf[idx + len(self.THINK_OPEN):]
                    self.mode = "think"
                else:
                    # Enter payload capture after marker
                    self.buf = self.buf[idx + len(self.MARKER):]
                    self.mode = "await_payload"
                continue
            else:
                if len(self.buf) > self.OVERLAP:
                    emit = self.buf[:-self.OVERLAP]
                    events.append({"type": StreamEvent.TEXT, "text": emit})
                    self.buf = self.buf[-self.OVERLAP:]
                break

        return events

    def finalize(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if self.mode == "json" and self.buf:
            start = self.buf.find("{")
            raw = self.buf[start:] if start != -1 else self.buf
            events.append({"type": StreamEvent.TOOL_COMPLETE, "style": "json", "raw": raw})
        elif self.mode == "py" and self.buf.strip():
            events.append({"type": StreamEvent.TOOL_COMPLETE, "style": "python", "raw": self.buf.strip()})
        elif self.buf:
            if self.mode == "think":
                events.append({"type": StreamEvent.THINK_STREAM, "text": self.buf})
            else:
                events.append({"type": StreamEvent.TEXT, "text": self.buf})
        self.buf = ""
        self.mode = "text"
        self._json_depth = 0
        self._in_str = False
        self._esc = False
        return events

    def _find_marker(self, s: str) -> int:
        return s.find(self.MARKER)
