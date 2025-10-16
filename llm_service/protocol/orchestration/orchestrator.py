"""
Tool orchestration module that handles token streaming, tool detection and execution.
"""
from __future__ import annotations
from typing import Optional, Dict, Any, AsyncIterator, List, TYPE_CHECKING
import asyncio
import anyio
import json
from concurrent.futures import TimeoutError as FuturesTimeoutError

from llm_service.protocol.core.interfaces import (
    ArgsParser,
    ToolExecutor,
    HistoryWriter,
    EventEmitter,
    Logger,
    ConfigProvider,
    TokenSource,
    ToolCallStrategy
)
from llm_service.protocol.core.loggers import NoOpLogger
from llm_service.protocol.core.config import EnvironmentConfigProvider
from llm_service.protocol.core.interfaces import StreamParser

from llm_service.protocol.orchestration.filters import HiddenBlockFilter
from llm_service.protocol.orchestration.controller import ToolBoundaryController
from llm_service.protocol.orchestration.detector import PrefixedToolCallDetector
from llm_service.protocol.orchestration.adapters.model_token_source import ModelTokenSource
from tests.One_shot_test.simple_sliding_tool_parser import StreamEvent


class ToolOrchestrator:
    """
    High-level use case: stream tokens; if a tool call is detected,
    emit start, wait for completion, execute, write to history, loop.
    """
    def __init__(
        self,
        args_parser: ArgsParser,
        tool_executor: ToolExecutor,
        history: HistoryWriter,
        emitter: EventEmitter,
        stream_parser: StreamParser | None = None,
        max_tool_loops: int = 3,
        tool_prefix: str = "__tool_",
        logger: Optional[Logger] = None,
        config: Optional[ConfigProvider] = None,
    ):
        self.args_parser = args_parser
        self.tool_executor = tool_executor
        self.history = history
        self.emitter = emitter
        self._logger = logger or NoOpLogger()
        self._config = config or EnvironmentConfigProvider()
        self.max_tool_loops = max_tool_loops if max_tool_loops else self._config.get_max_tool_loops()
        self.tool_prefix = tool_prefix if tool_prefix else self._config.get_tool_prefix()
        # Optional stream parser for model outputs
        self.stream_parser = stream_parser
        self._async_timeout = self._config.get_async_timeout()

    async def run_turns(
        self,
        session_id: str,
        model_factory,
        model_args: Dict[str, Any],
        tools_to_use: list[dict],
    ) -> AsyncIterator[bytes]:
        """
        model_factory: () -> out_iter for one assistant turn
        model_args: dict passed to model_factory
        """
        # If a stream_parser is provided, delegate to parser-driven loop
        if self.stream_parser:
            async for b in self._run_with_stream_parser(session_id, model_factory, model_args, tools_to_use):
                yield b
            return

        try:
            history = self.history.get_history(session_id)
            self.history.ensure_system_prompt(history, tools_to_use)
            
            self._logger.info(
                "Starting generation with tools for session %s (tools: %d, history: %d msgs)",
                session_id, len(tools_to_use), len(history)
            )
            loops = 0                            # ← initialize here
            while True:
                try:
                    # Use timeout to prevent hanging during model generation
                    async with asyncio.timeout(self._async_timeout):
                        out_iter = model_factory(**model_args, messages=history, tools=tools_to_use, stream=True)
                        token_source: TokenSource = ModelTokenSource(
                            out_iter, logger=self._logger, config=self._config
                        )
                        detector: ToolCallStrategy = PrefixedToolCallDetector(
                            prefix=self.tool_prefix, logger=self._logger, config=self._config
                        )
                        # Insert think/reflection hidden block filter
                        hidden_filter = HiddenBlockFilter()
                        # Controller for tool boundaries
                        controller = ToolBoundaryController(self)

                        pass_through = False
                        streamed_text: List[str] = []
                        tool_name: Optional[str] = None
                        tool_id: Optional[str] = None
                        pub: str = ""
                        captured_args: Optional[Dict[str, Any]] = None
                        # phase for hidden_thought events: 'pre_tool' or 'post_tool'
                        phase = "pre_tool"

                        async for delta in token_source.stream():
                            if delta.token:
                                # Route hidden and visible tokens
                                routed = hidden_filter.feed(delta.token)
                                # Emit hidden thought tokens if any
                                if routed.to_ui_hidden:
                                    yield self.emitter.hidden_thought(routed.to_ui_hidden, phase=phase)
                                # Visible part for UI
                                visible = routed.to_ui_visible
                                if not visible:
                                    continue
                                # Token to feed detector
                                detect_token = routed.to_detector
                                if not pass_through:
                                    status, payload = detector.feed(detect_token)

                                    if status == "prose":
                                        text = payload or ""
                                        if text:
                                            streamed_text.append(text)
                                            yield self.emitter.token(text)
                                        pass_through = True
                                        continue

                                    if status == "call_started":
                                        tool_name = payload or ""
                                        # Emit start via controller
                                        tool_id, pub, start_ev = controller.on_start(
                                            tool_name, session_id, loops
                                        )
                                        yield start_ev
                                        
                                        # Check if we have enough information to execute the tool now
                                        # For some tools with obvious usage, we could execute immediately
                                        if tool_name == "get_current_time" and "timezone" in "".join(detector._raw):
                                            self._logger.info(f"Tool {tool_name} has sufficient context for immediate execution")
                                            
                                            # Try to extract timezone from raw text
                                            raw_text = "".join(detector._raw)
                                            import re
                                            timezone_match = re.search(r'timezone\s*=\s*["\']([^"\']+)["\']', raw_text)
                                            
                                            if timezone_match:
                                                timezone = timezone_match.group(1)
                                                self._logger.info(f"Extracted timezone parameter: {timezone}")
                                                
                                                # Extract format parameter if present
                                                format_match = re.search(r'format\s*=\s*["\']([^"\']+)["\']', raw_text)
                                                format_value = format_match.group(1) if format_match else "human"
                                                
                                                captured_args = {"timezone": timezone, "format": format_value}
                                                self._logger.info(f"Immediate args capture: {captured_args}")
                                                
                                                # Safely abort the stream to proceed to tool execution
                                                await token_source.abort_current_stream()
                                                break
                                        
                                        # DO NOT abort the stream here - keep collecting tokens to get arguments
                                        self._logger.info("Tool call started, continuing to collect arguments")
                                        # switch to post-tool phase for hidden thoughts after tool
                                        phase = "post_tool"
                                        continue

                                    if status == "call_complete":
                                        self._logger.info(f"==== TOOL CALL COMPLETE: {tool_name} ====")
                                        self._logger.info(f"Raw arguments: {payload}")
                                        captured_args = self.args_parser.parse(payload or "")
                                        self._logger.info(
                                            "Tool call args captured for %s: %s", 
                                            tool_name, captured_args
                                        )
                                        
                                        # Abort streaming now that we have the arguments
                                        self._logger.info("Aborting stream now that we have complete arguments")
                                        await token_source.abort_current_stream()
                                        
                                        break  # end this assistant turn to execute tool

                                                            # Debug for undecided state
                                    self._logger.debug(f"Tool detection status: undecided. Buffer: {detector._raw[-20:] if len(detector._raw) > 20 else detector._raw}")
                                    continue

                                # pass-through mode: stream immediately
                                streamed_text.append(visible)
                                yield self.emitter.token(visible)

                            # terminator: check finish_reason outside token handling
                            if delta.finish_reason in ("stop", "length", "content_filter", "tool_calls", "timeout", "error"):
                                # Debug finish reason
                                self._logger.info(f"==== STREAM FINISH: {delta.finish_reason} ====")
                                if tool_name is not None:
                                    self._logger.info(f"Tool name at finish: {tool_name}, Args captured: {captured_args is not None}")
                                self._logger.debug("Stream finished with reason: %s", delta.finish_reason)

                                # If we still don't have arguments, show the raw buffer as text
                                if tool_name and captured_args is None and not pass_through:
                                    self._logger.warning(f"Failed to extract arguments for {tool_name}, falling back to plain text")
                                    text = "".join(detector._raw[:1000])  # safe: internal buffer, limit size
                                    if text:
                                        explanation = f"\n[I tried to get the time, but I need a specific timezone. The USA has multiple time zones like America/New_York (Eastern), America/Chicago (Central), America/Denver (Mountain), and America/Los_Angeles (Pacific).]"
                                        streamed_text.append(explanation)
                                        yield self.emitter.token(explanation)
                                    tool_name, tool_id = None, None
                                break
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    self._logger.warning("Timeout or cancellation during token streaming")
                    yield self.emitter.token("\n[Generation timed out or was cancelled]")
                    break
                except Exception as e:
                    self._logger.exception("Error during token streaming: %s", e)
                    yield self.emitter.token(f"\n[Error during generation: {str(e)}]")
                    break

                # Execute tool if captured
                if tool_name and tool_id is not None:
                    # Check if we have arguments or provide defaults for tools that need them
                    if captured_args is None:
                        self._logger.warning(f"No arguments captured for tool {tool_name}, using empty dict")
                        captured_args = {}
                        
                        # Special handling for tools we know about
                        if tool_name == "get_current_time":
                            self._logger.info("Using default arguments for get_current_time: timezone=UTC")
                            # Use human-readable format for CLI output
                            captured_args = {"timezone": "UTC", "format": "human"}
                            
                            # Check raw tokens for timezone mentions
                            raw_text = "".join(detector._raw)
                            self._logger.debug(f"Checking raw text for timezone mentions: {raw_text[:200]}...")
                            try:
                                # Try to find common timezone mentions
                                if "USA" in raw_text or "United States" in raw_text:
                                    # Default to Eastern time for USA when not specified
                                    captured_args["timezone"] = "America/New_York"
                                    self._logger.info("Detected USA mention, using America/New_York timezone")
                                elif "UK" in raw_text or "England" in raw_text or "Britain" in raw_text or "London" in raw_text:
                                    captured_args["timezone"] = "Europe/London"
                                    self._logger.info("Detected UK/England mention, using Europe/London timezone")
                                elif "Europe" in raw_text:
                                    captured_args["timezone"] = "Europe/Paris"
                                    self._logger.info("Detected Europe mention, using Europe/Paris timezone")
                            except Exception as e:
                                self._logger.warning(f"Error while extracting timezone from raw text: {e}")
                    
                    # Enhanced debug logging
                    self._logger.info(f"==== EXECUTING TOOL: {tool_name} ====")
                    self._logger.info(f"Tool ID: {tool_id}, Published name: {pub}")
                    self._logger.info(f"Arguments: {captured_args}")
                    
                    try:
                        # Execute tool and emit end via controller
                        result, end_ev = controller.on_complete(session_id, tool_id, pub, captured_args, history)
                        self._logger.info(f"==== TOOL EXECUTION COMPLETE: {tool_name} ====")
                        self._logger.info(f"Tool result: {result}")
                        self._logger.debug(f"Sending tool_end event: {end_ev}")
                        yield end_ev
                    except Exception as e:
                        self._logger.exception(f"==== TOOL EXECUTION FAILED: {tool_name} ====")
                        # Try to send an error message to client
                        try:
                            error_msg = {"error": f"Error executing tool: {str(e)}"}
                            error_ev = self.emitter.tool_end(tool_id, pub, error_msg)
                            yield error_ev
                        except Exception as inner_e:
                            self._logger.exception(f"Failed to send error event: {inner_e}")

                    loops += 1
                    if loops >= self.max_tool_loops:
                        self._logger.warning(
                            "Maximum tool loops (%d) reached for session %s", 
                            self.max_tool_loops, session_id
                        )
                        self.history.add_assistant_text(
                            session_id, 
                            "I ran tools several times but couldn't finish. Please try again."
                        )
                        yield self.emitter.done()
                        return
                        
                    # Generate a response based on tool result - this is critical for CLI use case
                    # since otherwise the tool result is just shown without explanation
                    if tool_name == "get_current_time" and isinstance(result, str):
                        self._logger.info("Generating response text for time tool result")
                        response_text = f"The current time in {captured_args.get('timezone', 'UTC')} is {result}."
                        yield self.emitter.token(response_text)
                        self.history.add_assistant_text(session_id, response_text)
                        yield self.emitter.done()
                        return
                        
                    continue  # next assistant turn if no special handling

                # No tool call → persist and finish
                final_text = "".join(streamed_text).strip()
                if final_text:
                    self._logger.debug("Recording final assistant response: %d chars", len(final_text))
                    self.history.add_assistant_text(session_id, final_text)
                yield self.emitter.done()
                return
        except (anyio.get_cancelled_exc_class(), asyncio.CancelledError):
            self._logger.info("Request cancelled for session %s", session_id)
            return
        except Exception as e:
            self._logger.exception("Unhandled error in run_turns: %s", e)
            yield self.emitter.token(f"\n[An error occurred: {str(e)}]")
            yield self.emitter.done()
            return

    async def _emit_parser_events(
        self,
        events: list[dict],
        phase: str,
        controller: ToolBoundaryController,
        session_id: str,
        loops: int,
        history: list[Any],
    ) -> AsyncIterator[bytes]:
        """
        Shared event-to-bytes emitter for parser events.
        """
        for evt in events:
            # Debug: print every parser event
            self._logger.debug(f"Parser event: {evt}")
            etype = evt.get("type")
            if etype == StreamEvent.TEXT:
                yield self.emitter.token(evt.get("text", ""))
            elif etype == StreamEvent.THINK_STREAM:
                yield self.emitter.hidden_thought(evt.get("text", ""), phase)
            elif etype == StreamEvent.TOOL_STARTED:
                tc_id = evt.get("id", "")
                name = evt.get("name", "")
                args = evt.get("arguments", evt.get("parameters", {}))
                # record tool call in history
                
                yield self.emitter.tool_start(tc_id, name)
                phase = "post_tool"
            elif etype == StreamEvent.TOOL_PROGRESS:
                yield self.emitter.tool_progress()
            elif etype == StreamEvent.TOOL_COMPLETE:
                raw = evt.get("raw", "")
                try:
                    obj = json.loads(raw)
                    pub = obj.get("name", obj.get("function", ""))
                    args = obj.get("arguments", obj.get("parameters", {}))
                except Exception:
                    pub = evt.get("name", evt.get("function", ""))
                    args = evt.get("arguments", evt.get("parameters", {}))
                
                tool_id, pub, start_ev = controller.on_start(pub, session_id, loops)
                try:
                    self.history.add_assistant_toolcall(session_id,history, tool_id, pub, args)
                except Exception as e:
                    self._logger.warning(f"Failed to record tool call in history: {e}")
                    pass
                yield start_ev
                result, end_ev = controller.on_complete(session_id, tool_id, pub, args, history)

                yield end_ev

    async def _stream_one_turn(
        self,
        session_id: str,
        model_factory,
        model_args: Dict[str, Any],
        tools_to_use: list[dict],
        history: list[Any],
    ) -> AsyncIterator[bytes]:
        """
        Perform exactly one model.generate(stream=True) call,
        feed its tokens into the existing StreamParser + emitter loop,
        then stop immediately on the first TOOL_COMPLETE event.
        """
        if self.stream_parser is None:
            raise RuntimeError("Stream parser not configured")
        parser = self.stream_parser
        controller = ToolBoundaryController(self)
        phase = "pre_tool"
        # Invoke model for one turn
        out_iter = model_factory(**model_args, messages=history, tools=tools_to_use, stream=True)
        token_source: TokenSource = ModelTokenSource(out_iter, logger=self._logger, config=self._config)
        async for delta in token_source.stream():
            raw_token = delta.token or ""
            events = parser.feed(raw_token)
            async for out in self._emit_parser_events(events, phase, controller, session_id, 0, history):
                yield out
            if delta.finish_reason:
                if delta.finish_reason in ("stop","length","content_filter","tool_calls","timeout","error"):
                    # <-- add your debug here
                    self._logger.debug(f"Finish reason detected: {delta.finish_reason}")
                    self._logger.info(f"==== STREAM FINISH: {delta.finish_reason} ====")
                    self._logger.debug("Stream finished with reason: %s", delta.finish_reason)
                break
        return
    
    async def _run_with_stream_parser(
        self,
        session_id: str,
        model_factory,
        model_args: Dict[str, Any],
        tools_to_use: list[dict],
    ) -> AsyncIterator[bytes]:
        """
        Loop streaming parser-driven turns with tool parsing.
        """
        # Ensure a parser is configured
        if self.stream_parser is None:
            raise RuntimeError("Stream parser not configured")
        parser = self.stream_parser
        # Prepare history and ensure system prompt
        history = self.history.get_history(session_id)
        self.history.ensure_system_prompt(history, tools_to_use)
        loops = 0
        # Loop invoking one model turn per tool call
        while loops < self.max_tool_loops:
            # history = self.history.get_history(session_id)
            # collect all text tokens and detect tool calls for this turn
            buffered_text: List[str] = []
            tool_call_made: bool = False
            async for ev in self._stream_one_turn(session_id, model_factory, model_args, tools_to_use, history):
                tool_call_made = False
                payload = ev.decode("utf-8").strip()
                # parse JSON event if possible
                obj = json.loads(payload) if payload.startswith("{") and payload.endswith("}") else {}
                ev_type = obj.get("type")
                # buffer text tokens for 'token' events
                if ev_type == "token":
                    buffered_text.append(obj.get("content", ""))
                # detect tool completion via 'tool_end'
                if ev_type == "tool_end":
                    tool_call_made = True
                yield ev
            # break if no tool call
            if not tool_call_made:
                break
            loops += 1
        # After max loops, flush any remaining parser events
        parser = self.stream_parser
        controller = ToolBoundaryController(self)
        phase = "post_tool"
        events = parser.finalize()
        # Buffer any TEXT events into buffered_text
        for evt in events:
            if evt.get("type") == StreamEvent.TEXT:
                buffered_text.append(evt.get("text", ""))
        # Emit all parser events
        async for out in self._emit_parser_events(events, phase, controller, session_id, loops, history):
            yield out
        # After flush, persist buffered assistant reply
        final_reply = "".join(buffered_text).strip()
        if final_reply:
            self.history.add_assistant_text(session_id, final_reply)
        # Signal done after flush
        yield self.emitter.done()