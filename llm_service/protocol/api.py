from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, AsyncIterator, Protocol, Iterable, Tuple, List, runtime_checkable, TypeVar, Callable, Union
import asyncio, json, logging, re, threading, time, inspect, anyio
from fastapi import HTTPException
from starlette.responses import StreamingResponse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import os
import sys
import importlib
import pkgutil
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llm_service import tools
from llm_service.context import ContextComponent
from llm_service.model.mlc_engine import ModelComponent
from llm_service.tools import execute_tool, get_all_tool_definitions

# -----------------------------
# ====== New Protocols ========
# -----------------------------

from llm_service.protocol.core.config import EnvironmentConfigProvider
from llm_service.protocol.core.execution import ThreadPoolExecutionStrategy
from llm_service.protocol.core.loggers import NoOpLogger, StandardLogger

from llm_service.protocol.core.interfaces import (
    Logger,
    ExecutionStrategy,
    ConfigProvider,
    TokenSource,
    ToolCallStrategy,
    ArgsParser,
    ToolExecutor,
    HistoryWriter,
    EventEmitter,
)
from llm_service.protocol.core.types import TokenDelta, RoutedChunk


logger = logging.getLogger(__name__)

class PrefixedToolCallDetector(ToolCallStrategy):
    """
    Detects __tool_<name>(...) via early tokens with balanced parens.
    """
    CALL_PREFIX_RE = re.compile(r'^__tool_[A-Za-z_]\w*\([^)]*$')
    CALL_FULL_RE   = re.compile(r'^__tool_[A-Za-z_]\w*\([^)]*\)\s*$')
    # Matches a complete tool call anywhere in the token sequence
    ANYWHERE_FULL  = re.compile(r'__tool_[A-Za-z_]\w*\([^)]*\)\s*')
    # Matches a partial tool call (no closing paren) anywhere
    ANYWHERE_PREFX = re.compile(r'__tool_[A-Za-z_]\w*\([^)]*$')

    def __init__(self, prefix: str = "__tool_", logger: Optional[Logger] = None, config: Optional[ConfigProvider] = None):
        self.prefix = prefix
        self._raw: List[str] = []
        self._compact: List[str] = []
        self._decided_prose = False
        self._started = False
        self._name: Optional[str] = None
        self._paren_balance = 0
        self._first_paren_ix: Optional[int] = None
        self._logger = logger or NoOpLogger()
        self._config = config or EnvironmentConfigProvider()
        self._max_buffer_size = self._config.get_max_token_buffer_size()

    def feed(self, token: str) -> Tuple[str, Optional[str]]:
        try:
            # Check buffer limit to prevent memory leaks
            if len(self._raw) >= self._max_buffer_size:
                self._logger.warning("Token buffer size exceeded maximum limit of %d", self._max_buffer_size)
                self._decided_prose = True
                return ("prose", "".join(self._raw[:100]) + "... [buffer overflow]")
            
            self._raw.append(token)
            if token and token.strip():
                self._compact.append(token.replace(" ", ""))

            raw = "".join(self._raw)
            compact = "".join(self._compact)

            if self._decided_prose:
                return ("undecided", None)

            if not self._started:
                # Relax early rejection: ignore leading whitespace, reject only if many newlines
                raw_stripped = raw.lstrip()
                newline_count = raw_stripped.count("\n")
                too_many_newlines = newline_count > 4
                too_long = len(raw) > 512 or len(compact) > 256
                # Detect tool call anywhere in prefix for complete or partial matches
                m_full = self.ANYWHERE_FULL.search(compact)
                if m_full:
                    self._started = True
                    self._name = self._extract_name(m_full.group(0))
                    self._logger.debug("Detected complete tool call anywhere: %s", self._name)
                    return ("call_started", self._name)
                m_pre = self.ANYWHERE_PREFX.search(compact)
                if m_pre:
                    self._started = True
                    self._name = self._extract_name(m_pre.group(0))
                    self._first_paren_ix = compact.find("(")
                    self._paren_balance = 1
                    self._logger.debug("Started tool call detection anywhere: %s", self._name)
                    return ("call_started", self._name)
                # Standard prefix/full-match detection
                looks_full = bool(self.CALL_FULL_RE.match(compact))
                looks_prefix = bool(self.CALL_PREFIX_RE.match(compact))
                if looks_full:
                    self._started = True
                    self._name = self._extract_name(compact)
                    self._logger.debug("Detected complete tool call: %s", self._name)
                    return ("call_started", self._name)
                if looks_prefix:
                    self._started = True
                    self._name = self._extract_name(compact)
                    self._first_paren_ix = compact.find("(")
                    self._paren_balance = 1
                    self._logger.debug("Started tool call detection: %s", self._name)
                    return ("call_started", self._name)
                # Immediate rejections: too many newlines or excessive length
                if too_many_newlines or too_long:
                    self._decided_prose = True
                    self._logger.debug("Rejecting as prose (too many newlines: %d or too long)", newline_count)
                    return ("prose", raw)
                
                # Be lenient with partial prefixes - check if what we have so far could
                # be the beginning of the tool prefix
                if len(compact) <= len(self.prefix) and self.prefix.startswith(compact):
                    self._logger.debug("Potential tool prefix start detected: %s", compact)
                    return ("undecided", None)
                
                # If we've collected enough and it doesn't match our prefix pattern
                if not compact.startswith(self.prefix):
                    self._decided_prose = True
                    self._logger.debug("Rejecting as prose: doesn't match tool prefix pattern")
                    return ("prose", raw)

                return ("undecided", None)

            # After start: update paren balance
            for ch in token:
                if ch == "(":
                    self._paren_balance += 1
                elif ch == ")":
                    self._paren_balance -= 1

            if self._paren_balance <= 0:
                compact = "".join(self._compact)
                if self._first_paren_ix is None:
                    self._first_paren_ix = compact.find("(")
                last_close = compact.rfind(")")
                inner = compact[self._first_paren_ix + 1:last_close] if last_close != -1 else ""
                self._logger.debug("Completed tool call: %s with args: %s", self._name, inner)
                return ("call_complete", inner)

            return ("undecided", None)
        except Exception as e:
            self._logger.exception("Error in tool call detection: %s", str(e))
            self._decided_prose = True
            return ("prose", "".join(self._raw[:100]) + f"... [error: {str(e)}]")

    def _extract_name(self, compact: str) -> str:
        open_ix = compact.find("(")
        return compact[len(self.prefix):open_ix] if open_ix != -1 else ""


class JsonArgsParser(ArgsParser):
    def __init__(self, logger: Optional[Logger] = None):
        self._logger = logger or NoOpLogger()
        
    def parse(self, raw: str) -> Dict[str, Any]:
        raw = (raw or "").strip()
        if not raw:
            self._logger.debug("Empty argument string received")
            return {}
            
        # First, try parsing as proper JSON
        try:
            return json.loads(raw)
        except Exception as e:
            self._logger.debug("Failed to parse JSON with standard quotes, trying with single quotes: %s", str(e))
            try:
                return json.loads(raw.replace("'", '"'))
            except Exception:
                pass
                
        # If JSON parsing fails, try to extract key=value pairs from the raw string
        self._logger.debug("Attempting to parse arguments from raw string: %s", raw)
        # Simple key=value extraction for common patterns like: arg1="value1", arg2=42
        arg_dict = {}
        
        try:
            # Remove outer parentheses if they exist
            cleaned = raw.strip('()')
            
            # Split by commas not inside quotes
            in_quotes = False
            quote_char = None
            parts = []
            current = []
            
            for char in cleaned:
                if char in ['"', "'"]:
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                        
                if char == ',' and not in_quotes:
                    parts.append(''.join(current))
                    current = []
                else:
                    current.append(char)
                    
            if current:
                parts.append(''.join(current))
                
            # Process each part to extract key=value
            for part in parts:
                part = part.strip()
                if '=' in part:
                    key, value = part.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    
                    # Try to convert to appropriate type
                    if value.lower() == 'true':
                        arg_dict[key] = True
                    elif value.lower() == 'false':
                        arg_dict[key] = False
                    elif value.isdigit():
                        arg_dict[key] = int(value)
                    else:
                        try:
                            arg_dict[key] = float(value)
                        except ValueError:
                            arg_dict[key] = value
            
            if arg_dict:
                self._logger.debug("Successfully extracted arguments: %s", arg_dict)
                return arg_dict
        except Exception as e:
            self._logger.warning("Error while parsing arguments: %s", str(e))
        
        self._logger.warning("Failed to parse arguments from raw string: %s", raw)
        return {}


class DefaultToolExecutor(ToolExecutor):
    """
    Adapts your existing execute_tool(name, args) and supports mapping "__tool_x" -> "x".
    Uses an ExecutionStrategy to run the tool function with timeout and error handling.
    """
    def __init__(self, executor_fn: Callable, execution_strategy: ExecutionStrategy, 
                 logger: Optional[Logger] = None, config: Optional[ConfigProvider] = None):
        self._exec = executor_fn
        self._execution_strategy = execution_strategy
        self._logger = logger or NoOpLogger()
        self._config = config or EnvironmentConfigProvider()
        self._timeout = self._config.get_tool_execution_timeout()

    def execute(self, published_name: str, args: Dict[str, Any]) -> Any:
        # published: "__tool_get_time" → canonical: "get_time"
        canonical = published_name.removeprefix("__tool_")
        self._logger.debug("Executing tool %s with args %s", canonical, args)
        
        # Handle common argument issues - remove '_raw' if it exists
        if '_raw' in args and len(args) == 1:
            self._logger.warning("Only '_raw' argument found, using empty args instead")
            args = {}
        elif '_raw' in args:
            self._logger.warning("Removing unexpected '_raw' argument")
            args.pop('_raw', None)
            
        try:
            result = self._execution_strategy.execute_with_timeout(
                func=self._exec,
                args=(canonical, args),
                kwargs={},
                timeout=self._timeout
            )
            self._logger.debug("Tool %s execution completed with result type: %s", 
                              canonical, type(result).__name__)
            return result
        except Exception as e:
            self._logger.exception("Error executing tool %s: %s", canonical, str(e))
            # Provide more helpful error message
            error_msg = f"Tool execution error: {str(e)}"
            suggestion = ""
            
            if "got an unexpected keyword argument" in str(e):
                suggestion = " Try providing arguments in the format: tool_name(key1=\"value1\", key2=value2)"
            elif "missing required argument" in str(e).lower() or "missing required positional argument" in str(e).lower():
                suggestion = " Please provide all required arguments for this tool."
                
            return error_msg + suggestion
    
    def cleanup(self) -> None:
        """Cleanup resources used by the executor"""
        try:
            self._execution_strategy.cleanup()
        except Exception as e:
            self._logger.exception("Error during executor cleanup: %s", str(e))


class NdjsonEventEmitter(EventEmitter):
    def __init__(self, logger: Optional[Logger] = None):
        self._logger = logger or NoOpLogger()
    
    def _line(self, payload: Dict[str, Any] | str) -> bytes:
        try:
            if isinstance(payload, str):
                result = (payload.rstrip("\n") + "\n").encode("utf-8")
                self._logger.debug("Emitting string event: %s", payload[:50])
                return result
                
            result = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            self._logger.debug("Emitting event: %s", payload.get("event", "unknown"))
            return result
        except Exception as e:
            self._logger.exception("Error formatting event: %s", str(e))
            # Fallback to simple format to avoid breaking the stream
            return f'{{"event":"error","message":"Event formatting error"}}\n'.encode("utf-8")

    def token(self, text: str) -> bytes:
        return self._line({"event": "token", "data": text})
    
    def hidden_thought(self, text: str, phase: str) -> bytes:
        """
        Emit hidden thought tokens with a phase tag ('pre_tool' or 'post_tool').
        """
        return self._line({"event": "hidden_thought", "phase": phase, "data": text})

    def tool_start(self, tc_id: str, published_name: str) -> bytes:
        self._logger.info("Tool call started: %s (id: %s)", published_name, tc_id)
        return self._line({"event": "tool_start", "id": tc_id, "name": published_name})

    def tool_end(self, tc_id: str, published_name: str, result: Any) -> bytes:
        self._logger.info("Tool call completed: %s (id: %s)", published_name, tc_id)
        # Ensure result is serializable - truncate if too long
        result_str = str(result)
        if len(result_str) > 4096:
            result_str = result_str[:4093] + "..."
            
        return self._line({
            "event": "tool_end", 
            "id": tc_id, 
            "name": published_name, 
            "result": result_str
        })

    def done(self) -> bytes:
        self._logger.debug("Stream completed")
        return self._line({"event": "done"})


# -----------------------------
#  Concrete adapters (your app)
# -----------------------------

class ModelTokenSource(TokenSource):
    """
    Adapts your model.generate(...) stream to TokenDelta.
    """
    def __init__(self, out_iter, logger: Optional[Logger] = None, config: Optional[ConfigProvider] = None):
        self.out_iter = out_iter
        self._logger = logger or NoOpLogger()
        self._config = config or EnvironmentConfigProvider()
        self._async_timeout = self._config.get_async_timeout()
        # current async iterator for streaming, used for abort
        self._current_aiter: Optional[AsyncIterator] = None

    def _is_async_iter(self, obj) -> bool:
        return hasattr(obj, "__aiter__") or inspect.isasyncgen(obj)

    def _aiter_sync(self, gen):
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=8)
        SENTINEL = object()
        stop = threading.Event()

        def produce():
            try:
                for item in gen:
                    if stop.is_set():
                        break
                    try:
                        loop.call_soon_threadsafe(q.put_nowait, item)
                    except asyncio.QueueFull:
                        self._logger.warning("Queue full, waiting before adding more items")
                        time.sleep(0.1)  # Brief pause to let queue clear
            except Exception as e:
                self._logger.exception("Error in producer thread: %s", str(e))
                try:
                    loop.call_soon_threadsafe(q.put_nowait, e)
                except:
                    pass  # If we can't queue the exception, we'll rely on timeouts
            finally:
                try:
                    loop.call_soon_threadsafe(q.put_nowait, SENTINEL)
                except:
                    pass  # Best effort

        t = threading.Thread(target=produce, name="sync-gen-producer", daemon=True)
        t.start()

        async def _aiter():
            try:
                while True:
                    try:
                        # Add timeout to prevent hanging
                        item = await asyncio.wait_for(q.get(), timeout=self._async_timeout)
                        if item is SENTINEL:
                            break
                        if isinstance(item, Exception):
                            raise item
                        yield item
                    except asyncio.TimeoutError:
                        self._logger.warning("Timeout waiting for next item in stream")
                        break
            finally:
                stop.set()
                try:
                    gen.close()
                except Exception as e:
                    self._logger.debug("Error closing generator: %s", str(e))
        return _aiter()

    async def stream(self) -> AsyncIterator[TokenDelta]:
        self._logger.debug("Starting token stream")
        try:
            aiter = self.out_iter if self._is_async_iter(self.out_iter) else self._aiter_sync(self.out_iter)
            # store for abort
            self._current_aiter = aiter
            
            # Wrap in timeout to avoid hanging
            async with asyncio.timeout(self._async_timeout):
                async for chunk in aiter:
                    try:
                        choice = chunk.choices[0] if hasattr(chunk, 'choices') and chunk.choices else None
                        if not choice:
                            self._logger.warning("Received chunk without choices")
                            continue
                            
                        delta = getattr(choice, "delta", None)
                        token = getattr(delta, "content", None)
                        finish = getattr(choice, "finish_reason", None)
                        
                        if finish:
                            self._logger.debug("Stream finished with reason: %s", finish)
                            
                        yield TokenDelta(token=token, finish_reason=finish)
                    except Exception as e:
                        self._logger.exception("Error processing chunk: %s", str(e))
                        # Continue streaming if possible
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            self._logger.warning("Stream operation timed out or was cancelled: %s", str(e))
            yield TokenDelta(token=None, finish_reason="timeout")
        except Exception as e:
            self._logger.exception("Unexpected error in token stream: %s", str(e))
            yield TokenDelta(token=None, finish_reason="error")
        finally:
            # clear current iterator on completion
            self._current_aiter = None

    async def abort_current_stream(self) -> None:
        """
        Abort the current token stream to stop consuming further tokens.
        """
        aiter = self._current_aiter
        self._current_aiter = None
        # if async generator, close it
        if aiter and hasattr(aiter, 'aclose'):
            try:
                # type ignore: aclose may not be on generic AsyncIterator
                await (aiter.aclose())  # type: ignore
            except Exception:
                pass  # best effort


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

    def add_tool_result(self, history: list, tool_id: str, pub_name: str, result: Any) -> None:
        try:
            # Convert result to string and truncate if too long
            result_str = str(result)
            if len(result_str) > 10000:  # Prevent excessively large results
                result_str = result_str[:9997] + "..."
                self._logger.warning("Tool result truncated from %d to 10000 chars", len(str(result)))
                
            history.append({
                "role": "tool", 
                "tool_call_id": tool_id, 
                "name": pub_name, 
                "content": result_str
            })
            self._logger.debug("Added tool result to history: %s result length: %d", 
                             pub_name, len(result_str))
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


"""
Tool boundary controller to handle start/end events and history around tool execution.
"""
class ToolBoundaryController:
    def __init__(self, orchestrator: 'ToolOrchestrator'):
        self.orch = orchestrator

    def on_start(self, tool_name: str, session_id: str, loops: int) -> Tuple[str, str, bytes]:
        """Generate tool_id, emit start event, and return identifiers and event bytes."""
        tool_id = f"tc_{loops}_{int(time.time())}"
        pub = f"{self.orch.tool_prefix}{tool_name}"
        self.orch._logger.info("Tool call detected: %s (session: %s)", tool_name, session_id)
        ev = self.orch.emitter.tool_start(tool_id, pub)
        return tool_id, pub, ev

    def on_complete(self, tool_id: str, pub: str, args: Dict[str, Any], history: list) -> Tuple[Any, bytes]:
        """Execute tool, emit end event, and update history. Returns result and event bytes."""
        # Record call in history
        self.orch.history.add_assistant_toolcall(history, tool_id, pub, args)
        # Execute tool
        try:
            self.orch._logger.info("Executing tool: %s with args: %s", pub, args)
            result = self.orch.tool_executor.execute(pub, args)
        except FuturesTimeoutError:
            # Handle tool execution timeout
            timeout = self.orch._config.get_tool_execution_timeout()
            self.orch._logger.warning("Tool %s timed out after %s seconds", pub, timeout)
            result = f"Tool execution timed out after {timeout} seconds"
        except Exception as e:
            self.orch._logger.exception("Error executing tool %s: %s", pub, str(e))
            result = f"Tool execution error: {e}"
        # Emit end event
        ev = self.orch.emitter.tool_end(tool_id, pub, result)
        # Write result to history
        self.orch.history.add_tool_result(history, tool_id, pub, result)
        return result, ev


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
        loops = 0
        try:
            history = self.history.get_history(session_id)
            self.history.ensure_system_prompt(history, tools_to_use)
            
            self._logger.info(
                "Starting generation with tools for session %s (tools: %d, history: %d msgs)",
                session_id, len(tools_to_use), len(history)
            )

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
                                        # Abort current generation stream
                                        await token_source.abort_current_stream()
                                        # switch to post-tool phase for hidden thoughts after tool
                                        phase = "post_tool"
                                        continue
                                        continue

                                    if status == "call_complete":
                                        captured_args = self.args_parser.parse(payload or "")
                                        self._logger.debug(
                                            "Tool call args captured for %s: %s", 
                                            tool_name, captured_args
                                        )
                                        break  # end this assistant turn to execute tool

                                    # undecided → keep buffering
                                    continue

                                # pass-through mode: stream immediately
                                streamed_text.append(visible)
                                yield self.emitter.token(visible)

                            # terminator: check finish_reason outside token handling
                            if delta.finish_reason in ("stop", "length", "content_filter", "tool_calls", "timeout", "error"):
                                self._logger.debug("Stream finished with reason: %s", delta.finish_reason)
                                # If a tool was started but never completed, flush raw buffer as prose
                                if tool_name and captured_args is None and not pass_through:
                                    text = "".join(detector._raw[:1000])  # safe: internal buffer, limit size
                                    if text:
                                        streamed_text.append(text)
                                        yield self.emitter.token(text)
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
                if tool_name and captured_args is not None and tool_id is not None:
                    # Execute tool and emit end via controller
                    result, end_ev = controller.on_complete(
                        tool_id,
                        pub,
                        captured_args,
                        history,
                    )
                    yield end_ev

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
                    continue  # next assistant turn

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

# --- Pydantic Models for API ---

class Message(BaseModel):
    """Message model for API responses."""
    id: int
    role: str
    content: str
    created_at: datetime


class CreateSessionResponse(BaseModel):
    """Response model for session creation."""
    session_id: str
    created_at: datetime


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    session_id: str = Field(...)
    prompt: str = Field(...)
    model_name: str = Field(...)
    device: Optional[str] = "auto"
    dll: Optional[str] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95
    # Add tools parameter for OpenAI-compatible tool definitions
    tools: Optional[List[dict]] = None
    # The mcp_server config is now handled by the client, which passes `tools`
    mcp_server: Optional[dict] = None


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    session_id: str
    reply: str
    messages: List[Message]


class UnloadModelRequest(BaseModel):
    """Request model for unloading a model."""
    model_name: str
    device: Optional[str] = "auto"

# --- Protocol Component ---

class ProtocolComponent:
    """Protocol component for MCP architecture handling API interactions."""
    
    def __init__(self, model_component: ModelComponent, context_component: ContextComponent):
        """Initialize with model and context components."""
        self.model = model_component
        self.context = context_component
        self.max_tool_loops = 3
        self._executed_tool_calls: set = set()
        self._execution_strategy_for_request = None
        
        # Register tools by auto-discovering modules
        self._register_tools()
        
        # Cache for dynamic system prompts with tool information (keyed by tool count/hash)
        self._system_prompt_cache = {}
    
    def _register_tools(self):
        """Auto-discover and register all tool modules."""
        # Auto-discover tool modules in the tools package
        for m in pkgutil.iter_modules(tools.__path__):
            importlib.import_module(f"{tools.__name__}.{m.name}")
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Get available models from the model component."""
        return self.model.get_available_models()
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from the tools registry."""
        return get_all_tool_definitions()
    
    def generate_dynamic_system_prompt(self, tools: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate a dynamic system prompt that includes instructions for tool calling
        based on the available tools.
        """
        tool_names = sorted([t.get("function", {}).get("name", "") for t in tools])
        cache_key = ",".join(tool_names)
        
        if cache_key in self._system_prompt_cache:
            return self._system_prompt_cache[cache_key]
        
        tool_descriptions = []
        
        for tool in tools:
            if "function" not in tool:
                continue
                
            func = tool["function"]
            name = func.get("name", "")
            description = func.get("description", "No description available")
            params = func.get("parameters", {})
            
            param_list = []
            if "properties" in params:
                for param_name, param_info in params["properties"].items():
                    param_type = param_info.get("type", "string")
                    param_desc = param_info.get("description", "")
                    required = param_name in params.get("required", [])
                    
                    req_text = "required" if required else "optional"
                    param_list.append(f"  - {param_name} ({param_type}, {req_text}): {param_desc}")
            
            tool_desc = f"• {name}: {description}"
            if param_list:
                tool_desc += "\n" + "\n".join(param_list)
            tool_descriptions.append(tool_desc)
        
        system_content = f"""You are a helpful assistant with access to these tools:

        ### AVAILABLE TOOLS:
        {chr(10).join(tool_descriptions)}
        Tool names always start with `__tool_`. When invoking a tool, output exactly one line like `__tool_get_current_time(timezone=\"Europe/Dublin\")`. Use keyword arguments only.

        ### TOOL CALL FORMAT (CRITICAL)
        - When you need a tool, output **exactly one line** containing **only** the function call (no prose, no code fences).
        - Use **keyword arguments only**. Do **NOT** pass a single dict as a positional arg.
        - Quote strings. Keep values as simple Python/JSON literals.

        ✅ Correct:
        __tool_get_current_time(timezone="Europe/Vienna")
        __tool_web_search(query="Bitcoin latest news", count=5)
        __tool_get_top_headlines(q="latest news", country="ie", page_size=5)
        __tool_list_sources(country="ie")

    ❌ Wrong:
    __tool_get_current_time({{"timezone":"Europe/Vienna"}})       # dict as positional arg
    ```python
    __tool_get_current_time(timezone="Europe/Vienna")
    ```                                             # code fence / extra text
    __tool_get_current_time(timezone="Europe/Vienna") Also... # any extra prose

        ### WHEN NO TOOL IS NEEDED
        - Respond with a single, concise answer (no alternative phrasings).
        """

        system_message = {
            "role": "system", 
            "content": system_content
        }
        
        self._system_prompt_cache[cache_key] = system_message
        return system_message

    def create_session(self) -> Dict[str, Any]:
        """Create a new session using the context component."""
        return self.context.create_session()
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions using the context component."""
        return self.context.list_sessions()
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session using the context component."""
        return self.context.delete_session(session_id)
    
    def delete_all_sessions(self) -> int:
        """Delete all sessions using the context component."""
        return self.context.delete_all_sessions()
    
    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get messages for a session using the context component."""
        return self.context.get_messages(session_id)
    
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Generate text using the model component based on the conversation history.
        """
        session_id = request.session_id
        
        session = self.context.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        self.context.add_message(session_id, "user", request.prompt)
        
        history = self.context.get_conversation_history(session_id, format="chat")

        available_tools = self.get_available_tools()
        
        if request.tools:
            reg_names = {t["function"]["name"] for t in available_tools}
            requested = [t.get("function", {}).get("name") for t in request.tools]
            missing = [n for n in requested if n not in reg_names]
            if missing:
                logger.debug("Ignoring unregistered tools requested by client: %s", missing)
            tools_to_use = [t for t in request.tools if t.get("function", {}).get("name") in reg_names]
        else:
            tools_to_use = available_tools
            
        if tools_to_use and not any(msg.get("role") == "system" for msg in history):
            system_message = self.generate_dynamic_system_prompt(tools_to_use)
            history.insert(0, system_message)

        reply = ""
        tool_loop_count = 0

        while True:
            out = self.model.generate(
                model_name=request.model_name,
                messages=history,
                device=request.device or "auto",
                dll_path=request.dll,
                max_tokens=request.max_tokens or 512,
                temperature=request.temperature or 0.2,
                top_p=request.top_p or 0.95,
                tools=tools_to_use
            )
            
            message = out.choices[0].message if out and out.choices else None

            if message and getattr(message, "tool_calls", None):
                history.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": (
                                    tc.function.arguments
                                    if isinstance(tc.function.arguments, str)
                                    else json.dumps(tc.function.arguments or {})
                                )
                            }
                        }
                        for tc in (message.tool_calls or [])
                    ],
                })

                for tc in (message.tool_calls or []):
                    name = tc.function.name
                    args = tc.function.arguments
                    
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    elif args is None:
                        args = {}
                    
                    try:
                        result = execute_tool(name, args)
                    except Exception as e:
                        result = f"Tool execution error: {e}"
                    history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": str(result),
                    })

                tool_loop_count += 1
                if tool_loop_count >= self.max_tool_loops:
                    reply = (
                        "I ran tools several times but couldn't finish the reasoning. "
                        "Try rephrasing or reducing steps."
                    )
                    break

                continue
            
            reply = (message.content or "") if message else ""
            break

        self.context.add_message(session_id, "assistant", reply)
        
        messages = self.context.get_messages(session_id)
        
        return GenerateResponse(
            session_id=session_id,
            reply=reply,
            messages=[
                Message(
                    id=m["id"],
                    role=m["role"],
                    content=m["content"],
                    created_at=m["created_at"]
                ) for m in messages
            ]
        )
    
    def unload_model(self, request: UnloadModelRequest) -> Dict[str, str]:
        """Unload a model from memory."""
        self.model.unload_model(request.model_name, request.device)
        return {"detail": f"Model '{request.model_name}' unloaded successfully"}
    
    # -----------------------------
    # ======== Composition =========
    # -----------------------------
    def _get_execution_strategy(self) -> ExecutionStrategy:
        """
        Get a shared execution strategy, using the one from the app state if available.
        """
        if hasattr(self, "_execution_strategy_for_request") and self._execution_strategy_for_request is not None:
            strategy = self._execution_strategy_for_request
            self._execution_strategy_for_request = None  # Clean up after use
            return strategy
        
        # Create a new one if not provided by app state
        config = EnvironmentConfigProvider()
        return ThreadPoolExecutionStrategy(max_workers=config.get_thread_pool_workers())
    
    async def generate_stream(self, request: "GenerateRequest") -> StreamingResponse:
        """
        Thin composition root: wires app deps into SOLID components.
        """
        session_id = request.session_id
        session = self.context.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Record user
        self.context.add_message(session_id, "user", request.prompt)

        # Tools filter
        available_tools = self.get_available_tools()
        if request.tools:
            reg = {t["function"]["name"] for t in available_tools}
            tools_to_use = [t for t in request.tools if t.get("function", {}).get("name") in reg]
        else:
            tools_to_use = available_tools
            
        # Create shared dependencies
        logger = StandardLogger(__name__)
        config = EnvironmentConfigProvider()
        execution_strategy = self._get_execution_strategy()

        # Orchestrator deps (via DI)
        orchestrator = ToolOrchestrator(
            args_parser=JsonArgsParser(logger=logger),
            tool_executor=DefaultToolExecutor(
                executor_fn=execute_tool,
                execution_strategy=execution_strategy,
                logger=logger,
                config=config
            ),
            history=ContextHistoryWriter(
                self.context,
                self.generate_dynamic_system_prompt,
                logger
            ),
            emitter=NdjsonEventEmitter(logger=logger),
            max_tool_loops=getattr(self, "max_tool_loops", 3),
            tool_prefix="__tool_",
            logger=logger,
            config=config
        )

        # Model factory (DI) – single place where vendor specifics live
        def model_factory(**kw):
            return self.model.generate(
                model_name=request.model_name,
                device=request.device or "auto",
                dll_path=request.dll,
                max_tokens=request.max_tokens or 512,
                temperature=request.temperature or 0.2,
                top_p=request.top_p or 0.95,
                **kw,
            )

        async def event_stream():
            logger.info(
                "Streaming generate request: session=%s, model=%s, tools=%s, prompt=%s",
                request.session_id,
                request.model_name,
                [t.get("function", {}).get("name") for t in tools_to_use],
                request.prompt[:100],
            )
            async for line in orchestrator.run_turns(
                session_id=session_id,
                model_factory=model_factory,
                model_args={},  # messages/tools/stream handled inside orchestrator
                tools_to_use=tools_to_use,
            ):
                yield line

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

# --- FastAPI Application Factory ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for resource cleanup."""
    # Initialize shared resources
    logger = StandardLogger("app_lifespan")
    app.state.logger = logger
    app.state.config = EnvironmentConfigProvider()
    
    # Create thread pool for tool execution
    workers = app.state.config.get_thread_pool_workers()
    logger.info("Initializing application with %d worker threads", workers)
    app.state.execution_strategy = ThreadPoolExecutionStrategy(max_workers=workers)
    
    try:
        logger.info("Application started")
        yield
    except Exception as e:
        logger.exception("Error during application lifecycle: %s", str(e))
    finally:
        # Clean up resources
        logger.info("Application shutting down, cleaning up resources")
        
        # Clean up execution strategy
        try:
            if hasattr(app.state, "execution_strategy"):
                app.state.execution_strategy.cleanup()
                logger.info("Thread pool execution strategy shut down")
        except Exception as e:
            logger.exception("Error cleaning up execution strategy: %s", str(e))
        
        # Clear model cache
        try:
            app.state.protocol.model.clear_engine_cache()
            logger.info("Model engine cache cleared")
        except Exception as e:
            logger.exception("Error clearing model cache: %s", str(e))


def create_api_app(protocol: ProtocolComponent) -> FastAPI:
    """
    Create a FastAPI application with all routes configured.
    
    Args:
        protocol: The protocol component instance
        
    Returns:
        A configured FastAPI application
    """
    app = FastAPI(
        title="MLC-LLM Session Service", 
        version="0.1.0", 
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.protocol = protocol
    
    # Register a shutdown event handler for additional cleanup
    @app.on_event("shutdown")
    async def shutdown_event():
        logger = getattr(app.state, "logger", StandardLogger("shutdown"))
        logger.info("Application shutdown event triggered")
    
    # Health check endpoint
    @app.get("/healthz")
    def healthz():
        return {"ok": True}
    
    # Model endpoints
    @app.get("/models")
    def list_models():
        return {"models": protocol.get_available_models()}
    
    @app.post("/models/unload")
    def unload_model(req: UnloadModelRequest):
        return protocol.unload_model(req)
    
    # Tools endpoint
    @app.get("/tools")
    def list_tools():
        return {"tools": protocol.get_available_tools()}
    
    # Execute tool endpoint for direct tool calls
    @app.post("/tools/execute")
    def execute_tool_endpoint(call: Dict[str, Any]):
        # Validate tool name
        name = call.get("name")
        if not isinstance(name, str):
            raise HTTPException(status_code=400, detail="Tool name must be a string")
        # Validate arguments
        params = call.get("arguments", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise HTTPException(status_code=400, detail="Tool arguments must be a dict")
        # Execute tool
        try:
            result = execute_tool(name, params)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"tool": name, "result": result}
    
    # Session endpoints
    @app.post("/sessions", response_model=CreateSessionResponse)
    def create_session():
        result = protocol.create_session()
        return CreateSessionResponse(
            session_id=result["session_id"],
            created_at=result["created_at"]
        )
    
    @app.get("/sessions")
    def list_sessions():
        return protocol.list_sessions()
    
    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str):
        if protocol.delete_session(session_id):
            return {"detail": "Session deleted"}
        raise HTTPException(status_code=404, detail="Session not found")
    
    @app.delete("/sessions")
    def delete_all_sessions():
        count = protocol.delete_all_sessions()
        return {"detail": f"Deleted {count} sessions."}
    
    @app.get("/sessions/{session_id}/messages", response_model=List[Message])
    def get_session_messages(session_id: str):
        messages = protocol.get_session_messages(session_id)
        if not messages:
            raise HTTPException(status_code=404, detail="Session not found")
        return [
            Message(
                id=m["id"],
                role=m["role"],
                content=m["content"],
                created_at=m["created_at"]
            ) for m in messages
        ]
    
    # Generation endpoints
    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest):
        return protocol.generate(req)
    
    @app.post("/generate_stream")
    async def generate_stream(request: Request, req: GenerateRequest):
        # Pass app state's shared execution strategy to protocol if needed
        if hasattr(app.state, "execution_strategy"):
            # Store the execution strategy temporarily if needed for this request
            app.state.protocol._execution_strategy_for_request = app.state.execution_strategy
        return await protocol.generate_stream(req)
    
    return app

class HiddenBlockFilter:
    """
    Strips tokens inside <think>...</think> (and similar) from the stream.
    Can optionally surface them to UI as 'hidden_thought' events.
    """
    START_TAGS = [re.compile(r'<think\b[^>]*>'), re.compile(r'<reflection\b[^>]*>')]
    END_TAGS   = [re.compile(r'</think\s*>'),     re.compile(r'</reflection\s*>')]
    START_HTML_COMMENT = re.compile(r'<!--')
    END_HTML_COMMENT   = re.compile(r'-->')

    def __init__(self, expose_hidden_thoughts: bool = True, on_hidden_chunk: Optional[Callable[[str], None]] = None):
        """Filter that routes hidden blocks/comments away from detector and optionally invokes callback."""
        # expose_hidden_thoughts controls whether hidden tokens are emitted to UI
        self.expose_hidden = expose_hidden_thoughts
        self._on_hidden = on_hidden_chunk
        self._in_hidden = False
        self._in_comment = False
        self._buf = []

    def feed(self, token: str) -> RoutedChunk:
        # Detect HTML comments irrespective of tag state
        if not self._in_hidden and self.START_HTML_COMMENT.search(token):
            self._in_comment = True
        if self._in_comment:
            # Accumulate comments as hidden tokens
            self._buf.append(token)
            if self.END_HTML_COMMENT.search(token):
                self._in_comment = False
                self._buf.clear()
            hidden = token if self.expose_hidden else ""
            if self._on_hidden:
                self._on_hidden(token)
            return RoutedChunk(to_detector="", to_ui_visible="", to_ui_hidden=hidden)
        # Detect think/reflection start tags
        if not self._in_hidden and any(p.search(token) for p in self.START_TAGS):
            self._in_hidden = True
        if self._in_hidden:
            # Accumulate hidden block tokens
            self._buf.append(token)
            if any(p.search(token) for p in self.END_TAGS):
                self._in_hidden = False
                self._buf.clear()
            hidden = token if self.expose_hidden else ""
            if self._on_hidden:
                self._on_hidden(token)
            return RoutedChunk(to_detector="", to_ui_visible="", to_ui_hidden=hidden)
        # Visible token routes to detector and visible UI
        return RoutedChunk(to_detector=token, to_ui_visible=token, to_ui_hidden="")
