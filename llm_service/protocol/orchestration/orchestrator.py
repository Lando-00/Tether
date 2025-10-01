"""
Tool orchestration module that handles token streaming, tool detection and execution.
"""
from __future__ import annotations
from typing import Optional, Dict, Any, AsyncIterator, List, TYPE_CHECKING
import asyncio
import anyio
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

from llm_service.protocol.orchestration.filters import HiddenBlockFilter
from llm_service.protocol.orchestration.controller import ToolBoundaryController
from llm_service.protocol.orchestration.detector import PrefixedToolCallDetector
from llm_service.protocol.orchestration.adapters.model_token_source import ModelTokenSource


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