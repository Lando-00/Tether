"""
Generation service for streaming responses.

This service handles text generation with streaming support, tool execution,
and integration with the ToolOrchestrator for managing complex generation flows.
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional, AsyncIterator, List, Set

from fastapi import HTTPException
from starlette.responses import StreamingResponse

from llm_service.protocol.core.interfaces import ExecutionStrategy, Logger
from llm_service.protocol.core.loggers import StandardLogger, NoOpLogger
from llm_service.protocol.core.config import EnvironmentConfigProvider

from llm_service.protocol.orchestration import (
    DefaultToolExecutor,
    NdjsonEventEmitter,
    ContextHistoryWriter,
    ToolOrchestrator,
)
from llm_service.protocol.orchestration.parsers import JsonArgsParser

from llm_service.tools import get_all_tool_definitions

from llm_service.protocol.api.schemas import GenerateRequest

from llm_service.protocol.service.helpers.event_parser import EventParser
# New streaming parser adapter
from llm_service.protocol.orchestration.parsers import SimpleSlidingStreamParser
from llm_service.protocol.service.helpers.event_logger import EventLogger

from llm_service.protocol.orchestration.system_prompt import build_tooling_system_prompt
from contextlib import asynccontextmanager

class GenerationService:
    """Service for handling text generation with streaming support."""

    def __init__(
        self, 
        model_component, 
        context_component,
        generate_dynamic_system_prompt_fn=None,
        get_available_tools_fn=None,
        execution_strategy_provider=None,
        logger=None,
    ):
        """
        Initialize the generation service with required components.
        
        Args:
            model_component: ModelComponent instance
            context_component: ContextComponent instance
            generate_dynamic_system_prompt_fn: Function to generate system prompts
            get_available_tools_fn: Function to get available tools
            execution_strategy_provider: Function that returns an ExecutionStrategy instance
            logger: Optional logger instance
        """
        self.model = model_component
        self.context = context_component
        self._generate_system_prompt = generate_dynamic_system_prompt_fn
        self._get_available_tools = get_available_tools_fn or get_all_tool_definitions
        self._execution_strategy_provider = execution_strategy_provider
        self.logger = logger or StandardLogger(__name__)
        self.max_tool_loops = 3

        # Active generation tasks that can be aborted
        self._active_generations = {}
        
    def _get_execution_strategy(self) -> ExecutionStrategy:
        """Get a shared execution strategy."""
        if self._execution_strategy_provider:
            return self._execution_strategy_provider()
        
        # Create a new one if not provided
        config = EnvironmentConfigProvider()
        from llm_service.protocol.core.execution import ThreadPoolExecutionStrategy
        return ThreadPoolExecutionStrategy(max_workers=config.get_thread_pool_workers())

    def _get_and_filter_tools(self, requested_tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Get and filter tools based on the request."""
        available_tools = self._get_available_tools()
        if requested_tools:
            reg = {t["function"]["name"] for t in available_tools}
            return [t for t in requested_tools if t.get("function", {}).get("name") in reg]
        return available_tools

    @asynccontextmanager
    async def _track_task(self, session_id: str):
        """
        Track the currently running asyncio.Task for this session so it can be aborted.
        Ensures add/remove is centralized and exception-safe.
        """
        task = asyncio.current_task()
        if task:
            self._active_generations.setdefault(session_id, set()).add(task)
        try:
            yield
        finally:
            if task and session_id in self._active_generations:
                self._active_generations[session_id].discard(task)
                if not self._active_generations[session_id]:
                    del self._active_generations[session_id]

    def generate_stream(self, request: GenerateRequest) -> StreamingResponse:
        """
        Stream generation results as they become available.

        Args:
            request: The generation request

        Returns:
            StreamingResponse with NDJSON events
        """
        session_id = request.session_id
        session = self.context.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Record user message
        self.context.add_message(session_id, "user", request.prompt)

        # Get and filter tools
        tools_to_use = self._get_and_filter_tools(request.tools)

        # Shared deps
        logger = self.logger
        config = EnvironmentConfigProvider()

        # Parsers and loggers
        parser = EventParser()
        stream_parser = SimpleSlidingStreamParser()
        evt_logger = EventLogger(logger)

        # Emitter for NDJSON events
        emitter = NdjsonEventEmitter(logger=logger)

        # Orchestrator deps (via DI)
        # Instantiate orchestrator with stream parser for interface-driven streaming
        orchestrator = ToolOrchestrator(
            args_parser=JsonArgsParser(logger=logger),
            tool_executor=DefaultToolExecutor(logger=logger),
            history=ContextHistoryWriter(
                self.context,
                # Ensure a callable even if self._generate_system_prompt is None
                self._generate_system_prompt or build_tooling_system_prompt,
                logger,
            ),
            emitter=emitter,  # reuse same emitter for error/cancel, too
            stream_parser=stream_parser,
            max_tool_loops=getattr(self, "max_tool_loops", 3),
            tool_prefix="__tool_",
            logger=logger,
            config=config,
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
            # Centralized, exception-safe task tracking
            async with self._track_task(session_id):
                try:
                    logger.info(
                        "Streaming generate request: session=%s, model=%s, tools=%s, prompt=%s",
                        request.session_id,
                        request.model_name,
                        [t.get("function", {}).get("name") for t in tools_to_use],
                        request.prompt[:100],
                    )

                    event_count = 0
                    tool_start_seen = False
                    tool_end_seen = False

                    logger.info("==== STARTING EVENT STREAM ====")
                    async for line in orchestrator.run_turns(
                        session_id=session_id,
                        model_factory=model_factory,
                        model_args={},  # messages/tools/stream handled inside orchestrator
                        tools_to_use=tools_to_use,
                    ):
                        event_count += 1
                        # Parse only for telemetry/logging; delivery uses the raw bytes from orchestrator.
                        event = parser.parse(line)  # dict (or {}) on failure
                        start, end = evt_logger.log(event, event_count)
                        tool_start_seen |= start
                        tool_end_seen |= end

                        yield line

                    logger.info("==== EVENT STREAM COMPLETED ====")
                    logger.info(
                        "Total events: %s, tool_start_seen: %s, tool_end_seen: %s",
                        event_count, tool_start_seen, tool_end_seen
                    )

                except asyncio.CancelledError:
                    logger.info("Generation for session %s was cancelled", session_id)
                    # Proper cancellation event via emitter
                    yield emitter.cancelled()
                    raise

                except Exception as e:
                    logger.exception("Error in generation stream for session %s: %s", session_id, str(e))
                    # Proper error event via emitter (no synthetic tool_end)
                    yield emitter.error(str(e))

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")

    def abort_generation(self, session_id: str) -> bool:
        """
        Abort an ongoing generation for a session.
        
        Args:
            session_id: The ID of the session with active generation
            
        Returns:
            True if any generations were aborted, False if no active generations found
        """
        if session_id not in self._active_generations:
            self.logger.warning(f"No active generation found for session {session_id}")
            return False
            
        tasks = self._active_generations[session_id]
        if not tasks:
            self.logger.warning(f"Empty task set for session {session_id}")
            return False
            
        aborted = 0
        for task in list(tasks):
            if not task.done():
                task.cancel()
                aborted += 1
                
        self.logger.info(f"Aborted {aborted} generation tasks for session {session_id}")
        return aborted > 0