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
    JsonArgsParser,
    DefaultToolExecutor,
    NdjsonEventEmitter,
    ContextHistoryWriter,
    ToolOrchestrator,
)

from llm_service.tools import execute_tool, get_all_tool_definitions

from llm_service.protocol.api.schemas import GenerateRequest


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
        self._active_generations: Dict[str, Set[asyncio.Task]] = {}
        
    def _get_execution_strategy(self) -> ExecutionStrategy:
        """Get a shared execution strategy."""
        if self._execution_strategy_provider:
            return self._execution_strategy_provider()
        
        # Create a new one if not provided
        config = EnvironmentConfigProvider()
        from llm_service.protocol.core.execution import ThreadPoolExecutionStrategy
        return ThreadPoolExecutionStrategy(max_workers=config.get_thread_pool_workers())

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
        available_tools = self._get_available_tools()
        if request.tools:
            reg = {t["function"]["name"] for t in available_tools}
            tools_to_use = [t for t in request.tools if t.get("function", {}).get("name") in reg]
        else:
            tools_to_use = available_tools
            
        # Create shared dependencies
        logger = self.logger
        config = EnvironmentConfigProvider()
        execution_strategy = self._get_execution_strategy()

        # Create a custom event emitter that implements the required methods
        from llm_service.protocol.core.interfaces import EventEmitter
        
        class CustomEventEmitter(EventEmitter):
            def __init__(self, logger=None):
                self._logger = logger or NoOpLogger()
                
            def _emit_event(self, event_type: str, data: Dict[str, Any]) -> bytes:
                """
                Formats and emits a single event as NDJSON.
                
                Args:
                    event_type: The type of event (e.g., "token", "hidden_thought")
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
                
                Args:
                    text: The token text
                    
                Returns:
                    Formatted NDJSON event as bytes
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
        
        # Orchestrator deps (via DI)
        orchestrator = ToolOrchestrator(
            args_parser=JsonArgsParser(logger=logger),
            tool_executor=DefaultToolExecutor(logger=logger),
            history=ContextHistoryWriter(
                self.context,
                self._generate_system_prompt or self._generate_default_system_prompt,
                logger
            ),
            emitter=CustomEventEmitter(logger=logger),
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

        # Track generation task for this session
        if session_id not in self._active_generations:
            self._active_generations[session_id] = set()
            
        async def event_stream():
            # Get the current task
            current_task = asyncio.current_task()
            if current_task:
                self._active_generations[session_id].add(current_task)
                
            try:
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
            except asyncio.CancelledError:
                logger.info(f"Generation for session {session_id} was cancelled")
                # Send cancellation event to client
                event = {"type": "cancelled"}
                yield f"data: {json.dumps(event)}\n\n".encode()
                raise
            except Exception as e:
                logger.exception(f"Error in generation stream for session {session_id}: {str(e)}")
                # Send error event to client
                event = {"type": "error", "error": str(e)}
                yield f"data: {json.dumps(event)}\n\n".encode()
            finally:
                # Clean up task tracking
                if current_task and session_id in self._active_generations:
                    self._active_generations[session_id].discard(current_task)
                    if not self._active_generations[session_id]:
                        del self._active_generations[session_id]

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
        
    def _generate_default_system_prompt(self, tools: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate a default system prompt if no custom function is provided.
        
        Args:
            tools: List of tool definitions
            
        Returns:
            Dict with "role" and "content" keys
        """
        tool_names = sorted([t.get("function", {}).get("name", "") for t in tools])
        
        tool_descriptions = []
        for tool in tools:
            if "function" not in tool:
                continue
                
            func = tool["function"]
            name = func.get("name", "")
            description = func.get("description", "No description available")
            
            tool_descriptions.append(f"• {name}: {description}")
            
        system_content = f"""You are a helpful assistant with access to these tools:

        {chr(10).join(tool_descriptions)}

        When you need to use a tool, output the function call in the format __tool_name(arg1="value", arg2="value").
        """
            
        return {
            "role": "system", 
            "content": system_content
        }