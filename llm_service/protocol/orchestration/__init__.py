"""
This module defines the orchestration components for the LLM service protocol.

It exports the following components:
- PrefixedToolCallDetector: Detects tool call patterns in LLM output
- JsonArgsParser: Parses JSON arguments for tool calls
- DefaultToolExecutor: Executes tools with validation and error handling
- NdjsonEventEmitter: Emits events in NDJSON format for streaming responses
- HiddenBlockFilter: Filters hidden blocks and comments from the token stream
- ToolBoundaryController: Handles tool boundaries, tool execution and event emission
- ModelTokenSource: Adapts model.generate output stream to TokenDelta objects
- ContextHistoryWriter: Adapts context component to HistoryWriter interface
- ToolOrchestrator: High-level orchestrator for token streaming and tool execution
"""

from llm_service.protocol.orchestration.detector import PrefixedToolCallDetector
from llm_service.protocol.orchestration.parsers import JsonArgsParser
from llm_service.protocol.orchestration.parsers import SimpleSlidingToolStreamParser
from llm_service.protocol.orchestration.parsers import StreamEvent
from llm_service.protocol.orchestration.executor import DefaultToolExecutor
from llm_service.protocol.orchestration.emitter import NdjsonEventEmitter
from llm_service.protocol.orchestration.filters import HiddenBlockFilter
from llm_service.protocol.orchestration.controller import ToolBoundaryController
from llm_service.protocol.orchestration.adapters.model_token_source import ModelTokenSource
from llm_service.protocol.orchestration.adapters.context_history import ContextHistoryWriter
from llm_service.protocol.orchestration.orchestrator import ToolOrchestrator

__all__ = [
    "PrefixedToolCallDetector",
    "JsonArgsParser",
    "SimpleSlidingToolStreamParser",
    "StreamEvent",
    "DefaultToolExecutor",
    "NdjsonEventEmitter",
    "HiddenBlockFilter",
    "ToolBoundaryController",
    "ModelTokenSource",
    "ContextHistoryWriter",
    "ToolOrchestrator"
]
