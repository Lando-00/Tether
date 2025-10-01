"""
Main package exports for protocol components.

This module provides public exports for backward compatibility
with the original api.py implementation.
"""

# For backward compatibility
from .compatibility import ProtocolComponent, create_api_app

# New modular approach
from llm_service.protocol.api.app_factory import create_app as create_new_app

# Core components
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
from llm_service.protocol.core.config import EnvironmentConfigProvider
from llm_service.protocol.core.execution import ThreadPoolExecutionStrategy
from llm_service.protocol.core.loggers import NoOpLogger, StandardLogger
from llm_service.protocol.core.types import TokenDelta, RoutedChunk

# Orchestration components
from llm_service.protocol.orchestration import (
    PrefixedToolCallDetector,
    JsonArgsParser,
    DefaultToolExecutor,
    NdjsonEventEmitter,
    HiddenBlockFilter,
    ToolBoundaryController,
    ModelTokenSource,
    ContextHistoryWriter,
    ToolOrchestrator,
)

# API schemas
from llm_service.protocol.api.schemas import (
    Message,
    CreateSessionResponse,
    GenerateRequest,
    GenerateResponse,
    UnloadModelRequest,
)

# Services
from llm_service.protocol.service import ProtocolService, GenerationService

# Export everything for backward compatibility
__all__ = [
    # Old API exports
    "ProtocolComponent",
    "create_api_app",
    
    # Core interfaces
    "Logger",
    "ExecutionStrategy",
    "ConfigProvider",
    "TokenSource", 
    "ToolCallStrategy",
    "ArgsParser",
    "ToolExecutor",
    "HistoryWriter",
    "EventEmitter",
    
    # Core implementations
    "EnvironmentConfigProvider",
    "ThreadPoolExecutionStrategy",
    "NoOpLogger",
    "StandardLogger",
    "TokenDelta",
    "RoutedChunk",
    
    # Orchestration components
    "PrefixedToolCallDetector",
    "JsonArgsParser", 
    "DefaultToolExecutor",
    "NdjsonEventEmitter",
    "HiddenBlockFilter",
    "ToolBoundaryController",
    "ModelTokenSource",
    "ContextHistoryWriter",
    "ToolOrchestrator",
    
    # API schemas
    "Message",
    "CreateSessionResponse",
    "GenerateRequest",
    "GenerateResponse",
    "UnloadModelRequest",
    
    # Services
    "ProtocolService",
    "GenerationService",
]
