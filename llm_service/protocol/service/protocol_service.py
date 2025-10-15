"""
Protocol service implementation for the MCP architecture.
"""
import json
import logging
import pkgutil
import importlib
from typing import List, Dict, Any, Optional, AsyncIterator

from fastapi import HTTPException
from llm_service import tools
from llm_service.context import ContextComponent
from llm_service.model.mlc_engine import ModelComponent
from llm_service.tools import execute_tool, get_all_tool_definitions

from llm_service.protocol.orchestration.system_prompt import build_tooling_system_prompt

from llm_service.protocol.api.schemas import (
    Message,
    GenerateRequest,
    GenerateResponse,
    UnloadModelRequest,
)

logger = logging.getLogger(__name__)


class ProtocolService:
    """Protocol service for MCP architecture handling API interactions."""
    
    def __init__(self, model_component: ModelComponent, context_component: ContextComponent):
        """Initialize with model and context components."""
        self.model = model_component
        self.context = context_component
        self._executed_tool_calls: set = set()
        
        # Register tools by auto-discovering modules
        self._register_tools()
        
        # Cache for dynamic system prompts with tool information (keyed by tool count/hash)
        self._system_prompt_cache = {}
        
        # Lazy-loaded generation service for streaming support
        self._generation_service = None
    
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
    
    def generate_dynamic_system_prompt(self, tools):
        return build_tooling_system_prompt(tools, locale_hint="Europe/Dublin")

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
    
    def unload_model(self, request: UnloadModelRequest) -> Dict[str, str]:
        """Unload a model from memory."""
        self.model.unload_model(request.model_name, request.device)
        return {"detail": f"Model '{request.model_name}' unloaded successfully"}
        
    def _get_generation_service(self):
        """
        Lazy-load the generation service.
        
        Returns:
            GenerationService instance
        """
        if self._generation_service is None:
            from llm_service.protocol.service.generation_service import GenerationService
            self._generation_service = GenerationService(
                self.model,
                self.context,
                self.generate_dynamic_system_prompt,
                self.get_available_tools
            )
        return self._generation_service
        
    def generate_stream(self, 
                      session_id: str, 
                      model_name: str,
                      prompt: str,
                      tools=None,
                      device=None,
                      dll=None,
                      max_tokens=None,
                      temperature=None,
                      top_p=None,
                      **kwargs):
        """
        Generate text with streaming support.
        
        Args:
            session_id: Session ID
            model_name: Model name
            prompt: Text prompt
            tools: List of tool definitions
            device: Device to use (auto, cpu, cuda)
            dll: Path to MLC DLL
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional parameters
            
        Returns:
            StreamingResponse with NDJSON events
        """
        generation_service = self._get_generation_service()
        
        request = GenerateRequest(
            session_id=session_id,
            prompt=prompt,
            model_name=model_name,
            tools=tools,
            device=device,
            dll=dll,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            **kwargs
        )
        
        # Return the streaming response directly
        return generation_service.generate_stream(request)
        
    def abort_generation(self, session_id: str) -> bool:
        """
        Abort an ongoing generation for a session.
        
        Args:
            session_id: The ID of the session with active generation
            
        Returns:
            True if any generations were aborted, False if no active generations found
        """
        generation_service = self._get_generation_service()
        return generation_service.abort_generation(session_id)