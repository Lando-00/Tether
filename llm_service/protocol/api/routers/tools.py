"""
Router for tool-related endpoints.
"""
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException

from llm_service.protocol.service.protocol_service import ProtocolService
from llm_service.tools import execute_tool


def get_tools_router(protocol_service_provider):
    """
    Creates a router for tool-related endpoints.
    
    Args:
        protocol_service_provider: Callable that returns a ProtocolService instance
        
    Returns:
        FastAPI APIRouter for tool-related endpoints
    """
    router = APIRouter(prefix="/tools", tags=["tools"])
    
    @router.get("")
    def list_tools(
        protocol: ProtocolService = Depends(protocol_service_provider)
    ):
        """List available tools."""
        return {"tools": protocol.get_available_tools()}
    
    @router.post("/execute")
    def execute_tool_endpoint(
        call: Dict[str, Any],
        _protocol: ProtocolService = Depends(protocol_service_provider)
    ):
        """
        Execute a tool directly.
        
        Args:
            call: Dict with "name" and "arguments" keys
            
        Returns:
            Dict with "tool" and "result" keys
        """
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
    
    return router