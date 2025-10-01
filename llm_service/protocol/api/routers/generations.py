"""
Router for text generation endpoints.
"""
from typing import Dict, Any, Optional, Union, List, AsyncIterator
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from llm_service.protocol.service.protocol_service import ProtocolService
from llm_service.protocol.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    Message
)


def get_generations_router(protocol_service_provider):
    """
    Creates a router for generation-related endpoints.
    
    Args:
        protocol_service_provider: Callable that returns a ProtocolService instance
        
    Returns:
        FastAPI APIRouter for generation-related endpoints
    """
    router = APIRouter(prefix="/generations", tags=["generations"])
    
    @router.post("")
    def generate(
        request: GenerateRequest,
        background_tasks: BackgroundTasks,
        protocol: ProtocolService = Depends(protocol_service_provider)
    ) -> Union[StreamingResponse, GenerateResponse]:
        """
        Generate text from a prompt using a model.
        
        Args:
            request: The generation request
            background_tasks: FastAPI background tasks
            protocol: ProtocolService for handling the generation
            
        Returns:
            Either a streaming response or a complete generation response
        """
        session_id = request.session_id
        
        # Validate session exists
        session = protocol.context.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # Get parameters from request
        model_name = request.model_name
        prompt = request.prompt
        tools = request.tools
        device = request.device
        dll = request.dll
        max_tokens = request.max_tokens
        temperature = request.temperature
        top_p = request.top_p
        
        # Check if streaming is requested
        stream = request.stream
        
        if stream:
            # Use the protocol service's stream implementation
            return protocol.generate_stream(
                session_id=session_id,
                model_name=model_name,
                prompt=prompt,
                tools=tools,
                device=device,
                dll=dll,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
        
        # Handle non-streaming request
        result = protocol.generate(request)
        
        return result
    
    @router.post("/abort")
    def abort_generation(
        data: Dict[str, str],
        protocol: ProtocolService = Depends(protocol_service_provider)
    ) -> Dict[str, bool]:
        """
        Abort a generation in progress.
        
        Args:
            data: Dict with "session_id" key
            protocol: ProtocolService for handling the abort request
            
        Returns:
            Dict with "success" key
        """
        session_id = data.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
            
        if not protocol.context.get_session(session_id):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            
        # Use the implemented abort_generation method
        success = protocol.abort_generation(session_id)
        
        return {"success": success}
    
    return router