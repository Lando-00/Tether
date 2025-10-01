"""
Router for model-related endpoints.
"""
from fastapi import APIRouter, Depends

from llm_service.protocol.api.schemas import UnloadModelRequest
from llm_service.protocol.service.protocol_service import ProtocolService


def get_models_router(protocol_service_provider):
    """
    Creates a router for model-related endpoints.
    
    Args:
        protocol_service_provider: Callable that returns a ProtocolService instance
        
    Returns:
        FastAPI APIRouter for model-related endpoints
    """
    router = APIRouter(prefix="/models", tags=["models"])
    
    @router.get("")
    def list_models(
        protocol: ProtocolService = Depends(protocol_service_provider)
    ):
        """List available models."""
        return {"models": protocol.get_available_models()}
    
    @router.post("/unload")
    def unload_model(
        req: UnloadModelRequest,
        protocol: ProtocolService = Depends(protocol_service_provider)
    ):
        """Unload a model from memory."""
        return protocol.unload_model(req)
    
    return router