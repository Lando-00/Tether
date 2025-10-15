"""
Factory for creating the FastAPI application with all routers and components.
"""
import logging
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI

from llm_service.context import ContextComponent
from llm_service.model.mlc_engine import ModelComponent
# Import protocol service inside function to avoid circular imports
from llm_service.protocol.api.routers.health import get_health_router
from llm_service.protocol.api.routers.models import get_models_router
from llm_service.protocol.api.routers.sessions import get_sessions_router
from llm_service.protocol.api.routers.generations import get_generations_router
from llm_service.protocol.api.routers.tools import get_tools_router

logger = logging.getLogger(__name__)


def create_app(
    model_component: Optional[ModelComponent] = None,
    context_component: Optional[ContextComponent] = None,
) -> FastAPI:
    """
    Create the FastAPI application with all routers and components.
    
    Args:
        model_component: Optional model component, will be created if None
        context_component: Optional context component, will be created if None
        
    Returns:
        FastAPI application
    """
    if not model_component:
        model_component = ModelComponent()
    
    if not context_component:
        context_component = ContextComponent()
    
    # Service provider function for dependency injection
    def get_protocol_service():
        return protocol_service
    
    # Create the protocol service (import here to avoid circular import)
    from llm_service.protocol.service.protocol_service import ProtocolService
    protocol_service = ProtocolService(
        model_component=model_component,
        context_component=context_component,
    )
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load any resources on startup
        logger.info("Starting MLC Protocol API")
        yield
        # Clean up resources on shutdown
        logger.info("Shutting down MLC Protocol API")
        model_component.unload_model()  # Unloads all models when no params provided
    
    # Create the FastAPI app
    app = FastAPI(
        title="MLC Protocol API",
        description="API for interacting with MLC LLM models using the Model Context Protocol",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Add routers
    app.include_router(get_health_router())
    app.include_router(get_models_router(get_protocol_service))
    app.include_router(get_sessions_router(get_protocol_service))
    app.include_router(get_generations_router(get_protocol_service))
    app.include_router(get_tools_router(get_protocol_service))
    
    # Add root endpoint
    @app.get("/")
    def read_root():
        return {
            "name": "MLC Protocol API",
            "version": "0.1.0",
            "status": "running",
            "endpoints": [
                "/models", 
                "/sessions", 
                "/generations", 
                "/tools"
            ]
        }
    
    return app