"""
Compatibility module for backward compatibility with the original api.py implementation.

This module provides compatibility classes and functions that allow existing code
to continue working with the refactored protocol components.
"""
import logging
from typing import Optional

from fastapi import FastAPI

from llm_service.context import ContextComponent
from llm_service.model.mlc_engine import ModelComponent
# Import service inside the class to avoid circular imports


logger = logging.getLogger(__name__)


class ProtocolComponent:
    """
    Legacy compatibility class for the ProtocolComponent from the original api.py.
    
    This class wraps the new ProtocolService and provides the same interface
    as the original ProtocolComponent.
    """
    
    def __init__(self, 
                 model_component: Optional[ModelComponent] = None, 
                 context_component: Optional[ContextComponent] = None):
        """
        Initialize the ProtocolComponent with model and context components.
        
        Args:
            model_component: Optional model component, will be created if None
            context_component: Optional context component, will be created if None
        """
        if not model_component:
            model_component = ModelComponent()
        
        if not context_component:
            context_component = ContextComponent()
            
        self.model = model_component
        self.context = context_component
        
        # Import here to avoid circular imports
        from llm_service.protocol.service.protocol_service import ProtocolService
        self.service = ProtocolService(model_component, context_component)
    
    def __getattr__(self, name):
        """
        Forward attribute access to the ProtocolService.
        
        Args:
            name: Attribute name
            
        Returns:
            Attribute from the ProtocolService
            
        Raises:
            AttributeError: If the attribute doesn't exist
        """
        if hasattr(self.service, name):
            return getattr(self.service, name)
        raise AttributeError(f"'ProtocolComponent' has no attribute '{name}'")


def create_api_app(protocol_component: ProtocolComponent) -> FastAPI:
    """
    Legacy compatibility function for creating the FastAPI application.
    
    Args:
        protocol_component: ProtocolComponent instance
        
    Returns:
        FastAPI application
    """
    logger.warning(
        "Using deprecated create_api_app function. Consider using create_new_app instead."
    )
    
    # Extract the service from the protocol component
    service = protocol_component.service
    
    # Use the new app factory (import here to avoid circular imports)
    from llm_service.protocol.api.app_factory import create_app as create_new_app
    return create_new_app(
        model_component=protocol_component.model,
        context_component=protocol_component.context,
    )