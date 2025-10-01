"""
API module for the protocol component.

This module contains:
- API schemas: Pydantic models for API requests and responses
- FastAPI application factory and router configuration
"""

from llm_service.protocol.api.schemas import (
    Message,
    CreateSessionResponse,
    GenerateRequest,
    GenerateResponse,
    UnloadModelRequest,
)
from llm_service.protocol.api.app_factory import create_app as create_new_app

__all__ = [
    "Message",
    "CreateSessionResponse",
    "GenerateRequest",
    "GenerateResponse",
    "UnloadModelRequest",
    "create_new_app",
]