"""
API schema models for the protocol component.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Message model for API responses."""
    id: int
    role: str
    content: str
    created_at: datetime


class CreateSessionResponse(BaseModel):
    """Response model for session creation."""
    session_id: str
    created_at: datetime


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    session_id: str = Field(...)
    prompt: str = Field(...)
    model_name: str = Field(...)
    device: Optional[str] = "auto"
    dll: Optional[str] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95
    # Add tools parameter for OpenAI-compatible tool definitions
    tools: Optional[List[dict]] = None
    # Stream flag to enable streaming responses
    stream: Optional[bool] = False
    # Stream options for configuring the streaming behavior
    stream_options: Optional[Dict[str, Any]] = None
    # The mcp_server config is now handled by the client, which passes `tools`
    mcp_server: Optional[dict] = None


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    session_id: str
    reply: str
    messages: List[Message]


class UnloadModelRequest(BaseModel):
    """Request model for unloading a model."""
    model_name: str
    device: Optional[str] = "auto"