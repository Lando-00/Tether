"""
Service module for the protocol component.

This module contains:
- ProtocolService: Main service for handling API interactions
- GenerationService: Service for streaming text generation
"""

from llm_service.protocol.service.protocol_service import ProtocolService
from llm_service.protocol.service.generation_service import GenerationService

__all__ = [
    "ProtocolService",
    "GenerationService",
]