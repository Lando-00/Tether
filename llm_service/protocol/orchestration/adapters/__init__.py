"""
This module defines adapter classes for the LLM service protocol.

It exports the following adapters:
- ModelTokenSource: Adapts model.generate output stream to TokenDelta objects
- ContextHistoryWriter: Adapts context component to HistoryWriter interface
"""

from llm_service.protocol.orchestration.adapters.model_token_source import ModelTokenSource
from llm_service.protocol.orchestration.adapters.context_history import ContextHistoryWriter

__all__ = [
    "ModelTokenSource",
    "ContextHistoryWriter"
]