"""Tether — local LLM inference with function calling.

Public API:
    from tether_service import Engine, Settings, load_settings

Importing this package MUST NOT pull in FastAPI, MLC, or Brave. Every
concrete provider/parser/store/tool is lazy-imported inside the consumer
(e.g., ``Engine.from_settings``). Per _synthesis.md §4 Phase 2 step 22
(library-first import surface).
"""
from tether_service.config import Settings, load_settings
from tether_service.engine import Engine

__all__ = ["Engine", "Settings", "load_settings"]
