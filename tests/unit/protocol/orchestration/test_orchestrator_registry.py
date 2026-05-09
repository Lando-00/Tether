"""Unit tests for orchestrator strategy registry.

Tests resolve_orchestrator_class() in isolation — no engine or HTTP
layer needed. Briefing §2 Seam B; synthesis §3.5.
"""
from __future__ import annotations

import pytest

from tether_service.protocol.orchestration.registry import (
    UnknownOrchestratorMode,
    resolve_orchestrator_class,
)

# Default registry matching OrchestratorSettings defaults.
_DEFAULT_REGISTRY = {
    "chat": "tether_service.protocol.orchestration.chatty.ChattyAgentOrchestrator",
    "research": "tether_service.protocol.orchestration.notebook.NotebookOrchestrator",
}


def test_resolve_chat_returns_chatty():
    from tether_service.protocol.orchestration.chatty import ChattyAgentOrchestrator

    cls = resolve_orchestrator_class("chat", _DEFAULT_REGISTRY)
    assert cls is ChattyAgentOrchestrator


def test_resolve_research_returns_notebook():
    from tether_service.protocol.orchestration.notebook import NotebookOrchestrator

    cls = resolve_orchestrator_class("research", _DEFAULT_REGISTRY)
    assert cls is NotebookOrchestrator


def test_resolve_unknown_raises():
    with pytest.raises(UnknownOrchestratorMode) as exc_info:
        resolve_orchestrator_class("unknown_mode", _DEFAULT_REGISTRY)

    assert "unknown_mode" in str(exc_info.value)
    assert "registered" in str(exc_info.value)


def test_resolve_invalid_dotted_path_raises_importerror():
    """A registry entry with no module part (no '.') raises ImportError."""
    bad_registry = {"chat": "NoModulePath"}
    with pytest.raises(ImportError):
        resolve_orchestrator_class("chat", bad_registry)


def test_resolve_nonexistent_module_raises_importerror():
    """A registry entry pointing to a missing module raises ImportError."""
    bad_registry = {"chat": "tether_service.nonexistent.module.SomeClass"}
    with pytest.raises((ImportError, ModuleNotFoundError)):
        resolve_orchestrator_class("chat", bad_registry)


def test_resolve_non_orchestrator_class_raises_typeerror():
    """A registry entry pointing to a non-Orchestrator class raises TypeError."""
    # Use a real importable class that is NOT an Orchestrator subclass.
    non_orch_registry = {"chat": "tether_service.protocol.parsers.sliding.SlidingParser"}
    with pytest.raises(TypeError) as exc_info:
        resolve_orchestrator_class("chat", non_orch_registry)

    assert "not an Orchestrator subclass" in str(exc_info.value)


def test_unknown_orchestrator_mode_is_valueerror_subclass():
    assert issubclass(UnknownOrchestratorMode, ValueError)
