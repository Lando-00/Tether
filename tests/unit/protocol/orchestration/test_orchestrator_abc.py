"""Tests for the Orchestrator ABC contract.

Briefing §2 Seam B: ABC has at least two concrete impls so it's not
a single-impl abstraction (R6 anti-overengineering satisfied).
"""
from __future__ import annotations

import pytest

from tether_service.core.interfaces import Orchestrator
from tether_service.protocol.orchestration.chatty import ChattyAgentOrchestrator
from tether_service.protocol.orchestration.notebook import NotebookOrchestrator


def test_orchestrator_abc_is_abstract():
    """Cannot instantiate the bare ABC."""
    with pytest.raises(TypeError):
        Orchestrator()  # type: ignore[abstract]


def test_orchestrator_abc_run_is_abstract():
    """run() is marked abstract."""
    assert hasattr(Orchestrator, "__abstractmethods__")
    assert "run" in Orchestrator.__abstractmethods__


def test_chatty_is_orchestrator():
    """ChattyAgentOrchestrator inherits from Orchestrator ABC."""
    assert issubclass(ChattyAgentOrchestrator, Orchestrator)


def test_notebook_is_orchestrator():
    """NotebookOrchestrator inherits from Orchestrator ABC."""
    assert issubclass(NotebookOrchestrator, Orchestrator)


def test_two_concrete_impls_exist():
    """Briefing §2 Seam B: at least 2 concrete impls (R6 satisfied).

    Counts direct __subclasses__() that have no abstract methods
    remaining (i.e., are fully concrete).
    """
    concrete = [
        cls
        for cls in Orchestrator.__subclasses__()
        if not getattr(cls, "__abstractmethods__", None)
    ]
    assert len(concrete) >= 2, (
        f"Expected >=2 concrete Orchestrator impls, got: "
        f"{[c.__name__ for c in concrete]}"
    )
