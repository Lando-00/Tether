"""Orchestrator strategy registry.

Resolves a mode string ("chat", "research", etc.) to an Orchestrator
class via the dotted impl path in the orchestrator registry dict.

Mirrors the ``tools.registry`` resolution pattern.

Briefing §2 Seam B; synthesis §3.5 (Orchestrator strategy seam).
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Dict, Type

if TYPE_CHECKING:
    from tether.core.interfaces import Orchestrator


class UnknownOrchestratorMode(ValueError):
    """Raised when a requested mode isn't in the registry."""


def resolve_orchestrator_class(
    mode: str,
    registry: Dict[str, str],
) -> Type["Orchestrator"]:
    """Resolve a mode name to its Orchestrator class.

    Args:
        mode: The requested mode (e.g., "chat", "research").
        registry: Dict mapping mode names to dotted impl paths
                  (e.g., {"chat": "pkg.module.Class"}).

    Returns:
        The Orchestrator class (NOT instance — caller constructs).

    Raises:
        UnknownOrchestratorMode: if ``mode`` isn't a key in the
            registry. Subclass of ValueError.
        ImportError: if the dotted path can't be resolved.
        AttributeError: if the module is reachable but the named
            symbol isn't there.
        TypeError: if the resolved symbol isn't an Orchestrator
            subclass.
    """
    if mode not in registry:
        raise UnknownOrchestratorMode(
            f"unknown orchestrator mode: {mode!r}; "
            f"registered: {sorted(registry.keys())}"
        )

    dotted = registry[mode]
    module_path, _, class_name = dotted.rpartition(".")
    if not module_path:
        raise ImportError(
            f"invalid registry entry for mode={mode!r}: {dotted!r}"
        )

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Defense: ensure the resolved class is actually an Orchestrator.
    from tether.core.interfaces import Orchestrator
    if not isinstance(cls, type) or not issubclass(cls, Orchestrator):
        raise TypeError(
            f"registry entry for mode={mode!r} resolves to {cls!r}, "
            f"which is not an Orchestrator subclass"
        )

    return cls


__all__ = ["UnknownOrchestratorMode", "resolve_orchestrator_class"]
