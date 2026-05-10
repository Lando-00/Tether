"""Dotted-path loader used by ``Engine.from_settings`` and ``ToolRegistry``.

Phase 2 cleanup removed the legacy factory class — the typed
``Settings``/``Engine`` pair now owns construction. Only the dotted-path
``load()`` helper remains (no other production code path needs it). Per
_synthesis.md §4 Phase 2 step 24.
"""
from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any


def load(dotted: str, **kwargs: Any) -> Any:
    """Import a dotted path and instantiate the class if callable.

    Filters kwargs to match the constructor signature (unless ``**kwargs`` is
    accepted).
    """
    module, cls = dotted.rsplit(".", 1)
    mod = import_module(module)
    obj = getattr(mod, cls)

    if isinstance(obj, type):
        sig = inspect.signature(obj.__init__)
        params = list(sig.parameters.values())
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params)
        if accepts_kwargs:
            return obj(**kwargs)
        allowed = {
            p.name
            for p in params
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            and p.name != "self"
        }
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return obj(**filtered)

    # callable or object (rare)
    return obj
