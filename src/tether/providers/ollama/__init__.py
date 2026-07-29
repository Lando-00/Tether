"""Ollama provider package (ADR-0022).

`OllamaProvider` is exposed via lazy ``__getattr__`` so that importing
``tether.providers.ollama`` does not transitively import httpx or pull in
the openai-compat client until they are actually needed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from tether.providers.ollama.provider import OllamaProvider


def __getattr__(name: str):
    if name == "OllamaProvider":
        from tether.providers.ollama.provider import OllamaProvider as _cls

        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["OllamaProvider"]
