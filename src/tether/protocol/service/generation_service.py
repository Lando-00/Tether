"""Deprecated alias for :class:`tether.engine.Engine`.

Kept for one cycle to avoid breaking downstream callers. Removal scheduled
for Phase 8 (per _synthesis.md §4 Phase 2 step 22 / §4 Phase 8 cleanup).
"""
import warnings
from typing import Dict

from tether.core.interfaces import (
    ModelProvider,
    SessionStore,
    StreamParser,
    Tool,
)
from tether.engine import Engine


class GenerationService(Engine):
    """Deprecated alias for :class:`tether.engine.Engine`.

    Construction emits a ``DeprecationWarning``. All methods are inherited
    from ``Engine``. New code should use ``Engine`` (or
    ``Engine.from_settings``) directly.
    """

    def __init__(
        self,
        provider: ModelProvider,
        parser: StreamParser,
        session_store: SessionStore,
        tools: Dict[str, Tool],
        system_prompt: str,
    ):
        warnings.warn(
            "GenerationService is deprecated; use tether.Engine "
            "(or Engine.from_settings) instead. Removal scheduled for Phase 8.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            provider=provider,
            parser=parser,
            session_store=session_store,
            tools=tools,
            system_prompt=system_prompt,
        )
