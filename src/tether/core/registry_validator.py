"""Shared registry-name validator (synthesis §13.4 M5).

A pure, dependency-free helper used by:

* :class:`tether.core.tool_registry.ToolRegistry` (Phase 4) — checks
  that no in-tree tool name collides with a forbidden set (empty in Phase 4;
  populated with connector-prefixed names by Phase 4.5).
* (future) ``ConnectorRegistry`` (Phase 4.5 — connector spec §3.3) — enforces
  that every connector tool's name starts with the connector's
  ``cid_`` prefix, and that none collide with the in-tree tool registry.

Splitting this out as a single function (no class hierarchy) keeps the
contract obvious: "one place that says what a valid set of registered names
looks like". Per R6 anti-overengineering: ABCs are reserved for true ≥2-impl
seams, not for one-callsite helpers.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional


def validate_unique_names(
    items: Mapping[str, Any],
    *,
    require_prefix: Optional[str] = None,
    forbidden: Iterable[str] = (),
) -> None:
    """Validate a mapping of registered names against three constraints.

    Args:
        items: Mapping whose keys are the registered names. Values are
            opaque (the validator never inspects them). Mappings cannot
            contain duplicate keys, so the duplicate check is implicit;
            kept in the signature for future tuple-of-tuples callers
            (synthesis §13.4 M5).
        require_prefix: If given, every key must start with this prefix.
            Used by Phase 4.5 ``ConnectorRegistry`` to enforce
            ``"<cid>_"`` prefixing per connector spec §3.3.
        forbidden: Names that MUST NOT appear in ``items``. Phase 4
            passes an empty set; Phase 4.5 passes the existing tool
            registry's names so connector tools cannot shadow in-tree
            tools.

    Raises:
        ValueError: With a message naming the offender and (when
            applicable) the required prefix or the forbidden set. The
            error fires at boot time so misconfigurations never reach
            request handling.

    Synthesis §13.4 M5; connector spec §3.3.
    """
    forbidden_set = set(forbidden)

    for name in items.keys():
        if require_prefix is not None and not name.startswith(require_prefix):
            raise ValueError(
                f"Registered name {name!r} does not start with required "
                f"prefix {require_prefix!r}."
            )
        if name in forbidden_set:
            raise ValueError(
                f"Registered name {name!r} is forbidden "
                f"(forbidden set: {sorted(forbidden_set)!r})."
            )


__all__ = ["validate_unique_names"]
