"""ToolRegistry — bridges Settings → constructed tools.

Two operating modes:

1. **Legacy** (Phase 0A): caller passes ``registry_cfg`` (list of
   ``{name, impl, args}``) and ``enabled`` (list of names). Each enabled
   tool is loaded via dotted-path. Construction failures raise
   :class:`RuntimeError` so the service never boots with a tool the
   model believes it has but the registry does not actually hold.

2. **Discover** (Phase 4): caller passes ``disabled`` (list of names);
   tools are auto-discovered via the ``@tool`` decorator + entry-points
   and filtered against the disabled list. Same fail-fast semantics:
   construction failures raise :class:`RuntimeError`.

The :meth:`from_settings` factory dispatches between the two paths
based on whether ``settings.tools.registry`` is non-empty (legacy) or
empty (discover). This lets the existing ``default.yml`` continue to
work for one cycle while we migrate it.

Synthesis §4 Phase 0A §tooling (legacy fail-fast contract); §4 Phase 4
step 42 + step 41 (discover); §13.4 M5 (validate_unique_names).
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Iterable, List, Optional, Type

from tether_service.core.factory import load
from tether_service.core.registry_validator import validate_unique_names

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Loads and provides available tools.

    Construction is fail-fast: if any *enabled* / *non-disabled* tool
    fails to construct, ``__init__`` raises :class:`RuntimeError`
    (chained from the original exception) rather than silently skipping
    the tool. This ensures the service never boots with a tool the
    model believes it has but the registry does not actually hold.

    Synthesis §4 Phase 0A §tooling.
    """

    def __init__(
        self,
        registry_cfg: Optional[List[Dict[str, Any]]] = None,
        enabled: Optional[List[str]] = None,
        *,
        disabled: Optional[List[str]] = None,
        discovered: Optional[Dict[str, Type]] = None,
    ):
        """Build a registry from either the legacy or the discover path.

        Args:
            registry_cfg: Legacy path. List of ``{name, impl, args}``
                dicts. When non-empty, the legacy load-by-dotted-path
                code runs. Pass ``None`` or ``[]`` to use discover.
            enabled: Legacy path. Names of tools to instantiate.
            disabled: Discover path. Names of decorated tools to skip.
                Defaults to no exclusions.
            discovered: Discover path. Pre-resolved
                ``{name: Type}`` mapping. When ``None`` (default), the
                registry calls :func:`discover` itself; tests inject a
                synthetic mapping to bypass package walking.

        Both paths construct each tool, set ``_registry_name`` on the
        instance (so ``BaseTool.name`` returns the registered name), and
        raise :class:`RuntimeError` chained from the original failure
        when a tool's ``__init__`` raises.
        """
        self.tools: Dict[str, Any] = {}

        if registry_cfg:
            self._build_from_legacy(registry_cfg, enabled or [])
        else:
            self._build_from_discover(
                disabled=disabled or [],
                discovered=discovered,
            )

        # Phase 4: forbidden=set() — Phase 4.5 will pass connector tool
        # names to enforce the connector spec §3.3 namespacing rule.
        # Mappings cannot have duplicate keys, so the duplicate check is
        # a no-op for this caller; included to exercise the M5 contract.
        validate_unique_names(self.tools, forbidden=())

    # ------------------------------------------------------------------
    # Legacy path
    # ------------------------------------------------------------------

    def _build_from_legacy(
        self, registry_cfg: List[Dict[str, Any]], enabled: List[str]
    ) -> None:
        """Phase 0A path: load by dotted path; only build entries in
        ``enabled``."""
        enabled_set = set(enabled)
        for tcfg in registry_cfg:
            name = tcfg.get("name")
            if name not in enabled_set:
                continue
            impl = tcfg.get("impl", "")
            args = tcfg.get("args", {}) or {}
            try:
                instance = load(impl, **args)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to construct tool {name!r} (impl={impl!r}): {exc}"
                ) from exc
            self._register(name, instance)

    # ------------------------------------------------------------------
    # Discover path (Phase 4 step 42)
    # ------------------------------------------------------------------

    def _build_from_discover(
        self,
        disabled: Iterable[str],
        discovered: Optional[Dict[str, Type]] = None,
    ) -> None:
        """Phase 4 path: instantiate every @tool-decorated class except
        those in ``disabled``."""
        if discovered is None:
            from tether_service.tools.registration import discover
            discovered = discover()

        disabled_set = set(disabled)
        for name, cls in discovered.items():
            if name in disabled_set:
                logger.info("ToolRegistry: skipping disabled tool %r", name)
                continue
            try:
                instance = self._instantiate(cls)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to construct tool {name!r} "
                    f"({cls.__module__}.{cls.__name__}): {exc}"
                ) from exc
            self._register(name, instance)

    @staticmethod
    def _instantiate(cls: Type) -> Any:
        """Instantiate a discovered tool class.

        Phase 4 keeps this simple: discovered tools are constructed with
        no arguments. Per-tool configuration is read from environment
        variables or the typed Settings sub-models (e.g.,
        ``tools.web_search`` already parses the Brave subtree). A future
        cycle may add a per-tool args mapping to ``ToolsSettings`` if a
        new tool needs YAML-driven construction args.
        """
        sig = inspect.signature(cls.__init__)
        # Filter to no-arg-or-defaults-only call.
        return cls()

    # ------------------------------------------------------------------
    # Common registration helper
    # ------------------------------------------------------------------

    def _register(self, name: str, instance: Any) -> None:
        if hasattr(instance, "_registry_name"):
            instance._registry_name = name
        self.tools[name] = instance

    # ------------------------------------------------------------------
    # Public API (unchanged)
    # ------------------------------------------------------------------

    def get(self, name: str) -> Any:
        return self.tools.get(name)

    def all(self) -> Dict[str, Any]:
        return self.tools

    @classmethod
    def from_settings(cls, tools_settings) -> "ToolRegistry":
        """Build a ToolRegistry from typed ``ToolsSettings``.

        Dispatch:

        * If ``tools_settings.registry`` is non-empty, use the legacy
          path (load by dotted-path; filter by ``enabled``). This keeps
          the existing ``default.yml`` working during the transition.
        * Otherwise, use the discover path (auto-discover via @tool +
          entry-points; filter by ``disabled``).
        """
        if tools_settings.registry:
            registry_cfg = [
                {"name": t.name, "impl": t.impl, "args": t.args}
                for t in tools_settings.registry
            ]
            return cls(registry_cfg, list(tools_settings.enabled))

        disabled = list(getattr(tools_settings, "disabled", []) or [])
        return cls(disabled=disabled)
