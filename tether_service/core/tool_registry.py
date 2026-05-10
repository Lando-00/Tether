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

Phase 7 step 78: ``_instantiate`` now injects ``settings`` and
``secrets`` into tool constructors that declare those kwargs (Option A).
Tools opt in by adding the keyword argument to ``__init__``; tools that
don't declare them receive neither. This lets :class:`WebSearchTool`
(and future tools) receive ``settings.security.outbound_allowlist`` at
construction time so policy enforcement is live in production.

Synthesis §4 Phase 0A §tooling (legacy fail-fast contract); §4 Phase 4
step 42 + step 41 (discover); §13.4 M5 (validate_unique_names);
§3 (security) Phase 7 step 78 (settings injection).
"""
from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Type

from tether_service.core.factory import load
from tether_service.core.registry_validator import validate_unique_names

if TYPE_CHECKING:
    from tether_service.config.settings import Settings
    from tether_service.core.secrets import SecretsProvider

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
        settings: Optional["Settings"] = None,
        secrets: Optional["SecretsProvider"] = None,
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
            settings: Optional full :class:`Settings` instance. Injected
                into tool constructors that declare a ``settings`` kwarg
                (Phase 7 step 78 — Option A opt-in). When ``None``, the
                tool's own default applies.
            secrets: Optional :class:`SecretsProvider`. Injected into tool
                constructors that declare a ``secrets`` kwarg. When ``None``,
                the tool's own default applies (typically
                :class:`EnvFileSecretsProvider`).

        Both paths construct each tool and raise :class:`RuntimeError`
        chained from the original failure when a tool's ``__init__``
        raises. Phase 4 step 43: the legacy per-instance registry-name
        injection is retired — the ``@tool(name=...)`` decorator sets
        the registry name at class definition time via the
        ``_tether_tool_registered_name`` class attribute consumed by
        :attr:`BaseTool.name`. Legacy dotted-path tools that aren't
        decorated still get registered under their YAML ``name``; the
        ``BaseTool.name`` property then falls back to the class name
        for those, which is acceptable since the orchestrator looks
        them up by the registry dict key, not by ``tool.name``.
        """
        self._settings = settings
        self._secrets = secrets
        self.tools: Dict[str, Any] = {}

        if registry_cfg is not None:
            # Legacy path — caller provided an explicit registry list (possibly
            # empty). Keeps ``ToolRegistry([], [])`` semantics from Phase 0A:
            # an empty legacy registry is an empty tools dict.
            self._build_from_legacy(registry_cfg, enabled or [])
        else:
            # Discover path — caller passed ``registry_cfg=None`` (the default
            # for ``ToolRegistry.from_settings`` when ``settings.tools.registry``
            # is empty). Tools come from @tool decorations + entry_points.
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
        ``enabled``.

        Phase 7 step 78: ``load`` already filters kwargs to the tool's
        ``__init__`` signature, so passing ``settings`` and ``secrets``
        here is safe — tools that don't declare those kwargs simply won't
        receive them.
        """
        enabled_set = set(enabled)
        for tcfg in registry_cfg:
            name = tcfg.get("name")
            if name not in enabled_set:
                continue
            impl = tcfg.get("impl", "")
            args = tcfg.get("args", {}) or {}
            try:
                instance = load(
                    impl,
                    settings=self._settings,
                    secrets=self._secrets,
                    **args,
                )
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

    def _instantiate(self, cls: Type) -> Any:
        """Instantiate a discovered tool class, injecting well-known kwargs.

        Phase 7 step 78 (Option A): introspects ``cls.__init__`` and passes
        ``settings`` and/or ``secrets`` when declared as keyword arguments.
        Tools opt in by declaring the kwarg; tools that don't declare it
        receive neither — zero changes required for existing tools.

        This closes the BLOCKER where ``ToolRegistry`` was always calling
        ``cls()`` with no args, making ``settings=None`` dead code for every
        tool instantiated through the registry.
        """
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        accepts_var_keyword = any(
            p.kind == p.VAR_KEYWORD for p in params.values()
        )
        kwargs: Dict[str, Any] = {}
        if accepts_var_keyword or "settings" in params:
            kwargs["settings"] = self._settings
        if accepts_var_keyword or "secrets" in params:
            kwargs["secrets"] = self._secrets
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Common registration helper
    # ------------------------------------------------------------------

    def _register(self, name: str, instance: Any) -> None:
        # Phase 4 step 43: no more per-instance registry-name injection.
        # The ``@tool(name=...)`` decorator sets the class-level
        # ``_tether_tool_registered_name`` marker at definition time;
        # ``BaseTool.name`` reads from it. For undecorated legacy
        # dotted-path tools, ``BaseTool.name`` falls back to the class
        # name — that's fine because the orchestrator looks tools up
        # by the registry-dict key (``name`` arg), not by ``tool.name``.
        self.tools[name] = instance

    # ------------------------------------------------------------------
    # Public API (unchanged)
    # ------------------------------------------------------------------

    def get(self, name: str) -> Any:
        return self.tools.get(name)

    def all(self) -> Dict[str, Any]:
        return self.tools

    @classmethod
    def from_settings(cls, settings: "Settings") -> "ToolRegistry":
        """Build a ToolRegistry from a typed ``Settings`` object.

        Dispatch:

        * If ``settings.tools.registry`` is non-empty, use the legacy
          path (load by dotted-path; filter by ``enabled``). This keeps
          the existing ``default.yml`` working during the transition.
        * Otherwise, use the discover path (auto-discover via @tool +
          entry-points; filter by ``disabled``). Note: this is the
          ONLY trigger for the discover path; direct
          ``ToolRegistry([], [])`` calls keep the legacy "empty list →
          empty registry" semantics from Phase 0A.

        Phase 7 step 78: passes ``settings`` to the registry so that tool
        constructors that declare a ``settings`` kwarg receive the full
        policy configuration (e.g., ``settings.security.outbound_allowlist``
        for :class:`~tether_service.tools.web_search_tool.WebSearchTool`).
        """
        tools_settings = settings.tools
        if tools_settings.registry:
            registry_cfg = [
                {"name": t.name, "impl": t.impl, "args": t.args}
                for t in tools_settings.registry
            ]
            return cls(registry_cfg, list(tools_settings.enabled), settings=settings)

        disabled = list(getattr(tools_settings, "disabled", []) or [])
        return cls(registry_cfg=None, disabled=disabled, settings=settings)
