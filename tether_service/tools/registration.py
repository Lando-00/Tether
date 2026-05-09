"""Decorator-based tool registration + discovery (synthesis §4 Phase 4 step 42).

Replaces the old "list every tool's dotted path in YAML" pattern with a
decorator that authors apply at class definition time:

    from tether_service.tools.base import BaseTool
    from tether_service.tools.registration import tool

    @tool()                         # name = ClassName w/ "Tool" stripped, lowered
    class WebSearchTool(BaseTool):
        ...

    @tool(name="search")            # explicit override
    class WebSearchTool(BaseTool):
        ...

The decorator is a *pure annotator*: it does NOT instantiate the class.
Instantiation happens lazily in :class:`tether_service.core.tool_registry.ToolRegistry`
when ``Engine.from_settings`` runs. That keeps decoration cheap and free
of side effects (no httpx clients, no DB connections at import time —
those belong in :meth:`BaseTool.startup`).

Discovery has two surfaces:

* **In-tree**: walk packages (default ``["tether_service.tools"]``) and
  let import-time decorator side-effects populate ``_DECORATED_TOOLS``.
* **Plugins**: scan ``importlib.metadata.entry_points(group="tether.tools")``
  for installed packages that publish ``tether.tools`` entries. Synthesis
  §13 reserves this group name; future Phase 4.5 connector tools may
  also use it. (No third-party plugin SDK is shipped — see R10 — but
  keeping the entry-point hook in place makes a future on/off switch a
  one-line change.)

The module-level ``_DECORATED_TOOLS`` is a small composition-root convention,
not "global state" in the harmful sense: only ``Engine.from_settings`` reads
it, and tests can use :func:`_clear_registry` for isolation.

Synthesis §4 Phase 4 step 42; A2 step 4 (decorator design).
"""
from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


# Marker attribute set on every @tool-decorated class. Survives
# ``_clear_registry()`` (which only empties ``_DECORATED_TOOLS``), so a
# subsequent ``discover()`` call can re-populate the dict by walking
# already-imported modules and finding marked classes. Defined here as a
# private constant rather than a string literal at the call sites.
_TOOL_MARKER_ATTR = "_tether_tool_registered_name"

# Composition-root registry. Populated by the decorator and by
# ``discover()`` from entry-points; consumed by
# :class:`tether_service.core.tool_registry.ToolRegistry`.
_DECORATED_TOOLS: Dict[str, Type] = {}


def _default_name_from_class(cls: Type) -> str:
    """Derive a registry name from a class name.

    Convention (synthesis §4 Phase 4 step 42 default):

    * Strip a trailing ``"Tool"`` suffix if present.
    * Lowercase the result.

    Examples:

    * ``TimeTool`` → ``"time"``
    * ``WebSearchTool`` → ``"websearch"`` (note: kebab/snake is NOT
      auto-derived; explicit ``name="web_search"`` is preferred when
      the desired registry name does not match the camel-cased class
      stem). The four in-tree tools all use explicit names.

    Authors who want a different name pass ``@tool(name="...")``.
    """
    raw = cls.__name__
    if raw.endswith("Tool"):
        raw = raw[: -len("Tool")]
    return raw.lower()


def tool(*, name: Optional[str] = None) -> Callable[[Type], Type]:
    """Class decorator that registers a :class:`BaseTool` subclass.

    Args:
        name: Explicit registry name. Defaults to
            ``cls.__name__`` with a trailing ``"Tool"`` stripped, lowered.

    Returns:
        The decorated class, unchanged except for a marker attribute
        used by :func:`discover` to re-populate the registry after a
        ``_clear_registry()`` call.

    Side effects:
        Inserts ``cls`` into ``_DECORATED_TOOLS[name]`` at decoration
        time. Raises :class:`ValueError` if a *different* class is
        already registered under ``name`` — re-decorating the same
        class with the same name is idempotent and silent (helpful
        when a module is reloaded during tests).

    Notes:
        The decorator does NOT instantiate the class. Construction is
        deferred to :class:`tether_service.core.tool_registry.ToolRegistry`,
        which runs inside ``Engine.from_settings``. This keeps decoration
        cheap and ensures resource-owning side effects live in
        :meth:`BaseTool.startup`, not at import time.

    Synthesis §4 Phase 4 step 42; A2 step 4.
    """
    def _wrap(cls: Type) -> Type:
        registry_name = name if name is not None else _default_name_from_class(cls)

        existing = _DECORATED_TOOLS.get(registry_name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Tool name collision: {registry_name!r} is already "
                f"registered to {existing.__module__}.{existing.__name__}; "
                f"cannot re-register {cls.__module__}.{cls.__name__}."
            )

        # Mark the class itself so discover() can re-populate the dict
        # after a _clear_registry() call without re-importing modules.
        # Set on __dict__ (not via setattr through inheritance) so that
        # subclasses don't accidentally inherit the parent's marker.
        cls.__dict__.setdefault if False else None  # documentation only
        type.__setattr__(cls, _TOOL_MARKER_ATTR, registry_name)
        _DECORATED_TOOLS[registry_name] = cls
        return cls

    return _wrap


def discover(packages: Optional[List[str]] = None) -> Dict[str, Type]:
    """Scan in-tree packages + entry-points and return all registered tools.

    Args:
        packages: Dotted package names to walk. Defaults to
            ``["tether_service.tools"]``. Tests may pass a
            single-package list to scope discovery (e.g., for fixture
            packages).

    Returns:
        A snapshot copy of ``_DECORATED_TOOLS`` after both surfaces
        (in-tree + entry-points) have populated it. Calling
        ``discover()`` is idempotent: a second call sees the same
        modules in ``sys.modules`` (no re-import side-effects fire),
        scans them for marker-bearing classes, and ``setdefault``s
        them into the registry — so re-discovery after a
        ``_clear_registry()`` reproduces the same set.

    Failure modes:
        Per-package import failures are warned but skipped; the
        function never raises during discovery itself. Callers
        (ToolRegistry) catch tool-construction failures separately.

    Synthesis §4 Phase 4 step 42; §13 (entry_points group reserved).
    """
    if packages is None:
        packages = ["tether_service.tools"]

    for pkg_name in packages:
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as exc:
            # Phase 4.5 follow-up (rubber-duck consensus, xhigh CONCERN):
            # broaden ImportError → Exception so a buggy module that
            # raises e.g. RuntimeError at top-level doesn't crash
            # discovery. Mirrors the entry-point branch below
            # (``_load_entry_point_tools``) which already used the broader
            # catch. Synthesis §4 Phase 4 step 42.
            logger.warning("discover: cannot import package %r: %s", pkg_name, exc)
            continue

        if hasattr(pkg, "__path__"):
            for _, mod_name, _ in pkgutil.walk_packages(
                pkg.__path__, prefix=pkg_name + "."
            ):
                try:
                    mod = importlib.import_module(mod_name)
                except Exception as exc:
                    # Same Phase 4.5 follow-up: a single buggy module
                    # must not abort the rest of the walk.
                    logger.warning(
                        "discover: cannot import module %r: %s", mod_name, exc
                    )
                    continue
                _scan_module_for_marked_classes(mod)
        else:
            _scan_module_for_marked_classes(pkg)

    _load_entry_point_tools()

    return dict(_DECORATED_TOOLS)


def _scan_module_for_marked_classes(mod) -> None:
    """Re-population path: find @tool-marked classes that aren't in the
    registry yet and insert them via ``setdefault``.

    Used so that ``_clear_registry()`` followed by ``discover()`` works
    even when the tool modules were already imported (cached in
    ``sys.modules``) — re-import wouldn't fire decorators a second time.
    Only checks ``__dict__`` (not inherited markers) so a subclass of a
    decorated class doesn't accidentally re-register under the parent's
    name.
    """
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name, None)
        if not inspect.isclass(obj):
            continue
        if _TOOL_MARKER_ATTR not in obj.__dict__:
            continue
        registry_name = obj.__dict__[_TOOL_MARKER_ATTR]
        _DECORATED_TOOLS.setdefault(registry_name, obj)


def _load_entry_point_tools() -> None:
    """Load tools from packages that advertise the ``tether.tools`` entry
    point group.

    The ``setdefault`` semantics mean that an in-tree decoration always
    wins over an installed plugin with the same name — collisions are
    silent (the plugin is simply not registered). Future hardening can
    surface those collisions via ``validate_unique_names``; Phase 4
    keeps it permissive while the entry-point surface settles.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover (Python <3.10 doesn't ship here)
        return

    try:
        eps = entry_points(group="tether.tools")
    except Exception as exc:
        # Some test environments stub entry_points oddly; warn and move on
        # rather than letting discovery fail.
        logger.warning("discover: entry_points lookup failed: %s", exc)
        return

    for ep in eps:
        try:
            cls = ep.load()
        except Exception as exc:
            logger.warning(
                "discover: entry point %r failed to load: %s", ep.name, exc
            )
            continue
        _DECORATED_TOOLS.setdefault(ep.name, cls)


def _clear_registry() -> None:
    """Empty ``_DECORATED_TOOLS`` for test isolation.

    Does NOT remove the marker attribute from previously-decorated
    classes — re-running ``discover()`` will re-populate the dict
    from those markers. Tests that need a *truly* empty discovery
    surface should also import only fixture modules whose classes
    they control.
    """
    _DECORATED_TOOLS.clear()


__all__ = [
    "tool",
    "discover",
    "_DECORATED_TOOLS",
    "_clear_registry",
]
