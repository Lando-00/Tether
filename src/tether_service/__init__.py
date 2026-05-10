"""Backward-compatibility alias for the renamed ``tether_service`` package.

The package was renamed to ``tether`` in Phase 8 of the refactor (synthesis
§4 step 80). This module makes the legacy import path keep working for one
release cycle so external callers — and any forgotten internal references —
keep functioning.

How it works
------------
We install a single :class:`importlib.abc.MetaPathFinder` that forwards
every ``tether_service.*`` import to the real ``tether.*`` module and
caches the resulting module object under *both* names in :data:`sys.modules`.
Because the same module object is reused, identity checks like
``isinstance(x, tether_service.tools.base.BaseTool)`` work correctly even
when the type was originally defined under ``tether``.

A single :class:`DeprecationWarning` is emitted the first time the alias
is touched. We don't emit per-submodule warnings because nesting imports
would spam the user.

Citations: _synthesis.md §4 Phase 8 step 81 (transitional alias).
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings


_PREFIX = "tether_service"
_TARGET_PREFIX = "tether"
_DEPRECATION_MSG = (
    "tether_service.* is deprecated; import from tether.* instead. "
    "The alias will be removed in a future release."
)
_warned = False


def _emit_warning_once() -> None:
    global _warned
    if _warned:
        return
    _warned = True
    # stacklevel=4 puts the warning on the caller's import statement,
    # not deep inside importlib internals.
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=4)


def _legacy_to_real(name: str) -> str:
    if name == _PREFIX:
        return _TARGET_PREFIX
    return _TARGET_PREFIX + name[len(_PREFIX):]


class _TetherServiceAliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``tether_service.*`` to the matching ``tether.*`` module."""

    def find_spec(
        self,
        fullname: str,
        path: list[str] | None = None,
        target: object | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != _PREFIX and not fullname.startswith(_PREFIX + "."):
            return None
        # If we already aliased this module, hand its spec back so the
        # import machinery short-circuits to the cached object.
        existing = sys.modules.get(fullname)
        if existing is not None and getattr(existing, "__spec__", None) is not None:
            return existing.__spec__

        spec = importlib.util.spec_from_loader(fullname, self, is_package=True)
        return spec

    # ---- Loader protocol -------------------------------------------------

    def create_module(self, spec: importlib.machinery.ModuleSpec):
        _emit_warning_once()
        target_name = _legacy_to_real(spec.name)
        target = importlib.import_module(target_name)
        # Cache under the legacy name so any subsequent lookups (including
        # the import machinery's own post-load sys.modules check) find the
        # same module object.
        sys.modules[spec.name] = target
        return target

    def exec_module(self, module) -> None:
        # The real module was already executed when imported under its
        # canonical ``tether.*`` name; nothing more to do.
        return None


_finder = _TetherServiceAliasFinder()
if not any(isinstance(f, _TetherServiceAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _finder)


# Replace this in-flight ``tether_service`` module with the real ``tether``
# package so ``import tether_service`` and ``import tether`` return the
# same object. Must come AFTER the finder install so submodule lookups
# (triggered by tether's own ``__init__``) are routed through the alias.
_tether = importlib.import_module(_TARGET_PREFIX)
sys.modules[_PREFIX] = _tether
_emit_warning_once()
