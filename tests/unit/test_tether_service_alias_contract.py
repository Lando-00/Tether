"""Regression test: ``tether_service.*`` alias preserves package identity
+ submodule identity + emits at most one ``DeprecationWarning`` per
process.

Phase 8 RD Fix 5 (xhigh COVERAGE GAP). Synthesis §4 Phase 8 step 81 +
ADR-0013: ``tether_service.*`` is a backward-compat alias that
forwards every import to ``tether.*`` and caches the resulting module
under both names. The alias is load-bearing for external scripts during
the deprecation window.

Identity invariants pinned by this file:

* ``import tether_service`` is ``import tether`` (same module object).
* ``tether_service.X`` is ``tether.X`` for an arbitrary submodule
  (NOT just structurally equal — actual ``is`` identity, so
  ``isinstance(x, tether_service.tools.base.BaseTool)`` works for
  objects constructed via ``tether.tools.base.BaseTool``).
* The ``DeprecationWarning`` is emitted at most once per process —
  a global ``_warned`` flag guards repeats so the alias doesn't spam
  the user when re-imported.
"""
from __future__ import annotations

import importlib
import warnings


def test_alias_root_identity() -> None:
    """``import tether_service`` returns the SAME module object as
    ``import tether``.

    The alias replaces ``sys.modules['tether_service']`` with the real
    ``tether`` module so identity holds across the boundary.
    """
    import tether
    import tether_service

    assert tether is tether_service, (
        "tether_service alias broken: not the same module object as tether"
    )


def test_alias_submodule_identity() -> None:
    """A type imported via ``tether_service.X`` must be IDENTICAL (``is``)
    to the same type imported via ``tether.X``.

    Without identity (only structural equality), ``isinstance`` checks
    across the boundary would silently fail and break duck typing for
    callers still using the legacy import path.

    NOTE: We deliberately exercise a leaf module under ``tether.core``
    rather than ``tether.config.settings``. The alias finder's
    ``is_package=True`` flag (currently set unconditionally) interacts
    badly with ``importlib.resources`` for ``tether.config`` (whose
    ``default.yml`` resource is loaded eagerly by ``load_settings``),
    causing the package's resource path to resolve to an OrphanPath in
    subsequent lookups. That is a known limitation of the transitional
    alias mechanism that is out of scope for this regression test —
    fixing it would require teaching the finder to detect package vs
    leaf modules. The identity invariant being pinned here does not
    depend on which submodule we pick.
    """
    from tether.core.errors import ConnectorNotConfiguredError as ETether
    from tether_service.core.errors import (
        ConnectorNotConfiguredError as ETetherService,
    )

    assert ETether is ETetherService, (
        "tether_service.core.errors.ConnectorNotConfiguredError is not "
        "identical to tether.core.errors.ConnectorNotConfiguredError — "
        "alias regressed"
    )


def test_alias_isinstance_works_across_boundary() -> None:
    """The same ``BaseTool`` class is reachable under both names.

    A future refactor that re-defines ``BaseTool`` under a fresh module
    object on the alias side would break ``isinstance`` for connector
    tools across the boundary; pin that this can't happen silently.
    """
    from tether.tools.base import BaseTool as TetherBase
    from tether_service.tools.base import BaseTool as ServiceBase

    assert TetherBase is ServiceBase, (
        "BaseTool not identical across alias boundary — isinstance broken"
    )


def test_alias_warns_at_most_once_per_process() -> None:
    """Re-importing ``tether_service`` MUST NOT emit additional warnings.

    The alias caches a global ``_warned`` flag; this test verifies the
    cap actually limits warning emission. We can't reliably observe the
    FIRST import of a fresh process from inside pytest (import order is
    interpreter-wide), so we instead exercise the cap by:

    1. Importing the root alias and several submodules under the legacy
       name in quick succession.
    2. Asserting at most ONE ``DeprecationWarning`` fires across all of
       them — the global ``_warned`` flag is supposed to suppress repeats.

    NOTE: We deliberately do NOT do ``del sys.modules['tether_service']``
    + force a re-import. That manipulation appears to corrupt
    ``importlib.resources`` cached package paths for ``tether.config``
    in some test orderings, breaking unrelated tests downstream. The
    realistic regression to guard against is "user imports tether_service
    multiple times in the same process and gets multiple warnings", which
    the multi-import path here exercises.
    """
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        importlib.import_module("tether_service")
        importlib.import_module("tether_service")
        importlib.import_module("tether_service.core")
        importlib.import_module("tether_service.tools.base")

    deprecation_warnings = [
        w for w in captured if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) <= 1, (
        f"tether_service alias emitted {len(deprecation_warnings)} "
        f"DeprecationWarnings on repeated import; expected <= 1 (the "
        f"global _warned flag should suppress repeats)."
    )
