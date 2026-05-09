"""Phase 4.5 follow-up (rubber-duck consensus, xhigh CONCERN):
verify ``discover()`` is resilient to per-module import failures.

Before the fix, the in-tree pkgutil walk caught ``ImportError`` only,
so a buggy module that raised e.g. ``RuntimeError`` at top-level would
crash discovery and prevent every other tool from being registered.
The entry-point branch already used the broader ``except Exception``;
this PR mirrors that for the in-tree walk so failures are isolated.

Synthesis §4 Phase 4 step 42; rubber-duck consensus xhigh.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from tether_service.tools.registration import (
    _DECORATED_TOOLS,
    _clear_registry,
    discover,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Save / restore the global registry around each test."""
    saved = dict(_DECORATED_TOOLS)
    _clear_registry()
    try:
        yield
    finally:
        _clear_registry()
        _DECORATED_TOOLS.update(saved)


# ---------------------------------------------------------------------------
# In-tree walk resilience
# ---------------------------------------------------------------------------


def test_discover_skips_buggy_in_tree_module(tmp_path: Path, caplog) -> None:
    """A module that raises ``RuntimeError`` at top-level (a non-Import
    exception) must NOT crash ``discover()``; the package walk skips
    that module, logs a warning, and still registers the surviving
    tools in the same package.

    Reproduces the original bug: before the fix, the in-tree branch
    caught ``ImportError`` only, so a bare ``RuntimeError`` propagated
    out of ``discover()`` and aborted the entire registration pass.
    """
    # Build a temp package on disk with one buggy module + one good module.
    pkg_dir = tmp_path / "fixture_resilience_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text(
        '"""Resilience fixture package."""\n', encoding="utf-8"
    )
    # Buggy module: raises at import time with a non-ImportError.
    (pkg_dir / "buggy.py").write_text(
        textwrap.dedent(
            '''
            """Buggy module that raises RuntimeError at top-level."""
            raise RuntimeError("intentional import-time failure")
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    # Good module: a decorated tool that should land in the registry.
    (pkg_dir / "good.py").write_text(
        textwrap.dedent(
            '''
            """Good fixture module — registers a tool via @tool()."""
            from tether_service.tools.base import BaseTool
            from tether_service.tools.registration import tool


            @tool(name="resilience_good")
            class _GoodTool(BaseTool):
                @property
                def schema(self):
                    return {}

                async def run(self):
                    return {}
            '''
        ).lstrip(),
        encoding="utf-8",
    )

    # Make the temp package importable.
    sys.path.insert(0, str(tmp_path))
    try:
        with caplog.at_level("WARNING"):
            result = discover(["fixture_resilience_pkg"])

        # discover() must NOT raise — the buggy module is logged + skipped.
        # The good module's tool must be registered.
        assert "resilience_good" in result, (
            f"Expected 'resilience_good' to register despite buggy sibling; "
            f"got {sorted(result.keys())}"
        )
        # Warning surfaces the offending module name + exception.
        warning_msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any(
            "fixture_resilience_pkg.buggy" in msg or "buggy" in msg
            for msg in warning_msgs
        ), (
            f"Expected a warning naming the buggy module; got "
            f"{warning_msgs!r}"
        )
    finally:
        sys.path.remove(str(tmp_path))
        # Drop the temp package's modules from sys.modules so other
        # tests don't see stale state. ``buggy`` may not be in
        # sys.modules if its import raised — guard with pop(...).
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("fixture_resilience_pkg"):
                sys.modules.pop(mod_name, None)


# ---------------------------------------------------------------------------
# Entry-point resilience (parallel to the in-tree case)
# ---------------------------------------------------------------------------


def test_discover_skips_buggy_entry_point(caplog) -> None:
    """A failing entry-point ``load()`` is logged + skipped; other entry
    points still register. The entry-point branch already used the
    broader ``except Exception`` before this PR — this test pins the
    behaviour so any future regression surfaces.
    """
    from tether_service.tools.base import BaseTool

    class _GoodEPTool(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    class _BadEP:
        name = "buggy_ep"

        def load(self):
            raise RuntimeError("intentional EP load failure")

    class _GoodEP:
        name = "good_ep"

        def load(self):
            return _GoodEPTool

    # Monkeypatch entry_points to return one bad + one good entry.
    import importlib.metadata as md

    original = md.entry_points
    try:
        md.entry_points = lambda group=None: (
            [_BadEP(), _GoodEP()] if group == "tether.tools" else []
        )
        _clear_registry()
        with caplog.at_level("WARNING"):
            result = discover([])
    finally:
        md.entry_points = original

    assert "good_ep" in result
    assert "buggy_ep" not in result
    warning_msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("buggy_ep" in msg for msg in warning_msgs), (
        f"Expected a warning naming the buggy EP; got {warning_msgs!r}"
    )
