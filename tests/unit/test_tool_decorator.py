"""Tests for the @tool decorator + discover() module.

Synthesis §4 Phase 4 step 42; A2 step 4 (decorator design).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from tether.tools.base import BaseTool
from tether.tools.registration import (
    _DECORATED_TOOLS,
    _clear_registry,
    discover,
    tool,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Save / restore the global registry around each test so order doesn't matter.

    The decorator and discover both write to ``_DECORATED_TOOLS``; tests
    that mutate it should not leak into siblings.
    """
    saved = dict(_DECORATED_TOOLS)
    _clear_registry()
    try:
        yield
    finally:
        _clear_registry()
        _DECORATED_TOOLS.update(saved)


# ---------------------------------------------------------------------------
# Decorator semantics
# ---------------------------------------------------------------------------


def test_tool_decorator_registers_class():
    """@tool() applied to a fresh BaseTool subclass: registered under
    the default-derived name (class name with 'Tool' stripped, lowered).
    """
    @tool()
    class FreshExperimentTool(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    assert "freshexperiment" in _DECORATED_TOOLS
    assert _DECORATED_TOOLS["freshexperiment"] is FreshExperimentTool


def test_tool_decorator_explicit_name():
    """@tool(name="custom") registers under the explicit name, not the
    default-derived one."""
    @tool(name="custom")
    class _MyTool(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    assert "custom" in _DECORATED_TOOLS
    assert "_my" not in _DECORATED_TOOLS  # default-derived name not used
    assert _DECORATED_TOOLS["custom"] is _MyTool


def test_tool_decorator_default_name_strips_Tool_suffix():
    """@tool() on TimeTool → 'time' (strip 'Tool' suffix; lowercase)."""
    # We use a fresh subclass to avoid touching the real TimeTool registration.
    @tool()
    class TimeTool(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    assert "time" in _DECORATED_TOOLS
    assert _DECORATED_TOOLS["time"] is TimeTool


def test_tool_decorator_collision_raises():
    """Two @tool(name='x') on different classes → ValueError on the second."""
    @tool(name="dup")
    class _A(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    with pytest.raises(ValueError, match="dup"):
        @tool(name="dup")
        class _B(BaseTool):
            @property
            def schema(self):
                return {}

            async def run(self):
                return {}


def test_tool_decorator_re_decoration_idempotent():
    """Re-applying the SAME decorator+name to the SAME class is silent
    (helpful when a module is reloaded in tests)."""
    @tool(name="reapp")
    class _T(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    # Re-apply manually — should NOT raise.
    tool(name="reapp")(_T)
    assert _DECORATED_TOOLS["reapp"] is _T


# ---------------------------------------------------------------------------
# discover() — in-tree walk
# ---------------------------------------------------------------------------


def test_discover_walks_packages():
    """clear registry; call discover(['tests.unit._decorator_fixtures']);
    the fixture-decorated tools (fixture_alpha, fixture_beta) are
    registered.

    A separate integration test in ``test_tool_registry_discover.py``
    asserts the four real in-tree tools (time, weather, forecast,
    web_search) auto-discover after Commit 6.
    """
    # Import the fixture package eagerly so its decorations fire at least
    # once; the test then clears + re-discovers via marker scanning.
    import tests.unit._decorator_fixtures.decorated_tools  # noqa: F401

    _clear_registry()
    result = discover(["tests.unit._decorator_fixtures"])

    expected = {"fixture_alpha", "fixture_beta"}
    assert expected.issubset(result.keys()), (
        f"Expected {expected} in discover result; got {sorted(result.keys())}"
    )


def test_discover_returns_snapshot_dict():
    """discover() returns a dict copy so callers can mutate without
    affecting the global registry."""
    @tool(name="snap")
    class _S(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    result = discover([])  # No package walk; just return current registry.
    assert "snap" in result
    result.pop("snap")
    # Mutating the returned dict must not affect the global registry.
    assert "snap" in _DECORATED_TOOLS


def test_discover_idempotent():
    """Calling discover() twice returns the same dict; doesn't double-register."""
    import tests.unit._decorator_fixtures.decorated_tools  # noqa: F401

    _clear_registry()
    first = discover(["tests.unit._decorator_fixtures"])
    size_after_first = len(_DECORATED_TOOLS)
    second = discover(["tests.unit._decorator_fixtures"])
    size_after_second = len(_DECORATED_TOOLS)

    assert size_after_first == size_after_second
    assert set(first.keys()) == set(second.keys())


def test_discover_re_populates_after_clear():
    """clear → discover repopulates from class markers (no re-import needed)."""
    # Eagerly import fixture so its @tool calls fire.
    import tests.unit._decorator_fixtures.decorated_tools  # noqa: F401

    discover(["tests.unit._decorator_fixtures"])
    assert _DECORATED_TOOLS, "first discover should populate the registry"
    snapshot = set(_DECORATED_TOOLS.keys())

    _clear_registry()
    assert _DECORATED_TOOLS == {}

    discover(["tests.unit._decorator_fixtures"])
    # Modules already in sys.modules; re-population came from class markers.
    assert set(_DECORATED_TOOLS.keys()) >= snapshot


def test_discover_unknown_package_warns_does_not_raise(caplog):
    """A bad package name is logged + skipped, never raised."""
    with caplog.at_level("WARNING"):
        result = discover(["this.package.does.not.exist"])
    assert any("cannot import package" in rec.message for rec in caplog.records)
    # No raise; result is whatever's in the registry (probably empty).
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# discover() — entry points
# ---------------------------------------------------------------------------


def test_discover_loads_entry_points():
    """Monkeypatch entry_points to return a fake EP; discover() loads it
    and registers it under EP.name.
    """
    class _FakeTool(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    class _FakeEP:
        name = "fake_plugin"

        def load(self):
            return _FakeTool

    with patch(
        "tether.tools.registration.entry_points",
        return_value=[_FakeEP()],
        create=True,
    ):
        # Ensure importlib.metadata.entry_points is imported lazily inside
        # _load_entry_point_tools — patch the import path the function uses.
        # (It does `from importlib.metadata import entry_points` at call
        # time; we patch via sys.modules to intercept.)
        import importlib.metadata as md
        original = md.entry_points
        try:
            md.entry_points = lambda group=None: [_FakeEP()] if group == "tether.tools" else []
            _clear_registry()
            result = discover([])  # No package walk; only entry-point load.
        finally:
            md.entry_points = original

    assert "fake_plugin" in result
    assert result["fake_plugin"] is _FakeTool


def test_discover_entry_point_failure_does_not_break_others(caplog):
    """An EP that raises on load is logged + skipped; others still load."""
    class _BadEP:
        name = "bad"

        def load(self):
            raise RuntimeError("plugin import failed")

    class _GoodTool(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    class _GoodEP:
        name = "good"

        def load(self):
            return _GoodTool

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

    assert "good" in result
    assert "bad" not in result
    assert any("bad" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Test-isolation helper
# ---------------------------------------------------------------------------


def test_clear_registry_for_test_isolation():
    """_clear_registry() helper exists and empties _DECORATED_TOOLS."""
    @tool(name="will_clear")
    class _T(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    assert "will_clear" in _DECORATED_TOOLS
    _clear_registry()
    assert _DECORATED_TOOLS == {}
