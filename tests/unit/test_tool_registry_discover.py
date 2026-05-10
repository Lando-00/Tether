"""Tests for ToolRegistry's discover-based path + back-compat path.

F5 (synthesis §4 Phase 4 step 42): ToolRegistry now supports two paths:

* **Legacy** (Phase 0A): construct by dotted-path, filter by enabled.
* **Discover** (Phase 4): construct from @tool-decorated classes,
  filter by disabled, validate names via M5.

Both paths must be fail-fast on construction errors. Existing
``test_tool_registry_fail_fast.py`` covers the legacy path; this file
covers the new discover path + the dispatcher.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tether.core.tool_registry import ToolRegistry
from tether.tools.base import BaseTool
from tether.tools.registration import (
    _DECORATED_TOOLS,
    _clear_registry,
    tool,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Save / restore the global decorator registry around each test."""
    saved = dict(_DECORATED_TOOLS)
    _clear_registry()
    try:
        yield
    finally:
        _clear_registry()
        _DECORATED_TOOLS.update(saved)


# ---------------------------------------------------------------------------
# Helper tools
# ---------------------------------------------------------------------------


class _GoodA(BaseTool):
    @property
    def schema(self):
        return {}

    async def run(self):
        return {"a": True}


class _GoodB(BaseTool):
    @property
    def schema(self):
        return {}

    async def run(self):
        return {"b": True}


class _BadInit(BaseTool):
    """A tool whose __init__ raises — for fail-fast tests."""

    def __init__(self):
        raise RuntimeError("bad init in discover path")

    @property
    def schema(self):
        return {}

    async def run(self):
        return {}


# ---------------------------------------------------------------------------
# Discover path
# ---------------------------------------------------------------------------


def test_tool_registry_discover_path():
    """Empty legacy registry → discover path runs; injected discovered
    map produces a registry with those tools."""
    discovered = {"a": _GoodA, "b": _GoodB}
    reg = ToolRegistry(discovered=discovered)
    assert set(reg.all().keys()) == {"a", "b"}
    assert isinstance(reg.get("a"), _GoodA)
    assert isinstance(reg.get("b"), _GoodB)


def test_tool_registry_discover_uses_decorator_name():
    """Discovered + @tool-decorated tools report the registry name via
    :attr:`BaseTool.name` (the marker installed by ``@tool(name=...)``).

    Phase 4 step 43: ``ToolRegistry`` no longer post-hoc injects
    ``_registry_name`` on the instance — the decorator already set the
    class-level marker at definition time.
    """
    @tool(name="alpha")
    class _Alpha(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    reg = ToolRegistry(discovered={"alpha": _Alpha})
    inst = reg.get("alpha")
    assert inst.name == "alpha"
    assert not hasattr(inst, "_registry_name"), (
        "Phase 4 step 43 retired the _registry_name post-hoc injection; "
        "the decorator's class-level marker is now the only naming surface."
    )


def test_tool_registry_disabled_filters():
    """``disabled`` removes the named tool from the constructed registry."""
    discovered = {"a": _GoodA, "b": _GoodB}
    reg = ToolRegistry(disabled=["a"], discovered=discovered)
    assert set(reg.all().keys()) == {"b"}


def test_tool_registry_discover_construction_failure_raises():
    """Discover path is fail-fast: __init__ failure → RuntimeError."""
    with pytest.raises(RuntimeError) as exc_info:
        ToolRegistry(discovered={"bad": _BadInit})
    msg = str(exc_info.value)
    assert "'bad'" in msg
    assert "_BadInit" in msg
    # Original cause is chained.
    assert isinstance(exc_info.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# Legacy path still works (back-compat)
# ---------------------------------------------------------------------------


def test_tool_registry_legacy_path_works():
    """With non-empty registry/enabled, legacy load() path still runs."""
    cfg = [{"name": "x", "impl": "x.X", "args": {}}]
    sentinel = object()

    with patch(
        "tether.core.tool_registry.load",
        return_value=sentinel,
    ):
        reg = ToolRegistry(cfg, ["x"])

    assert reg.get("x") is sentinel
    assert "x" not in [k for k in reg.all().keys() if k != "x"]


def test_tool_registry_legacy_construction_failure_still_raises():
    """Legacy path's fail-fast (Phase 0A) still works."""
    cfg = [{"name": "boom", "impl": "x.B"}]
    with patch(
        "tether.core.tool_registry.load",
        side_effect=ValueError("legacy boom"),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            ToolRegistry(cfg, ["boom"])
    assert "boom" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValueError)


# ---------------------------------------------------------------------------
# from_settings dispatcher
# ---------------------------------------------------------------------------


def test_from_settings_legacy_when_registry_non_empty():
    """from_settings uses legacy path when settings.tools.registry is non-empty."""
    sentinel = object()

    class _ToolSpec:
        def __init__(self, name, impl, args):
            self.name = name
            self.impl = impl
            self.args = args

    tools_settings = SimpleNamespace(
        registry=[_ToolSpec("legacy", "x.X", {})],
        enabled=["legacy"],
        disabled=[],
    )
    # Phase 7 step 78: from_settings now takes a full Settings-like object.
    settings = SimpleNamespace(tools=tools_settings)

    with patch(
        "tether.core.tool_registry.load",
        return_value=sentinel,
    ):
        reg = ToolRegistry.from_settings(settings)

    assert reg.get("legacy") is sentinel


def test_from_settings_discover_when_registry_empty():
    """from_settings uses discover path when settings.tools.registry is empty."""
    @tool(name="auto1")
    class _Auto1(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    @tool(name="auto2")
    class _Auto2(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    tools_settings = SimpleNamespace(
        registry=[],
        enabled=[],
        disabled=[],
    )
    # Phase 7 step 78: from_settings now takes a full Settings-like object.
    settings = SimpleNamespace(tools=tools_settings)

    reg = ToolRegistry.from_settings(settings)
    assert "auto1" in reg.all()
    assert "auto2" in reg.all()


def test_from_settings_discover_honors_disabled():
    """from_settings discover path filters out names in disabled list."""
    @tool(name="keep_me")
    class _Keep(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    @tool(name="drop_me")
    class _Drop(BaseTool):
        @property
        def schema(self):
            return {}

        async def run(self):
            return {}

    tools_settings = SimpleNamespace(
        registry=[],
        enabled=[],
        disabled=["drop_me"],
    )
    # Phase 7 step 78: from_settings now takes a full Settings-like object.
    settings = SimpleNamespace(tools=tools_settings)
    reg = ToolRegistry.from_settings(settings)
    assert "keep_me" in reg.all()
    assert "drop_me" not in reg.all()


# ---------------------------------------------------------------------------
# M5 validate_unique_names integration
# ---------------------------------------------------------------------------


def test_tool_registry_validates_unique_names():
    """Two @tool(name='x') on different classes raise ValueError at the
    decorator step (the validator is the contractual safety net for the
    Phase 4.5 connector path; the decorator catches in-tree collisions
    earlier and more clearly)."""
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


def test_tool_registry_calls_validate_unique_names_with_forbidden(monkeypatch):
    """ToolRegistry calls validate_unique_names. Verified by patching the
    validator and asserting it sees the constructed tools dict.
    """
    captured = {}

    def _spy(items, *, require_prefix=None, forbidden=()):
        captured["items"] = dict(items)
        captured["forbidden"] = tuple(forbidden)

    monkeypatch.setattr(
        "tether.core.tool_registry.validate_unique_names", _spy
    )

    ToolRegistry(discovered={"a": _GoodA, "b": _GoodB})
    assert set(captured["items"].keys()) == {"a", "b"}
    assert captured["forbidden"] == ()
