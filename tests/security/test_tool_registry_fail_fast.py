"""
Security tests: ToolRegistry fails fast on construction errors.

§4 Phase 0A §tooling: if an enabled tool fails to construct, ToolRegistry.__init__
must RAISE (not silently skip) so the service does not boot with a ghost tool.
A non-enabled tool's construction failure is irrelevant and must be ignored.
"""

import pytest
from unittest.mock import patch


def _make_registry(registry_cfg, enabled):
    from tether_service.core.tool_registry import ToolRegistry
    return ToolRegistry(registry_cfg, enabled)


def _cfg(name, impl):
    return {"name": name, "impl": impl}


# ---------------------------------------------------------------------------
# Fail-fast: enabled tool that raises in __init__
# ---------------------------------------------------------------------------

def test_enabled_tool_construction_error_raises():
    """ToolRegistry must raise RuntimeError when an enabled tool fails to construct."""
    cfg = [_cfg("bad_tool", "some.module.BadTool")]
    original_error = ValueError("boom — bad config")

    with patch("tether_service.core.tool_registry.load", side_effect=original_error):
        with pytest.raises(RuntimeError) as exc_info:
            _make_registry(cfg, enabled=["bad_tool"])

    err = exc_info.value
    assert "bad_tool" in str(err)
    assert "some.module.BadTool" in str(err)
    # The original exception must be chained (PEP 3134)
    assert err.__cause__ is original_error


def test_enabled_tool_error_is_chained():
    """The RuntimeError must chain the original cause via `raise ... from exc`."""
    cfg = [_cfg("exploding", "a.b.C")]
    original = ImportError("no module named 'a'")

    with patch("tether_service.core.tool_registry.load", side_effect=original):
        with pytest.raises(RuntimeError) as exc_info:
            _make_registry(cfg, enabled=["exploding"])

    assert isinstance(exc_info.value.__cause__, ImportError)


def test_error_message_includes_impl():
    """Error message must include both tool name and impl path."""
    cfg = [_cfg("my_tool", "path.to.MyTool")]

    with patch("tether_service.core.tool_registry.load", side_effect=RuntimeError("bad")):
        with pytest.raises(RuntimeError) as exc_info:
            _make_registry(cfg, enabled=["my_tool"])

    msg = str(exc_info.value)
    assert "my_tool" in msg
    assert "path.to.MyTool" in msg


# ---------------------------------------------------------------------------
# Non-enabled tool construction errors must be IGNORED
# ---------------------------------------------------------------------------

def test_non_enabled_tool_error_is_ignored():
    """If a tool is registered but not enabled, its construction is never attempted."""
    cfg = [
        _cfg("good_tool", "good.module.GoodTool"),
        _cfg("bad_tool", "bad.module.BadTool"),
    ]
    # Patch load to succeed for good_tool and fail for bad_tool.
    # Since bad_tool is not enabled, load should NOT be called for it at all.
    good_instance = object()

    def _fake_load(dotted, **kwargs):
        if "GoodTool" in dotted:
            return good_instance
        raise RuntimeError("should never be called for non-enabled tool")

    with patch("tether_service.core.tool_registry.load", side_effect=_fake_load):
        registry = _make_registry(cfg, enabled=["good_tool"])

    assert registry.get("good_tool") is good_instance
    assert registry.get("bad_tool") is None


# ---------------------------------------------------------------------------
# Happy path: all enabled tools construct successfully
# ---------------------------------------------------------------------------

def test_all_enabled_tools_loaded():
    """When all enabled tools construct cleanly, registry holds them all."""
    cfg = [
        _cfg("tool_a", "a.A"),
        _cfg("tool_b", "b.B"),
        _cfg("tool_c", "c.C"),   # not enabled
    ]
    instances = {name: object() for name in ("tool_a", "tool_b")}

    def _fake_load(dotted, **kwargs):
        for name, inst in instances.items():
            if name.split("_")[1].upper() in dotted:
                return inst
        raise AssertionError(f"unexpected load call for {dotted!r}")

    with patch("tether_service.core.tool_registry.load", side_effect=_fake_load):
        registry = _make_registry(cfg, enabled=["tool_a", "tool_b"])

    assert registry.get("tool_a") is instances["tool_a"]
    assert registry.get("tool_b") is instances["tool_b"]
    assert registry.get("tool_c") is None
    assert set(registry.all().keys()) == {"tool_a", "tool_b"}


def test_empty_registry_config():
    """Empty config must produce an empty registry without error."""
    registry = _make_registry([], enabled=[])
    assert registry.all() == {}


def test_enabled_but_not_in_registry():
    """Enabling a tool not in registry cfg is a no-op (nothing to load)."""
    # load is never called since no cfg entries match the enabled name
    with patch("tether_service.core.tool_registry.load", side_effect=AssertionError("should not be called")):
        registry = _make_registry([], enabled=["ghost_tool"])

    assert registry.get("ghost_tool") is None
