"""Engine-level runtime tool enable/disable.

Toggling must take effect on the *next turn* without a restart, because
orchestrators are constructed per turn from ``Engine.enabled_tools()``.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

from tether.engine import Engine
from tether.protocol.parsers.sliding import SlidingParser
from tether.providers.dummy.provider import DummyProvider
from tether.tools.time_tool import TimeTool
from tether.tools.weather_tool import GetWeatherTool


def _engine() -> Engine:
    return Engine(
        provider=DummyProvider(),
        parser=SlidingParser(),
        session_store=AsyncMock(),
        tools={"time": TimeTool(), "weather": GetWeatherTool()},
        system_prompt="You are a helpful assistant.",
    )


def test_all_tools_start_enabled() -> None:
    engine = _engine()
    assert set(engine.enabled_tools()) == {"time", "weather"}
    assert engine.disabled_tool_names() == set()


def test_disabling_hides_a_tool_from_the_model() -> None:
    engine = _engine()

    assert engine.set_tool_enabled("weather", False) is True

    # Gone from what the orchestrator (and therefore the provider) sees...
    assert set(engine.enabled_tools()) == {"time"}
    assert engine.disabled_tool_names() == {"weather"}
    # ...but still constructed, so it can be switched back on and so shutdown
    # still tears it down.
    assert "weather" in engine.tools


def test_re_enabling_restores_the_tool() -> None:
    engine = _engine()
    engine.set_tool_enabled("weather", False)

    assert engine.set_tool_enabled("weather", True) is True

    assert set(engine.enabled_tools()) == {"time", "weather"}
    assert engine.disabled_tool_names() == set()


def test_unknown_tool_is_reported_not_silently_accepted() -> None:
    engine = _engine()
    assert engine.set_tool_enabled("nope", False) is False
    assert engine.disabled_tool_names() == set()


def test_toggling_is_idempotent() -> None:
    engine = _engine()
    engine.set_tool_enabled("time", False)
    engine.set_tool_enabled("time", False)
    assert engine.disabled_tool_names() == {"time"}


def test_list_tool_states_reports_enabled_flag() -> None:
    engine = _engine()
    engine.set_tool_enabled("time", False)

    states = {row["name"]: row for row in engine.list_tool_states()}

    assert states["time"]["enabled"] is False
    assert states["weather"]["enabled"] is True
    # Schema detail is carried through so a client can render the list.
    assert "parameters" in states["weather"]
    assert states["weather"]["description"]


def test_disabled_tool_is_excluded_from_history_view() -> None:
    """The disabled set is what gets handed to get_history(exclude_tools=...)."""
    engine = _engine()
    engine.set_tool_enabled("weather", False)
    assert engine.disabled_tool_names() == {"weather"}
