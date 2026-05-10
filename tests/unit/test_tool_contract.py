"""
Tests for the aligned Tool contract.

Acceptance A4: Synthesis §6 row 4 / A2 step 1.

After F4:
- Tool ABC abstractmethod is 'invoke', NOT 'run'
- BaseTool.invoke(args: dict) unpacks into BaseTool.run(**kwargs)
- ToolRunner.run() calls tool.invoke(args), not tool.run(**args)
- All bundled tools are still constructible; schema property returns valid dict
"""
import asyncio
import inspect
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from tether.core.interfaces import Tool
from tether.tools.base import BaseTool
from tether.tools.time_tool import TimeTool
from tether.protocol.orchestration.tool_runner import ToolRunner


# ---------------------------------------------------------------------------
# Helper: minimal concrete tool for contract tests
# ---------------------------------------------------------------------------

class _AddTool(BaseTool):
    """Adds two integers."""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(self, a: int, b: int = 0, **kwargs) -> dict:
        return {"result": a + b}


# ---------------------------------------------------------------------------
# Tool ABC contract
# ---------------------------------------------------------------------------

class TestToolABCContract:
    def test_invoke_is_abstractmethod_on_tool_abc(self):
        """Tool ABC must declare 'invoke' as an abstractmethod."""
        abstract_methods = {
            name for name, method in vars(Tool).items()
            if getattr(method, "__isabstractmethod__", False)
        }
        assert "invoke" in abstract_methods, (
            f"'invoke' not in Tool ABC abstractmethods: {abstract_methods}"
        )

    def test_run_is_not_abstractmethod_on_tool_abc(self):
        """Tool ABC must NOT declare 'run' as an abstractmethod (run lives on BaseTool)."""
        abstract_methods = {
            name for name, method in vars(Tool).items()
            if getattr(method, "__isabstractmethod__", False)
        }
        assert "run" not in abstract_methods, (
            f"'run' should not be in Tool ABC abstractmethods: {abstract_methods}"
        )


# ---------------------------------------------------------------------------
# BaseTool.invoke shim
# ---------------------------------------------------------------------------

class TestBaseToolInvokeShim:
    def test_invoke_same_result_as_run(self):
        """BaseTool.invoke({'a': 3, 'b': 4}) == BaseTool.run(a=3, b=4)."""
        tool = _AddTool()
        via_invoke = asyncio.get_event_loop().run_until_complete(
            tool.invoke({"a": 3, "b": 4})
        )
        via_run = asyncio.get_event_loop().run_until_complete(
            tool.run(a=3, b=4)
        )
        assert via_invoke == via_run == {"result": 7}

    def test_invoke_with_defaults(self):
        """invoke with only required args uses defaults from run()."""
        tool = _AddTool()
        result = asyncio.get_event_loop().run_until_complete(
            tool.invoke({"a": 5})
        )
        assert result == {"result": 5}  # b defaults to 0

    def test_timetool_invoke_and_run_agree(self):
        """TimeTool.invoke({'timezone': 'UTC'}) == TimeTool.run(timezone='UTC')."""
        tool = TimeTool()
        r_invoke = asyncio.get_event_loop().run_until_complete(
            tool.invoke({"timezone": "UTC"})
        )
        r_run = asyncio.get_event_loop().run_until_complete(
            tool.run(timezone="UTC")
        )
        # Both return dicts with a 'time' key; values may differ by milliseconds
        assert "time" in r_invoke
        assert "time" in r_run


# ---------------------------------------------------------------------------
# ToolRunner calls invoke, not run
# ---------------------------------------------------------------------------

class TestToolRunnerUsesInvoke:
    def test_tool_runner_calls_invoke(self):
        """ToolRunner.run() must call tool.invoke(args), not tool.run(**args).

        Phase 4 step 41a: the call now also threads ``context=None`` (or a
        :class:`ToolExecutionContext` when the orchestrator supplies one).
        Synthesis §4 Phase 4 step 41a.
        """
        mock_tool = MagicMock(spec=Tool)
        mock_tool.invoke = AsyncMock(return_value={"ok": True})
        runner = ToolRunner(tools={"my_tool": mock_tool})
        runner.timeout = 5

        result = asyncio.get_event_loop().run_until_complete(
            runner.run("my_tool", {"x": 1})
        )

        mock_tool.invoke.assert_called_once_with({"x": 1}, context=None)
        assert result == {"ok": True}

    def test_tool_runner_raises_on_unknown_tool(self):
        """ToolRunner.run raises ValueError for unknown tool names."""
        runner = ToolRunner(tools={})
        with pytest.raises(ValueError, match="not found"):
            asyncio.get_event_loop().run_until_complete(
                runner.run("nonexistent", {})
            )


# ---------------------------------------------------------------------------
# Bundled tools are still constructible + have valid schemas
# ---------------------------------------------------------------------------

class TestBundledToolsStillWork:
    @pytest.mark.parametrize("tool_cls,registry_name", [
        (TimeTool, "time_tool"),
    ])
    def test_bundled_tool_constructible_with_valid_schema(self, tool_cls, registry_name):
        """Each bundled tool can be constructed and its schema is a non-empty dict."""
        tool = tool_cls()
        schema = tool.schema
        assert isinstance(schema, dict), f"{tool_cls.__name__}.schema is not a dict"
        assert schema, f"{tool_cls.__name__}.schema is empty"
        # Validate expected schema structure
        assert "type" in schema or "function" in schema, (
            f"{tool_cls.__name__}.schema missing 'type' or 'function' key: {schema}"
        )

    def test_weather_tools_constructible(self):
        """WeatherTool and GetForecastTool are constructible."""
        from tether.tools.weather_tool import GetWeatherTool, GetForecastTool
        for cls in (GetWeatherTool, GetForecastTool):
            tool = cls()
            assert isinstance(tool, BaseTool)

    def test_web_search_tool_constructible(self):
        """WebSearchTool is constructible (even without API key)."""
        from tether.tools.web_search_tool import WebSearchTool
        tool = WebSearchTool()
        assert isinstance(tool, BaseTool)
        schema = tool.schema
        assert isinstance(schema, dict)
