"""Functional tests for :class:`tether_service.tools.time_tool.TimeTool`.

Style B Annotated migration (synthesis §4 Phase 4 step 43; A2 step 7).
Covers:

* Default args (UTC + human format) return a non-empty string.
* IANA timezones (Europe/Dublin, America/New_York, UTC) work.
* The lowercased alias map (eastern, est, pacific, london, ...)
  resolves to canonical IANA names.
* ``format='iso'`` returns ISO-8601, ``'rfc2822'`` returns RFC 2822.
* Unknown timezone returns ``{"time": None, "error": ...}``.
* The schema reports ``timezone`` and ``format`` with descriptions and
  the format enum.
"""
from __future__ import annotations

import re

import pytest

from tether_service.tools.time_tool import TimeTool


@pytest.fixture
def tool() -> TimeTool:
    return TimeTool()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestTimeToolSchema:
    def test_schema_function_name_from_decorator(self, tool: TimeTool):
        assert tool.auto_schema["function"]["name"] == "time"

    def test_schema_lifts_timezone_field_description(self, tool: TimeTool):
        params = tool.auto_schema["function"]["parameters"]["properties"]
        assert "IANA timezone" in params["timezone"]["description"]

    def test_schema_lifts_format_enum(self, tool: TimeTool):
        params = tool.auto_schema["function"]["parameters"]["properties"]
        fmt = params["format"]
        assert fmt["type"] == "string"
        assert set(fmt["enum"]) == {"iso", "rfc2822", "human"}
        assert fmt["default"] == "human"

    def test_schema_required_is_empty(self, tool: TimeTool):
        # Both params have defaults, so neither is required.
        assert tool.auto_schema["function"]["parameters"]["required"] == []


# ---------------------------------------------------------------------------
# Functional behavior
# ---------------------------------------------------------------------------


class TestTimeToolDefaults:
    @pytest.mark.asyncio
    async def test_default_args_human_utc(self, tool: TimeTool):
        result = await tool.invoke({})
        assert isinstance(result, dict)
        assert "time" in result
        assert isinstance(result["time"], str)
        # Human format mentions UTC
        assert "UTC" in result["time"]

    @pytest.mark.asyncio
    async def test_default_args_via_invoke_with_no_args(self, tool: TimeTool):
        """``await tool.invoke({})`` with no args uses both defaults."""
        result = await tool.invoke({})
        assert "time" in result
        assert "error" not in result


class TestTimeToolTimezones:
    @pytest.mark.asyncio
    async def test_iana_europe_dublin(self, tool: TimeTool):
        result = await tool.invoke({"timezone": "Europe/Dublin"})
        assert "time" in result
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_iana_america_new_york(self, tool: TimeTool):
        result = await tool.invoke({"timezone": "America/New_York"})
        assert "time" in result
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_alias_eastern_resolves(self, tool: TimeTool):
        """The 'eastern' alias resolves to America/New_York."""
        result = await tool.invoke({"timezone": "eastern"})
        assert "error" not in result
        assert "time" in result
        # The human-readable string should include the resolved zone
        assert "America" in result["time"] or "New York" in result["time"]

    @pytest.mark.asyncio
    async def test_alias_london_resolves(self, tool: TimeTool):
        result = await tool.invoke({"timezone": "london"})
        assert "error" not in result
        assert "Europe" in result["time"] or "London" in result["time"]

    @pytest.mark.asyncio
    async def test_unknown_timezone_returns_error_dict(self, tool: TimeTool):
        result = await tool.invoke({"timezone": "Nonexistent/Timezone"})
        assert result["time"] is None
        assert "error" in result
        assert "Nonexistent/Timezone" in result["error"]


class TestTimeToolFormats:
    @pytest.mark.asyncio
    async def test_iso_format_returns_iso8601(self, tool: TimeTool):
        result = await tool.invoke({"timezone": "UTC", "format": "iso"})
        # ISO 8601: 2026-05-09T16:38:43.407+00:00
        assert re.match(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
            result["time"],
        ), f"expected ISO-8601 prefix, got {result['time']!r}"

    @pytest.mark.asyncio
    async def test_rfc2822_format(self, tool: TimeTool):
        result = await tool.invoke({"timezone": "UTC", "format": "rfc2822"})
        # RFC 2822 begins with a 3-letter weekday: "Sat, 09 May 2026 ..."
        assert re.match(
            r"^[A-Z][a-z]{2},\s+\d{1,2}\s+\w{3}\s+\d{4}",
            result["time"],
        ), f"expected RFC 2822 shape, got {result['time']!r}"

    @pytest.mark.asyncio
    async def test_human_format_default(self, tool: TimeTool):
        result = await tool.invoke({"timezone": "UTC"})
        # Human format example: "12:34:56 PM on Saturday, May 09, 2026 (UTC - UTC)"
        assert " on " in result["time"]
        assert "(" in result["time"] and ")" in result["time"]
