
from typing import Annotated, Any, Dict, Literal

from pydantic import Field

from tether.tools.base import BaseTool
from tether.tools.registration import tool


@tool(name="time")
class TimeTool(BaseTool):
    """Get the current time for a timezone in various formats (ISO, RFC2822, or human-readable)."""
    
    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(
        self,
        timezone: Annotated[
            str,
            Field(
                description=(
                    "IANA timezone (e.g., 'Europe/Dublin', 'America/New_York', 'UTC'). "
                    "Defaults to UTC."
                ),
            ),
        ] = "UTC",
        format: Annotated[
            Literal["iso", "rfc2822", "human"],
            Field(
                description=(
                    "Output format - 'iso' (ISO 8601), 'rfc2822' (RFC 2822), "
                    "or 'human' (readable). Defaults to 'human'."
                ),
            ),
        ] = "human",
    ) -> dict:
        """Get the current time for a timezone.

        Style B (synthesis §4 Phase 4 step 43; A2 step 7): parameters use
        ``Annotated[T, Field(...)]`` so :meth:`BaseTool.auto_schema` lifts
        descriptions and constraints into the JSON schema. Pydantic does
        not validate Style B kwargs at invoke-time — that is reserved for
        Style A (the ``Inputs`` ClassVar). Behavior preserved: unknown
        timezones return ``{"time": None, "error": ...}``.
        """
        import datetime
        import email.utils as eut
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo  # type: ignore

        # Normalize common timezone names
        timezone_map = {
            "eastern": "America/New_York",
            "central": "America/Chicago",
            "mountain": "America/Denver",
            "pacific": "America/Los_Angeles",
            "est": "America/New_York",
            "cst": "America/Chicago",
            "mst": "America/Denver",
            "pst": "America/Los_Angeles",
            "uk": "Europe/London",
            "london": "Europe/London",
            "england": "Europe/London",
            "britain": "Europe/London",
        }
        tz_lower = timezone.lower() if isinstance(timezone, str) else ""
        if tz_lower in timezone_map:
            timezone = timezone_map[tz_lower]

        try:
            now = datetime.datetime.now(ZoneInfo(timezone))
        except Exception:
            return {"time": None, "error": f"Unknown timezone: {timezone}"}

        if format == "iso":
            return {"time": now.isoformat()}
        if format == "rfc2822":
            return {"time": eut.format_datetime(now)}

        # Human readable format (default)
        time_str = now.strftime("%I:%M:%S %p")
        date_str = now.strftime("%A, %B %d, %Y")
        zone_str = now.strftime("%Z")
        readable_tz = timezone.replace("_", " ").replace("/", ", ")
        return {"time": f"{time_str} on {date_str} ({zone_str} - {readable_tz})"}