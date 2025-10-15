# tools/time_tools.py
from datetime import datetime
from zoneinfo import ZoneInfo
import email.utils as eut
from typing import Literal, Optional
from . import register_tool

@register_tool
def get_current_time(timezone: Optional[str] = "UTC", format: Literal["iso", "rfc2822", "human"] = "human") -> str:
    """Get the current time for a timezone.
    
    Args:
        timezone: IANA timezone (e.g., Europe/Dublin, America/New_York, UTC). Defaults to UTC if not provided.
        format: The format for the returned time string. (e.g., "iso", "rfc2822", "human")
    """
    # Default to UTC if timezone is None or empty
    if not timezone:
        timezone = "UTC"
    
    # Convert common timezone references to proper IANA names
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
    
    # Try to normalize timezone if it's a common name
    timezone_lower = timezone.lower() if isinstance(timezone, str) else ""
    if timezone_lower in timezone_map:
        timezone = timezone_map[timezone_lower]
        
    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception as e:
        return f"Error: unknown timezone: {timezone}"
        
    if format == "iso":
        return now.isoformat()
    if format == "rfc2822":
        return eut.format_datetime(now)
    
    # Human readable format (default)
    time_str = now.strftime("%I:%M:%S %p")
    date_str = now.strftime("%A, %B %d, %Y")
    zone_str = now.strftime("%Z")
    
    # Add more readable timezone name
    readable_tz = timezone.replace("_", " ").replace("/", ", ")
    return f"{time_str} on {date_str} ({zone_str} - {readable_tz})"