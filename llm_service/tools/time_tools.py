# tools/time_tools.py
from datetime import datetime
from zoneinfo import ZoneInfo
import email.utils as eut
from typing import Literal
from . import register_tool

@register_tool
def get_current_time(timezone: str, format: Literal["iso", "rfc2822", "human"] = "iso") -> str:
    """Get the current time for a timezone.
    
    Args:
        timezone: IANA timezone (e.g., Europe/Dublin, America/New_York)
        format: The format for the returned time string.
    """
    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception as e:
        return f"Error: unknown timezone: {timezone}"
        
    if format == "iso":
        return now.isoformat()
    if format == "rfc2822":
        return eut.format_datetime(now)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")