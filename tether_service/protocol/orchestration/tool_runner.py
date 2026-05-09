import asyncio
from typing import Any, Dict

from tether_service.core.config import load_settings
from tether_service.core.interfaces import Tool


class ToolRunner:
    """Execute tools with timeout"""

    def __init__(self, tools: Dict[str, Tool]):
        self.settings = load_settings()
        limits = self.settings.get("limits", {})
        self.timeout = limits.get("tool_timeout_sec", 5)
        self.tools = tools

    async def run(self, name: str, args: Dict[str, Any]) -> Any:
        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        # Synthesis §6 row 4: orchestrator-facing API is invoke(args).
        # BaseTool.invoke unpacks the dict into kwargs for the typed run() method.
        return await asyncio.wait_for(tool.invoke(args), timeout=self.timeout)