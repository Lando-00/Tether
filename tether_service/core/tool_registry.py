from typing import Dict, Any, List
from tether_service.core.factory import load

class ToolRegistry:
    """Loads and provides available tools based on config"""
    def __init__(self, registry_cfg: List[Dict[str, Any]], enabled: List[str]):
        self.tools: Dict[str, Any] = {}
        for tcfg in registry_cfg:
            name = tcfg.get('name')
            if name in enabled:
                impl = tcfg.get('impl', '')
                args = tcfg.get('args', {}) or {}
                try:
                    tool = load(impl, **args)
                    # Set the tool's name to match the registry name
                    if hasattr(tool, '_registry_name'):
                        tool._registry_name = name
                    self.tools[name] = tool
                except Exception:
                    # skip invalid tool
                    continue

    def get(self, name: str) -> Any:
        return self.tools.get(name)

    def all(self) -> Dict[str, Any]:
        return self.tools
