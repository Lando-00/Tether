from typing import Dict, Any, List
from tether_service.core.factory import load

class ToolRegistry:
    """
    Loads and provides available tools based on config.

    Fail-fast contract: if any *enabled* tool fails to construct, __init__
    raises RuntimeError (chained from the original exception) rather than
    silently skipping the tool. This ensures the service never boots with a
    tool the model believes it has but the registry does not actually hold.
    Tools that are listed in the registry but not in `enabled` are ignored.

    §4 Phase 0A §tooling.
    """
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
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to construct tool '{name}' (impl={impl!r}): {exc}"
                    ) from exc

    def get(self, name: str) -> Any:
        return self.tools.get(name)

    def all(self) -> Dict[str, Any]:
        return self.tools
