from abc import abstractmethod
from typing import Dict, Any, Optional
from tether_service.core.interfaces import Tool

# =============================
# Tool Authoring Guidelines
# =============================
#
# To create a new tool:
# 1. Subclass BaseTool and implement the async run() method with explicit, type-annotated arguments.
# 2. Use a Google-style docstring for run() with an Args: section, e.g.:
#
#     async def run(self, timezone: str = "UTC", format: str = "human") -> dict:
#         """
#         Get the current time for a timezone in various formats.
#         Args:
#             timezone: IANA timezone (e.g., Europe/Dublin, America/New_York, UTC). Defaults to UTC if not provided.
#             format: The format for the returned time string. (e.g., "iso", "rfc2822", "human")
#         Returns:
#             dict: {"time": <formatted time string>}
#         """
#         ...
#
# 3. The schema will be generated automatically from the run() signature and docstring.
# 4. The class-level docstring will be used as the tool's description in the schema.
#
# Best Practices:
# - Always provide type hints for all arguments.
# - Use clear, concise parameter descriptions in the Args: section.
# - Document the return value in a Returns: section (optional, for clarity).
# - Avoid *args/**kwargs in run().
# - Use only JSON-serializable types for arguments and return values.

class BaseTool(Tool):
    
    def __init__(self):
        self._registry_name: str | None = None

    @staticmethod
    def _extract_param_descriptions(docstring: str) -> dict:
        """
        Parse the docstring for an Args: section and return a mapping of param name to description.
        Supports Google-style docstrings.
        """
        import re
        if not docstring:
            return {}
        param_desc = {}
        # Google style: Args:
        args_section = re.search(r"Args?:\s*(.*?)(^\S|\Z)", docstring, re.DOTALL | re.MULTILINE)
        if args_section:
            args_text = args_section.group(1)
            for line in args_text.splitlines():
                match = re.match(r"\s*(\w+)\s*:\s*(.*)", line)
                if match:
                    name, desc = match.groups()
                    param_desc[name] = desc.strip()
        return param_desc

    @property
    def auto_schema(self) -> Dict[str, Any]:
        import inspect
        from typing import get_type_hints
        sig = inspect.signature(self.run)
        hints = get_type_hints(self.run)
        docstring = self.run.__doc__ or self.__doc__ or ""
        param_docs = self._extract_param_descriptions(docstring)
        params = {}
        required = []
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            param_type = hints.get(name, str)
            # Map Python types to JSON schema types
            if param_type is str:
                typ = 'string'
            elif param_type is int:
                typ = 'integer'
            elif param_type is float:
                typ = 'number'
            elif param_type is bool:
                typ = 'boolean'
            else:
                typ = 'string'
            params[name] = {
                'type': typ,
                'description': param_docs.get(name, '')
            }
            if param.default is inspect.Parameter.empty:
                required.append(name)
        return self.build_schema(
            function_name=self.name,
            description=self.__doc__ or '',
            parameters=params,
            required=required
        )

    @property
    def name(self) -> str:
        # Use registry name if set, otherwise fall back to class name
        if self._registry_name:
            return self._registry_name
        return self.__class__.__name__


    @staticmethod
    def build_schema(
        function_name: str,
        description: str,
        parameters: dict,
        required: 'Optional[list[str]]' = None,
    ) -> dict:
        """
        Build a standard function tool schema.
        Args:
            function_name: Name of the function/tool.
            description: Description of the tool.
            parameters: Dict of parameter names to their JSON schema (type, description, etc).
            required: List of required parameter names.
        Returns:
            dict: Schema for the tool.
        """
        return {
            "type": "function",
            "function": {
                "name": function_name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required or [],
                },
            },
        }

    @property
    def schema(self) -> Dict[str, Any]:
        """
        Override in subclasses. Should return a standards-compliant tool schema dict.
        Use build_schema() for convenience.
        """
        return {}

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """Execute tool with given arguments (auto-schema will match signature)."""
        raise NotImplementedError()