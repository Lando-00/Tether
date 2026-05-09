from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, get_args, get_origin
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

    async def invoke(self, args: Dict[str, Any]) -> Any:
        """Registry-facing shim. Synthesis §6 row 4: unpacks the args dict
        into keyword arguments and delegates to the author-defined run().
        The orchestrator always calls tool.invoke(args); concrete tool authors
        always write a typed run(**kwargs) signature."""
        return await self.run(**args)

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

    @staticmethod
    def _python_type_to_json_schema(t: Any, default: Any = ...) -> Dict[str, Any]:
        """Recursively convert a Python type annotation to a JSON Schema fragment.

        Synthesis §6 row 5 / R8 / connector spec §8.1:
        Single function — no class hierarchy (R6 anti-overengineering rule).

        Nullable convention: Optional[T] adds "nullable": true to the schema
        (OpenAPI 3.0 style). anyOf with null would be equally valid; we use the
        simpler nullable flag to keep emitted schemas compact.
        """
        import types as _types

        # Annotated[T, ...]: strip metadata, recurse on the inner type
        if get_origin(t) is _types.UnionType or get_origin(t) is None:
            pass  # handled below in the Union branch

        # typing.get_origin helpers for stdlib generics
        origin = get_origin(t)
        args = get_args(t)

        # Annotated[T, metadata, ...]  — ignore metadata, recurse on T
        try:
            import typing
            if origin is typing.Annotated:
                return BaseTool._python_type_to_json_schema(args[0], default)
        except AttributeError:
            pass

        # Optional[T]  ≡  Union[T, None]  (also handles T | None in 3.10+)
        is_union = (origin is getattr(__import__('typing'), 'Union', None))
        is_pep604_union = (
            hasattr(__import__('types'), 'UnionType') and
            isinstance(t, __import__('types').UnionType)
        )
        if is_union or is_pep604_union:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                inner = BaseTool._python_type_to_json_schema(non_none[0])
                inner["nullable"] = True
                return inner
            # multi-type union without None: fall through to string fallback

        # list[T] / List[T]
        if origin is list:
            item_schema = (
                BaseTool._python_type_to_json_schema(args[0]) if args else {"type": "string"}
            )
            return {"type": "array", "items": item_schema}

        # dict[str, Any] / Dict[str, Any]
        if origin is dict:
            return {"type": "object"}

        # Literal["a", "b", ...]
        if origin is Literal:
            values = list(args)
            # Infer JSON type from first member
            first = values[0] if values else ""
            if isinstance(first, int):
                json_type = "integer"
            elif isinstance(first, float):
                json_type = "number"
            elif isinstance(first, bool):
                json_type = "boolean"
            else:
                json_type = "string"
            return {"type": json_type, "enum": values}

        # Primitive types
        _PRIMITIVES: Dict[Any, str] = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }
        if t in _PRIMITIVES:
            return {"type": _PRIMITIVES[t]}

        # Fallback: unknown type maps to string (preserves backward compat)
        return {"type": "string"}

    @property
    def auto_schema(self) -> Dict[str, Any]:
        import inspect
        from typing import get_type_hints
        sig = inspect.signature(self.run)
        # get_type_hints resolves string annotations and Annotated metadata
        try:
            hints = get_type_hints(self.run, include_extras=True)
        except Exception:
            hints = {}
        docstring = self.run.__doc__ or self.__doc__ or ""
        param_docs = self._extract_param_descriptions(docstring)
        params = {}
        required = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            # Skip *args and **kwargs — not tool parameters
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            t = hints.get(name, str)
            prop = self._python_type_to_json_schema(t)
            prop["description"] = param_docs.get(name, "")
            if param.default is not inspect.Parameter.empty:
                prop["default"] = param.default
            else:
                required.append(name)
            params[name] = prop
        return self.build_schema(
            function_name=self.name,
            description=self.__doc__ or "",
            parameters=params,
            required=required,
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
        """Author-facing API. Concrete tools override with typed signatures
        (e.g., `async def run(self, timezone: str = 'UTC')`).

        NOT part of the Tool ABC — do not call from the orchestrator or
        ToolRunner. Use tool.invoke(args) instead (synthesis §6 row 4).
        """
        raise NotImplementedError()