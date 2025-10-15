# tools/__init__.py
from typing import List, Dict, Any, Callable, get_type_hints, Optional, get_origin, get_args, Literal
import inspect

# Prefix for published tool names
TOOL_PREFIX = "__tool_"

# Keep canonical registry keyed by *function name*
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}
# Map published (prefixed) name -> function name
PUBLISHED_TO_FUNC: Dict[str, str] = {}

def register_tool(_func: Optional[Callable] = None, *, name: Optional[str] = None):
    """Decorator to register a function as a tool.
    Usage:
      @register_tool
      def foo(...): ...

      @register_tool(name="__tool_custom_name")
      def bar(...): ...
    """
    def _wrap(func: Callable) -> Callable:
        tool_name = func.__name__
        publish_name = name or f"{TOOL_PREFIX}{tool_name}"

        # Inspect signature and type hints
        type_hints = get_type_hints(func)
        signature = inspect.signature(func)
        # Parse docstring for parameter descriptions
        doc = inspect.getdoc(func) or ""
        param_descriptions: Dict[str, str] = {}
        for line in doc.splitlines():
            # match lines like 'param: description'
            parts = [p.strip() for p in line.split(':', 1)]
            if len(parts) == 2 and parts[0] in signature.parameters:
                param_descriptions[parts[0]] = parts[1]

        # Build JSON Schema parameters
        parameters: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        for param_name, param in signature.parameters.items():
            if param_name not in type_hints:
                continue
            param_type = type_hints[param_name]
            is_optional = param.default != inspect.Parameter.empty

            # Determine JSON Schema for parameter
            prop_schema: Dict[str, Any] = {}
            origin = get_origin(param_type)
            if origin is list or origin is List:
                # Typed list
                args = get_args(param_type)
                item_type = args[0] if args else Any
                # map Python type to JSON type
                json_type = 'string'
                if item_type == int:
                    json_type = 'integer'
                elif item_type == float:
                    json_type = 'number'
                elif item_type == bool:
                    json_type = 'boolean'
                prop_schema['type'] = 'array'
                prop_schema['items'] = {'type': json_type}
            elif get_origin(param_type) is Literal:
                # Enum values
                enum_values = list(get_args(param_type))
                # infer type from first enum value
                sample = enum_values[0]
                val_type = type(sample)
                json_type = 'string'
                if isinstance(sample, int):
                    json_type = 'integer'
                elif isinstance(sample, float):
                    json_type = 'number'
                elif isinstance(sample, bool):
                    json_type = 'boolean'
                prop_schema['type'] = json_type
                prop_schema['enum'] = enum_values
            else:
                # Simple types
                if param_type == int:
                    prop_schema['type'] = 'integer'
                elif param_type == float:
                    prop_schema['type'] = 'number'
                elif param_type == bool:
                    prop_schema['type'] = 'boolean'
                elif param_type == list:
                    prop_schema['type'] = 'array'
                else:
                    prop_schema['type'] = 'string'

            # Include default value if provided
            if param.default != inspect.Parameter.empty:
                prop_schema['default'] = param.default
            # Include description if parsed from docstring
            if param_name in param_descriptions:
                prop_schema['description'] = param_descriptions[param_name]

            parameters['properties'][param_name] = prop_schema
            if not is_optional:
                parameters['required'].append(param_name)

        # Use the first non-empty line of the docstring as the description
        raw_doc = func.__doc__ or ""
        doc_lines = [line.strip() for line in raw_doc.splitlines() if line.strip()]
        short_desc = doc_lines[0] if doc_lines else ""
        tool_definition = {
            "type": "function",
            "function": {
                "name": publish_name,
                "description": short_desc,
                "parameters": parameters,
            },
        }

        # Register in both registries
        TOOL_REGISTRY[tool_name] = {"function": func, "definition": tool_definition, "published_name": publish_name}
        PUBLISHED_TO_FUNC[publish_name] = tool_name
        return func

    # Support both @register_tool and @register_tool(...)
    return _wrap if _func is None else _wrap(_func)

def get_all_tool_definitions() -> List[Dict[str, Any]]:
    """Return optimized tool definitions for LLM consumption"""
    definitions = [tool["definition"] for tool in TOOL_REGISTRY.values()]
    for defn in definitions:
        fn = defn.get("function", {})
        # strip TOOL_PREFIX
        name = fn.get("name", "")
        if name.startswith(TOOL_PREFIX):
            fn["name"] = name[len(TOOL_PREFIX):]
        # collapse description to first nonempty line
        desc = fn.get("description", "") or ""
        desc_lines = [line.strip() for line in desc.split("\n") if line.strip()]
        fn["description"] = desc_lines[0] if desc_lines else ""
        # prune parameters schema
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        for p_schema in props.values():
            # remove default:null noise
            if p_schema.get("default") is None:
                p_schema.pop("default", None)
            # collapse parameter description to first nonempty line
            if "description" in p_schema:
                parts = [l.strip() for l in p_schema["description"].split("\n") if l.strip()]
                p_schema["description"] = parts[0] if parts else p_schema["description"]
        # remove empty required lists
        if params.get("required") == []:
            params.pop("required", None)
    return definitions

def execute_tool(tool_name: str, params: Dict[str, Any]) -> Any:
    """Execute a tool by published or canonical name."""
    # Resolve canonical function key
    func_key = PUBLISHED_TO_FUNC.get(tool_name) or (
        tool_name[len(TOOL_PREFIX):] if tool_name.startswith(TOOL_PREFIX) else tool_name
    )
    if func_key not in TOOL_REGISTRY:
        raise ValueError(f"Tool {tool_name} not found")
    fn = TOOL_REGISTRY[func_key]["function"]
    return fn(**params)