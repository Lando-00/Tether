# tools/__init__.py
from typing import List, Dict, Any, Callable, get_type_hints, Optional
import inspect
from pydantic import BaseModel, create_model

# Registry to store all tools
TOOL_REGISTRY = {}

def register_tool(func: Callable) -> Callable:
    """Decorator to register a function as a tool"""
    tool_name = func.__name__
    
    # Get parameter info from type hints and docstring
    type_hints = get_type_hints(func)
    signature = inspect.signature(func)
    
    # Create OpenAI tool schema
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    for param_name, param in signature.parameters.items():
        if param_name not in type_hints:
            continue
            
        param_type = type_hints[param_name]
        is_optional = param.default != inspect.Parameter.empty
        
        # Convert Python types to JSON Schema types
        json_type = "string"  # default
        if param_type == int:
            json_type = "integer"
        elif param_type == float:
            json_type = "number"
        elif param_type == bool:
            json_type = "boolean"
        elif param_type == list:
            json_type = "array"
        
        parameters["properties"][param_name] = {"type": json_type}
        
        # Add description from docstring if available
        if func.__doc__:
            # Parse docstring to find parameter descriptions
            # (simplified implementation)
            pass
            
        if not is_optional:
            parameters["required"].append(param_name)
    
    # Create the OpenAI tool definition
    tool_definition = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": func.__doc__.split("\n\n")[0] if func.__doc__ else "",
            "parameters": parameters
        }
    }
    
    # Store both the function and its definition
    TOOL_REGISTRY[tool_name] = {
        "function": func,
        "definition": tool_definition
    }
    
    return func

def get_all_tool_definitions() -> List[Dict[str, Any]]:
    """Return all tool definitions in OpenAI format"""
    return [tool["definition"] for tool in TOOL_REGISTRY.values()]

def execute_tool(tool_name: str, params: Dict[str, Any]) -> Any:
    """Execute a tool by name with given parameters"""
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Tool {tool_name} not found")
    
    tool = TOOL_REGISTRY[tool_name]["function"]
    return tool(**params)