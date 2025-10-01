import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from llm_service.protocol.core.interfaces import ToolExecutor, Logger
from llm_service.protocol.core.loggers import NoOpLogger


class DefaultToolExecutor(ToolExecutor):
    """
    Default implementation for executing tools with validation and error handling.
    """
    def __init__(self, logger: Optional[Logger] = None):
        self._logger = logger or NoOpLogger()

    def execute(self, tool_name: str, args: Dict[str, Any], tool_registry: Dict[str, Callable]) -> Tuple[bool, Any]:
        """
        Execute a tool given its name, arguments, and a registry of available tools.
        
        Returns:
            Tuple[bool, Any]: First element indicates success (True) or failure (False).
                              Second element is either the result or error information.
        """
        if not tool_name or not isinstance(tool_name, str):
            self._logger.warning("Invalid tool name provided: %s", tool_name)
            return False, {"error": "Invalid tool name"}
        
        tool_fn = tool_registry.get(tool_name)
        if not tool_fn:
            self._logger.warning("Tool not found: %s", tool_name)
            return False, {"error": f"Tool not found: {tool_name}", "available_tools": list(tool_registry.keys())}
        
        try:
            # Validate arguments against function signature
            validated_args = self._validate_args(tool_fn, args)
            if isinstance(validated_args, dict) and "_error" in validated_args:
                self._logger.warning("Invalid arguments for tool %s: %s", tool_name, validated_args["_error"])
                return False, {"error": f"Invalid arguments: {validated_args['_error']}"}
            
            # Execute the tool
            self._logger.debug("Executing tool %s with args: %s", tool_name, json.dumps(args))
            result = tool_fn(**validated_args)
            return True, result
        except Exception as e:
            self._logger.exception("Error executing tool %s: %s", tool_name, str(e))
            return False, {"error": f"Execution error: {str(e)}"}
    
    def _validate_args(self, tool_fn: Callable, args: Dict[str, Any]) -> Union[Dict[str, Any], Dict[str, str]]:
        """
        Validate and potentially transform arguments based on the function signature.
        
        Returns:
            Union[Dict[str, Any], Dict[str, str]]: Either validated arguments or error information
        """
        if not args:
            args = {}
            
        try:
            sig = inspect.signature(tool_fn)
            validated_args = {}
            
            # Check for missing required parameters
            missing_required = []
            for param_name, param in sig.parameters.items():
                # Skip self parameter for methods
                if param_name == "self":
                    continue
                    
                if param_name not in args and param.default is inspect.Parameter.empty:
                    missing_required.append(param_name)
            
            if missing_required:
                return {"_error": f"Missing required arguments: {', '.join(missing_required)}"}
            
            # Filter out unknown parameters
            param_names = set(p for p in sig.parameters.keys() if p != "self")
            for arg_name in args:
                if arg_name in param_names:
                    validated_args[arg_name] = args[arg_name]
                # Ignore extra arguments
            
            return validated_args
        except Exception as e:
            self._logger.exception("Error validating arguments: %s", str(e))
            return {"_error": f"Argument validation error: {str(e)}"}
