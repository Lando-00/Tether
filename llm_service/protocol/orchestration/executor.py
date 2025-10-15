import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Union

from llm_service.protocol.core.interfaces import ToolExecutor, Logger
from llm_service.protocol.core.loggers import NoOpLogger
from llm_service.tools import execute_tool


class DefaultToolExecutor(ToolExecutor):
    """
    Default implementation for executing tools with validation and error handling.
    """
    def __init__(self, logger: Optional[Logger] = None):
        self._logger = logger or NoOpLogger()
        
    def execute(self, published_name: str, args: Dict[str, Any]) -> Any:
        """
        Execute a tool given its published name and arguments.
        
        Args:
            published_name: The published name of the tool (e.g., "__tool_get_current_time")
            args: Dictionary of arguments to pass to the tool
            
        Returns:
            The result of the tool execution or an error message
        """
        if not published_name or not isinstance(published_name, str):
            self._logger.warning("Invalid tool name provided: %s", published_name)
            return {"error": "Invalid tool name"}
        
        try:
            # Execute the tool using the central registry
            self._logger.info("Executing tool %s with args: %s", published_name, json.dumps(args))
            result = execute_tool(published_name, args)
            self._logger.info("Tool %s execution result: %s", published_name, result)
            return result
        except ValueError as e:
            # Tool not found
            self._logger.warning("Tool not found: %s (%s)", published_name, str(e))
            return {"error": f"Tool not found: {published_name}"}
        except TypeError as e:
            # Argument validation error
            self._logger.warning("Invalid arguments for tool %s: %s", published_name, str(e))
            return {"error": f"Invalid arguments: {str(e)}"}
        except Exception as e:
            # General execution error
            self._logger.exception("Error executing tool %s: %s", published_name, str(e))
            return {"error": f"Execution error: {str(e)}"}
