import time
from typing import Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_service.protocol.orchestration.orchestrator import ToolOrchestrator

class ToolBoundaryController:
    """
    Tool boundary controller to handle start/end events and history around tool execution.
    """
    def __init__(self, orchestrator: 'ToolOrchestrator'):
        self.orch = orchestrator

    def on_start(self, tool_name: str, session_id: str, loops: int) -> Tuple[str, str, bytes]:
        """Generate tool_id, emit start event, and return identifiers and event bytes."""
        tool_id = f"tc_{loops}_{int(time.time())}"
        pub = f"{self.orch.tool_prefix}{tool_name}"
        self.orch._logger.info("Tool call detected: %s (session: %s)", tool_name, session_id)
        ev = self.orch.emitter.tool_start(tool_id, pub)
        return tool_id, pub, ev

    def on_complete(self, tool_id: str, pub: str, args: Dict[str, Any], history: list) -> Tuple[Any, bytes]:
        """Execute tool, emit end event, and update history. Returns result and event bytes."""
        # Record call in history
        self.orch.history.add_assistant_toolcall(history, tool_id, pub, args)
        # Execute tool
        try:
            result = self.orch.tool_executor.execute(pub, args)
            # Record result in history
            self.orch.history.add_tool_result(history, tool_id, pub, result)
            # Emit result event
            ev = self.orch.emitter.tool_end(tool_id, pub, result)
            return result, ev
        except Exception as e:
            self.orch._logger.exception("Error executing tool %s: %s", pub, str(e))
            error_msg = f"Error: {str(e)}"
            # Record error in history
            self.orch.history.add_tool_result(history, tool_id, pub, error_msg)
            # Emit result event with error
            ev = self.orch.emitter.tool_end(tool_id, pub, error_msg)
            return error_msg, ev