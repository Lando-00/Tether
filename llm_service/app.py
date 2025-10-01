"""
app.py - Main Application Entry Point

This module ties together all MCP components:
- Model: Handles language model inference
- Context: Manages conversation state and persistence
- Protocol: Provides API endpoints and client interface

TODO: Add configuration validation on startup
TODO: Implement health check endpoints
TODO: Create environment variable validation

The MCP architecture allows each component to be developed, tested, and replaced
independently, promoting separation of concerns.
"""

import os
import sys
import uvicorn
import logging
import subprocess
import time
import signal
from pathlib import Path
from typing import Optional

# Add the project root to the Python path to ensure tools can be imported
# This is needed so both the root-level 'tools' and llm_service modules can be found
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from llm_service.model import ModelComponent
from llm_service.context import ContextComponent
from llm_service.protocol import ProtocolComponent, create_api_app
from mlc_llm.protocol import openai_api_protocol
from mlc_llm.protocol.error_protocol import BadRequestError


try:
    import faulthandler

    faulthandler.enable()
except Exception:
    pass


# Monkey-patch to allow assistant.tool_calls in incoming messages
_original_check = openai_api_protocol.ChatCompletionRequest.check_message_validity


def _patched_check_message_validity(self):
    # same as original, but skip raising on assistant.tool_calls
    for i, message in enumerate(self.messages):
        if message.role == "system" and i != 0:
            raise BadRequestError(
                f"System prompt at position {i} in the message list is invalid."
            )
        if message.tool_call_id is not None and message.role != "tool":
            raise BadRequestError("Non-tool message having `tool_call_id` is invalid.")
        if isinstance(message.content, list) and message.role != "user":
            raise BadRequestError("Non-user message having a list of content is invalid.")
        # skip tool_calls check entirely
    # no exception for tool_calls


openai_api_protocol.ChatCompletionRequest.check_message_validity = (
    _patched_check_message_validity
)

# e.g., in llm_service/app.py (or wherever you bootstrap)
try:
    import model.mlc_stream_patch as mlc_stream_patch
    mlc_stream_patch.apply()
    print("[mlc-stream-patch] Enabled tool-call synthesis for streaming.")
except Exception as e:
    print("[mlc-stream-patch] Patch failed:", e)



def create_mcp_app(dist_path: str = "dist", database_url: Optional[str] = None):
    """
    Create the complete MCP application.

    Args:
        dist_path: Path to models directory
        database_url: Optional database connection URL

    Returns:
        Configured FastAPI application
    """
    # Initialize Model component
    model_component = ModelComponent(dist_path=Path(dist_path))

    # Initialize Context component
    context_component = ContextComponent(database_url=database_url)

    # Initialize Protocol component
    protocol_component = ProtocolComponent(
        model_component=model_component, context_component=context_component
    )

    # Create FastAPI app
    app = create_api_app(protocol_component)

    return app


def supervise():
    """
    Parent watchdog: restarts this module when it crashes (non-zero exit).
    """
    min_delay, max_delay = 1.0, 30.0
    delay = min_delay
    child = None
    stopping = False

    def stop_handler(sig, frame):
        nonlocal stopping, child
        stopping = True
        if child and child.poll() is None:
            try:
                child.terminate()
                child.wait(timeout=10)
            except Exception:
                pass
        os._exit(0)

    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_handler)

    env = dict(os.environ)
    env["MLC_SUPERVISED_CHILD"] = "1"
    cmd = [sys.executable, "-m", "llm_service.app"]

    while not stopping:
        print(f"[supervisor] starting: {' '.join(cmd)}")
        child = subprocess.Popen(cmd, env=env)
        rc = child.wait()
        if stopping or rc == 0:
            break
        print(f"[supervisor] child crashed (exit={rc}); restarting in {delay:.1f}s")
        time.sleep(delay)
        delay = min(delay * 2, max_delay)


def main():
    """Application entry point."""
    logging.basicConfig(level=logging.INFO)
    # If supervision requested and not already child, run watchdog
    if os.environ.get("MLC_SUPERVISE") == "1" and os.environ.get("MLC_SUPERVISED_CHILD") != "1":
        supervise()
        return

    # Allow configuration via environment variables
    dist_path = os.environ.get("MLC_DIST_PATH", "dist")
    database_url = os.environ.get("MLC_SQLITE_URL", "sqlite:///mlc_sessions.db")
    host = os.environ.get("MLC_HOST", "127.0.0.1")
    port = int(os.environ.get("MLC_PORT", "8090"))

    # Create the application
    app = create_mcp_app(dist_path, database_url)

    # Run with uvicorn
    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
