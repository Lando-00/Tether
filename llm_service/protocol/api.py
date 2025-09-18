"""
api.py - Protocol Component for MCP Architecture

This module represents the Protocol component in a Model-Context-Protocol architecture.
It handles all API interactions, including:
1. Request/response models
2. API endpoint definitions
3. HTTP interface
4. Tool registration and execution

TODO: Implement rate limiting for API endpoints
TODO: Add input validation with detailed error messages
TODO: Add authentication/authorization system
TODO: Improve error handling and add structured logging

The ProtocolComponent connects the Model and Context components but doesn't contain 
business logic itself. It's responsible for translating API requests into the appropriate
model and context operations.
"""

import os
import sys
import json
import time
import importlib
import pkgutil
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Callable
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from ..model.mlc_engine import ModelComponent
from ..context import ContextComponent
from contextlib import asynccontextmanager

# Add the project root to sys.path to ensure tools module is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import tools module directly
import tools
from tools import get_all_tool_definitions, execute_tool

# Initialize logger
logger = logging.getLogger(__name__)

def _format_model_output(out: Any) -> str:
    """Extract and format key fields from model output for concise, readable logging."""
    parts = []
    # Model ID (shortened for readability)
    model_id = getattr(out, 'id', 'unknown')
    short_id = model_id[:8] + '...' if isinstance(model_id, str) and len(model_id) > 8 else model_id
    parts.append(f"id={short_id}")
    # Process choices with better formatting
    for choice in getattr(out, 'choices', []):
        reason = getattr(choice, 'finish_reason', 'unknown')
        idx = getattr(choice, 'index', 0)
        msg = getattr(choice, 'message', None)
        if not msg:
            parts.append(f"choice[{idx}]: no message")
            continue
        role = getattr(msg, 'role', 'unknown')
        content = getattr(msg, 'content', None)
        tool_calls = getattr(msg, 'tool_calls', None)
        choice_info = f"choice[{idx}]: {reason}"
        if tool_calls:
            tool_names = [tc.function.name for tc in tool_calls if hasattr(tc, 'function')]
            choice_info += f" → tools: {tool_names}"
        elif content:
            preview = content if len(content) <= 50 else content[:47] + '...'
            choice_info += f" → {role}: \"{preview}\""
        else:
            choice_info += f" → {role}: <empty>"
        parts.append(choice_info)
    # Usage stats with better formatting
    usage = getattr(out, 'usage', None)
    if usage:
        prompt_tokens = getattr(usage, 'prompt_tokens', 0)
        completion_tokens = getattr(usage, 'completion_tokens', 0)
        total_tokens = getattr(usage, 'total_tokens', 0)
        extra = getattr(usage, 'extra', {})
        usage_info = f"tokens: {prompt_tokens}+{completion_tokens}={total_tokens}"
        if extra:
            decode_speed = extra.get('decode_tokens_per_s', 0)
            ttft = extra.get('ttft_s', 0)
            if decode_speed:
                usage_info += f" | speed: {decode_speed:.1f}t/s"
            if ttft:
                usage_info += f" | ttft: {ttft:.2f}s"
        parts.append(usage_info)
    return " | ".join(parts)

def _format_tool_call_log(tool_name: str, tool_args: dict, result_preview: str = None) -> str:
    """Format tool call information for clean logging."""
    if len(tool_args) <= 2:
        args_str = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
    else:
        args_str = f"{len(tool_args)} args"
    log_msg = f"🔧 {tool_name}({args_str})"
    if result_preview:
        preview = result_preview if len(result_preview) <= 60 else result_preview[:57] + '...'
        log_msg += f" → {preview}"
    return log_msg

def _format_session_log(session_id: str, action: str, details: str = None) -> str:
    """Format session management logs consistently."""
    short_session = session_id[:8] + '...' if isinstance(session_id, str) and len(session_id) > 8 else session_id
    log_msg = f"📋 Session {short_session}: {action}"
    if details:
        log_msg += f" | {details}"
    return log_msg


# --- Pydantic Models for API ---

class Message(BaseModel):
    """Message model for API responses."""
    id: int
    role: str
    content: str
    created_at: datetime


class CreateSessionResponse(BaseModel):
    """Response model for session creation."""
    session_id: str
    created_at: datetime


class GenerateRequest(BaseModel):
    """Request model for text generation."""
    session_id: str = Field(...)
    prompt: str = Field(...)
    model_name: str = Field(...)
    device: Optional[str] = "auto"
    dll: Optional[str] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95
    # Add tools parameter for OpenAI-compatible tool definitions
    tools: Optional[List[dict]] = None
    # The mcp_server config is now handled by the client, which passes `tools`
    mcp_server: Optional[dict] = None


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    session_id: str
    reply: str
    messages: List[Message]


class UnloadModelRequest(BaseModel):
    """Request model for unloading a model."""
    model_name: str
    device: Optional[str] = "auto"


# --- Protocol Component ---

class ProtocolComponent:
    """Protocol component for MCP architecture handling API interactions."""
    
    def __init__(self, model_component: ModelComponent, context_component: ContextComponent):
        """Initialize with model and context components."""
        self.model = model_component
        self.context = context_component
        self.max_tool_loops = 3
        
        # Register tools by auto-discovering modules
        self._register_tools()
        
        # Cache for dynamic system prompts with tool information (keyed by tool count/hash)
        self._system_prompt_cache = {}
    
    def _register_tools(self):
        """Auto-discover and register all tool modules."""
        # Auto-discover tool modules in the tools package
        for m in pkgutil.iter_modules(tools.__path__):
            importlib.import_module(f"{tools.__name__}.{m.name}")
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Get available models from the model component."""
        return self.model.get_available_models()
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get available tools from the tools registry."""
        return get_all_tool_definitions()
    
#     def generate_dynamic_system_prompt(self, tools: List[Dict[str, Any]]) -> Dict[str, str]:
#         """
#         Generate a dynamic system prompt that includes instructions for tool calling
#         based on the available tools.
        
#         Args:
#             tools: List of tool definitions to include in the prompt
            
#         Returns:
#             A system message object with role and content
#         """
#         # Create a cache key based on tool names to avoid regenerating the same prompt
#         tool_names = sorted([t.get("function", {}).get("name", "") for t in tools])
#         cache_key = ",".join(tool_names)
        
#         # Return cached prompt if available
#         if cache_key in self._system_prompt_cache:
#             return self._system_prompt_cache[cache_key]
        
#         # Start building the system prompt
#         tool_descriptions = []
#         tool_examples = []
        
#         for tool in tools:
#             if "function" not in tool:
#                 continue
                
#             func = tool["function"]
#             name = func.get("name", "")
#             description = func.get("description", "No description available")
#             params = func.get("parameters", {})
            
#             # Build parameter list with types and descriptions
#             param_list = []
#             if "properties" in params:
#                 for param_name, param_info in params["properties"].items():
#                     param_type = param_info.get("type", "string")
#                     param_desc = param_info.get("description", "")
#                     required = param_name in params.get("required", [])
                    
#                     # Format as "name (type, required/optional): description"
#                     req_text = "required" if required else "optional"
#                     param_list.append(f"{param_name} ({param_type}, {req_text}): {param_desc}")
            
#             # Format tool description
#             tool_desc = f"- {name}: {description}\n"
#             if param_list:
#                 tool_desc += "  Parameters:\n" + "\n".join([f"    - {p}" for p in param_list])
#             tool_descriptions.append(tool_desc)
            
#             # Create a simple example for this tool
#             example_params = {}
#             for param_name, param_info in params.get("properties", {}).items():
#                 if param_name in params.get("required", []):
#                     param_type = param_info.get("type", "string")
#                     if param_type == "string":
#                         example_params[param_name] = "example_value"
#                     elif param_type == "integer":
#                         example_params[param_name] = 42
#                     elif param_type == "number":
#                         example_params[param_name] = 3.14
#                     elif param_type == "boolean":
#                         example_params[param_name] = True
            
#             example = {
#                 "thought": f"I need to use the {name} tool to get information.",
#                 "tool_call": {
#                     "name": name,
#                     "parameters": example_params
#                 }
#             }
#             tool_examples.append(json.dumps(example, indent=2))
        
#         # Build the complete system prompt
#         system_content = f"""You are an assistant with access to tools that help you answer user questions.

# Available tools:
# {chr(10).join(tool_descriptions)}

# When you need to use a tool, respond with valid JSON in this exact format:
# {{
#   "thought": "Your reasoning for using the tool",
#   "tool_call": {{
#     "name": "tool_name",
#     "parameters": {{
#       "param1": "value1",
#       "param2": "value2"
#     }}
#   }}
# }}

# Example tool usage:
# {tool_examples[0] if tool_examples else ''}

# After receiving tool results, provide your final answer as normal text.
# If you don't need to use a tool, simply respond conversationally.

# Remember to:
# 1. Always use valid JSON format for tool calls
# 2. Include all required parameters
# 3. Use the exact tool names as provided
# 4. Return to normal text responses after tool usage
# """
        
#         # Create the system message
#         system_message = {
#             "role": "system",
#             "content": system_content
#         }
        
#         # Cache the prompt for future use
#         self._system_prompt_cache[cache_key] = system_message
#         return system_message
    
    def generate_dynamic_system_prompt(self, tools: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate a dynamic system prompt that includes instructions for tool calling
        based on the available tools.
        
        Args:
            tools: List of tool definitions to include in the prompt
            
        Returns:
            A system message object with role and content
        """
        # Create a cache key based on tool names to avoid regenerating the same prompt
        tool_names = sorted([t.get("function", {}).get("name", "") for t in tools])
        cache_key = ",".join(tool_names)
        
        # Return cached prompt if available
        if cache_key in self._system_prompt_cache:
            return self._system_prompt_cache[cache_key]
        
        # Build tool descriptions
        tool_descriptions = []
        
        for tool in tools:
            if "function" not in tool:
                continue
                
            func = tool["function"]
            name = func.get("name", "")
            description = func.get("description", "No description available")
            params = func.get("parameters", {})
            
            # Build parameter list with types and descriptions
            param_list = []
            if "properties" in params:
                for param_name, param_info in params["properties"].items():
                    param_type = param_info.get("type", "string")
                    param_desc = param_info.get("description", "")
                    required = param_name in params.get("required", [])
                    
                    req_text = "required" if required else "optional"
                    param_list.append(f"  - {param_name} ({param_type}, {req_text}): {param_desc}")
            
            # Format tool description
            tool_desc = f"• {name}: {description}"
            if param_list:
                tool_desc += "\n" + "\n".join(param_list)
            tool_descriptions.append(tool_desc)
        
        # Build the system prompt for native function calling
        system_content = f"""You are a helpful assistant with access to these tools:

        ### AVAILABLE TOOLS:
        {chr(10).join(tool_descriptions)}

    ### FUNCTION CALLING INSTRUCTIONS:
    When you need to use a tool, write ONLY the function call with no markdown or code fences—plain text only.

    CORRECT examples:
    get_current_time(timezone="Europe/Dublin")
    web_search(query="Bitcoin latest news", count=5)
    get_top_headlines(q="latest news", country="ie", page_size=5)
    list_sources(country="ie")

    WRONG examples (do NOT use code blocks):
    ```python
    get_current_time(timezone="Europe/Dublin")
    ```
        """

        # Cache and return the system message
        system_message = {
            "role": "system", 
            "content": system_content
        }
        
        self._system_prompt_cache[cache_key] = system_message
        return system_message

    def create_session(self) -> Dict[str, Any]:
        """Create a new session using the context component."""
        return self.context.create_session()
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions using the context component."""
        return self.context.list_sessions()
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session using the context component."""
        return self.context.delete_session(session_id)
    
    def delete_all_sessions(self) -> int:
        """Delete all sessions using the context component."""
        return self.context.delete_all_sessions()
    
    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get messages for a session using the context component."""
        return self.context.get_messages(session_id)
    
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Generate text using the model component based on the conversation history.
        
        Args:
            request: The generation request containing all parameters
            
        Returns:
            A response with the generated text and updated message list
        """
        session_id = request.session_id
        
        # Validate session exists
        session = self.context.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Add user message
        self.context.add_message(session_id, "user", request.prompt)
        
        # Get conversation history
        history = self.context.get_conversation_history(session_id, format="chat")


        # Get available tools
        available_tools = self.get_available_tools()
        
        # Filter client-provided tools to those registered, else use all available
        if request.tools:
            reg_names = {t["function"]["name"] for t in available_tools}
            requested = [t.get("function", {}).get("name") for t in request.tools]
            missing = [n for n in requested if n not in reg_names]
            if missing:
                # optional: use your logger instead of print
                logger.debug("Ignoring unregistered tools requested by client: %s", missing)
            tools_to_use = [t for t in request.tools if t.get("function", {}).get("name") in reg_names]
        else:
            tools_to_use = available_tools
            
        # Generate and add dynamic system prompt to history if tools are available
        if tools_to_use and not any(msg.get("role") == "system" for msg in history):
            system_message = self.generate_dynamic_system_prompt(tools_to_use)
            history.insert(0, system_message)

        # OpenAI-style tool loop
        reply = ""
        tool_loop_count = 0

        while True:
            # Generate response from model
            out = self.model.generate(
                model_name=request.model_name,
                messages=history,
                device=request.device or "auto",
                dll_path=request.dll,
                max_tokens=request.max_tokens or 512,
                temperature=request.temperature or 0.2,
                top_p=request.top_p or 0.95,
                tools=tools_to_use
            )
            
            logger.info("Model output: %s", _format_model_output(out))
            
            first = out.choices[0] if out and out.choices else None
            message = out.choices[0].message if out and out.choices else None

            # Check for standard OpenAI-style tool calls
            if message and getattr(message, "tool_calls", None):
                # 1) Record assistant tool-call message
                history.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": (
                                    tc.function.arguments
                                    if isinstance(tc.function.arguments, str)
                                    else json.dumps(tc.function.arguments or {})
                                )
                            }
                        }
                        for tc in (message.tool_calls or [])
                    ],
                })

                # 2) Execute each tool and append a tool message
                for tc in (message.tool_calls or []):
                    # Log tool call using formatter
                    name = tc.function.name
                    args = tc.function.arguments if isinstance(tc.function.arguments, dict) else json.loads(tc.function.arguments or "{}")
                    logger.info(_format_tool_call_log(name, args))
                    name = tc.function.name
                    args = tc.function.arguments
                    
                    # Parse arguments if they're in string format
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    elif args is None:
                        args = {}
                    
                    # Execute the tool and handle errors
                    try:
                        result = execute_tool(name, args)
                    except Exception as e:
                        result = f"Tool execution error: {e}"
                    # Log tool result
                    logger.info(_format_tool_call_log(name, args, str(result)))
                    # Add tool result to history
                    history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": str(result),
                    })

                tool_loop_count += 1
                if tool_loop_count >= self.max_tool_loops:
                    reply = (
                        "I ran tools several times but couldn't finish the reasoning. "
                        "Try rephrasing or reducing steps."
                    )
                    break

                continue
            
            # Fallback: attempt to parse JSON tool_call(s) from assistant content
            content = message.content if message else ""
            trimmed = content.strip()
            parsed_calls = []
            # Try parse JSON object or array
            try:
                data = json.loads(trimmed)
                # Single tool_call object
                if isinstance(data, dict) and "tool_call" in data:
                    tc = data["tool_call"]
                    if isinstance(tc, dict) and "name" in tc and "parameters" in tc:
                        parsed_calls.append({"name": tc["name"], "arguments": tc["parameters"]})
                # List of calls
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "name" in item and "arguments" in item:
                            parsed_calls.append({"name": item["name"], "arguments": item["arguments"]})
            except (json.JSONDecodeError, TypeError):
                parsed_calls = []
            # Process parsed calls
            if parsed_calls:
                registered = {t["function"]["name"] for t in tools_to_use}
                for call in parsed_calls:
                    name = call.get("name")
                    params = call.get("arguments", {})
                    if name not in registered:
                        continue
                    tool_call_id = f"call_{tool_loop_count}_{int(time.time())}"
                    # Add synthetic tool call message
                    history.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": tool_call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(params)}}]
                    })
                    logger.info("Fallback calling tool '%s' with params %s", name, params)
                    # Execute the tool
                    try:
                        result = execute_tool(name, params)
                    except Exception as e:
                        result = f"Tool execution error: {e}"
                    history.append({"role": "tool", "tool_call_id": tool_call_id, "name": name, "content": str(result)})
                tool_loop_count += 1
                if tool_loop_count >= self.max_tool_loops:
                    reply = ("I ran tools several times but couldn't finish the reasoning. Try rephrasing or reducing steps.")
                    break
                continue
            
            # No more tool_calls → final assistant reply
            reply = (message.content or "") if message else ""
            break

        # Add assistant message
        self.context.add_message(session_id, "assistant", reply)
        
        # Return with full message history
        messages = self.context.get_messages(session_id)
        
        # Convert to API response format
        return GenerateResponse(
            session_id=session_id,
            reply=reply,
            messages=[
                Message(
                    id=m["id"],
                    role=m["role"],
                    content=m["content"],
                    created_at=m["created_at"]
                ) for m in messages
            ]
        )
    
    def unload_model(self, request: UnloadModelRequest) -> Dict[str, str]:
        """Unload a model from memory."""
        self.model.unload_model(request.model_name, request.device)
        return {"detail": f"Model '{request.model_name}' unloaded successfully"}
    
    async def generate_stream(self, request: GenerateRequest) -> StreamingResponse:
        """
        Stream generation results token-by-token.

        Args:
            request: The generation request containing all parameters

        Returns:
            A streaming response with generated tokens
        """
        session_id = request.session_id
        
        # Validate session exists
        session = self.context.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Add user message
        self.context.add_message(session_id, "user", request.prompt)
        
        # Get conversation history
        history = self.context.get_conversation_history(session_id, format="chat")
        # --- Tool setup for streaming endpoint ---
        available_tools = self.get_available_tools()
        if request.tools:
            reg_names = {t["function"]["name"] for t in available_tools}
            requested = [t.get("function", {}).get("name") for t in request.tools]
            missing = [n for n in requested if n not in reg_names]
            if missing:
                logger.debug("Ignoring unregistered tools requested by client: %s", missing)
            tools_to_use = [t for t in request.tools if t.get("function", {}).get("name") in reg_names]
        else:
            tools_to_use = available_tools

        # Inject dynamic system prompt if tools are present and no system message yet
        if tools_to_use and not any(msg.get("role") == "system" for msg in history):
            system_message = self.generate_dynamic_system_prompt(tools_to_use)
            history.insert(0, system_message)

        # Log streaming generate request context
        logger.info(
            "Streaming generate request: session=%s, model=%s, tools=%s, prompt=%s",
            request.session_id,
            request.model_name,
            [t.get("function", {}).get("name") for t in tools_to_use],
            request.prompt[:100]
        )

        async def token_stream():
            # Loop for tool calls, feeding results back into model before streaming tokens
            tool_loop = 0
            final_text = None
            while True:
                out = self.model.generate(
                    model_name=request.model_name,
                    messages=history,
                    device=request.device or "auto",
                    dll_path=request.dll,
                    max_tokens=request.max_tokens or 512,
                    temperature=request.temperature or 0.2,
                    top_p=request.top_p or 0.95,
                    tools=tools_to_use
                )
                msg = out.choices[0].message
                # Log the model output for debugging
                logger.info("Stream model output: %s", _format_model_output(out))
                # Execute any tool calls and append results
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        name = tc.function.name
                        args = tc.function.arguments if isinstance(tc.function.arguments, dict) else json.loads(tc.function.arguments or "{}")
                        yield json.dumps({"event": "tool_start", "tool": name}) + "\n"
                        logger.info("--- ---\n Calling tool : '%s' with args %s \n--- ---", tc.function.name, tc.function.arguments)
                        try:
                            result = execute_tool(name, args)
                            logger.info("Tool '%s' result: %s", name, str(result)[:100])
                        except Exception as e:
                            result = f"Tool execution error: {e}"
                        # Notify UI that tool finished without exposing result
                        yield json.dumps({"event": "tool_end", "tool": name, "result": str(result)}) + "\n"
                        history.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": str(result)})
                    tool_loop += 1
                    if tool_loop >= self.max_tool_loops:
                        final_text = "I ran tools several times but couldn't finish reasoning."
                        break
                    continue  # regenerate with updated history
                # No more tool calls, capture text and break
                final_text = msg.content or ""
                break
            # Stream assistant tokens
            for ch in final_text:
                yield json.dumps({"event": "token", "data": ch}) + "\n"
            # Persist final assistant message
            self.context.add_message(session_id, "assistant", final_text)
        
        # Stream NDJSON events: tokens, tool signals, etc.
        return StreamingResponse(token_stream(), media_type="application/x-ndjson")


# --- FastAPI Application Factory ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for resource cleanup."""
    yield
    # On shutdown, clear model cache
    app.state.protocol.model.clear_engine_cache()


def create_api_app(protocol: ProtocolComponent) -> FastAPI:
    """
    Create a FastAPI application with all routes configured.
    
    Args:
        protocol: The protocol component instance
        
    Returns:
        A configured FastAPI application
    """
    app = FastAPI(title="MLC-LLM Session Service", version="0.1.0", lifespan=lifespan)
    app.state.protocol = protocol
    
    # Health check endpoint
    @app.get("/healthz")
    def healthz():
        return {"ok": True}
    
    # Model endpoints
    @app.get("/models")
    def list_models():
        return {"models": protocol.get_available_models()}
    
    @app.post("/models/unload")
    def unload_model(req: UnloadModelRequest):
        return protocol.unload_model(req)
    
    # Tools endpoint
    @app.get("/tools")
    def list_tools():
        return {"tools": protocol.get_available_tools()}
    
    # Session endpoints
    @app.post("/sessions", response_model=CreateSessionResponse)
    def create_session():
        result = protocol.create_session()
        return CreateSessionResponse(
            session_id=result["session_id"],
            created_at=result["created_at"]
        )
    
    @app.get("/sessions")
    def list_sessions():
        return protocol.list_sessions()
    
    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str):
        if protocol.delete_session(session_id):
            return {"detail": "Session deleted"}
        raise HTTPException(status_code=404, detail="Session not found")
    
    @app.delete("/sessions")
    def delete_all_sessions():
        count = protocol.delete_all_sessions()
        return {"detail": f"Deleted {count} sessions."}
    
    @app.get("/sessions/{session_id}/messages", response_model=List[Message])
    def get_session_messages(session_id: str):
        messages = protocol.get_session_messages(session_id)
        if not messages:
            raise HTTPException(status_code=404, detail="Session not found")
        return [
            Message(
                id=m["id"],
                role=m["role"],
                content=m["content"],
                created_at=m["created_at"]
            ) for m in messages
        ]
    
    # Generation endpoints
    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest):
        return protocol.generate(req)
    
    @app.post("/generate_stream")
    async def generate_stream(request: Request, req: GenerateRequest):
        return await protocol.generate_stream(req)
    
    return app
