"""
Protocol service implementation for the MCP architecture.
"""
import json
import logging
import pkgutil
import importlib
from typing import List, Dict, Any, Optional, AsyncIterator

from fastapi import HTTPException
from llm_service import tools
from llm_service.context import ContextComponent
from llm_service.model.mlc_engine import ModelComponent
from llm_service.tools import execute_tool, get_all_tool_definitions

from llm_service.protocol.api.schemas import (
    Message,
    GenerateRequest,
    GenerateResponse,
    UnloadModelRequest,
)

logger = logging.getLogger(__name__)


class ProtocolService:
    """Protocol service for MCP architecture handling API interactions."""
    
    def __init__(self, model_component: ModelComponent, context_component: ContextComponent):
        """Initialize with model and context components."""
        self.model = model_component
        self.context = context_component
        self.max_tool_loops = 3
        self._executed_tool_calls: set = set()
        
        # Register tools by auto-discovering modules
        self._register_tools()
        
        # Cache for dynamic system prompts with tool information (keyed by tool count/hash)
        self._system_prompt_cache = {}
        
        # Lazy-loaded generation service for streaming support
        self._generation_service = None
    
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
    
    def generate_dynamic_system_prompt(self, tools: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate a dynamic system prompt that includes instructions for tool calling
        based on the available tools.
        """
        tool_names = sorted([t.get("function", {}).get("name", "") for t in tools])
        cache_key = ",".join(tool_names)
        
        if cache_key in self._system_prompt_cache:
            return self._system_prompt_cache[cache_key]
        
        tool_descriptions = []
        
        for tool in tools:
            if "function" not in tool:
                continue
                
            func = tool["function"]
            name = func.get("name", "")
            description = func.get("description", "No description available")
            params = func.get("parameters", {})
            
            param_list = []
            if "properties" in params:
                for param_name, param_info in params["properties"].items():
                    param_type = param_info.get("type", "string")
                    param_desc = param_info.get("description", "")
                    required = param_name in params.get("required", [])
                    
                    req_text = "required" if required else "optional"
                    param_list.append(f"  - {param_name} ({param_type}, {req_text}): {param_desc}")
            
            tool_desc = f"• {name}: {description}"
            if param_list:
                tool_desc += "\n" + "\n".join(param_list)
            tool_descriptions.append(tool_desc)
        
        system_content = f"""You are a helpful assistant with access to these tools:

        ### AVAILABLE TOOLS:
        {chr(10).join(tool_descriptions)}
        Tool names always start with `__tool_`. When invoking a tool, output exactly one line like `__tool_get_current_time(timezone=\"Europe/Dublin\")`. Use keyword arguments only.

        ### TOOL CALL FORMAT (CRITICAL)
        - When you need a tool, output **exactly one line** containing **only** the function call (no prose, no code fences).
        - Use **keyword arguments only**. Do **NOT** pass a single dict as a positional arg.
        - Quote strings. Keep values as simple Python/JSON literals.

        ✅ Correct:
        __tool_get_current_time(timezone="Europe/Vienna")
        __tool_web_search(query="Bitcoin latest news", count=5)
        __tool_get_top_headlines(q="latest news", country="ie", page_size=5)
        __tool_list_sources(country="ie")

    ❌ Wrong:
    __tool_get_current_time({{"timezone":"Europe/Vienna"}})       # dict as positional arg
    ```python
    __tool_get_current_time(timezone="Europe/Vienna")
    ```                                             # code fence / extra text
    __tool_get_current_time(timezone="Europe/Vienna") Also... # any extra prose

        ### WHEN NO TOOL IS NEEDED
        - Respond with a single, concise answer (no alternative phrasings).
        """

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
        """
        session_id = request.session_id
        
        session = self.context.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        self.context.add_message(session_id, "user", request.prompt)
        
        history = self.context.get_conversation_history(session_id, format="chat")

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
            
        if tools_to_use and not any(msg.get("role") == "system" for msg in history):
            system_message = self.generate_dynamic_system_prompt(tools_to_use)
            history.insert(0, system_message)

        reply = ""
        tool_loop_count = 0

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
            
            message = out.choices[0].message if out and out.choices else None

            if message and getattr(message, "tool_calls", None):
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

                for tc in (message.tool_calls or []):
                    name = tc.function.name
                    args = tc.function.arguments
                    
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    elif args is None:
                        args = {}
                    
                    try:
                        result = execute_tool(name, args)
                    except Exception as e:
                        result = f"Tool execution error: {e}"
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
            
            reply = (message.content or "") if message else ""
            break

        self.context.add_message(session_id, "assistant", reply)
        
        messages = self.context.get_messages(session_id)
        
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
        
    def _get_generation_service(self):
        """
        Lazy-load the generation service.
        
        Returns:
            GenerationService instance
        """
        if self._generation_service is None:
            from llm_service.protocol.service.generation_service import GenerationService
            self._generation_service = GenerationService(
                self.model,
                self.context,
                self.generate_dynamic_system_prompt,
                self.get_available_tools
            )
        return self._generation_service
        
    def generate_stream(self, 
                      session_id: str, 
                      model_name: str,
                      prompt: str,
                      tools=None,
                      device=None,
                      dll=None,
                      max_tokens=None,
                      temperature=None,
                      top_p=None,
                      **kwargs):
        """
        Generate text with streaming support.
        
        Args:
            session_id: Session ID
            model_name: Model name
            prompt: Text prompt
            tools: List of tool definitions
            device: Device to use (auto, cpu, cuda)
            dll: Path to MLC DLL
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional parameters
            
        Returns:
            StreamingResponse with NDJSON events
        """
        generation_service = self._get_generation_service()
        
        request = GenerateRequest(
            session_id=session_id,
            prompt=prompt,
            model_name=model_name,
            tools=tools,
            device=device,
            dll=dll,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            **kwargs
        )
        
        # Return the streaming response directly
        return generation_service.generate_stream(request)
        
    def abort_generation(self, session_id: str) -> bool:
        """
        Abort an ongoing generation for a session.
        
        Args:
            session_id: The ID of the session with active generation
            
        Returns:
            True if any generations were aborted, False if no active generations found
        """
        generation_service = self._get_generation_service()
        return generation_service.abort_generation(session_id)