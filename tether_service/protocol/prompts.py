# tether_service/protocol/prompts.py
"""
System prompt construction with tool schema injection.

Provides utilities to build system prompts that instruct the model on:
1. How to emit tool calls (sentinel format)
2. What tools are available (names, descriptions, parameters)
3. When to use tools vs. respond directly
"""
from typing import Dict, List, Any


def _render_tool_catalog(tools: Dict[str, Any]) -> str:
    """
    Render a readable list of tools from the tools registry.
    Each tool is rendered with name, description, and parameters.
    
    Format:
      • tool_name: Short description
        - param (type, required/optional): description
    """
    if not tools:
        return ""
    
    lines: List[str] = []
    for tool_name, tool_obj in tools.items():
        schema = tool_obj.schema
        fn = schema.get("function") or {}
        name = fn.get("name", tool_name)
        desc = fn.get("description", "No description provided.")
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        required = set(params.get("required") or [])

        lines.append(f"• {name}: {desc}")

        # Render parameters (sorted for stability)
        for pname in sorted(props.keys()):
            pinfo = props.get(pname) or {}
            ptype = pinfo.get("type", "string")
            pdesc = pinfo.get("description", "").strip()
            req = "required" if pname in required else "optional"
            if pdesc:
                lines.append(f"  - {pname} ({ptype}, {req}): {pdesc}")
            else:
                lines.append(f"  - {pname} ({ptype}, {req})")
    
    return "\n".join(lines)


def build_system_prompt_with_tools(
    tools: Dict[str, Any],
    base_instruction: str = "You are a helpful assistant.",
) -> str:
    """
    Build a system prompt that includes:
    - Base instruction
    - Tool catalog with descriptions and parameters
    - Explicit tool-calling format instructions
    
    Args:
        tools: Dictionary of tool name -> Tool object
        base_instruction: Base assistant behavior description
    
    Returns:
        Complete system prompt string
    """
    if not tools:
        return base_instruction
    
    catalog = _render_tool_catalog(tools)
    
    prompt = f"""{base_instruction}

You have access to the following tools:

{catalog}

To call a tool, emit EXACTLY this format (replace with actual tool name and arguments):
<<function_call>> {{"name": "tool_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}

Do not add commentary on the same line as the tool call.
After you receive the tool result, continue the conversation normally with a natural response.
Only use tools when they are necessary to answer the user's question.
""".strip()
    
    return prompt
