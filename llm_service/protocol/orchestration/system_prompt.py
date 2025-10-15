# llm_service/protocol/orchestration/system_prompt.py
from __future__ import annotations

import os
from textwrap import dedent
from typing import Dict, List, Any, Iterable


def _render_tool_catalog(tools: Iterable[Dict[str, Any]]) -> str:
    """
    Render a readable list of tools from OpenAI-style function schemas.
    Each item looks like:
      • __tool_foo: Short description
        - param (type, required/optional): description
    """
    lines: list[str] = []
    for tool in tools or []:
        fn = tool.get("function") or {}
        name = fn.get("name", "")
        desc = fn.get("description", "No description provided.")
        params = fn.get("parameters") or {}
        props = params.get("properties") or {}
        required = set(params.get("required") or [])

        if not name:
            # Skip entries that aren't well-formed
            continue

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


def build_tooling_system_prompt(
    tool_defs: List[Dict[str, Any]],
    *,
    tool_prefix: str = "__tool_",
    locale_hint: str | None = None,
) -> Dict[str, str]:
    """
    Build a single 'system' message that:
      - Lists available tools (from OpenAI-style function schemas)
      - Defines the *exact* tool call format
      - Tells the model to produce concise prose when no tool is needed

    Args:
        tool_defs: List of tool definitions with the shape:
                   {"type":"function","function":{"name": str, "description": str,
                    "parameters":{"type":"object","properties":{...},"required":[...]}}}
        tool_prefix: The prefix your runtime expects for tool calls (default: "__tool_")
        locale_hint: Optional text like "Europe/Dublin" you want to include as a hint.

    Returns:
        {"role":"system","content": "..."} — ready to prepend to your chat history.
    """
    # return {"role": "system", "content": "You are a helpful assistant that uses tools when appropriate."}

    # Render tool catalog
    # catalog = _render_tool_catalog(tool_defs)
    # Neutral, format-agnostic prompt
    # content = dedent(f"""\n
    #     You are a helpful assistant with access to external tools.
    #     Tool specifications are provided to the model via the tools parameter.
    #     - Respond normally when no tools are required.
    #     - Use tools as specified when appropriate.

    #     Output the tool-call and wait for tool response.
    #             """).strip("\n")
    content = "You are a helpful assistant that uses tools when appropriate. You can use a tool by outputting '<<function_call>> \\{function name with params in structured json\\}".strip()
    return {"role": "system", "content": content}


def extract_tool_schemas(tool_defs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return tool definitions in OpenAI-compatible schema form.
    """
    # Our tool_defs are already in the correct OpenAI function schema format
    return tool_defs
