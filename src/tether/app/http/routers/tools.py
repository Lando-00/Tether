"""``/api/v1/tools`` route — list registered tools with their auto-schemas.

Mirrors the ``/api/v1/models`` and ``/api/v1/connectors`` pattern: read
the live registry off ``app.state.gen_svc`` (the Engine), serialize each
tool's ``auto_schema`` (name, description, parameters JSON Schema).

Used by ``tether-cli``'s ``\\tools`` slash command, and by ad-hoc
clients that want to introspect the running server without sending a
chat turn (the same information appears on every ``message_start`` event,
but having a standalone GET keeps client UX simple).

Disabling a tool is a context-budget feature as much as a permission one: a
disabled tool is not advertised to the provider, cannot be dispatched, and its
past calls/results are dropped from the model-facing history. On small models
that matters — every unused schema and stale tool result competes for
attention. See :meth:`tether.engine.Engine.set_tool_enabled`.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/tools", tags=["tools"])


class ToolEnableRequest(BaseModel):
    """Body for ``POST /tools/{name}/enabled``."""

    enabled: bool = Field(
        description="True to enable the tool, False to disable it.",
    )


@router.get("", response_model=List[Dict[str, Any]])
def list_tools(request: Request) -> List[Dict[str, Any]]:
    """Return the registered tools, sorted by name.

    Each entry contains:
      * ``name``: tool name (registry key, e.g. ``"web_search"``).
      * ``description``: first paragraph of the tool's docstring.
      * ``parameters``: JSON Schema object describing arguments.
      * ``enabled``: whether the model can currently see and call it.

    ``BaseTool.auto_schema`` emits the OpenAI-style nested shape
    (``{"type": "function", "function": {...}}``). This route flattens it so
    clients get ``name``/``description``/``parameters`` at the top level
    regardless of the tool implementation.

    Empty list when no tools are configured (settings.tools.registry / .enabled
    both empty). 200 always; this route never errors on a healthy server.
    """
    engine = request.app.state.gen_svc
    states = getattr(engine, "list_tool_states", None)
    if callable(states):
        return list(states())

    # Fallback for engines predating runtime toggling (and for test doubles).
    tools_dict = getattr(engine, "tools", {}) or {}
    out: List[Dict[str, Any]] = []
    for name in sorted(tools_dict.keys()):
        schema = getattr(tools_dict[name], "auto_schema", None)
        if schema is None:
            continue
        body = schema.get("function", schema)
        out.append(
            {
                "name": body.get("name", name),
                "enabled": True,
                "description": body.get("description", ""),
                "parameters": body.get("parameters", {}),
            }
        )
    return out


@router.post("/{name}/enabled", response_model=Dict[str, Any])
def set_tool_enabled(
    name: str, body: ToolEnableRequest, request: Request
) -> Dict[str, Any]:
    """Enable or disable a tool for subsequent turns.

    Takes effect on the next turn — orchestrators are constructed per turn, so
    no restart is needed. Returns 404 for an unknown tool name.
    """
    engine = request.app.state.gen_svc
    setter = getattr(engine, "set_tool_enabled", None)
    if not callable(setter):
        raise HTTPException(
            status_code=501,
            detail="This server does not support runtime tool toggling.",
        )
    if not setter(name, body.enabled):
        known = sorted(getattr(engine, "tools", {}) or {})
        raise HTTPException(
            status_code=404,
            detail=f"Unknown tool '{name}'. Known tools: {known}.",
        )
    return {"name": name, "enabled": body.enabled}
