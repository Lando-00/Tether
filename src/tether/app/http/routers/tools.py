"""``/api/v1/tools`` route — list registered tools with their auto-schemas.

Mirrors the ``/api/v1/models`` and ``/api/v1/connectors`` pattern: read
the live registry off ``app.state.gen_svc`` (the Engine), serialize each
tool's ``auto_schema`` (name, description, parameters JSON Schema).

Used by ``tether-cli``'s ``\\tools`` slash command, and by ad-hoc
clients that want to introspect the running server without sending a
chat turn (the same information appears on every ``message_start`` event,
but having a standalone GET keeps client UX simple).
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Request

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("", response_model=List[Dict[str, Any]])
def list_tools(request: Request) -> List[Dict[str, Any]]:
    """Return the registered tools' auto-schemas, sorted by name.

    Each entry contains:
      * ``name``: tool name (registry key, e.g. ``"web_search"``).
      * ``description``: first paragraph of the tool's docstring.
      * ``parameters``: JSON Schema object describing arguments.

    Empty list when no tools are configured (settings.tools.registry / .enabled
    both empty). 200 always; this route never errors on a healthy server.
    """
    engine = request.app.state.gen_svc
    tools_dict = getattr(engine, "tools", {}) or {}
    out: List[Dict[str, Any]] = []
    for name in sorted(tools_dict.keys()):
        tool = tools_dict[name]
        schema = getattr(tool, "auto_schema", None)
        if schema is None:
            continue
        out.append(schema)
    return out
