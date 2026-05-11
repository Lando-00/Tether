"""Integration tests for ``GET /api/v1/tools`` (cli-polish branch).

Mirrors the test layout in test_connector_routes.py: minimal Engine with
mocked dependencies + the tools router under /api/v1, then asserts the
route returns the tool auto-schemas sorted by name.
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from tether.app.http.routers.tools import router as tools_router
from tether.engine import Engine


class _StubTool:
    """Bare-minimum tool stand-in: exposes ``auto_schema`` as expected by the route."""

    def __init__(self, name: str, description: str, properties: Dict[str, Any]) -> None:
        self._schema = {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
            },
        }

    @property
    def auto_schema(self) -> Dict[str, Any]:
        return self._schema


def _make_app(tools: Dict[str, Any]) -> FastAPI:
    """Construct a minimal FastAPI app with the tools router and a fake Engine."""
    from tether.protocol.parsers.sliding import SlidingParser

    engine = Engine(
        provider=AsyncMock(),
        parser=SlidingParser(),
        session_store=AsyncMock(),
        tools=tools,
        system_prompt="",
    )
    app = FastAPI()
    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(tools_router)
    app.include_router(v1)
    app.state.gen_svc = engine
    return app


def test_get_tools_empty_registry_returns_empty_list():
    """With no tools registered, /api/v1/tools returns []."""
    app = _make_app({})
    client = TestClient(app)
    resp = client.get("/api/v1/tools")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_tools_returns_schemas_sorted_by_name():
    """Tools are listed with their auto_schema, sorted by name for stable output."""
    tools = {
        "weather": _StubTool(
            "weather",
            "Get weather at a location.",
            {"location": {"type": "string"}},
        ),
        "time": _StubTool(
            "time",
            "Get current time.",
            {"timezone": {"type": "string"}},
        ),
    }
    app = _make_app(tools)
    client = TestClient(app)

    resp = client.get("/api/v1/tools")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    # Sorted by name: "time" before "weather".
    assert [t["name"] for t in body] == ["time", "weather"]
    # Each entry carries the full schema shape the auto_schema property emits.
    time_schema = body[0]
    assert time_schema["description"] == "Get current time."
    assert "parameters" in time_schema
    assert "timezone" in time_schema["parameters"]["properties"]


def test_get_tools_skips_tools_missing_auto_schema():
    """Defensive: a tool object without ``auto_schema`` is silently omitted."""

    class _BareTool:
        pass

    tools = {
        "good": _StubTool("good", "ok", {"x": {"type": "string"}}),
        "broken": _BareTool(),  # no auto_schema attribute
    }
    app = _make_app(tools)
    client = TestClient(app)
    resp = client.get("/api/v1/tools")
    assert resp.status_code == 200
    body = resp.json()
    assert [t["name"] for t in body] == ["good"]
