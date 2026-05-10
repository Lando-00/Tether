"""Schema-only unit tests for the EchoConnector tools.

Validates the connector spec §8.1 acceptance contract that
:class:`tether.tools.base.BaseTool.auto_schema` produces correct
JSON Schema for the parameter shapes connector tools care about:

* Plain ``str``                          — ``{"type": "string"}``.
* ``list[str]``                          — ``{"type": "array", "items": {"type": "string"}}``.
* ``Optional[str]`` (with default None)  — ``{"type": "string", "nullable": true}``.
* Required vs optional                   — params without defaults appear
                                           in ``required``; params with
                                           defaults do not.

These complement the Phase 4.5 lifecycle integration tests in
``tests/integration/test_connectors_lifecycle.py``: the schema layer is
checked here directly so a regression in ``BaseTool._python_type_to_json_schema``
fails loudly on a single, fast unit test rather than buried inside an
HTTP/lifecycle scenario.

Citations: connector spec §8.1; synthesis §4 Phase 4.5 step 47e-47f.
"""
from __future__ import annotations

from tether.connectors.types import ConnectorState
from tests.fixtures.echo_connector import (
    EchoConfirmSendTool,
    EchoMarkSeenTool,
    EchoSendTool,
    EchoWithOptionalTool,
)


def _ready() -> ConnectorState:
    return ConnectorState.READY


# ---------------------------------------------------------------------------
# echo_send — plain str
# ---------------------------------------------------------------------------


def test_echo_send_schema_basic_string() -> None:
    tool = EchoSendTool(outbox=[], state_provider=_ready)
    schema = tool.auto_schema

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo_send"

    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["message"]["type"] == "string"
    assert "message" in params["required"]


# ---------------------------------------------------------------------------
# echo_mark_seen — list[str]  (spec §8.1 acceptance)
# ---------------------------------------------------------------------------


def test_echo_mark_seen_schema_is_array_of_string() -> None:
    """Connector spec §8.1: list[str] arg advertises array of string."""
    tool = EchoMarkSeenTool(seen=set(), state_provider=_ready)
    schema = tool.auto_schema

    assert schema["function"]["name"] == "echo_mark_seen"

    params = schema["function"]["parameters"]
    event_ids = params["properties"]["event_ids"]
    assert event_ids["type"] == "array"
    assert event_ids["items"]["type"] == "string"
    assert "event_ids" in params["required"]


# ---------------------------------------------------------------------------
# echo_with_optional — Optional[str] = None  (spec §8.1 acceptance)
# ---------------------------------------------------------------------------


def test_echo_with_optional_schema_handles_optional() -> None:
    """Connector spec §8.1: Optional[T] with default produces nullable
    schema and is excluded from ``required``."""
    tool = EchoWithOptionalTool(log=[], state_provider=_ready)
    schema = tool.auto_schema

    assert schema["function"]["name"] == "echo_with_optional"

    params = schema["function"]["parameters"]
    props = params["properties"]

    # text is plain str, required.
    assert props["text"]["type"] == "string"
    assert "text" in params["required"]

    # label is Optional[str] with default None: nullable + optional.
    label = props["label"]
    assert label["type"] == "string"
    assert label.get("nullable") is True
    assert "label" not in params["required"]


# ---------------------------------------------------------------------------
# echo_confirm_send — context kw is excluded from schema
# ---------------------------------------------------------------------------


def test_echo_confirm_send_schema_excludes_context() -> None:
    """The ``context`` kw-only parameter is plumbed by ``BaseTool.invoke``
    and MUST NOT leak into the schema (synthesis §4 Phase 4 step 41a)."""
    tool = EchoConfirmSendTool(confirmed_drafts=[], state_provider=_ready)
    schema = tool.auto_schema

    assert schema["function"]["name"] == "echo_confirm_send"

    params = schema["function"]["parameters"]
    props = params["properties"]
    assert props["draft_id"]["type"] == "string"
    assert "draft_id" in params["required"]
    assert "context" not in props
    assert "context" not in params["required"]
