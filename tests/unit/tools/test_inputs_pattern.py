"""Tests for the ``BaseTool.Inputs`` ClassVar pattern (Style A).

A1 acceptance: synthesis §4 Phase 4 step 43; A2 step 5.

Covers:

* The ClassVar default is ``None`` (Style B remains the default).
* :meth:`BaseTool.invoke` dispatches to ``run(inputs)`` when ``Inputs``
  is set, and to ``run(**args)`` when it is ``None``.
* Pydantic ``ValidationError`` surfaces from ``invoke`` for invalid
  args under Style A.
* ``context`` keyword threading works for both Style A and Style B
  ``run`` signatures.
* Auto-generated schemas strip Pydantic noise (``$defs``, ``title``,
  ``anyOf [..., {"type": "null"}]``).
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Type

import pytest
from pydantic import Field, ValidationError

from tether.core.types import ToolExecutionContext
from tether.tools.base import BaseTool, ToolInputs

# ---------------------------------------------------------------------------
# Fixture inputs models + tools
# ---------------------------------------------------------------------------


class _FakeInputs(ToolInputs):
    """Minimal Style A inputs model for dispatch tests."""

    x: int = Field(..., ge=0, le=100, description="Bounded integer input.")
    label: Optional[str] = Field(default=None, description="Optional label.")


class _StyleATool(BaseTool):
    """Fixture: Style A tool — declares ``Inputs = _FakeInputs``."""

    Inputs: ClassVar[Optional[Type[ToolInputs]]] = _FakeInputs

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(self, inputs: _FakeInputs) -> Dict[str, Any]:
        return {"got_x": inputs.x, "got_label": inputs.label}


class _StyleAContextTool(BaseTool):
    """Fixture: Style A tool whose ``run`` opts into ``context``."""

    Inputs: ClassVar[Optional[Type[ToolInputs]]] = _FakeInputs

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(
        self,
        inputs: _FakeInputs,
        *,
        context: Optional[ToolExecutionContext] = None,
    ) -> Dict[str, Any]:
        return {
            "got_x": inputs.x,
            "got_context": context is not None,
            "session_id": context.session_id if context else None,
        }


class _StyleBTool(BaseTool):
    """Fixture: legacy Style B tool — kwargs unpack, no ``Inputs``."""

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(self, x: int, label: Optional[str] = None) -> Dict[str, Any]:
        return {"got_x": x, "got_label": label}


class _StyleBContextTool(BaseTool):
    """Fixture: Style B tool that opts into ``context``."""

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(
        self,
        x: int,
        *,
        context: Optional[ToolExecutionContext] = None,
    ) -> Dict[str, Any]:
        return {
            "got_x": x,
            "got_context": context is not None,
            "session_id": context.session_id if context else None,
        }


# ---------------------------------------------------------------------------
# A1.1 — Inputs default
# ---------------------------------------------------------------------------


class TestInputsClassVarDefault:
    def test_basetool_inputs_default_none(self):
        """A bare ``BaseTool`` subclass without override has Inputs=None."""
        assert _StyleBTool.Inputs is None

    def test_basetool_inputs_class_attribute(self):
        """The ClassVar is reachable via both the class and the instance."""
        assert BaseTool.Inputs is None
        tool = _StyleBTool()
        assert tool.Inputs is None


# ---------------------------------------------------------------------------
# A1.2 — invoke dispatch (Style A vs Style B)
# ---------------------------------------------------------------------------


class TestInvokeDispatchStyleA:
    @pytest.mark.asyncio
    async def test_invoke_dispatches_to_run_inputs_when_set(self):
        """Style A: ``invoke({'x': 1})`` calls ``run(_FakeInputs(x=1))``."""
        tool = _StyleATool()
        result = await tool.invoke({"x": 1, "label": "hi"})
        assert result == {"got_x": 1, "got_label": "hi"}

    @pytest.mark.asyncio
    async def test_invoke_validates_via_pydantic(self):
        """Style A: invalid args raise ``pydantic.ValidationError``."""
        tool = _StyleATool()
        with pytest.raises(ValidationError):
            await tool.invoke({"x": -1})  # ge=0 fails
        with pytest.raises(ValidationError):
            await tool.invoke({"x": 200})  # le=100 fails

    @pytest.mark.asyncio
    async def test_invoke_extra_field_rejected(self):
        """Style A inherits ``extra='forbid'`` from ``ToolInputs``."""
        tool = _StyleATool()
        with pytest.raises(ValidationError):
            await tool.invoke({"x": 1, "unknown_field": "boom"})

    @pytest.mark.asyncio
    async def test_invoke_missing_required_raises(self):
        """Style A: missing required field surfaces as ValidationError."""
        tool = _StyleATool()
        with pytest.raises(ValidationError):
            await tool.invoke({})


class TestInvokeDispatchStyleB:
    @pytest.mark.asyncio
    async def test_invoke_dispatches_to_kwargs_when_inputs_none(self):
        """Style B (legacy): ``invoke({'x': 1})`` calls ``run(x=1)``."""
        tool = _StyleBTool()
        result = await tool.invoke({"x": 1, "label": "hello"})
        assert result == {"got_x": 1, "got_label": "hello"}


# ---------------------------------------------------------------------------
# A1.3 — context threading for both styles
# ---------------------------------------------------------------------------


class TestInvokeContextThreading:
    @pytest.mark.asyncio
    async def test_invoke_passes_context_to_style_a(self):
        """Style A tool whose run accepts ``context`` receives it via
        the dispatch."""
        tool = _StyleAContextTool()
        ctx = ToolExecutionContext(session_id="sid-42", turn_id="t-1")
        result = await tool.invoke({"x": 7}, context=ctx)
        assert result["got_x"] == 7
        assert result["got_context"] is True
        assert result["session_id"] == "sid-42"

    @pytest.mark.asyncio
    async def test_invoke_style_a_context_omitted_when_run_excludes_it(self):
        """Style A tool whose ``run(inputs)`` does NOT declare context
        must not break when invoke is called WITH a context."""
        tool = _StyleATool()  # run signature has no context param
        ctx = ToolExecutionContext(session_id="sid-x", turn_id="t-x")
        result = await tool.invoke({"x": 5}, context=ctx)
        assert result == {"got_x": 5, "got_label": None}

    @pytest.mark.asyncio
    async def test_invoke_passes_context_to_style_b(self):
        """Style B tool whose run accepts ``context`` receives it."""
        tool = _StyleBContextTool()
        ctx = ToolExecutionContext(session_id="sid-99", turn_id="t-99")
        result = await tool.invoke({"x": 3}, context=ctx)
        assert result == {
            "got_x": 3,
            "got_context": True,
            "session_id": "sid-99",
        }

    @pytest.mark.asyncio
    async def test_invoke_style_b_no_context_param_does_not_leak_kwarg(self):
        """Style B tool whose ``run(x, label=None)`` doesn't declare
        context must NOT receive an unexpected ``context`` kwarg
        (would raise TypeError)."""
        tool = _StyleBTool()
        ctx = ToolExecutionContext(session_id="sid", turn_id="t")
        # Just must not raise
        result = await tool.invoke({"x": 1}, context=ctx)
        assert result["got_x"] == 1


# ---------------------------------------------------------------------------
# A1.4 — auto_schema strips Pydantic noise (Style A only)
# ---------------------------------------------------------------------------


class TestAutoSchemaStripsPydanticNoise:
    def _params(self, tool: BaseTool) -> Dict[str, Any]:
        return tool.auto_schema["function"]["parameters"]

    def test_auto_schema_no_top_level_title(self):
        """Pydantic adds a top-level ``title`` (e.g. "FakeInputs"); the
        emitted schema strips it because OpenAI tool schemas don't
        carry it."""
        params = self._params(_StyleATool())
        assert "title" not in params

    def test_auto_schema_no_per_property_title(self):
        """Pydantic adds a per-property ``title`` (e.g. "X"); strip it."""
        params = self._params(_StyleATool())
        for prop in params["properties"].values():
            assert "title" not in prop, f"unexpected title in {prop!r}"

    def test_auto_schema_no_additional_properties(self):
        """Pydantic encodes ``extra='forbid'`` as
        ``additionalProperties: false``; strip it (validate-time
        concern, not wire schema)."""
        params = self._params(_StyleATool())
        assert "additionalProperties" not in params

    def test_auto_schema_no_dollar_defs(self):
        """Flat input models don't reference nested types; assert
        ``$defs`` is absent (defensive — emitted by Pydantic for
        nested schemas)."""
        params = self._params(_StyleATool())
        assert "$defs" not in params

    def test_optional_collapses_to_nullable(self):
        """``Optional[str]`` → ``{"type": "string", "nullable": true}``,
        not ``{"anyOf": [{"type": "string"}, {"type": "null"}]}``."""
        params = self._params(_StyleATool())
        label = params["properties"]["label"]
        assert label["type"] == "string"
        assert label.get("nullable") is True
        assert "anyOf" not in label

    def test_required_propagates(self):
        """Required Pydantic fields surface in the JSON schema's
        ``required`` array."""
        params = self._params(_StyleATool())
        assert "x" in params["required"]
        assert "label" not in params["required"]

    def test_field_constraints_propagate(self):
        """Pydantic ``ge``/``le`` constraints surface as
        ``minimum``/``maximum`` in the JSON schema."""
        params = self._params(_StyleATool())
        x = params["properties"]["x"]
        assert x["type"] == "integer"
        assert x["minimum"] == 0
        assert x["maximum"] == 100

    def test_field_description_propagates(self):
        """Pydantic ``Field(description=...)`` surfaces as the property
        ``description`` key."""
        params = self._params(_StyleATool())
        assert params["properties"]["x"]["description"] == "Bounded integer input."


# ---------------------------------------------------------------------------
# A1.5 — Style A schema also reports the @tool decorator name
# ---------------------------------------------------------------------------


class TestStyleAToolNameInSchema:
    def test_style_a_tool_uses_class_name_as_default(self):
        """Undecorated Style A tools fall back to the class name."""
        tool = _StyleATool()
        # Not decorated, so name falls through to the class name.
        assert tool.auto_schema["function"]["name"] == "_StyleATool"
