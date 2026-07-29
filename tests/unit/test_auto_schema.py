"""
Tests for BaseTool.auto_schema type mapping.

Acceptance A5: Synthesis §6 row 5 / R8 / connector spec §8.1.

Tests cover: list[str], Optional[str], Literal[...], int=5 default,
*args/**kwargs filtered, list[int], Optional[Literal[...]], dict[str,Any],
and TimeTool's actual schema.
"""
from typing import Any, Dict, Literal, Optional

from tether.tools.base import BaseTool
from tether.tools.time_tool import TimeTool

# ---------------------------------------------------------------------------
# Minimal fixture tools for schema testing
# ---------------------------------------------------------------------------

class _ListStrTool(BaseTool):
    """Connector spec §8.1 acceptance: list[str] param."""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(self, event_ids: list[str], wait_seconds: int = 5, **kwargs) -> dict:
        return {"got": event_ids}


class _OptionalStrTool(BaseTool):
    """Optional[str] param."""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(self, label: Optional[str] = None, **kwargs) -> dict:
        return {}


class _LiteralTool(BaseTool):
    """Literal param."""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(self, mode: Literal["a", "b", "c"], **kwargs) -> dict:
        return {}


class _DefaultIntTool(BaseTool):
    """int with default — must not appear in required."""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(self, count: int = 5, **kwargs) -> dict:
        return {}


class _VarArgsTool(BaseTool):
    """*args and **kwargs must be filtered from schema."""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(self, name: str, *args, **kwargs) -> dict:
        return {}


class _ListIntTool(BaseTool):
    """list[int] param."""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(self, counts: list[int], **kwargs) -> dict:
        return {}


class _OptionalLiteralTool(BaseTool):
    """Optional[Literal[...]] composition."""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(self, fmt: Optional[Literal["json", "text"]] = None, **kwargs) -> dict:
        return {}


class _DictTool(BaseTool):
    """dict[str, Any] param."""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(self, metadata: Dict[str, Any], **kwargs) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _props(tool: BaseTool) -> dict:
    return tool.auto_schema["function"]["parameters"]["properties"]


def _required(tool: BaseTool) -> list:
    return tool.auto_schema["function"]["parameters"]["required"]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestAutoSchemaListStr:
    """A5a — connector spec §8.1 acceptance test."""

    def test_list_str_is_array_of_string(self):
        props = _props(_ListStrTool())
        ei = props["event_ids"]
        assert ei["type"] == "array"
        assert ei["items"] == {"type": "string"}
        assert "description" in ei  # description field present (may be empty)

    def test_list_str_is_required(self):
        assert "event_ids" in _required(_ListStrTool())

    def test_int_with_default_not_required(self):
        props = _props(_ListStrTool())
        assert "wait_seconds" in props
        assert props["wait_seconds"]["type"] == "integer"
        assert props["wait_seconds"]["default"] == 5
        assert "wait_seconds" not in _required(_ListStrTool())


class TestAutoSchemaOptionalStr:
    """A5b — Optional[str] → string with nullable: true."""

    def test_optional_str_is_nullable_string(self):
        props = _props(_OptionalStrTool())
        label = props["label"]
        assert label["type"] == "string"
        assert label.get("nullable") is True

    def test_optional_str_has_default_none(self):
        props = _props(_OptionalStrTool())
        assert props["label"]["default"] is None

    def test_optional_str_not_required(self):
        assert "label" not in _required(_OptionalStrTool())


class TestAutoSchemaLiteral:
    """A5c — Literal["a","b","c"] → string with enum."""

    def test_literal_has_enum(self):
        props = _props(_LiteralTool())
        mode = props["mode"]
        assert mode["type"] == "string"
        assert set(mode["enum"]) == {"a", "b", "c"}

    def test_literal_is_required(self):
        assert "mode" in _required(_LiteralTool())


class TestAutoSchemaDefaultInt:
    """A5d — int = 5 → integer + default: 5, NOT in required."""

    def test_int_with_default_has_default_key(self):
        props = _props(_DefaultIntTool())
        assert props["count"]["type"] == "integer"
        assert props["count"]["default"] == 5

    def test_int_with_default_not_in_required(self):
        assert "count" not in _required(_DefaultIntTool())


class TestAutoSchemaVarArgsFiltered:
    """A5e — *args and **kwargs must not appear in schema."""

    def test_varargs_filtered(self):
        props = _props(_VarArgsTool())
        assert "args" not in props
        assert "kwargs" not in props

    def test_named_param_still_present(self):
        props = _props(_VarArgsTool())
        assert "name" in props
        assert props["name"]["type"] == "string"


class TestAutoSchemaListInt:
    """A5f — list[int] → array of integer."""

    def test_list_int_is_array_of_integer(self):
        props = _props(_ListIntTool())
        counts = props["counts"]
        assert counts["type"] == "array"
        assert counts["items"] == {"type": "integer"}


class TestAutoSchemaOptionalLiteral:
    """A5g — Optional[Literal["json","text"]] composes correctly."""

    def test_optional_literal_is_nullable_with_enum(self):
        props = _props(_OptionalLiteralTool())
        fmt = props["fmt"]
        assert fmt["type"] == "string"
        assert set(fmt["enum"]) == {"json", "text"}
        assert fmt.get("nullable") is True

    def test_optional_literal_not_required(self):
        assert "fmt" not in _required(_OptionalLiteralTool())


class TestAutoSchemaDictType:
    """A5h — dict[str, Any] → {"type": "object"}."""

    def test_dict_param_maps_to_object(self):
        props = _props(_DictTool())
        assert props["metadata"]["type"] == "object"


class TestTimeToolSchema:
    """Verify TimeTool's schema after the auto_schema upgrade."""

    def setup_method(self):
        self.tool = TimeTool()
        self.schema = self.tool.auto_schema
        self.props = self.schema["function"]["parameters"]["properties"]
        self.req = self.schema["function"]["parameters"]["required"]

    def test_timezone_is_string_with_default_utc(self):
        tz = self.props["timezone"]
        assert tz["type"] == "string"
        assert tz["default"] == "UTC"

    def test_timezone_not_required(self):
        assert "timezone" not in self.req

    def test_format_has_enum(self):
        fmt = self.props["format"]
        assert fmt["type"] == "string"
        assert set(fmt["enum"]) == {"iso", "rfc2822", "human"}

    def test_format_has_default_human(self):
        assert self.props["format"]["default"] == "human"

    def test_format_not_required(self):
        assert "format" not in self.req
