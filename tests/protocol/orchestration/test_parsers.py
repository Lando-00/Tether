import pytest
from llm_service.protocol.orchestration.parsers import JsonArgsParser

@pytest.fixture
def parser():
    """Pytest fixture to create a JsonArgsParser instance."""
    return JsonArgsParser()

class TestJsonArgsParser:
    """Test suite for the JsonArgsParser class."""

    def test_parse_empty_string(self, parser: JsonArgsParser):
        """Tests that parsing an empty string returns an empty dictionary."""
        assert parser.parse("") == {}
        assert parser.parse("   ") == {}

    def test_parse_valid_json(self, parser: JsonArgsParser):
        """Tests parsing of a valid JSON string."""
        json_string = '{"key": "value", "number": 123}'
        expected = {"key": "value", "number": 123}
        assert parser.parse(json_string) == expected

    def test_parse_timezone_argument(self, parser: JsonArgsParser):
        """Tests the special handling for timezone arguments."""
        arg_string = 'timezone="Europe/Vienna"'
        expected = {"timezone": "Europe/Vienna", "format": "iso"}
        assert parser.parse(arg_string) == expected

    def test_parse_timezone_and_format_argument(self, parser: JsonArgsParser):
        """Tests the special handling for timezone and format arguments."""
        arg_string = 'timezone="Europe/Vienna", format="rfc3339"'
        expected = {"timezone": "Europe/Vienna", "format": "rfc3339"}
        assert parser.parse(arg_string) == expected

    def test_parse_quoted_string_with_equals(self, parser: JsonArgsParser):
        """Tests parsing of a quoted string with key=value format."""
        arg_string = '"foo=bar, baz=qux"'
        expected = {"foo": "bar", "baz": "qux"}
        assert parser.parse(arg_string) == expected

    def test_parse_key_value_without_braces(self, parser: JsonArgsParser):
        """Tests parsing of key:value format without surrounding braces."""
        arg_string = '"key": "value"'
        expected = {"key": "value"}
        assert parser.parse(arg_string) == expected

    def test_parse_invalid_json_with_fallback(self, parser: JsonArgsParser):
        """Tests that invalid JSON is handled by the fallback parser."""
        # JSON is invalid because key is not a string
        arg_string = "{key: 'value'}"
        expected = {"key": "value"}
        # The _fallback_parse method is called internally
        assert parser.parse(arg_string) == expected

    def test_fallback_parse_method(self, parser: JsonArgsParser):
        """Tests the _fallback_parse method directly."""
        arg_string = "key: 'value', number: 123, bool_true: true, bool_false: false, null_val: null"
        expected = {
            "key": "value",
            "number": 123,
            "bool_true": True,
            "bool_false": False,
            "null_val": None,
        }
        assert parser._fallback_parse(arg_string) == expected
