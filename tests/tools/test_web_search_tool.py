"""
Integration tests for WebSearchTool.

Tests tool behavior including:
- Schema generation matches new signature
- Tool invocation via the registry-facing API (BaseTool.invoke)
- Pydantic validation rejecting invalid inputs (Style A)
- Mocked tool execution
- Contract test for normalized JSON structure

Style A migration (synthesis §4 Phase 4 step 43): tests now exercise the
public ``tool.invoke({...})`` API rather than calling the typed ``run``
helper directly. Bounds previously checked by hand inside ``run`` are now
enforced by ``WebSearchInputs`` (Pydantic v2) and surface as
:class:`pydantic.ValidationError` from ``invoke``.

Lifecycle migration (synthesis §4 Phase 4 step 44; §6 row 17): the legacy
``_get_client()`` helper has been removed in favour of a long-lived
:class:`BraveSearchClient` opened in :meth:`startup`. Tests that need a
mock client now assign ``tool._client = mock_client`` directly (bypassing
:meth:`startup` for targeted unit tests). The lifecycle itself is covered
by ``tests/unit/tools/test_web_search_tool_lifecycle.py``.
"""
import pytest
from pydantic import ValidationError
from unittest.mock import AsyncMock
from tether.tools.web_search_tool import WebSearchTool, WebSearchInputs
from tether.tools.base import BaseTool


class TestWebSearchToolSchema:
    """Test schema generation and tool metadata."""
    
    def test_tool_inherits_from_base(self):
        """Verify WebSearchTool properly inherits from BaseTool."""
        tool = WebSearchTool()
        assert isinstance(tool, BaseTool)
    
    def test_schema_has_correct_parameters(self):
        """Test that auto-generated schema includes new Brave params."""
        tool = WebSearchTool()
        schema = tool.auto_schema
        
        # Verify schema structure
        assert "function" in schema
        assert "name" in schema["function"]
        assert "parameters" in schema["function"]
        
        params = schema["function"]["parameters"]
        properties = params["properties"]
        
        # New Brave parameters should be present
        assert "query" in properties
        assert "count" in properties
        assert "country" in properties
        assert "search_lang" in properties
        assert "freshness" in properties
        
        # Old NewsAPI parameters should NOT be present
        assert "sources" not in properties
        assert "domains" not in properties
        assert "exclude_domains" not in properties
        assert "sort_by" not in properties
        assert "from_param" not in properties
        assert "to" not in properties
        assert "page" not in properties
    
    def test_schema_required_fields(self):
        """Test that only 'query' is required."""
        tool = WebSearchTool()
        schema = tool.auto_schema
        
        required = schema["function"]["parameters"]["required"]
        assert required == ["query"]
        assert len(required) == 1
    
    def test_registry_name_injection(self):
        """Test that the @tool decorator name is reflected in the schema."""
        tool = WebSearchTool()
        # The @tool(name="web_search") decorator on WebSearchTool sets the
        # registry name at class definition time; no per-instance plumbing.
        schema = tool.auto_schema
        assert schema["function"]["name"] == "web_search"


class TestWebSearchToolParameterValidation:
    """Test parameter validation via Pydantic (Style A)."""

    @pytest.mark.asyncio
    async def test_empty_query_raises_validation_error(self):
        """Empty query must raise ValidationError (Pydantic enforces
        ``min_length=1`` after ``_strip_query``)."""
        tool = WebSearchTool()

        with pytest.raises(ValidationError):
            await tool.invoke({"query": ""})

    @pytest.mark.asyncio
    async def test_whitespace_only_query_raises_validation_error(self):
        """Whitespace-only query is rejected (matches the previous
        hand-rolled ``query.strip()`` + emptiness check)."""
        tool = WebSearchTool()

        with pytest.raises(ValidationError):
            await tool.invoke({"query": "   "})

    @pytest.mark.asyncio
    async def test_count_above_max_raises_validation_error(self):
        """count > 20 now raises ValidationError instead of clamping
        silently. Pydantic ``le=20`` enforces the bound declared in
        the schema (``maximum: 20``)."""
        tool = WebSearchTool()

        with pytest.raises(ValidationError):
            await tool.invoke({"query": "test", "count": 100})

    @pytest.mark.asyncio
    async def test_count_below_one_raises_validation_error(self):
        """count < 1 raises ValidationError (Pydantic ``ge=1``)."""
        tool = WebSearchTool()

        with pytest.raises(ValidationError):
            await tool.invoke({"query": "test", "count": 0})

    @pytest.mark.asyncio
    async def test_valid_params_accepted(self):
        """Valid parameters are accepted via the invoke shim."""
        tool = WebSearchTool()

        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [],
            "meta": {"engine": "brave", "query": "test", "took_ms": 100},
            "articles": []
        }

        # Bypass startup() — we're unit-testing run() dispatch only.
        tool._client = mock_client

        result = await tool.invoke({
            "query": "test query",
            "count": 5,
            "country": "gb",
            "search_lang": "en",
            "freshness": "pd",
        })

        assert "results" in result
        mock_client.search.assert_called_once()


class TestWebSearchToolDeprecation:
    """Test handling of language-related parameters."""

    @pytest.mark.asyncio
    async def test_search_lang_parameter(self):
        """Verify 'search_lang' parameter is passed through to the client."""
        tool = WebSearchTool()

        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [],
            "meta": {"engine": "brave", "query": "test", "took_ms": 100},
            "articles": []
        }

        tool._client = mock_client

        await tool.invoke({"query": "test", "search_lang": "fr"})

        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs['search_lang'] == "fr"


class TestWebSearchToolExecution:
    """Test actual tool execution and response handling."""

    @pytest.mark.asyncio
    async def test_successful_search_returns_structured_format(self):
        """Successful search returns the results/meta structure."""
        tool = WebSearchTool()

        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [
                {
                    "url": "https://example.com/1",
                    "title": "Test Result 1",
                    "snippet": "This is a test",
                    "rank": 1
                }
            ],
            "meta": {
                "engine": "brave",
                "query": "test query",
                "took_ms": 123
            }
        }

        tool._client = mock_client

        result = await tool.invoke({"query": "test query", "count": 1})

        assert "results" in result
        assert "meta" in result

        assert len(result["results"]) == 1
        assert result["results"][0]["url"] == "https://example.com/1"
        assert result["results"][0]["title"] == "Test Result 1"

        assert result["meta"]["engine"] == "brave"
        assert result["meta"]["query"] == "test query"

    @pytest.mark.asyncio
    async def test_api_key_error_handling(self):
        """Missing/invalid API key returns a clear error dict (transport
        layer; NOT the validation layer).

        Lifecycle migration: missing-key now manifests as ``_client``
        staying ``None`` after :meth:`startup`, and :meth:`run` returns
        the structured error dict directly. Synthesis §4 Phase 4
        step 44; §6 row 17."""
        tool = WebSearchTool()

        # Simulate the missing-key branch of startup(): the tool was
        # constructed but no BraveSearchClient was opened.
        assert tool._client is None

        result = await tool.invoke({"query": "test"})

        assert "error" in result
        error_msg = result["error"].lower()
        assert "brave_api_key" in error_msg

    @pytest.mark.asyncio
    async def test_http_error_propagation(self):
        """HTTP errors from BraveSearchClient surface as error dicts."""
        tool = WebSearchTool()

        mock_client = AsyncMock()
        mock_client.search.side_effect = ValueError("Invalid API key")

        tool._client = mock_client

        result = await tool.invoke({"query": "test"})
        assert "error" in result
        assert "Invalid API key" in result["error"]


class TestWebSearchToolContractTest:
    """Contract tests to ensure response format stability."""

    @pytest.mark.asyncio
    async def test_response_format_contract(self):
        """Snapshot test for normalized JSON structure."""
        tool = WebSearchTool()

        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [
                {
                    "url": "https://example.com",
                    "title": "Test",
                    "snippet": "Description",
                    "rank": 1
                }
            ],
            "meta": {
                "engine": "brave",
                "query": "test",
                "took_ms": 100
            },
            "articles": ["Test: Description - https://example.com"]
        }

        tool._client = mock_client

        result = await tool.invoke({"query": "test"})

        assert set(result.keys()) == {"results", "meta", "articles"}

        assert isinstance(result["results"], list)
        if result["results"]:
            result_keys = set(result["results"][0].keys())
            assert result_keys == {"url", "title", "snippet", "rank"}

        assert "engine" in result["meta"]
        assert "query" in result["meta"]
        assert "took_ms" in result["meta"]

        assert isinstance(result["articles"], list)
        if result["articles"]:
            assert isinstance(result["articles"][0], str)


class TestWebSearchToolWithToolRunner:
    """Test tool execution via the registry-facing invoke API."""

    @pytest.mark.asyncio
    async def test_tool_execution_with_timeout(self):
        """Tool completes within timeout when executed via invoke()."""
        tool = WebSearchTool()

        async def slow_search(*args, **kwargs):
            import asyncio
            await asyncio.sleep(0.1)
            return {
                "results": [],
                "meta": {"engine": "brave", "query": "test", "took_ms": 100},
                "articles": []
            }

        mock_client = AsyncMock()
        mock_client.search = slow_search

        tool._client = mock_client

        result = await tool.invoke({"query": "test"})
        assert "results" in result

    @pytest.mark.asyncio
    async def test_tool_args_dispatch(self):
        """Tool methods receive the dict via the Style A invoke dispatch."""
        tool = WebSearchTool()

        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [],
            "meta": {"engine": "brave", "query": "test", "took_ms": 100},
            "articles": []
        }

        tool._client = mock_client

        args = {
            "query": "test query",
            "count": 5,
            "country": "us"
        }
        result = await tool.invoke(args)

        assert "results" in result

        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs['q'] == "test query"
        assert call_kwargs['count'] == 5
        assert call_kwargs['country'] == "us"


class TestWebSearchToolEdgeCases:
    """Test edge cases and unusual inputs."""

    @pytest.mark.asyncio
    async def test_unicode_query(self):
        """Unicode characters in query are handled correctly."""
        tool = WebSearchTool()

        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [],
            "meta": {"engine": "brave", "query": "test", "took_ms": 100},
            "articles": []
        }

        tool._client = mock_client

        result = await tool.invoke({"query": "日本語 search 中文"})

        assert "results" in result
        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs['q'] == "日本語 search 中文"

    @pytest.mark.asyncio
    async def test_special_chars_in_query(self):
        """Special characters in query are handled correctly."""
        tool = WebSearchTool()

        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [],
            "meta": {"engine": "brave", "query": "test", "took_ms": 100},
            "articles": []
        }

        tool._client = mock_client

        result = await tool.invoke({"query": 'test "quotes" & symbols <>'})

        assert "results" in result

    @pytest.mark.asyncio
    async def test_long_query_within_bounds(self):
        """Long queries (≤400 chars) are accepted; queries above the
        bound raise ValidationError (Pydantic ``max_length=400``)."""
        tool = WebSearchTool()

        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [],
            "meta": {"engine": "brave", "query": "test", "took_ms": 100},
            "articles": []
        }

        tool._client = mock_client

        # 300 chars — within max_length=400
        long_query = "abc " * 75
        result = await tool.invoke({"query": long_query.strip()})
        assert "results" in result

        with pytest.raises(ValidationError):
            # 500 chars — exceeds max_length=400
            await tool.invoke({"query": "x" * 500})
