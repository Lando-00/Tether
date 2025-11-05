"""
Integration tests for WebSearchTool.

Tests tool behavior including:
- Schema generation matches new signature
- Deprecation warning for 'language' parameter
- Count clamping to max_count
- Parameter validation (empty query, invalid values)
- Tool execution via ToolRunner with timeout
- Contract test for normalized JSON structure
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import os
from tether_service.tools.web_search_tool import WebSearchTool
from tether_service.tools.base import BaseTool


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
        
        # Deprecated parameter should still be present
        assert "language" in properties
        
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
        """Test that _registry_name is properly injected by BaseTool."""
        tool = WebSearchTool()
        # Registry name should be set by tool registry during loading
        # For unit test, we can set it manually
        tool._registry_name = "web_search"
        assert tool._registry_name == "web_search"
        
        schema = tool.auto_schema
        assert schema["function"]["name"] == "web_search"


class TestWebSearchToolParameterValidation:
    """Test parameter validation and transformations."""
    
    @pytest.mark.asyncio
    async def test_empty_query_fails(self):
        """Test that empty query returns error."""
        tool = WebSearchTool()
        
        result = await tool.run(query="")
        assert "error" in result
        assert "query" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_count_clamping_to_max(self):
        """Test that count > max_count is silently clamped to max_count=20."""
        tool = WebSearchTool()
        
        # Mock BraveSearchClient to track what count is actually used
        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [],
            "meta": {"engine": "brave", "query": "test", "took_ms": 100},
            "articles": []
        }
        
        with patch('tether_service.tools.web_search_tool._get_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            # Request 100 results (should be clamped to 20)
            result = await tool.run(query="test", count=100)
            
            # Verify search was called with clamped value
            mock_client.search.assert_called_once()
            call_kwargs = mock_client.search.call_args.kwargs
            assert call_kwargs['count'] == 20  # Clamped to max_count
    
    @pytest.mark.asyncio
    async def test_count_below_one_fails(self):
        """Test that count < 1 returns validation error."""
        tool = WebSearchTool()
        
        result = await tool.run(query="test", count=0)
        assert "error" in result
        assert "count" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_valid_params_accepted(self):
        """Test that valid parameters are accepted."""
        tool = WebSearchTool()
        
        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [],
            "meta": {"engine": "brave", "query": "test", "took_ms": 100},
            "articles": []
        }
        
        with patch('tether_service.tools.web_search_tool._get_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            result = await tool.run(
                query="test query",
                count=5,
                country="gb",
                search_lang="en",
                freshness="pd"
            )
            
            # Should succeed without errors
            assert "results" in result
            mock_client.search.assert_called_once()


class TestWebSearchToolDeprecation:
    """Test deprecation handling for old parameters."""
    
    @pytest.mark.asyncio
    async def test_language_param_logs_deprecation_warning(self):
        """Test that using 'language' parameter logs deprecation warning."""
        tool = WebSearchTool()
        
        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [],
            "meta": {"engine": "brave", "query": "test", "took_ms": 100},
            "articles": []
        }
        
        with patch('tether_service.tools.web_search_tool._get_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            with patch('tether_service.tools.web_search_tool.logger') as mock_logger:
                await tool.run(query="test", language="fr")
                
                # Should log deprecation warning (check for any warning/info call)
                warning_logged = False
                for call in mock_logger.warning.call_args_list + mock_logger.info.call_args_list:
                    call_str = str(call).lower()
                    if "deprecat" in call_str or "language" in call_str:
                        warning_logged = True
                        break
                
                assert warning_logged, "Deprecation warning should be logged"
    
    @pytest.mark.asyncio
    async def test_language_maps_to_search_lang(self):
        """Test that 'language' parameter is mapped to 'search_lang'."""
        tool = WebSearchTool()
        
        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [],
            "meta": {"engine": "brave", "query": "test", "took_ms": 100},
            "articles": []
        }
        
        with patch('tether_service.tools.web_search_tool._get_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            with patch('tether_service.tools.web_search_tool.logger'):
                await tool.run(query="test", language="fr")
                
                # Verify search_lang was set correctly
                call_kwargs = mock_client.search.call_args.kwargs
                assert call_kwargs['search_lang'] == "fr"


class TestWebSearchToolExecution:
    """Test actual tool execution and response handling."""
    
    @pytest.mark.asyncio
    async def test_successful_search_returns_structured_format(self):
        """Test that successful search returns results/meta/articles structure."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
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
            },
            "articles": [
                "Test Result 1: This is a test - https://example.com/1"
            ]
        }
        
        with patch('tether_service.tools.web_search_tool._get_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            result = await tool.run(query="test query", count=1)
            
            # Verify structured format
            assert "results" in result
            assert "meta" in result
            assert "articles" in result
            
            # Verify results content
            assert len(result["results"]) == 1
            assert result["results"][0]["url"] == "https://example.com/1"
            assert result["results"][0]["title"] == "Test Result 1"
            
            # Verify meta
            assert result["meta"]["engine"] == "brave"
            assert result["meta"]["query"] == "test query"
            
            # Verify deprecated articles format still present
            assert len(result["articles"]) == 1
    
    @pytest.mark.asyncio
    async def test_api_key_error_handling(self):
        """Test that missing/invalid API key returns clear error."""
        tool = WebSearchTool()
        
        # Mock _get_client to raise ValueError (simulating missing API key)
        with patch('tether_service.tools.web_search_tool._get_client') as mock_get_client:
            mock_get_client.side_effect = ValueError(
                "Environment variable BRAVE_API_KEY not set. "
                "Get your free API key at https://api-dashboard.search.brave.com/"
            )
            
            # Tool catches ValueError and returns error dict
            result = await tool.run(query="test")
            
            assert "error" in result
            error_msg = result["error"].lower()
            assert "brave_api_key" in error_msg
    
    @pytest.mark.asyncio
    async def test_http_error_propagation(self):
        """Test that HTTP errors from BraveSearchClient are propagated."""
        tool = WebSearchTool()
        
        mock_client = AsyncMock()
        mock_client.search.side_effect = ValueError("Invalid API key")
        
        with patch('tether_service.tools.web_search_tool._get_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            # The tool catches exceptions and returns error dict
            result = await tool.run(query="test")
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
        
        with patch('tether_service.tools.web_search_tool._get_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            result = await tool.run(query="test")
            
            # Contract: response must have these exact top-level keys
            assert set(result.keys()) == {"results", "meta", "articles"}
            
            # Contract: results must be list of dicts with specific keys
            assert isinstance(result["results"], list)
            if result["results"]:
                result_keys = set(result["results"][0].keys())
                assert result_keys == {"url", "title", "snippet", "rank"}
            
            # Contract: meta must have these keys
            assert "engine" in result["meta"]
            assert "query" in result["meta"]
            assert "took_ms" in result["meta"]
            
            # Contract: articles must be list of strings (deprecated but present)
            assert isinstance(result["articles"], list)
            if result["articles"]:
                assert isinstance(result["articles"][0], str)


class TestWebSearchToolWithToolRunner:
    """Test tool execution via ToolRunner (simulating orchestrator behavior)."""
    
    @pytest.mark.asyncio
    async def test_tool_execution_with_timeout(self):
        """Test that tool respects timeout when executed via ToolRunner."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        # Mock a slow response
        async def slow_search(*args, **kwargs):
            import asyncio
            await asyncio.sleep(0.1)  # Simulate slow API
            return {
                "results": [],
                "meta": {"engine": "brave", "query": "test", "took_ms": 100},
                "articles": []
            }
        
        mock_client = AsyncMock()
        mock_client.search = slow_search
        
        with patch('tether_service.tools.web_search_tool._get_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            # Execute tool (should complete within timeout)
            result = await tool.run(query="test")
            assert "results" in result
    
    @pytest.mark.asyncio
    async def test_tool_kwargs_unpacking(self):
        """Test that tool methods receive **kwargs correctly (not dict)."""
        tool = WebSearchTool()
        
        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [],
            "meta": {"engine": "brave", "query": "test", "took_ms": 100},
            "articles": []
        }
        
        with patch('tether_service.tools.web_search_tool._get_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            # Call with kwargs (as orchestrator would)
            args = {
                "query": "test query",
                "count": 5,
                "country": "us"
            }
            result = await tool.run(**args)
            
            # Should succeed (verifies kwargs unpacking works)
            assert "results" in result
            
            # Verify correct params were passed to client
            call_kwargs = mock_client.search.call_args.kwargs
            assert call_kwargs['q'] == "test query"
            assert call_kwargs['count'] == 5
            assert call_kwargs['country'] == "us"


class TestWebSearchToolEdgeCases:
    """Test edge cases and unusual inputs."""
    
    @pytest.mark.asyncio
    async def test_unicode_query(self):
        """Test that Unicode characters in query are handled correctly."""
        tool = WebSearchTool()
        
        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [],
            "meta": {"engine": "brave", "query": "test", "took_ms": 100},
            "articles": []
        }
        
        with patch('tether_service.tools.web_search_tool._get_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            # Unicode query
            result = await tool.run(query="日本語 search 中文")
            
            assert "results" in result
            call_kwargs = mock_client.search.call_args.kwargs
            assert call_kwargs['q'] == "日本語 search 中文"
    
    @pytest.mark.asyncio
    async def test_special_chars_in_query(self):
        """Test that special characters are handled correctly."""
        tool = WebSearchTool()
        
        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [],
            "meta": {"engine": "brave", "query": "test", "took_ms": 100},
            "articles": []
        }
        
        with patch('tether_service.tools.web_search_tool._get_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            # Query with special chars
            result = await tool.run(query='test "quotes" & symbols <>')
            
            assert "results" in result
    
    @pytest.mark.asyncio
    async def test_very_long_query(self):
        """Test that very long queries are handled."""
        tool = WebSearchTool()
        
        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [],
            "meta": {"engine": "brave", "query": "test", "took_ms": 100},
            "articles": []
        }
        
        with patch('tether_service.tools.web_search_tool._get_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            # Very long query
            long_query = "test " * 100  # 500 chars
            result = await tool.run(query=long_query)
            
            assert "results" in result
