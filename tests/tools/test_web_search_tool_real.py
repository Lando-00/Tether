"""
Real HTTP integration tests for WebSearchTool.

These tests exercise the complete WebSearchTool with actual Brave API calls:
- Full tool execution flow
- Schema validation
- Parameter validation with real responses
- Error handling with real API
- Tool execution timing

Marked as @pytest.mark.integration and require BRAVE_API_KEY.
"""
import pytest
import os
import asyncio
from tether_service.tools.web_search_tool import WebSearchTool


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("BRAVE_API_KEY"),
        reason="BRAVE_API_KEY not set - real API tests require valid key"
    )
]


@pytest.mark.asyncio
class TestWebSearchToolRealAPI:
    """Integration tests using real Brave Search API through WebSearchTool."""
    
    async def test_basic_search_with_real_api(self):
        """Test basic web search with real API."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        result = await tool.run(query="Python programming", count=3)
        
        # Verify response structure
        assert "results" in result
        assert "meta" in result
        assert "articles" in result
        
        # Verify we got results
        assert len(result["results"]) > 0
        assert len(result["results"]) <= 3
        
        # Verify result content
        first = result["results"][0]
        assert first["url"].startswith("http")
        assert len(first["title"]) > 0
        assert len(first["snippet"]) > 0
        assert first["rank"] >= 1
        
        print(f"\n✓ Web search returned {len(result['results'])} results")
        print(f"✓ First: {first['title'][:50]}...")
    
    async def test_search_with_all_parameters(self):
        """Test search with all available parameters."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        await asyncio.sleep(1)
        
        result = await tool.run(
            query="artificial intelligence news",
            count=5,
            country="us",
            search_lang="en",
            freshness="pw"  # past week
        )
        
        assert "results" in result
        assert len(result["results"]) >= 0  # May have 0 if no recent results
        
        print(f"\n✓ Full param search: {len(result['results'])} results")
    
    async def test_count_clamping_with_real_api(self):
        """Test that count > 20 is clamped correctly with real API."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        await asyncio.sleep(1)
        
        # Request 100 results (should be clamped to 20)
        result = await tool.run(query="technology", count=100)
        
        # Should get at most 20 results
        assert len(result["results"]) <= 20
        
        print(f"\n✓ Count clamping: requested 100, got {len(result['results'])}")
    
    async def test_deprecated_language_param_with_real_api(self):
        """Test deprecated 'language' parameter still works."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        await asyncio.sleep(1)
        
        # Use deprecated 'language' parameter
        result = await tool.run(
            query="machine learning",
            count=2,
            language="en"  # Deprecated, should map to search_lang
        )
        
        assert "results" in result
        assert len(result["results"]) >= 0
        
        print(f"\n✓ Deprecated language param: {len(result['results'])} results")
    
    async def test_unicode_query_with_real_api(self):
        """Test Unicode characters in query."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        await asyncio.sleep(1)
        
        result = await tool.run(
            query="東京 Tokyo",
            count=3
        )
        
        assert "results" in result
        
        print(f"\n✓ Unicode query: {len(result['results'])} results")
    
    async def test_special_characters_with_real_api(self):
        """Test special characters in query."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        await asyncio.sleep(1)
        
        result = await tool.run(
            query='"Python 3.12" features',
            count=3
        )
        
        assert "results" in result
        
        print(f"\n✓ Special chars query: {len(result['results'])} results")
    
    async def test_country_filter_with_real_api(self):
        """Test country filter with real API."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        await asyncio.sleep(1)
        
        # Search with UK country filter
        result = await tool.run(
            query="weather",
            count=3,
            country="gb"
        )
        
        assert "results" in result
        assert len(result["results"]) > 0
        
        print(f"\n✓ Country filter (GB): {len(result['results'])} results")
    
    async def test_freshness_filter_with_real_api(self):
        """Test time-based freshness filter."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        await asyncio.sleep(1)
        
        # Search for recent content only
        result = await tool.run(
            query="AI developments",
            count=5,
            freshness="pm"  # past month
        )
        
        assert "results" in result
        # May have 0 results if nothing found in timeframe
        
        print(f"\n✓ Freshness filter (pm): {len(result['results'])} results")
    
    async def test_schema_matches_execution(self):
        """Test that tool schema matches actual execution."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        # Get schema
        schema = tool.auto_schema
        params = schema["function"]["parameters"]["properties"]
        
        await asyncio.sleep(1)
        
        # Execute with all schema params
        result = await tool.run(
            query="test",
            count=2,
            country="us",
            search_lang="en",
            freshness="pd"
        )
        
        # Verify schema params are actually used
        assert "query" in params
        assert "count" in params
        assert "country" in params
        assert "search_lang" in params
        assert "freshness" in params
        
        # Verify execution succeeded
        assert "results" in result
        
        print(f"\n✓ Schema matches execution")
    
    async def test_response_format_contract_with_real_api(self):
        """Test that real API responses match expected contract."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        await asyncio.sleep(1)
        
        result = await tool.run(query="Python", count=2)
        
        # Contract: Must have these top-level keys
        assert set(result.keys()) == {"results", "meta", "articles"}
        
        # Contract: results structure
        assert isinstance(result["results"], list)
        if result["results"]:
            first = result["results"][0]
            assert set(first.keys()) == {"url", "title", "snippet", "rank"}
            assert isinstance(first["url"], str)
            assert isinstance(first["title"], str)
            assert isinstance(first["snippet"], str)
            assert isinstance(first["rank"], int)
        
        # Contract: meta structure
        assert "engine" in result["meta"]
        assert "query" in result["meta"]
        assert "took_ms" in result["meta"]
        assert result["meta"]["engine"] == "brave"
        
        # Contract: articles (deprecated) structure
        assert isinstance(result["articles"], list)
        if result["articles"]:
            assert isinstance(result["articles"][0], str)
        
        print(f"\n✓ Response format contract verified")
    
    async def test_execution_timing_with_real_api(self):
        """Test that tool execution completes in reasonable time."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        await asyncio.sleep(1)
        
        import time
        start = time.time()
        
        result = await tool.run(query="test", count=3)
        
        elapsed = time.time() - start
        
        # Should complete within tool timeout (15s)
        assert elapsed < 15.0
        
        # Typically much faster
        assert "results" in result
        
        print(f"\n✓ Execution time: {elapsed:.2f}s")


@pytest.mark.asyncio
class TestWebSearchToolRealAPIErrors:
    """Test error handling with real API."""
    
    async def test_empty_query_validation(self):
        """Test that empty query is rejected before API call."""
        tool = WebSearchTool()
        
        # Tool returns error dict instead of raising exception
        result = await tool.run(query="")
        
        assert "error" in result
        assert "query" in result["error"].lower()
        
        print(f"\n✓ Empty query validation works")
    
    async def test_invalid_count_validation(self):
        """Test that invalid count is rejected."""
        tool = WebSearchTool()
        
        # Test count=0
        result = await tool.run(query="test", count=0)
        assert "error" in result
        assert "count" in result["error"].lower()
        
        # Test count=-1
        result = await tool.run(query="test", count=-1)
        assert "error" in result
        assert "count" in result["error"].lower()
        
        print(f"\n✓ Invalid count validation works")
    
    async def test_api_key_missing_error(self):
        """Test clear error when API key is missing."""
        # Temporarily remove API key
        original_key = os.environ.get("BRAVE_API_KEY")
        
        try:
            if "BRAVE_API_KEY" in os.environ:
                del os.environ["BRAVE_API_KEY"]
            
            tool = WebSearchTool()
            
            # Tool should return error dict or raise exception
            try:
                result = await tool.run(query="test")
                # If it returns a result, it should be an error
                assert "error" in result
                error_msg = result["error"].lower()
                assert "brave_api_key" in error_msg or "api key" in error_msg
            except (ValueError, KeyError) as e:
                # Or it might raise an exception
                error_msg = str(e).lower()
                assert "brave_api_key" in error_msg or "api key" in error_msg
            
            print(f"\n✓ Missing API key error is clear")
        
        finally:
            # Restore API key
            if original_key:
                os.environ["BRAVE_API_KEY"] = original_key


@pytest.mark.asyncio
class TestWebSearchToolRealAPIOrchestration:
    """Test tool behavior in orchestration-like scenarios."""
    
    async def test_multiple_sequential_searches(self):
        """Test multiple searches in sequence (simulating conversation)."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        # First search
        result1 = await tool.run(query="Python", count=2)
        assert len(result1["results"]) > 0
        
        await asyncio.sleep(1)
        
        # Second search (different query)
        result2 = await tool.run(query="JavaScript", count=2)
        assert len(result2["results"]) > 0
        
        # Results should be different
        assert result1["meta"]["query"] != result2["meta"]["query"]
        
        print(f"\n✓ Sequential searches: Query 1: {len(result1['results'])}, Query 2: {len(result2['results'])}")
    
    async def test_search_result_as_context(self):
        """Test that search results could be used as context in conversation."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        await asyncio.sleep(1)
        
        result = await tool.run(query="climate change", count=3)
        
        # Verify results are suitable for use as context
        assert len(result["results"]) > 0
        
        # Each result should have meaningful content
        for res in result["results"]:
            assert len(res["title"]) > 0
            assert len(res["snippet"]) > 10  # Substantial snippet
            assert res["url"].startswith("http")
        
        # Articles format should be readable
        for article in result["articles"]:
            assert len(article) > 20  # Meaningful text
            assert "http" in article  # Contains URL
        
        print(f"\n✓ Search results suitable for context")
    
    async def test_tool_kwargs_format(self):
        """Test that tool works with kwargs dict (as orchestrator passes)."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        await asyncio.sleep(1)
        
        # Orchestrator passes args as dict that gets unpacked
        args = {
            "query": "machine learning",
            "count": 3,
            "country": "us"
        }
        
        result = await tool.run(**args)
        
        assert "results" in result
        assert len(result["results"]) > 0
        
        print(f"\n✓ Tool kwargs format works")


@pytest.mark.asyncio
class TestWebSearchToolRealAPIBackwardCompatibility:
    """Test backward compatibility with real responses."""
    
    async def test_articles_format_stability(self):
        """Test that deprecated 'articles' format remains stable."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        await asyncio.sleep(1)
        
        result = await tool.run(query="Python tutorial", count=3)
        
        # Both formats should be present
        assert "results" in result
        assert "articles" in result
        
        # Articles should match results count
        assert len(result["articles"]) == len(result["results"])
        
        # Each article should contain key info
        for article in result["articles"]:
            assert isinstance(article, str)
            assert "http" in article  # Contains URL
            assert len(article) > 20  # Has title and snippet
        
        print(f"\n✓ Articles format stable: {len(result['articles'])} items")
    
    async def test_transition_period_compatibility(self):
        """Test that both old and new consumers can use the response."""
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        await asyncio.sleep(1)
        
        result = await tool.run(query="AI", count=2)
        
        # New consumer: uses results/meta
        for res in result["results"]:
            new_format_url = res["url"]
            new_format_title = res["title"]
            assert new_format_url and new_format_title
        
        # Old consumer: uses articles
        for article in result["articles"]:
            assert isinstance(article, str)
            assert "http" in article
        
        print(f"\n✓ Transition period compatibility verified")


# Usage instructions
"""
To run these real API tests:

    # Run all real API tests
    pytest tests/tools/test_web_search_tool_real.py -v -m integration

    # Run specific test class
    pytest tests/tools/test_web_search_tool_real.py::TestWebSearchToolRealAPI -v

    # Run with output printed
    pytest tests/tools/test_web_search_tool_real.py -v -s -m integration

Note: These tests consume API quota. Each test run uses ~15-20 API calls.
With 2,000 calls/month free tier, you can run this suite ~100-130 times/month.
"""
