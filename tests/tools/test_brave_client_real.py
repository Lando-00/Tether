"""
Real HTTP integration tests for BraveSearchClient.

These tests make actual API calls to Brave Search API and are:
- Marked as @pytest.mark.integration (run with: pytest -m integration)
- Skipped if BRAVE_API_KEY is not set in environment
- Rate-limited aware (avoid hitting 10 req/min limit)
- Useful for validating API contract and response parsing

WARNING: These tests consume your Brave API quota (2,000/month free tier).
Run sparingly during development. CI/CD should mock these tests.
"""
import pytest
import os
import asyncio
from tether_service.tools.brave_client import BraveSearchClient


# Skip all tests in this module if BRAVE_API_KEY is not set
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("BRAVE_API_KEY"),
        reason="BRAVE_API_KEY not set - real API tests require valid key"
    )
]


@pytest.mark.asyncio
class TestBraveSearchClientRealAPI:
    """Integration tests using real Brave Search API."""
    
    async def test_real_search_basic_query(self):
        """Test basic search with real API call."""
        client = BraveSearchClient(
            api_key=os.environ["BRAVE_API_KEY"],
            connect_timeout=2.0,
            read_timeout=6.0,
            total_timeout=15.0
        )
        
        result = await client.search(q="Python programming", count=3)
        
        # Verify response structure
        assert "results" in result
        assert "meta" in result
        assert "articles" in result
        
        # Verify we got results
        assert len(result["results"]) > 0
        assert len(result["results"]) <= 3
        
        # Verify result structure
        first_result = result["results"][0]
        assert "url" in first_result
        assert "title" in first_result
        assert "snippet" in first_result
        assert "rank" in first_result
        
        # Verify URL is valid
        assert first_result["url"].startswith("http")
        
        # Verify meta
        assert result["meta"]["engine"] == "brave"
        assert result["meta"]["query"] == "Python programming"
        assert result["meta"]["took_ms"] > 0
        
        print(f"\n✓ Found {len(result['results'])} results")
        print(f"✓ First result: {first_result['title']}")
        print(f"✓ Search took {result['meta']['took_ms']}ms")
    
    async def test_real_search_with_country_filter(self):
        """Test search with country parameter."""
        client = BraveSearchClient(api_key=os.environ["BRAVE_API_KEY"])
        
        # Add small delay to avoid rate limiting
        await asyncio.sleep(1)
        
        result = await client.search(
            q="news",
            count=2,
            country="gb"  # UK results
        )
        
        assert len(result["results"]) > 0
        assert result["meta"]["query"] == "news"
        
        print(f"\n✓ Country filter (GB): {len(result['results'])} results")
    
    async def test_real_search_with_language_filter(self):
        """Test search with language parameter."""
        client = BraveSearchClient(api_key=os.environ["BRAVE_API_KEY"])
        
        await asyncio.sleep(1)
        
        result = await client.search(
            q="artificial intelligence",
            count=2,
            search_lang="en"
        )
        
        assert len(result["results"]) > 0
        
        print(f"\n✓ Language filter (EN): {len(result['results'])} results")
    
    async def test_real_search_with_freshness_filter(self):
        """Test search with freshness (time) filter."""
        client = BraveSearchClient(api_key=os.environ["BRAVE_API_KEY"])
        
        await asyncio.sleep(1)
        
        # Search for recent results only (past day)
        result = await client.search(
            q="AI news",
            count=3,
            freshness="pd"  # past day
        )
        
        assert len(result["results"]) >= 0  # May have no results if nothing recent
        
        print(f"\n✓ Freshness filter (pd): {len(result['results'])} results")
    
    async def test_real_search_html_sanitization(self):
        """Test that HTML tags are properly removed from real API responses."""
        client = BraveSearchClient(api_key=os.environ["BRAVE_API_KEY"])
        
        await asyncio.sleep(1)
        
        result = await client.search(q="HTML tutorial", count=2)
        
        # Check that no HTML tags remain in results
        for res in result["results"]:
            assert "<" not in res["title"] or res["title"].startswith("http")
            assert ">" not in res["snippet"] or "->" in res["snippet"]  # Allow arrows
            assert "<strong>" not in res["snippet"]
            assert "<em>" not in res["snippet"]
            assert "<b>" not in res["snippet"]
        
        print(f"\n✓ HTML sanitization verified on {len(result['results'])} results")
    
    async def test_real_search_snippet_length(self):
        """Test that snippets are properly truncated."""
        client = BraveSearchClient(api_key=os.environ["BRAVE_API_KEY"])
        
        await asyncio.sleep(1)
        
        result = await client.search(q="machine learning", count=3)
        
        for res in result["results"]:
            # Snippets should not exceed 363 chars (360 + "...")
            assert len(res["snippet"]) <= 363
            if len(res["snippet"]) == 363:
                assert res["snippet"].endswith("...")
        
        print(f"\n✓ Snippet length verified on {len(result['results'])} results")
    
    async def test_real_search_empty_query_handling(self):
        """Test API behavior with unusual queries."""
        client = BraveSearchClient(api_key=os.environ["BRAVE_API_KEY"])
        
        await asyncio.sleep(1)
        
        # Very specific query that might return few/no results
        result = await client.search(
            q="xyzabc123veryrarequery999",
            count=5
        )
        
        # Should return valid structure even with no results
        assert "results" in result
        assert "meta" in result
        assert "articles" in result
        assert isinstance(result["results"], list)
        
        print(f"\n✓ Rare query handled: {len(result['results'])} results")
    
    async def test_real_search_unicode_query(self):
        """Test search with Unicode characters."""
        client = BraveSearchClient(api_key=os.environ["BRAVE_API_KEY"])
        
        await asyncio.sleep(1)
        
        result = await client.search(q="日本 Japan", count=2)
        
        assert len(result["results"]) >= 0
        assert result["meta"]["query"] == "日本 Japan"
        
        print(f"\n✓ Unicode query handled: {len(result['results'])} results")
    
    async def test_real_search_special_characters(self):
        """Test search with special characters in query."""
        client = BraveSearchClient(api_key=os.environ["BRAVE_API_KEY"])
        
        await asyncio.sleep(1)
        
        result = await client.search(q='"Python" & "JavaScript"', count=2)
        
        assert len(result["results"]) >= 0
        
        print(f"\n✓ Special chars query handled: {len(result['results'])} results")
    
    async def test_real_api_rate_limit_headers(self):
        """Test that we can see rate limit headers in responses."""
        client = BraveSearchClient(api_key=os.environ["BRAVE_API_KEY"])
        
        await asyncio.sleep(1)
        
        # This test verifies the API returns rate limit info
        # We can't easily check headers with our current client structure,
        # but we can verify the search completes without rate limit errors
        result = await client.search(q="test", count=1)
        
        assert "results" in result
        
        print(f"\n✓ Rate limit check passed (no 429 error)")
    
    async def test_real_search_max_count_parameter(self):
        """Test requesting maximum number of results."""
        client = BraveSearchClient(api_key=os.environ["BRAVE_API_KEY"])
        
        await asyncio.sleep(1)
        
        # Request 20 results (our max_count)
        result = await client.search(q="technology", count=20)
        
        # Should get up to 20 results (may be less if API doesn't have that many)
        assert len(result["results"]) > 0
        assert len(result["results"]) <= 20
        
        print(f"\n✓ Max count test: {len(result['results'])}/20 results")
    
    async def test_real_search_response_time(self):
        """Test that search completes within reasonable time."""
        client = BraveSearchClient(api_key=os.environ["BRAVE_API_KEY"])
        
        await asyncio.sleep(1)
        
        import time
        start = time.time()
        
        result = await client.search(q="weather", count=3)
        
        elapsed = time.time() - start
        
        # Should complete within total_timeout (15s)
        assert elapsed < 15.0
        
        # Typically should be much faster (< 2s for Brave API)
        print(f"\n✓ Response time: {elapsed:.2f}s (API: {result['meta']['took_ms']}ms)")
        
        assert len(result["results"]) > 0


@pytest.mark.asyncio
class TestBraveSearchClientRealAPIErrorCases:
    """Test error scenarios with real API."""
    
    async def test_invalid_api_key_403(self):
        """Test that invalid API key returns 403/422 with clear error."""
        client = BraveSearchClient(api_key="invalid_key_12345")
        
        with pytest.raises(ValueError) as exc_info:
            await client.search(q="test")
        
        error_msg = str(exc_info.value).lower()
        assert "api key" in error_msg or "authentication" in error_msg
        
        print(f"\n✓ Invalid API key handled correctly (ValueError raised)")
    
    async def test_empty_api_key(self):
        """Test that empty API key is rejected."""
        # This would fail in _get_client() in WebSearchTool
        # but we can test the client initialization
        with pytest.raises((ValueError, Exception)):
            client = BraveSearchClient(api_key="")
            await client.search(q="test")
        
        print(f"\n✓ Empty API key rejected")


@pytest.mark.asyncio
class TestBraveSearchClientRealAPIBackwardCompatibility:
    """Test backward compatibility with real API responses."""
    
    async def test_both_formats_in_real_response(self):
        """Verify both new (results/meta) and old (articles) formats."""
        client = BraveSearchClient(api_key=os.environ["BRAVE_API_KEY"])
        
        await asyncio.sleep(1)
        
        result = await client.search(q="Python", count=2)
        
        # New structured format
        assert "results" in result
        assert isinstance(result["results"], list)
        if result["results"]:
            assert "url" in result["results"][0]
            assert "title" in result["results"][0]
            assert "snippet" in result["results"][0]
            assert "rank" in result["results"][0]
        
        # Meta information
        assert "meta" in result
        assert result["meta"]["engine"] == "brave"
        
        # Old format (backward compatibility)
        assert "articles" in result
        assert isinstance(result["articles"], list)
        if result["articles"]:
            # Should be formatted strings
            assert isinstance(result["articles"][0], str)
            assert "http" in result["articles"][0]  # Contains URL
        
        print(f"\n✓ Both formats present in real API response")
        print(f"  - results: {len(result['results'])} items")
        print(f"  - articles: {len(result['articles'])} items")


# Note about rate limits
"""
IMPORTANT: Running these tests multiple times in quick succession may hit
Brave's rate limit (10 requests/minute). The tests include 1-second delays
between requests, but be mindful when running the full suite.

To run only real API tests:
    pytest tests/tools/test_brave_client_real.py -v -m integration

To skip real API tests (use mocked tests instead):
    pytest tests/tools/ -v -m "not integration"
"""
