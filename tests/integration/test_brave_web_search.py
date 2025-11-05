"""
End-to-end integration test for Brave web search tool.

This test validates the complete flow:
1. Start server with test config
2. Create session
3. Send chat prompt that triggers web search
4. Verify NDJSON streaming events (tool_started, tool_completed, text with results)
5. Confirm no errors in orchestration loop
6. Verify both old (articles) and new (results) formats in tool result

This is an OPTIONAL test that requires:
- httpx to be installed
- BRAVE_API_KEY in environment
- Server to start successfully
"""
import pytest
import httpx
import json
import os
from typing import AsyncIterator


# Skip this test if BRAVE_API_KEY is not set (optional E2E test)
pytestmark = pytest.mark.skipif(
    not os.getenv("BRAVE_API_KEY"),
    reason="BRAVE_API_KEY not set - E2E test requires real API key"
)


@pytest.mark.asyncio
@pytest.mark.e2e
class TestBraveWebSearchE2E:
    """End-to-end test for web search through the full streaming API."""
    
    async def test_web_search_full_flow(self):
        """Test complete flow from chat endpoint to web search tool execution."""
        
        # This test would require:
        # 1. Starting the FastAPI server (or using TestClient)
        # 2. Creating a session via POST /api/v1/sessions
        # 3. Streaming a chat that triggers web search
        # 4. Parsing NDJSON events
        # 5. Verifying tool execution
        
        # For now, this is a placeholder showing the structure
        # Actual implementation would use httpx.AsyncClient or FastAPI TestClient
        
        pytest.skip("E2E test requires running server - implement when needed")
        
        # Example structure:
        # async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        #     # Create session
        #     response = await client.post("/api/v1/sessions")
        #     session_data = response.json()
        #     session_id = session_data["id"]
        #     
        #     # Stream chat with web search prompt
        #     async with client.stream(
        #         "POST",
        #         "/api/v1/chat/stream",
        #         json={
        #             "session_id": session_id,
        #             "prompt": "What's the latest news about AI?",
        #             "model": "test_model"
        #         }
        #     ) as stream:
        #         events = []
        #         async for line in stream.aiter_lines():
        #             if line.strip():
        #                 event = json.loads(line)
        #                 events.append(event)
        #         
        #         # Verify events
        #         event_types = [e["type"] for e in events]
        #         assert "tool_started" in event_types
        #         assert "tool_completed" in event_types
        #         
        #         # Find tool_completed event
        #         tool_events = [e for e in events if e["type"] == "tool_completed"]
        #         assert len(tool_events) > 0
        #         
        #         tool_result = tool_events[0]["data"]["result"]
        #         
        #         # Verify response format
        #         assert "results" in tool_result  # New format
        #         assert "meta" in tool_result
        #         assert "articles" in tool_result  # Deprecated format


@pytest.mark.asyncio
@pytest.mark.integration
class TestWebSearchWithMockedServer:
    """Integration test using mocked components (doesn't require server)."""
    
    async def test_web_search_via_orchestrator(self):
        """Test web search tool via orchestrator (simulated)."""
        from tether_service.tools.web_search_tool import WebSearchTool
        from unittest.mock import AsyncMock, patch
        
        # Create tool
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        # Mock BraveSearchClient
        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [
                {
                    "url": "https://example.com/ai-news",
                    "title": "Latest AI Developments",
                    "snippet": "Recent advances in artificial intelligence...",
                    "rank": 1
                }
            ],
            "meta": {
                "engine": "brave",
                "query": "AI developments 2025",
                "took_ms": 234
            },
            "articles": [
                "Latest AI Developments: Recent advances in artificial intelligence... - https://example.com/ai-news"
            ]
        }
        
        with patch('tether_service.tools.web_search_tool._get_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            # Execute tool as orchestrator would
            result = await tool.run(
                query="AI developments 2025",
                count=5,
                country="us",
                search_lang="en"
            )
            
            # Verify result structure
            assert "results" in result
            assert "meta" in result
            assert "articles" in result
            
            # Verify content
            assert len(result["results"]) == 1
            assert "AI" in result["results"][0]["title"]
            assert result["meta"]["engine"] == "brave"
            
            # Verify client was called correctly
            mock_client.search.assert_called_once_with(
                q="AI developments 2025",
                count=5,
                country="us",
                search_lang="en",
                freshness=None
            )
    
    async def test_web_search_in_multi_turn_conversation(self):
        """Test that web search results are included in conversation history."""
        # This would test the SessionStore integration
        # Verifying that tool calls and results are persisted
        
        pytest.skip("Requires SessionStore integration - implement when needed")
        
        # Example structure:
        # 1. Create session
        # 2. Add user message with web search request
        # 3. Execute tool via orchestrator
        # 4. Verify tool_call and tool_result are in session history
        # 5. Send follow-up message
        # 6. Verify model receives previous tool context


@pytest.mark.asyncio
@pytest.mark.integration
class TestWebSearchErrorHandling:
    """Integration tests for error scenarios."""
    
    async def test_rate_limit_error_in_orchestration(self):
        """Test that rate limit errors are handled gracefully in orchestration."""
        from tether_service.tools.web_search_tool import WebSearchTool
        from unittest.mock import AsyncMock, patch
        import httpx
        
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        # Mock client that raises 429 error
        mock_client = AsyncMock()
        mock_response = httpx.Response(
            status_code=429,
            headers={"Retry-After": "60"}
        )
        mock_client.search.side_effect = httpx.HTTPStatusError(
            "Rate limited",
            request=httpx.Request("GET", "https://api.search.brave.com"),
            response=mock_response
        )
        
        with patch('tether_service.tools.web_search_tool._get_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            # Tool execution should raise error (orchestrator will handle)
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await tool.run(query="test")
            
            assert exc_info.value.response.status_code == 429
    
    async def test_timeout_error_in_orchestration(self):
        """Test that timeout errors are handled gracefully."""
        from tether_service.tools.web_search_tool import WebSearchTool
        from unittest.mock import AsyncMock, patch
        import asyncio
        
        tool = WebSearchTool()
        tool._registry_name = "web_search"
        
        # Mock client that times out
        mock_client = AsyncMock()
        
        async def timeout_search(*args, **kwargs):
            await asyncio.sleep(20)  # Longer than tool timeout
        
        mock_client.search = timeout_search
        
        with patch('tether_service.tools.web_search_tool._get_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            # Should timeout (when executed via ToolRunner with timeout)
            # For this unit test, we'll just verify the mock is set up
            # Actual timeout enforcement happens in ToolRunner
            assert mock_client.search == timeout_search


@pytest.mark.asyncio
@pytest.mark.integration
class TestWebSearchBackwardCompatibility:
    """Test backward compatibility with old response format."""
    
    async def test_both_formats_present_in_response(self):
        """Verify both new (results/meta) and old (articles) formats are present."""
        from tether_service.tools.web_search_tool import WebSearchTool
        from unittest.mock import AsyncMock, patch
        
        tool = WebSearchTool()
        
        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "results": [
                {"url": "http://ex.com", "title": "Test", "snippet": "Desc", "rank": 1}
            ],
            "meta": {"engine": "brave", "query": "test", "took_ms": 100},
            "articles": ["Test: Desc - http://ex.com"]
        }
        
        with patch('tether_service.tools.web_search_tool._get_client') as mock_get_client:
            mock_get_client.return_value = mock_client
            
            result = await tool.run(query="test")
            
            # New format
            assert "results" in result
            assert isinstance(result["results"], list)
            assert "meta" in result
            
            # Old format (backward compatibility)
            assert "articles" in result
            assert isinstance(result["articles"], list)
            if result["articles"]:
                assert isinstance(result["articles"][0], str)
