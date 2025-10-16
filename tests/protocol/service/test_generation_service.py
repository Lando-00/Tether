import pytest
from unittest.mock import MagicMock, AsyncMock
from llm_service.protocol.service.generation_service import GenerationService
from llm_service.protocol.api.schemas import GenerateRequest

@pytest.fixture
def generation_service():
    model = MagicMock()
    context = MagicMock()
    return GenerationService(model, context)

def test_get_and_filter_tools(generation_service):
    # Should return all available tools if none requested
    generation_service._get_available_tools = MagicMock(return_value=[{"function": {"name": "foo"}}])
    tools = generation_service._get_and_filter_tools(None)
    assert tools == [{"function": {"name": "foo"}}]
    # Should filter requested tools
    requested = [{"function": {"name": "foo"}}, {"function": {"name": "bar"}}]
    filtered = generation_service._get_and_filter_tools(requested)
    assert filtered == [{"function": {"name": "foo"}}]

def test_abort_generation(generation_service):
    # Simulate active tasks
    mock_task = MagicMock()
    mock_task.done.return_value = False
    generation_service._active_generations["session1"] = {mock_task}
    assert generation_service.abort_generation("session1") is True
    # No active generation
    assert generation_service.abort_generation("session2") is False

def test_generate_stream_session_not_found(generation_service):
    # Context returns None for session
    generation_service.context.get_session.return_value = None
    req = GenerateRequest(session_id="bad", prompt="hi", model_name="m", tools=None)
    with pytest.raises(Exception):
        generation_service.generate_stream(req)
