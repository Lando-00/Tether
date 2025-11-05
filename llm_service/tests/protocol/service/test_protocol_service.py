import pytest
from unittest.mock import MagicMock
from llm_service.protocol.service.protocol_service import ProtocolService

@pytest.fixture
def protocol_service():
    # Use MagicMock for model and context components
    model = MagicMock()
    context = MagicMock()
    return ProtocolService(model, context)

def test_get_available_models(protocol_service):
    protocol_service.model.get_available_models.return_value = [{"name": "test-model"}]
    models = protocol_service.get_available_models()
    assert models == [{"name": "test-model"}]

def test_get_available_tools(protocol_service):
    # Should call get_all_tool_definitions (mocked via import)
    tools = protocol_service.get_available_tools()
    assert isinstance(tools, list)

def test_create_session(protocol_service):
    protocol_service.context.create_session.return_value = {"session_id": "abc123"}
    session = protocol_service.create_session()
    assert session["session_id"] == "abc123"

def test_list_sessions(protocol_service):
    protocol_service.context.list_sessions.return_value = [{"session_id": "abc123"}]
    sessions = protocol_service.list_sessions()
    assert sessions == [{"session_id": "abc123"}]

def test_delete_session(protocol_service):
    protocol_service.context.delete_session.return_value = True
    assert protocol_service.delete_session("abc123") is True

def test_delete_all_sessions(protocol_service):
    protocol_service.context.delete_all_sessions.return_value = 2
    assert protocol_service.delete_all_sessions() == 2

def test_get_session_messages(protocol_service):
    protocol_service.context.get_messages.return_value = ["msg1", "msg2"]
    msgs = protocol_service.get_session_messages("abc123")
    assert msgs == ["msg1", "msg2"]

def test_unload_model(protocol_service):
    result = protocol_service.unload_model(MagicMock(model_name="test-model", device="cpu"))
    assert "unloaded successfully" in result["detail"]

def test_abort_generation(protocol_service):
    # Patch the _get_generation_service method to return a mock with abort_generation
    mock_gen_service = MagicMock()
    mock_gen_service.abort_generation.return_value = True
    protocol_service._get_generation_service = MagicMock(return_value=mock_gen_service)
    assert protocol_service.abort_generation("abc123") is True
