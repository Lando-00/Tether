import pytest
from unittest.mock import patch, MagicMock
from llm_service.protocol.orchestration.executor import DefaultToolExecutor

@pytest.fixture
def executor():
    """Provides a DefaultToolExecutor instance for tests."""
    return DefaultToolExecutor()

def test_execute_success(executor):
    """Tests successful execution of a tool."""
    tool_name = "__tool_test_tool"
    args = {"arg1": "value1"}
    expected_result = {"status": "success"}

    with patch('llm_service.protocol.orchestration.executor.execute_tool', return_value=expected_result) as mock_execute:
        result = executor.execute(tool_name, args)
        
        mock_execute.assert_called_once_with(tool_name, args)
        assert result == expected_result

def test_execute_tool_not_found(executor):
    """Tests the response when a tool is not found (ValueError)."""
    tool_name = "__tool_non_existent_tool"
    args = {}

    with patch('llm_service.protocol.orchestration.executor.execute_tool', side_effect=ValueError("Tool not found")) as mock_execute:
        result = executor.execute(tool_name, args)
        
        mock_execute.assert_called_once_with(tool_name, args)
        assert "error" in result
        assert "Tool not found" in result["error"]

def test_execute_invalid_arguments(executor):
    """Tests the response when invalid arguments are provided (TypeError)."""
    tool_name = "__tool_test_tool"
    args = {"wrong_arg": "value"}
    error_message = "Invalid arguments for tool"

    with patch('llm_service.protocol.orchestration.executor.execute_tool', side_effect=TypeError(error_message)) as mock_execute:
        result = executor.execute(tool_name, args)
        
        mock_execute.assert_called_once_with(tool_name, args)
        assert "error" in result
        assert "Invalid arguments" in result["error"]
        assert error_message in result["error"]

def test_execute_general_error(executor):
    """Tests the response for a general exception during execution."""
    tool_name = "__tool_test_tool"
    args = {}
    error_message = "Something went wrong"

    with patch('llm_service.protocol.orchestration.executor.execute_tool', side_effect=Exception(error_message)) as mock_execute:
        result = executor.execute(tool_name, args)
        
        mock_execute.assert_called_once_with(tool_name, args)
        assert "error" in result
        assert "Execution error" in result["error"]
        assert error_message in result["error"]

@pytest.mark.parametrize("invalid_name", [None, "", 123, {}])
def test_execute_invalid_tool_name(executor, invalid_name):
    """Tests that an invalid tool name is handled correctly."""
    with patch('llm_service.protocol.orchestration.executor.execute_tool') as mock_execute:
        result = executor.execute(invalid_name, {})
        
        mock_execute.assert_not_called()
        assert "error" in result
        assert result["error"] == "Invalid tool name"
