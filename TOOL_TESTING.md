# Tool Calling Testing for MLC-LLM Models

This directory contains scripts to test and evaluate tool calling capabilities of MLC-LLM models within the MCP architecture, including both standard function calling and our custom JSON-based approach.

## Overview

The tool calling functionality lets models interact with external functions, allowing them to perform tasks like retrieving weather information, performing calculations, and more. These scripts help you:

1. Test how well your models understand when to use tools
2. Evaluate the accuracy of tool parameter extraction
3. Measure performance across different models
4. Visualize tool calling results

## Tool Calling Approaches

Our system supports two approaches for tool calling:

1. **Standard Function Calling**: For models that natively support OpenAI-style function calling (like GPT-4, Claude, etc.)
2. **Custom JSON-based Tool Calls**: For models without native function calling support (like Qwen2.5-Instruct), we implement a custom approach where:
   - A dynamic system prompt is generated with tool definitions
   - The model is instructed to output tool calls in a special JSON format
   - The system parses these JSON blocks and executes the corresponding tools

## Available Test Tools

Several tools are provided for testing:

- `time_tools.py`: Time-related functions like getting the current time for a timezone
- `weather_tools.py`: Weather forecasting and condition reporting (simulated data)
- Additional test tools embedded in the test script:
  - `calculate_sum`: Simple addition function
  - `calculate_product`: Multiplication function
  - `extract_entities`: Simple named entity extraction
  - `search_knowledge_base`: Simulated knowledge base search

## How to Run Tests

### Step 1: Start the MCP Server

First, ensure the MLC-LLM service is running:

```bash
python -m llm_service.app
```

### Step 2: Run the Test Script

#### For Standard Function Calling Tests:

```bash
python test_tool_call.py --model "Qwen2.5-7B-q4f16_0-MLC" --save
```

#### For Custom JSON-based Tool Calling Tests:

```bash
python test_dynamic_tool_prompt.py --model "Qwen2.5-7B-q4f16_0-MLC"
```

Arguments:
- `--model`: The name of the model to test (required)
- `--save`: Save results to a JSON file for later analysis (optional, for test_tool_call.py)

### Step 3: Visualize Results

After running tests on one or more models, visualize the results:

```bash
python visualize_tool_results.py
```

This generates three visualization files:
- `tool_calling_success_rates.png`: Overall success rates by model
- `tool_calling_execution_times.png`: Average execution times by model
- `tool_performance_by_model.png`: Breakdown of success rates by tool and model

## Creating Custom Tools

You can extend the test suite by creating custom tools. Follow this pattern:

1. Create a new file in the `llm_service/tools/` directory
2. Import the `register_tool` decorator
3. Define your function with type annotations
4. Add a descriptive docstring
5. Apply the `@register_tool` decorator

Example:

```python
from typing import List, Dict
from ..tools import register_tool

@register_tool
def my_custom_tool(param1: str, param2: int = 0) -> Dict[str, Any]:
    """Clear description of what your tool does.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        
    Returns:
        Description of the return value
    """
    # Tool implementation
    return {"result": f"Processed {param1} with value {param2}"}
```

## Adding Test Cases

To add new test cases, modify the `get_test_cases()` function in `test_tool_calling.py`:

```python
def get_test_cases() -> List[ToolTestCase]:
    return [
        # Existing test cases...
        ToolTestCase(
            prompt="Your test prompt here",
            expected_tool="your_tool_name",
            expected_params={"param1": "value1", "param2": "value2"}
        ),
    ]
```

## Dynamic System Prompt Example

For models that don't support native function calling, our system generates a dynamic system prompt with all available tools:

```
You are an AI assistant with access to the following tools:

Tool: get_current_time
Description: Get the current time for a specific timezone
Parameters:
  - timezone (string): The timezone to get the current time for (e.g., "America/New_York", "Europe/London")

Tool: calculate_sum
Description: Calculate the sum of two numbers
Parameters:
  - a (number): First number
  - b (number): Second number

To call a function, respond with:
<tool_call>
{
  "name": "function_name",
  "arguments": {
    "arg1": "value1",
    "arg2": "value2"
  }
}
</tool_call>
```

## Custom JSON-based Tool Call Format

For models without native function calling support, we use a special format in the system prompt:

```
<tool_call>
{
  "name": "function_name",
  "arguments": {
    "arg1": "value1",
    "arg2": "value2"
  }
}
</tool_call>
```

The protocol layer automatically:
1. Generates this system prompt with all available tools
2. Parses the output for tool call blocks
3. Executes the corresponding tools
4. Updates the conversation history

## Performance Considerations

- Lower the temperature (e.g., 0.1) when testing tool calling to get more consistent results
- Tool calling typically requires more tokens and processing time than regular completions
- Models may vary significantly in their tool calling capabilities
- The custom JSON-based approach works well with instruction-tuned models like Qwen2.5-Instruct
- Not all models may respond correctly to the custom format; experiment with different prompt formats if needed

## Troubleshooting

If you encounter issues:

1. Verify the server is running and accessible at the configured URL
2. Check that all required packages are installed
3. Examine the model's raw responses if tools aren't being called
4. Try different prompt formulations if a specific tool isn't being used
5. For custom JSON format issues, check if the model is correctly following the format instructions
6. Increase temperature slightly if the model is too rigid in its responses

## Customizing the Tool Call Format

You can customize the tool call format by modifying the `generate_dynamic_system_prompt` method in the `ProtocolComponent` class. This allows you to experiment with different formats that might work better with specific models.
