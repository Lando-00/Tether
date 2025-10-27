# Tether

**A personal experiment in local LLM inference with function calling**

Tether is a FastAPI-based service that provides session-based, streaming access to language models compiled with [MLC-LLM](https://github.com/mlc-ai/mlc-llm). This is a personal project for experimenting with running AI models locally with complete control over your data and conversations.

## 🎯 What is Tether?

Tether is my experimental playground for running large language models locally on my **Snapdragon X Elite** hardware. It started as a way to test model inference on the NPU/GPU and has evolved into a flexible service with function calling, persistent chat history, and streaming responses. 

The project explores how to give models access to personal data (like emails, calendars, files) while keeping everything local and private—no data leaves your machine.

**Current Features:**
- 🔒 **Privacy First**: Your data never leaves your machine
- 🛠️ **Function Calling**: Models can use tools (weather, web search, custom functions)
- 💬 **Session Management**: Persistent conversation history stored in SQLite
- 🌊 **Streaming API**: Token-by-token responses via NDJSON
- 🏗️ **MCP Architecture**: Clean, extensible Model-Context-Protocol design
- ⚙️ **Config-Driven**: YAML-based configuration for easy customization
- 🔌 **Extensible**: Add custom tools, providers, and parsers

## 🚀 Background

This project started as an experiment with the **Snapdragon X Elite GPU** to see how well local model inference could work on ARM-based NPU hardware. While MLC-LLM works great for this, I'm planning to add support for other providers like **Ollama** to make the tool system and personal data features accessible to anyone, regardless of hardware.

## 🎯 Next Steps

Current experiments and planned features:
- **Email Integration**: Building tools to read, organize, and summarize emails
- **Email Management**: Let models help categorize, prioritize, and draft responses
- **Ollama Support**: Add Ollama as an alternative provider for broader hardware compatibility
- **Calendar Tools**: Access and manage calendar events
- **File System Tools**: Search and summarize local documents
- **Personal Knowledge Base**: Use your own files as context for queries

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [API Documentation](#-api-documentation)
- [Tool System](#-tool-system)
- [Configuration](#%EF%B8%8F-configuration)
- [Adding Custom Tools](#-adding-custom-tools)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)

## 🏁 Quick Start

### Prerequisites

- **Python 3.10+**
- **MLC-LLM** installed ([installation guide](https://llm.mlc.ai/docs/install/mlc_llm.html))
- **A compiled MLC model** (see [Model Setup](#model-setup))
- **Windows/Linux/macOS** (tested on Windows with Snapdragon X Elite)

### Installation

1. **Clone the repository:**
```powershell
git clone https://github.com/Lando-00/Tether.git
cd Tether
```

2. **Create and activate a Python environment:**
```powershell
# Using conda (recommended)
conda create -n tether python=3.11
conda activate tether

# Or using venv
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/macOS
```

3. **Install dependencies:**
```powershell
pip install -r requirements.txt
```

4. **Set up MLC-LLM models** (see [Model Setup](#model-setup) below)

5. **Run the service:**
```powershell
python -m tether_service.app
```

The API will be available at `http://localhost:8080`. Visit `http://localhost:8080/docs` for interactive API documentation.

### Model Setup

Tether uses [MLC-LLM](https://github.com/mlc-ai/mlc-llm) for efficient model inference. You'll need compiled models in the MLC format.

#### Option 1: Download Pre-compiled Models

Visit the [MLC-LLM model repository](https://huggingface.co/mlc-ai) and download a pre-compiled model. Popular options:
- `Llama-3-8B-Instruct-q4f16_1-MLC`
- `Phi-3-mini-4k-instruct-q4f16_1-MLC`
- `Mistral-7B-Instruct-v0.3-q4f16_1-MLC`

#### Option 2: Compile Your Own Model

```powershell
# Install MLC-LLM compilation tools
pip install mlc-llm

# Compile a model (example: Llama-3-8B with 4-bit quantization)
mlc_llm compile meta-llama/Meta-Llama-3-8B-Instruct \
  --quantization q4f16_1 \
  --device auto \
  -o dist/Llama-3-8B-Instruct-q4f16_1-MLC
```

#### Expected Directory Structure

Place compiled models in the `dist/` directory:

```
Tether/
├── dist/
│   ├── libs/                           # Shared libraries
│   │   └── Llama-3-8B-q4f16_1.dll     # Model-specific DLL
│   └── Llama-3-8B-Instruct-q4f16_1-MLC/
│       ├── mlc-chat-config.json       # Model configuration
│       ├── ndarray-cache.json         # Weight metadata
│       └── params_*.bin               # Model weights
```

Configure the model in `tether_service/config/default.yml`:

```yaml
providers:
  model:
    impl: "tether_service.providers.mlc.provider.MLCProvider"
    args:
      dist_root: "dist"
      model_name: "Llama-3-8B-Instruct-q4f16_1-MLC"
      device: "auto"  # or "cuda", "vulkan", "metal"
```

### Basic Usage Example

Here's a simple Python script to interact with the API:

```python
import requests
import json

# 1. Create a session
response = requests.post("http://localhost:8080/sessions")
session = response.json()
session_id = session["session_id"]

# 2. Stream a chat completion
stream_request = {
    "session_id": session_id,
    "prompt": "What's the weather like in Dublin?",
    "model_name": "Llama-3-8B-Instruct-q4f16_1-MLC"
}

response = requests.post(
    "http://localhost:8080/chat/stream",
    json=stream_request,
    stream=True
)

# 3. Process NDJSON events
for line in response.iter_lines():
    if line:
        event = json.loads(line)
        if event["type"] == "token":
            print(event["data"]["text"], end="", flush=True)
        elif event["type"] == "tool_call":
            print(f"\n[Tool: {event['data']['name']}]")
        elif event["type"] == "tool_result":
            print(f"[Result: {event['data']['result']}]")
```

## 🏗️ Architecture

Tether follows the **Model-Context-Protocol (MCP)** architecture, which cleanly separates concerns into distinct layers. This design makes the system highly maintainable, testable, and extensible.

### MCP Layers

```
┌─────────────────────────────────────────────────────────────┐
│                      HTTP API Layer                          │
│              (FastAPI Routes, WebSocket, REST)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   Protocol Layer                             │
│  • Orchestration (tool loop, streaming coordination)        │
│  • Parsers (<<function_call>> detection)                    │
│  • Event Emission (NDJSON formatting)                       │
└──────────────┬──────────────────────┬───────────────────────┘
               │                      │
       ┌───────▼────────┐     ┌──────▼─────────┐
       │  Model Layer   │     │  Context Layer  │
       │  (Providers)   │     │  (Storage)      │
       │                │     │                 │
       │  • MLCProvider │     │  • SQLite Store │
       │  • Streaming   │     │  • Sessions     │
       │  • Inference   │     │  • History      │
       └────────────────┘     └─────────────────┘
```

### Directory Structure

```
tether_service/
├── app/                          # HTTP API layer
│   ├── __main__.py              # Entry point
│   └── http/
│       └── routers/             # FastAPI route handlers
│           ├── chat.py          # /chat/stream endpoint
│           ├── sessions.py      # Session CRUD
│           ├── models.py        # Model discovery
│           └── health.py        # Health checks
│
├── config/                       # Configuration files
│   ├── default.yml              # Main config (system prompt, tools, limits)
│   └── testing.yml              # Test configuration
│
├── core/                         # Core infrastructure
│   ├── interfaces.py            # Abstract interfaces (ModelProvider, SessionStore, etc.)
│   ├── types.py                 # Type definitions and data classes
│   ├── factory.py               # Dependency injection container
│   ├── config.py                # Config loading utilities
│   ├── logging.py               # Structured logging
│   └── tool_registry.py         # Tool discovery and registration
│
├── providers/                    # Model providers (implements ModelProvider)
│   ├── mlc/
│   │   └── provider.py          # MLC-LLM integration
│   └── dummy/
│       └── provider.py          # Mock provider for testing
│
├── context/                      # Context storage (implements SessionStore)
│   ├── sqlite_store.py          # SQLite-based persistence with WAL mode
│   ├── memory_store.py          # In-memory store for testing
│   └── schema.sql               # Database schema
│
├── protocol/                     # Protocol layer (orchestration & parsing)
│   ├── orchestration/
│   │   ├── orchestrator.py      # Main coordination loop (model → parser → tools)
│   │   └── tool_runner.py       # Tool execution with timeout
│   ├── parsers/
│   │   └── sliding.py           # Stateful parser for <<function_call>> detection
│   ├── service/
│   │   └── generation_service.py # High-level streaming API
│   └── prompts.py               # System prompt utilities
│
└── tools/                        # Tool implementations (extends BaseTool)
    ├── base.py                  # Abstract tool base class
    ├── time_tool.py             # Get current time
    ├── weather_tool.py          # Weather and forecast
    └── web_search_tool.py       # News search via NewsAPI
```

### Key Design Decisions

1. **Interface-Driven**: All major components implement abstract interfaces (`ModelProvider`, `SessionStore`, `Parser`), making them swappable and testable.

2. **Config-First**: YAML configuration drives component wiring via dependency injection (see `core/factory.py`).

3. **Stateful Parsing**: The `SlidingParser` maintains state across chunks to handle tool calls split across stream boundaries.

4. **Tool Loop**: The orchestrator can execute multiple tool calls in sequence (up to `max_tool_loops`), allowing the model to "think with tools."

5. **Event Streaming**: All outputs use a structured NDJSON event format (`token`, `tool_call`, `tool_result`, `done`).

## 📡 API Documentation

### Endpoints

#### `POST /sessions`
Create a new conversation session.

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-10-27T14:30:00Z"
}
```

#### `GET /sessions`
List all sessions.

**Response:**
```json
[
  {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "created_at": "2025-10-27T14:30:00Z"
  }
]
```

#### `GET /sessions/{session_id}/messages`
Get conversation history for a session.

**Response:**
```json
[
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi! How can I help you today?"}
]
```

#### `DELETE /sessions/{session_id}`
Delete a specific session.

#### `DELETE /sessions`
Delete all sessions.

#### `POST /chat/stream`
Stream a chat completion with function calling support.

**Request:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "prompt": "What's the weather in Dublin?",
  "model_name": "Llama-3-8B-Instruct-q4f16_1-MLC"
}
```

**Response (NDJSON stream):**
```json
{"type": "token", "data": {"text": "Let"}}
{"type": "token", "data": {"text": " me"}}
{"type": "token", "data": {"text": " check"}}
{"type": "tool_call", "data": {"name": "weather", "arguments": {"location": "Dublin"}}}
{"type": "tool_result", "data": {"tool_name": "weather", "result": {"temp": 12, "condition": "Rainy"}}}
{"type": "token", "data": {"text": "It's"}}
{"type": "token", "data": {"text": " 12°C"}}
{"type": "token", "data": {"text": " and"}}
{"type": "token", "data": {"text": " rainy"}}
{"type": "done", "data": {"finish_reason": "stop"}}
```

### Event Types

| Event Type | Description | Data Fields |
|------------|-------------|-------------|
| `token` | Model-generated text token | `text`: string |
| `tool_call` | Model requests tool execution | `name`: string, `arguments`: dict |
| `tool_result` | Tool execution completed | `tool_name`: string, `result`: any |
| `done` | Generation finished | `finish_reason`: "stop" \| "length" \| "error" |
| `error` | Error occurred | `message`: string |

## 🛠️ Tool System

Tether's function calling system allows models to interact with external tools. The model can call tools, receive results, and incorporate them into its response.

### How It Works

1. **System Prompt**: The model is instructed to emit tool calls in the format:
   ```
   <<function_call>> {"name": "tool_name", "arguments": {...}}
   ```

2. **Stream Parsing**: The `SlidingParser` detects `<<function_call>>` markers in real-time, even across chunk boundaries.

3. **Tool Execution**: The `ToolRunner` executes the tool with a timeout and returns the result.

4. **Loop Continuation**: The tool result is added to the conversation history, and the model continues generating.

5. **History Persistence**: Both tool calls and results are saved to the session store for multi-turn tool use.

### Built-in Tools

#### TimeTool
Get the current time in any timezone.

```json
{
  "name": "get_current_time",
  "arguments": {
    "timezone": "America/New_York",
    "format": "human"
  }
}
```

#### WeatherTool
Get current weather conditions.

```json
{
  "name": "weather",
  "arguments": {
    "location": "London, UK"
  }
}
```

#### GetForecastTool
Get weather forecast.

```json
{
  "name": "forecast",
  "arguments": {
    "location": "Tokyo",
    "days": 3
  }
}
```

#### WebSearchTool
Search news via NewsAPI (requires API key in environment).

```json
{
  "name": "web_search",
  "arguments": {
    "query": "AI developments 2025",
    "max_results": 5
  }
}
```

## 🔧 Configuration

Configuration is managed through `tether_service/config/default.yml`. Here's a breakdown of key sections:

### Server Configuration
```yaml
host: "127.0.0.1"
port: 8080
```

### System Prompt
```yaml
system:
  prompt: |
    You are a helpful assistant that uses tools when appropriate.
    To call a tool, output exactly one line that starts with:
    <<function_call>>
    followed by a single JSON object on the same line.
```

### Model Provider
```yaml
providers:
  model:
    impl: "tether_service.providers.mlc.provider.MLCProvider"
    args:
      dist_root: "dist"              # Model directory
      model_name: "Llama-3-8B-q4f16_1-MLC"
      device: "auto"                 # auto | cuda | vulkan | metal | cpu | opencl
      max_tokens: 1024               # Max generation length
```

### Tool Configuration
```yaml
tools:
  registry:
    - name: "time"
      impl: "tether_service.tools.time_tool.TimeTool"
    - name: "weather"
      impl: "tether_service.tools.weather_tool.WeatherTool"
  enabled:
    - "time"
    - "weather"
```

### Limits
```yaml
limits:
  tool_timeout_sec: 15    # Max execution time per tool
  max_tool_loops: 5       # Max consecutive tool calls
```

## 🎨 Adding Custom Tools

Creating custom tools is straightforward. Follow these steps:

### 1. Create Your Tool Class

Create a new file in `tether_service/tools/`, e.g., `calculator_tool.py`:

```python
from tether_service.tools.base import BaseTool

class CalculatorTool(BaseTool):
    """Performs basic arithmetic operations."""
    
    def __init__(self):
        super().__init__()  # Required for registry name injection
    
    def run(
        self,
        operation: str,
        a: float,
        b: float
    ) -> dict:
        """
        Execute a calculation.
        
        Args:
            operation: One of "add", "subtract", "multiply", "divide"
            a: First number
            b: Second number
        
        Returns:
            dict: Result of the calculation
        """
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else None
        }
        
        if operation not in operations:
            return {"error": f"Unknown operation: {operation}"}
        
        result = operations[operation](a, b)
        
        if result is None:
            return {"error": "Division by zero"}
        
        return {
            "operation": operation,
            "result": result,
            "expression": f"{a} {operation} {b} = {result}"
        }
```

### 2. Register in Configuration

Add your tool to `tether_service/config/default.yml`:

```yaml
tools:
  registry:
    - name: "calculator"
      impl: "tether_service.tools.calculator_tool.CalculatorTool"
    # ... other tools
  enabled:
    - "calculator"
    # ... other enabled tools
```

### 3. Update System Prompt (Optional)

Add a description to the system prompt so the model knows when to use it:

```yaml
system:
  prompt: |
    You are a helpful assistant with access to these tools:
    
    - calculator: Perform arithmetic (add, subtract, multiply, divide)
    
    To use a tool, output:
    <<function_call>> {"name":"calculator","arguments":{"operation":"add","a":5,"b":3}}
```

### 4. Restart the Service

```powershell
python -m tether_service.app
```

### Tool Implementation Guidelines

✅ **Do:**
- Inherit from `BaseTool`
- Call `super().__init__()` in your `__init__`
- Use type hints for all parameters (used for auto-schema generation)
- Accept `**kwargs` in `run()` method
- Return dictionaries or JSON-serializable objects
- Handle errors gracefully and return error messages

❌ **Don't:**
- Accept raw dictionaries as arguments (use `**kwargs` unpacking)
- Perform long-running operations without considering the timeout
- Raise exceptions (return error dicts instead)
- Modify global state

### Advanced: Async Tools

For I/O-bound tools, make `run()` async:

```python
class AsyncWebTool(BaseTool):
    async def run(self, url: str) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                return {"content": await resp.text()}
```

The `ToolRunner` automatically handles both sync and async tools.

## 👨‍💻 Development

### Running Tests

```powershell
# All tests
pytest

# Parser unit tests (27 tests)
pytest tests/protocol/parsers/ -v

# Integration tests (tool calling end-to-end)
pytest tests/integration/ -v

# Specific test file
pytest tests/protocol/parsers/test_sliding_parser.py -v
```

### Project Structure Deep Dive

- **`app/`**: HTTP layer - FastAPI routes and application setup
- **`config/`**: YAML configurations - system prompts, tool registries, limits
- **`core/`**: Infrastructure - interfaces, DI, logging, type definitions
- **`providers/`**: Model implementations - MLC provider, dummy provider for testing
- **`context/`**: Persistence - SQLite store with WAL mode for concurrent access
- **`protocol/`**: Orchestration - the "brain" coordinating model, parser, and tools
- **`tools/`**: Tool implementations - inherit from `BaseTool` with auto-schema

### Debugging Tips

**Enable Verbose Logging:**
Edit `tether_service/core/logging.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

**Inspect Database:**
```powershell
sqlite3 data/tether.db
sqlite> SELECT * FROM messages WHERE session_id='...' ORDER BY ts;
```

**Trace a Request:**
1. HTTP request → `app/http/routers/chat.py`
2. → `protocol/service/generation_service.py`
3. → `protocol/orchestration/orchestrator.py`
4. → Model streaming + parser + tool execution

**Test Parser Directly:**
```python
from tether_service.protocol.parsers.sliding import SlidingParser

parser = SlidingParser(max_tool_chars=1024)
parser.feed("Some text <<function_call>> {")
parser.feed('"name":"time"}')

if parser.has_tool_call():
    call = parser.extract_tool_call()
    print(call)  # ToolCall(name='time', args_json='{}', raw=...)
```

### Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Model doesn't call tools | Generates descriptions instead of `<<function_call>>` | Check system prompt includes explicit format instructions |
| Tool not found | `"Tool X not found"` error | Verify tool is in both `registry` and `enabled` in config |
| Tool execution fails | `'dict' object has no attribute...` | Ensure tool uses `**kwargs`, not `args: dict` |
| Repeated failed calls | Model keeps trying same failed tool | Check `get_history()` includes `tool_result` messages |
| Parser misses tool calls | Tool call split across chunks | This is handled automatically by `SlidingParser` buffer |

## 🐛 Troubleshooting

### MLC-LLM Model Issues

**Problem:** Model fails to load
```
Solution:
1. Verify model directory structure matches expected format
2. Check that DLL/shared library is in dist/libs/
3. Ensure model name in config matches directory name
4. Try different device options: "cuda", "vulkan", "cpu"
```

**Problem:** Slow inference
```
Solution:
1. Check device is set correctly (GPU vs CPU)
2. Reduce max_tokens in config
3. Use smaller quantized models (q4f16 vs q0f16)
4. Enable model caching (happens automatically)
```

### Tool Calling Issues

**Problem:** Model describes actions but doesn't call tools
```
Solution: The system prompt must explicitly show the tool call format.
Edit config/default.yml and ensure the prompt includes:
  <<function_call>> {"name":"tool_name","arguments":{...}}
```

**Problem:** Tools timing out
```
Solution: Increase timeout in config/default.yml:
  limits:
    tool_timeout_sec: 30  # Increase from default 15
```

### Database Issues

**Problem:** Database locked errors
```
Solution: Tether uses WAL mode by default. If issues persist:
1. Ensure only one Tether instance is running
2. Delete data/tether.db-wal and data/tether.db-shm
3. Restart the service
```

### API Issues

**Problem:** Connection refused
```
Solution:
1. Check service is running: python -m tether_service.app
2. Verify port isn't in use: netstat -ano | findstr :8080
3. Check firewall settings
```

## 🤝 Contributing

This is a personal experiment, but contributions and ideas are welcome! Areas I'm exploring:
- Additional tool implementations (especially email/calendar tools)
- Alternative model providers (Ollama, llama.cpp)
- Performance optimizations
- Documentation improvements

Feel free to open an issue to discuss ideas or share your own experiments.

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- [MLC-LLM](https://github.com/mlc-ai/mlc-llm) for enabling efficient local inference on Snapdragon X Elite
- Qualcomm for the Snapdragon X Elite hardware that inspired this project
- FastAPI for making API development straightforward
- The open-source AI community for inspiration and tools

## 📞 Questions or Ideas?

- **Issues**: [GitHub Issues](https://github.com/Lando-00/Tether/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Lando-00/Tether/discussions)

---

**A personal experiment in local AI, privacy, and giving models access to your data - safely.**
