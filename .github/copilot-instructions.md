# Copilot Instructions — Tether

**Tether** is a Python/FastAPI service for session-based, streaming access to MLC-LLM models with function-calling support. It uses SQLite for chat history, follows Model-Context-Protocol (MCP) architecture, and streams NDJSON events.

## Quick Start

```powershell
# Environment
conda activate mlc-venv2
pip install -e ".[server,cli,brave,dev]"
# MLC CodeLinaro wheels are installed separately; see environment.yml.

# Run tether_service (new implementation)
python -m tether_service.app      # http://localhost:8000
# Or debug: .\run_debug.ps1

# Run tests
pytest -q                          # All tests
pytest tests/protocol/parsers/     # Parser unit tests
pytest tests/integration/          # Integration tests
```

## Architecture (MCP Layers)

### 1. Model (`tether_service/providers/`)
- `mlc/provider.py`: MLC-LLM streaming interface
- Implements `ModelProvider` interface from `core/interfaces.py`

### 2. Context (`tether_service/context/`)
- `sqlite_store.py`: Session + message persistence with WAL
- **Critical**: `get_history()` must include tool calls and results for multi-turn tool execution

### 3. Protocol (`tether_service/protocol/`)
- `orchestration/orchestrator.py`: Main loop coordinating model → parser → tool execution
- `parsers/sliding.py`: Stateful parser detecting `<<function_call>>` markers in streams
- `orchestration/tool_runner.py`: Executes tools with timeout

### 4. Tools (`tether_service/tools/`)
- `base.py`: `BaseTool` abstract class with auto-schema generation
- `core/tool_registry.py`: Loads tools from config, injects registry names

### 5. Config (`tether_service/config/`)
- `default.yml`: System prompt, tool registry, limits (max_tool_loops)
- `core/factory.py`: DI container for wiring dependencies

## Tool Calling System (Critical)

### How It Works
1. **System Prompt**: Must explicitly specify `<<function_call>> {"name":"...", "arguments":{...}}` format in `config/default.yml`
2. **Parser Detection**: `SlidingParser.feed()` detects markers, handles chunk boundaries (17-char overlap buffer)
3. **Orchestration Loop**: Up to `max_tool_loops` iterations (default 5)
   - Get history (including previous tool calls/results)
   - Stream from model
   - Parse for tool calls
   - Execute tool via `ToolRunner`
   - Persist tool call + result to session store
   - Continue loop or exit if no more tool calls
4. **History Format**: 
   - Tool calls stored as: `role="tool"` with `tool_name`, `args`
   - Results stored as: `role="tool_result"` with `tool_name`, `result`
   - `get_history()` converts to model-compatible messages:
     - Tool call → assistant message with `<<function_call>>` syntax
     - Tool result → user message with formatted JSON

### Tool Implementation Checklist
- [ ] Inherit from `BaseTool` in `tether_service/tools/base.py`
- [ ] Call `super().__init__()` in `__init__` to enable registry name injection
- [ ] Use type hints for parameters (auto-generates schema)
- [ ] Tool methods receive `**kwargs`, not dict
- [ ] Register in `config/default.yml` under `tools.registry` and `tools.enabled`

### Common Tool Calling Issues
| Issue | Symptom | Fix |
|-------|---------|-----|
| Model describes actions instead of emitting calls | No `<<function_call>>` in stream | Add explicit format to system prompt |
| Tool call detected but not executed | "Tool X not found" | Check registry name matches schema name |
| Tool execution fails with `'dict' object has no attribute...` | Args not unpacked | Use `tool.run(**args)` not `tool.run(args)` |
| Model repeats failed tool calls | No learning from errors | Verify `get_history()` includes tool_result messages |

## Directory Structure

```
tether_service/           # New implementation (SOLID, config-driven)
├── app/                  # FastAPI app + HTTP routers
├── config/               # YAML configs (default.yml, testing.yml)
├── context/              # SqliteSessionStore (chat history)
├── core/                 # Interfaces, types, factory, logging, tool registry
├── protocol/             # Orchestration, parsers, service layer
│   ├── orchestration/    # orchestrator.py, tool_runner.py
│   └── parsers/          # sliding.py (<<function_call>> detection)
├── providers/            # Model providers (mlc/, dummy/)
└── tools/                # BaseTool + concrete tools (time, weather, web_search, etc.)

llm_service/              # Legacy reference implementation
tests/                    # pytest tests (use anyio for async)
├── protocol/parsers/     # Parser unit tests (27 tests)
└── integration/          # End-to-end tool calling tests (4 tests)
```

## Development Tips

- **Trace a request**: `app/__main__.py` → `app/http/routers/chat.py` → `protocol/service/generation_service.py` → `protocol/orchestration/orchestrator.py`
- **Debug tool calls**: Enable logging in `core/logging.py`, check for `tool_call` / `tool_result` events (v2 vocab)
- **Add new tool**: Create in `tools/`, register in `config/default.yml`, restart server
- **Test parser**: `pytest tests/protocol/parsers/test_sliding_parser.py -v` (tests chunk boundaries, nested JSON, etc.)
- **Inspect DB**: `sqlite3 data/tether.db "SELECT * FROM messages WHERE session_id='...' ORDER BY ts"`

## Reference: Legacy vs New
- `llm_service/`: **DO NOT MODIFY** — Original implementation kept for reference only (shows working patterns and lessons learned)
- `tether_service/`: Active codebase — All new work happens here (interface-based with DI)
- Use `llm_service/` to understand behavior in legacy code, feature implement/fixes only in `tether_service/`

---

## Tool System Architecture

### How Tools Are Described to the Model

Tools are described to the model through **auto-generated JSON schemas** created by the `BaseTool` class:

**Schema Generation Flow:**
1. `BaseTool.auto_schema` property (lines 61-96 in `tether_service/tools/base.py`)
2. Introspects the `run()` method signature using Python's `inspect` module
3. Extracts parameter names, types, defaults, and docstrings
4. Generates JSON Schema with:
   - `name`: Injected by `ToolRegistry` from config (e.g., "web_search")
   - `description`: From tool class docstring
   - `parameters`: Auto-generated from `run()` method signature
     - Type mapping: `str → "string"`, `int → "integer"`, `Optional[str] → nullable`, etc.
     - Required vs optional based on presence of default values
     - Parameter descriptions from docstring (if available)

**To Modify Tool Description:**
- **Change tool name**: Update `name` field in `config/default.yml::tools.registry`
- **Change tool description**: Update the docstring of the tool class (first line after `class WebSearchTool(BaseTool):`)
- **Change parameter names/types**: Update the `run()` method signature
- **Change parameter descriptions**: Update the `run()` method's docstring (param sections)

**Example - WebSearchTool Schema Generation:**
```python
# In tether_service/tools/web_search_tool.py
class WebSearchTool(BaseTool):
    """Search the web using Brave Search API."""  # ← Tool description
    
    async def run(
        self,
        query: str,                    # Required (no default)
        count: int = 5,                # Optional (has default)
        country: str = "us",           # Optional
        search_lang: str = "en",       # Optional
        freshness: Optional[str] = None,  # Optional + nullable
        **kwargs
    ) -> dict:
        # Docstring here can provide parameter descriptions
```

**Resulting Schema:**
```json
{
  "name": "web_search",  // From config registry
  "description": "Search the web using Brave Search API.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "count": {"type": "integer", "default": 5},
      "country": {"type": "string", "default": "us"},
      ...
    },
    "required": ["query"]  // Only params without defaults
  }
}
```

### Tool Execution Flow

**Complete Request Flow:**
```
1. User sends message → POST /api/v1/chat/stream
   ↓
2. app/http/routers/chat.py::stream()
   ↓
3. protocol/service/generation_service.py::stream()
   ↓
4. protocol/orchestration/orchestrator.py::orchestrate()
   │
   ├─ Loop (up to max_tool_loops=5):
   │   ├─ Get history (including previous tool calls/results)
   │   ├─ Stream from model (MLCProvider)
   │   ├─ Parse stream for <<function_call>> markers (SlidingParser)
   │   ├─ If tool call detected:
   │   │   ├─ Emit `tool_call` event (v2 vocab; `message_start`/`text_delta`/`message_stop` per turn)
   │   │   ├─ protocol/orchestration/tool_runner.py::run()
   │   │   │   ├─ Lookup tool in ToolRegistry
   │   │   │   ├─ Call tool.run(**args) with timeout
   │   │   │   └─ Return result or error dict
   │   │   ├─ Persist tool call + result to SessionStore
   │   │   ├─ Emit `tool_result` event (v2 vocab; status="ok"/"error" field)
   │   │   └─ Continue loop (model sees result in next iteration)
   │   └─ If no tool call: exit loop
   │
5. Stream text response back to client as NDJSON events
```

**Key Files:**
- `tether_service/core/tool_registry.py`: Loads tools from config, injects registry names
- `tether_service/protocol/orchestration/orchestrator.py`: Main orchestration loop
- `tether_service/protocol/orchestration/tool_runner.py`: Tool execution with timeout
- `tether_service/protocol/parsers/sliding.py`: Detects `<<function_call>>` in stream
- `tether_service/context/sqlite_store.py`: Persists tool calls/results for multi-turn

### Available Tools

#### Web Search Tool (Brave Search API)
**Provider:** Brave Search API (https://api.search.brave.com/)  
**File:** `tether_service/tools/web_search_tool.py`  
**HTTP Client:** `tether_service/tools/brave_client.py` (315 lines, httpx-based)

**Configuration:**
```yaml
# In tether_service/config/default.yml
tools:
  registry:
    - name: "web_search"  # Name exposed to model
      impl: "tether_service.tools.web_search_tool.WebSearchTool"
  enabled:
    - "web_search"
  web_search:
    provider: "brave"
    timeouts:
      connect_sec: 2
      read_sec: 6
    retries:
      max_attempts: 3
      backoff_base_sec: 0.5
    defaults:
      count: 5
      max_count: 20
      country: "us"
      search_lang: "en"
```

**Environment Variables:**
- `BRAVE_API_KEY` (required) — Get from https://api-dashboard.search.brave.com/
- Loaded via python-dotenv in `conftest.py`

**Parameters (Auto-Generated Schema):**
- `query` (str, required): Search query
- `count` (int, default 5): Results to return (clamped to 1-20)
- `country` (str, default "us"): 2-letter country code
- `search_lang` (str, default "en"): Language code
- `freshness` (str, optional): Time filter - "pd"(day), "pw"(week), "pm"(month), "py"(year)
- `language` (str, deprecated): Use `search_lang` instead

**Response Format:**
```json
{
  "results": [
    {"url": "...", "title": "...", "snippet": "...", "rank": 1}
  ],
  "meta": {"took_ms": 123, "engine": "brave", "query": "..."},
  "articles": ["..."]  // Deprecated, for backward compatibility
}
```

**Error Handling:**
- Returns `{"error": "..."}` dicts (not exceptions)
- 403/422 → Friendly ValueError about API key
- 429 → Retry with exponential backoff
- 5xx → Retry with exponential backoff
- Other 4xx → Fail fast

**Testing:**
- `tests/tools/test_brave_client.py`: 17 unit tests (mocked)
- `tests/tools/test_brave_client_real.py`: 15 real API tests
- `tests/tools/test_web_search_tool.py`: 18 integration tests (mocked)
- `tests/tools/test_web_search_tool_real.py`: 19 real API tests

**Implementation Notes:**
- `BraveSearchClient`: Async httpx client with separate timeouts (connect/read/total)
- HTML tag removal and 360-char snippet truncation
- Param mapping: `country→cc`, `search_lang→hl` (Brave API params)
- Rate limits: 2k queries/month, 10 req/min (free tier)

#### Other Tools
- **Time Tool** (`tether_service/tools/time_tool.py`): Get current time
- **Weather Tool** (`tether_service/tools/weather_tool.py`): Get weather information

### Adding a New Tool

1. **Create tool class** in `tether_service/tools/your_tool.py`:
   ```python
   from tether_service.tools.base import BaseTool
   
   class YourTool(BaseTool):
       """Your tool description for the model."""
       
       def __init__(self):
           super().__init__()  # CRITICAL: Enables registry name injection
       
       async def run(
           self,
           param1: str,           # Required
           param2: int = 10,      # Optional with default
           **kwargs               # REQUIRED: Catch extra args
       ) -> dict:
           """
           Tool implementation.
           
           Args:
               param1: Description of param1
               param2: Description of param2
           """
           # Return dict, not raise exceptions for user-facing errors
           return {"result": "..."}
   ```

2. **Register in config** (`tether_service/config/default.yml`):
   ```yaml
   tools:
     registry:
       - name: "your_tool"  # Name model will see
         impl: "tether_service.tools.your_tool.YourTool"
     enabled:
       - "your_tool"
   ```

3. **Restart server** — Schema auto-generated on startup

4. **Test** — Create tests in `tests/tools/test_your_tool.py`

**Important:** Legacy reference code exists in `llm_service/tools/` — **DO NOT MODIFY**. Only work in `tether_service/`.

---

## Model-Specific Behavior & Shutdown

### Critical: Model-Dependent Shutdown Hang Fixed

**Issue**: Qwen2.5-7B was hanging on Ctrl+C shutdown while Qwen3-4B exited cleanly.

**Root Cause**: Different `prefill_chunk_size` in model configs (256 vs 2048) caused different OpenCL driver states. When Python GC ran destructors during shutdown, Qwen2.5-7B's driver hung.

**Fix**: Disabled GC in daemon shutdown thread (`api.py::shutdown_provider_with_timeout()`). The daemon thread is killed on process exit, so OS handles GPU resource cleanup.

**Key Model Differences**:

| Model | Prefill Chunk | Context Window | DLL |
|-------|---------------|----------------|-----|
| Qwen3-4B | 2048 | 40960 | `Qwen3-4B-q4f16_0-adreno.dll` |
| Qwen2.5-7B | 256 | 12288 | `Qwen2.5-7B-q4f16_0-MLC-adreno.dll` |

**Shutdown Architecture**:

1. **Force-exit handler** (5s max) - catches second Ctrl+C or timeout
2. **Provider timeout** (3s) - abandons cleanup if stuck
3. **Per-engine timeout** (0.75s each) - bounds native `terminate()` calls
4. **GC disabled** - prevents hanging in destructors
5. **Daemon thread** - can be killed without waiting

**Testing**: Use `test_model_shutdown.py` to verify both models:
```powershell
python test_model_shutdown.py Qwen2.5-7B-q4f16_0-MLC
python test_model_shutdown.py Qwen3-4B-q4f16_0-MLC
```

**Documentation**: See `SHUTDOWN_HANG_FIX_SUMMARY.md` and `MODEL_DEPENDENT_SHUTDOWN_FIX.md` for full analysis.

**Never re-enable GC** in the daemon shutdown thread - this is critical for models with smaller prefill chunks.
