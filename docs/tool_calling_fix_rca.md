# Root Cause Analysis: Tool Calling Regression

## Executive Summary

**Issue**: `tether_service` fails to invoke tools despite using the same model and tool configurations as the working `llm_service`.

**Root Cause**: System prompt does not instruct the model how to emit tool calls in the expected format (`<<function_call>> {json}`).

**Impact**: No tool calls detected → orchestrator persists plain text → no tool execution → degraded functionality.

---

## Detailed Analysis

### What Works (llm_service)

1. **System Prompt** (`llm_service/protocol/orchestration/system_prompt.py:79`):
   ```python
   "You can use a tool by outputting '<<function_call>> \{function name with params in structured json\}'"
   ```
   - Explicitly tells the model the **sentinel format**
   - Provides clear emission contract

2. **Streaming Parser** (`llm_service/protocol/orchestration/parsers/simple_sliding_tool_parser.py`):
   - Detects `<<function_call>>` marker across chunk boundaries
   - Captures balanced JSON payload (string-aware, handles nested braces)
   - Emits `TOOL_COMPLETE` events with parsed `tool_name` and `tool_args`

3. **Orchestration Loop** (`llm_service/protocol/orchestration/orchestrator.py:405-463`):
   - Receives parser events
   - On `TOOL_COMPLETE`: executes tool, adds result to history, continues loop
   - Emits NDJSON: `tool_start`, `tool_complete`, `done`

4. **Tool Schema Injection**:
   - System prompt includes tool catalog: names, descriptions, parameters
   - Model has context to ground tool calls

### What Breaks (tether_service - BEFORE FIX)

1. **System Prompt** (`tether_service/config/default.yml:4`):
   ```yaml
   prompt: "You are a helpful assistant. Only call tools when necessary!"
   ```
   - **Missing**: No sentinel format instruction
   - **Missing**: No tool catalog
   - Model has no idea **how** to emit a tool call

2. **Parser** (`tether_service/protocol/parsers/sliding.py`):
   - **Correctly** detects `<<function_call>>` and captures JSON
   - **But**: Model never emits this format because it wasn't instructed to

3. **Orchestration** (`tether_service/protocol/orchestration/orchestrator.py`):
   - Loop exists, event handling correct
   - **But**: Never receives `TOOL_COMPLETE` events → falls through to text persistence

4. **Tool Schemas**:
   - Passed to MLC engine (`provider.stream(tools=tool_schemas)`)
   - **But**: Not documented in system prompt for model's reasoning

### Common Misconceptions

- ❌ **"The MLC patch is needed for streaming"**  
  → NO. The `mlc_stream_patch.py` is for **non-streaming mode** only. It buffers entire output, then synthesizes tool_calls from `__tool_*` format at the end. Streaming uses the parser directly.

- ❌ **"The parser is broken"**  
  → NO. The `SlidingParser` correctly detects the marker and JSON. The model just never emits it.

- ❌ **"Tool schemas aren't passed to the engine"**  
  → They are, but the model can't use them without explicit instructions in the prompt.

---

## The Fix

### 1. System Prompt Enhancement (`tether_service/protocol/prompts.py`)

**New File**: `tether_service/protocol/prompts.py`

```python
def build_system_prompt_with_tools(
    tools: Dict[str, Any],
    base_instruction: str = "You are a helpful assistant.",
) -> str:
    """
    Build system prompt with:
    - Tool catalog (names, descriptions, parameters)
    - Explicit sentinel format: <<function_call>> {json}
    - Usage guidelines
    """
    if not tools:
        return base_instruction
    
    catalog = _render_tool_catalog(tools)  # Formats tool details
    
    prompt = f"""{base_instruction}

You have access to the following tools:

{catalog}

To call a tool, emit EXACTLY this format:
<<function_call>> {{"name": "tool_name", "arguments": {{"param1": "value1"}}}}

Do not add commentary on the same line as the tool call.
After you receive the tool result, continue normally.
Only use tools when necessary.
"""
    return prompt
```

### 2. Factory Integration (`tether_service/core/factory.py:74-81`)

```python
# OLD:
system_prompt = self.config.get("system", {}).get("prompt", "")

# NEW:
from tether_service.protocol.prompts import build_system_prompt_with_tools

base_prompt = self.config.get("system", {}).get("prompt", "You are a helpful assistant.")
system_prompt = build_system_prompt_with_tools(tools, base_prompt)
```

### 3. Config Update (`tether_service/config/default.yml:4`)

```yaml
# OLD:
prompt: "You are a helpful assistant. Only call tools when necessary!"

# NEW:
prompt: "You are a helpful assistant that uses tools when appropriate."
```

### 4. Event Name Consistency (`tether_service/protocol/orchestration/orchestrator.py`)

```python
# OLD:
"type": "tool_start"     # CLI expects "tool_started"
"type": "tool_complete"  # CLI expects "tool_completed"

# NEW:
"type": "tool_started"
"type": "tool_completed"
```

### 5. Logging (`tether_service/providers/mlc/provider.py:172-176`)

```python
if tools:
    tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
    print(f"Available tools: {tool_names}")
```

---

## Migration Guide

### For Existing Deployments

1. **Update `default.yml`**:
   ```yaml
   system:
     prompt: "You are a helpful assistant that uses tools when appropriate."
   ```

2. **No code changes needed** if using `ServiceFactory`.  
   The factory now calls `build_system_prompt_with_tools` automatically.

3. **Custom Prompts**: If you override `system.prompt` in your config, ensure it includes:
   - The sentinel: `<<function_call>> {json}`
   - Tool catalog (or use `build_system_prompt_with_tools`)

### Verification

**Check Logs** for these lines (with `INFO` level):

```
Available tools: ['get_current_time', 'weather', 'web_search']
Tool call detected: tool_name=get_current_time, tool_args={'timezone': 'UTC'}
Tool execution complete: get_current_time, result={'time': '...'}
```

**CLI Output** should show:
```
╭─ Calling tool: get_current_time ─╮
│ Tool get_current_time output: ... │
╰───────────────────────────────────╯
```

---

## Testing

### Unit Tests

- `tether_service/tests/test_sliding_parser.py`:  
  Validates parser detects `<<function_call>>`, captures JSON, handles edge cases.

### Integration Tests

- `tether_service/tests/test_tool_integration.py`:  
  Verifies end-to-end: prompt → tool detection → execution → result → model follow-up.

### Manual Test

```powershell
# Start service
python -m tether_service.app

# In another terminal
python tether_cli.py

# Ask: "What time is it in UTC?"
# Expect: Tool invocation visible, then natural language reply
```

---

## Key Takeaways

1. **Small models need explicit instructions**.  
   They won't infer structured output formats. The sentinel `<<function_call>> {json}` must be in the prompt.

2. **Tool schemas alone aren't enough**.  
   Even when passed to the engine, the model needs a **human-readable catalog** in the prompt to reason about which tool to use.

3. **Streaming parsers work without patches**.  
   The `mlc_stream_patch.py` is a red herring for streaming mode. The parser handles live token detection.

4. **Event naming matters**.  
   Ensure NDJSON event types match what the CLI/client expects (`tool_started` vs `tool_start`).

---

## Files Modified

| File | Change |
|------|--------|
| `tether_service/protocol/prompts.py` | **NEW**: System prompt builder with tool instructions |
| `tether_service/core/factory.py` | Import and call `build_system_prompt_with_tools` |
| `tether_service/config/default.yml` | Update base prompt |
| `tether_service/protocol/orchestration/orchestrator.py` | Fix event names (`tool_started`/`tool_completed`), add logging |
| `tether_service/providers/mlc/provider.py` | Add "Available tools" logging |
| `tether_service/tests/test_sliding_parser.py` | **NEW**: Parser unit tests |
| `tether_service/tests/test_tool_integration.py` | **NEW**: Integration tests (note: requires interface fixes) |

---

## Success Criteria

✅ Running `tether_cli.py` with "What time is it?" triggers tool invocation  
✅ Logs show `Available tools:`, `Tool call detected:`, `Tool execution complete:`  
✅ CLI displays "Calling tool: ..." and "Tool output: ..."  
✅ Same prompt in both services yields identical tool selection  
✅ Parser tests pass: `pytest tether_service/tests/test_sliding_parser.py -v`

---

## References

- `llm_service/protocol/orchestration/system_prompt.py:79` (working prompt)
- `llm_service/protocol/orchestration/parsers/simple_sliding_tool_parser.py` (parser reference)
- `llm_service/model/mlc_stream_patch.py` (non-streaming patch, not used here)
- MCP Architecture: Model (provider) → Context (store) → Protocol (orchestrator + parser)
