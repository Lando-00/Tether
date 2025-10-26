# Tool Calling Fix: Root Cause Analysis and Solution

## Summary

Fixed tool calling in `tether_service` to match `llm_service` behavior. The model now correctly emits structured tool calls that are detected, parsed, and executed.

## Root Causes Identified

### 1. **System Prompt Contract Missing** (PRIMARY ISSUE)
- **Problem:** `tether_service` config had vague prompt: "You are a helpful assistant. Only call tools when necessary!"
- **Impact:** Model didn't know HOW to emit tool calls, so it either:
  - Responded with normal text instead of calling tools
  - Fabricated fake tool results instead of using actual tools
  - Described tool usage verbally without structured emission
- **Fix:** Updated default system prompt to explicitly instruct:
  ```
  To call a tool, output exactly one line that starts with:
  <<function_call>>
  followed by a single JSON object on the same line, e.g.:
  <<function_call>> {"name":"get_current_time","arguments":{"timezone":"Europe/Dublin","format":"human"}}
  ```

### 2. **Missing TOOL_STARTED Event**
- **Problem:** Parser detected `<<function_call>>` but didn't emit intermediate event
- **Impact:** No visibility into when tool parsing begins vs when JSON completes
- **Fix:** Added `TOOL_STARTED` event emission when marker is detected, before JSON parsing

### 3. **Insufficient Logging**
- **Problem:** Hard to debug why tools weren't being invoked
- **Impact:** No clear signal showing where the pipeline failed
- **Fix:** Added comprehensive logging at INFO and DEBUG levels throughout parser and orchestrator

### 4. **Parser Edge Cases Not Handled**
- **Problem:** Chunk boundaries, large JSON, non-whitespace before `{`, etc.
- **Impact:** Parser could miss tool calls or fail silently
- **Fix:** Hardened parser with:
  - Truncation detection for oversized payloads
  - Proper handling of whitespace/newlines after marker
  - Warning when non-whitespace appears before JSON
  - Balanced brace tracking with string-awareness

## Changes Made

### Configuration (`tether_service/config/default.yml`)

1. **Updated system prompt** with explicit tool call format
2. **Added config switches:**
   ```yaml
   tools:
     call_contract: "marker_json"  # Tool emission format
     continue_after_tool: true     # Continue after tool execution
   limits:
     max_tool_loops: 5
     tool_timeout_sec: 15
   stream:
     parser:
       strict_balancing: true      # Enforce balanced JSON
   ```

### Parser (`tether_service/protocol/parsers/sliding.py`)

1. **Added TOOL_STARTED event:** Emitted when `<<function_call>>` marker detected
2. **Enhanced logging:** DEBUG logs for state transitions, INFO for tool detection
3. **Improved JSON parsing:**
   - Check buffer size before and during JSON parsing
   - Log warnings for non-whitespace before `{`
   - Better error messages with context
4. **Reset tool_started flag** on completion/error

### Orchestrator (`tether_service/protocol/orchestration/orchestrator.py`)

1. **Handle TOOL_STARTED events:** Emit `tool_marker_detected` to client
2. **Handle ERROR events:** Forward parser errors to client
3. **Enhanced logging:** Log tool call detection, execution, and result persistence

### Tests

#### Parser Unit Tests (`tests/protocol/parsers/test_sliding_parser.py`)
- 27 tests covering:
  - Basic text streaming
  - Tool marker detection (single chunk, split across chunks, with newlines)
  - JSON payload parsing (nested objects, arrays, escaped quotes, quoted braces)
  - Error cases (incomplete JSON, missing payload, invalid syntax, truncation)
  - Think mode (`<think>...</think>`)
  - Finalize behavior
  - Real-world scenarios

#### Integration Tests (`tests/integration/test_tool_calling.py`)
- 4 tests covering:
  - Simple tool call execution
  - Tool call with marker split across chunks
  - Normal text without tool calls
  - Tool call with newlines

### Documentation (`docs/troubleshooting-tools.md`)

Comprehensive troubleshooting guide covering:
- Quick verification test with curl
- Common issues and fixes
- Debug logging instructions
- Parser state inspection
- Migration notes from `llm_service`
- Testing checklist

## Key Differences: llm_service vs tether_service

| Aspect | llm_service | tether_service |
|--------|------------|----------------|
| **System Prompt** | Explicit `<<function_call>>` instruction | Now explicit (was vague) |
| **Stream Patch** | Uses `mlc_stream_patch` to synthesize calls | Pure parser-based detection |
| **Event Schema** | `{"type": "tool_complete", "style": "json", "obj": {...}}` | `{"type": "tool_complete", "data": {"tool_name": "...", "tool_args": {...}}}` |
| **Config** | Hardcoded thresholds | YAML config-driven |
| **Tool Events** | `tool_start`, `tool_end` | `tool_started`, `tool_completed` |

## Acceptance Criteria Met

✅ **Parser tests:** 27/27 passing  
✅ **Integration tests:** 4/4 passing (asyncio variant)  
✅ **System prompt:** Explicit tool call contract  
✅ **Event parity:** TOOL_STARTED and TOOL_COMPLETE events  
✅ **Config-driven:** All thresholds and options configurable  
✅ **Logging:** Comprehensive DEBUG and INFO logs  
✅ **Documentation:** Troubleshooting guide created  

## Testing Instructions

### 1. Run Unit Tests
```powershell
pytest tests/protocol/parsers/test_sliding_parser.py -v
```

### 2. Run Integration Tests
```powershell
pytest tests/integration/test_tool_calling.py -k asyncio -v
```

### 3. Manual Verification (with actual MLC model)
```powershell
# Start service
python -m tether_service.app

# In another terminal
curl -N http://localhost:8080/api/v1/chat/stream `
  -H "Content-Type: application/json" `
  -d '{"session_id":"test","prompt":"What time is it?"}'
```

Expected output should include:
- `{"type":"tool_marker_detected",...}`
- `{"type":"tool_started","data":{"tool_name":"time",...}}`
- `{"type":"tool_completed","data":{"tool_result":{...}}}`

## Migration Notes

For teams migrating from `llm_service`:

1. **Update system prompts** in all configs to include explicit tool call format
2. **Update client code** to handle new event schema (if consuming events)
3. **Review config files** for new options: `tools.call_contract`, `stream.parser.*`
4. **Avoid vague prompts** like "Only call tools when necessary" without format spec

## Future Improvements (Not in Scope)

- Support for alternative tool call formats (e.g., raw JSON without marker)
- Tool call chaining/composition
- Streaming of partial tool results
- Tool call validation/schema enforcement before execution
- Retry logic for transient tool failures
