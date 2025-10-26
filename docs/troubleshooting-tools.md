# Troubleshooting Tool Calls in Tether

This document provides guidance for debugging and verifying tool call behavior in `tether_service`.

## Quick Verification Test

To verify tool calls work correctly, use this simple time query:

```powershell
# Start the service
python -m tether_service.app

# In another terminal, test tool calling
curl -N http://localhost:8080/api/v1/chat/stream `
  -H "Content-Type: application/json" `
  -d '{"session_id":"test","prompt":"What time is it?"}'
```

### Expected Behavior

You should see NDJSON events in this order:

1. **text** events: Model's natural language response
2. **tool_marker_detected**: Parser found `<<function_call>>`
3. **tool_started**: Tool execution beginning with parsed arguments
4. **tool_completed**: Tool result returned
5. **text** events: Model continues with result

Example output:
```json
{"type":"text","session_id":"test","data":{"delta":"Let me check the time for you.\n"}}
{"type":"tool_marker_detected","session_id":"test","data":{}}
{"type":"tool_started","session_id":"test","data":{"tool_name":"time","tool_args":{"timezone":"Europe/Dublin","format":"human"}}}
{"type":"tool_completed","session_id":"test","data":{"tool_name":"time","tool_result":{"time":"..."}}}
{"type":"done","session_id":"test","data":{}}
```

## Common Issues

### 1. Model Emits Random Tool-Like Text Instead of Structured Calls

**Symptom:** Model outputs something like "I'll use the time function" but no actual tool execution happens.

**Root Cause:** The system prompt doesn't specify the exact emission format.

**Fix:** Verify `tether_service/config/default.yml` contains the explicit tool call contract:

```yaml
system:
  prompt: |
    You are a helpful assistant that uses tools when appropriate.
    To call a tool, output exactly one line that starts with:
    <<function_call>>
    followed by a single JSON object on the same line, e.g.:
    <<function_call>> {"name":"get_current_time","arguments":{"timezone":"Europe/Dublin","format":"human"}}
    Do not include extra commentary on that line. After I run the tool, I will stream the result back to you.
```

**Why This Works:** The model needs explicit instructions on the literal format. Vague prompts like "use tools when necessary" cause the model to describe actions instead of emitting structured calls.

### 2. Parser Doesn't Detect Tool Call Marker

**Symptom:** Logs show text streaming but no "TOOL_STARTED" or parser state transitions.

**Enable Debug Logging:**

```python
# In tether_service/core/logging.py, temporarily set level to DEBUG
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check Parser Logs:**

Look for these debug messages in parser output:
- `Parser feed: mode=text, buf_len=...` - Parser receiving chunks
- `Parser: detected <<function_call>> marker` - Marker found
- `Parser: entering think mode` / `mode=await_payload` - State transitions
- `Parser: JSON capture complete` - JSON parsed successfully

**Common Causes:**

1. **Marker split across chunks badly:** The parser keeps an overlap buffer (default 17 chars) to handle this. If your marker is longer or chunks are tiny, adjust:
   ```yaml
   providers:
     parser:
       args:
         max_tool_chars: 32768  # Increase if large JSON
   ```

2. **Non-whitespace before JSON:** Parser expects `<<function_call>>` followed by optional whitespace, then `{`. Check logs for warnings:
   ```
   Parser: non-whitespace before JSON: "some text{..."
   ```

3. **Incomplete JSON:** If stream ends mid-JSON, finalize() emits an error:
   ```
   Parser finalize: incomplete tool payload
   ```

### 3. Tool Detection Works But Execution Doesn't Happen

**Symptom:** Logs show `TOOL_COMPLETE` event but no `tool_started` or `tool_completed` in output.

**Check Orchestrator Logs:**

```
Tool call detected: {'tool_name': 'time', 'tool_args': {...}}
Assistant tool call persisted: session_id=..., tool_name=time
Tool executed: time, result={...}
```

**If Missing:**

1. **Tool not registered:** Verify `tether_service/config/default.yml`:
   ```yaml
   tools:
     registry:
       - name: "time"
         impl: "tether_service.tools.time_tool.TimeTool"
     enabled:
       - "time"  # Must be in enabled list
   ```

2. **Tool name mismatch:** Model emitted `get_current_time` but registry has `time`. Check the tool's `name` property matches what the model calls.

3. **Tool execution exception:** Check for stack traces in logs. Common issues:
   - Missing required arguments
   - Type mismatches (e.g., expecting int but got string)
   - Tool timeout (default 15s in config)

### 4. Tool Loop Runs Too Many Times or Not Enough

**Configure Loop Limits:**

```yaml
limits:
  max_tool_loops: 5  # Max turns before stopping
  tool_timeout_sec: 15  # Per-tool timeout
```

**Debugging:**

Enable INFO logging to see loop iterations:
```
Tool loop 1/5 for session_id=...
Tool loop 2/5 for session_id=...
```

If it stops after 1 loop even though model wants to continue:
- Check `tools.continue_after_tool` in config (default: true)
- Verify `full_response_text` is being persisted correctly

### 5. JSON Parsing Errors

**Symptom:** Parser emits `ERROR` event with "json_parse_error"

**Enable Verbose Errors:**

In parser logs (DEBUG level), you'll see:
```
Parser: JSON parse error: <exception details>, raw=<first 100 chars>
```

**Common Causes:**

1. **Unescaped quotes in strings:**
   ```json
   {"msg": "He said "hi""}  ❌
   {"msg": "He said \"hi\""}  ✅
   ```

2. **Trailing commas:**
   ```json
   {"name":"test","args":{},}  ❌
   {"name":"test","args":{}}   ✅
   ```

3. **Single quotes instead of double:**
   ```json
   {'name':'test'}  ❌
   {"name":"test"}  ✅
   ```

**Fix:** These are usually model fine-tuning issues. You can:
- Add few-shot examples to system prompt showing correct JSON
- Use a model checkpoint specifically trained on function calling
- Implement a lenient JSON parser (not recommended for production)

## Advanced Debugging

### Inspect Parser State Mid-Stream

Add a breakpoint or log statement in `SlidingParser.feed()`:

```python
def feed(self, chunk: str | List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # ... existing code ...
    logger.debug(f"Parser state: mode={self.mode}, buf='{self.buf[:50]}...', json_depth={self._json_depth}")
```

### Trace Event Flow

To see complete event flow from provider → parser → orchestrator → emitter:

```python
# In orchestrator.py
async for chunk in provider.stream(...):
    logger.debug(f"RAW CHUNK: {repr(chunk)}")
    events = parser.feed(chunk)
    for evt in events:
        logger.debug(f"PARSER EVENT: {evt['type']} | {evt.get('data', {})}")
```

### Validate Model Output Manually

If unsure whether issue is in model or parser, capture raw model stream:

```python
# Temporary debug script
async def capture_raw():
    provider = factory.get_provider()
    async for chunk in provider.stream(model_name="...", messages=[...], tools=[...]):
        print(repr(chunk), end='', flush=True)
```

Save output and manually check for `<<function_call>>` markers and JSON structure.

## Migration Notes

### From llm_service to tether_service

If migrating from the legacy `llm_service`:

1. **System Prompt Update Required:** The old vague prompt won't work. Use the explicit contract above.

2. **No More mlc_stream_patch:** `tether_service` doesn't use the stream patch. Tool detection is pure parser-based. This is more reliable but requires the model to emit the exact marker.

3. **Event Schema Changes:**
   - Old: `{"type": "tool_complete", "style": "json", "raw": "...", "obj": {...}}`
   - New: `{"type": "tool_complete", "data": {"tool_name": "...", "tool_args": {...}, "raw": "..."}}`

4. **Config-Driven:** All thresholds, timeouts, and parser options are now in YAML config, not hardcoded.

### Risk of "Only call tools when necessary!"

This instruction **without an explicit emission format** is dangerous because:
- Model may describe tool usage verbally instead of emitting structured calls
- Model may invent fake results instead of triggering actual execution
- Ambiguity leads to hallucinations

**Always pair permissive instructions with exact format specs.**

## Testing Checklist

Before deploying:

- [ ] Run `pytest tests/protocol/parsers/test_sliding_parser.py` - All parser unit tests pass
- [ ] Run `pytest tests/integration/test_tool_calling.py` - Integration tests pass
- [ ] Manual test with time tool succeeds
- [ ] Manual test with weather tool succeeds
- [ ] Test with prompt requiring 2+ tool calls (e.g., "What's the weather and time in Dublin?")
- [ ] Test with prompt not needing tools (e.g., "Hello, how are you?")
- [ ] Verify logs show correct event sequences
- [ ] Check session history persists tool calls correctly

## Support

For additional help:
1. Check logs at INFO level first
2. Enable DEBUG logging for detailed parser traces
3. Run unit and integration tests to isolate component failures
4. Compare working `llm_service` logs with `tether_service` for parity
