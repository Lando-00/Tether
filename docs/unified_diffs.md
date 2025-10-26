# Tool Calling Fix - Unified Diffs

## Summary of Changes

This patch restores tool calling functionality in `tether_service` by:
1. Adding explicit tool-calling instructions to the system prompt
2. Injecting tool catalog with descriptions and parameters
3. Fixing NDJSON event naming consistency
4. Adding logging for debugging

---

## Files Changed

- **NEW**: `tether_service/protocol/prompts.py` (system prompt builder)
- **MODIFIED**: `tether_service/core/factory.py` (integrate prompt builder)
- **MODIFIED**: `tether_service/config/default.yml` (base prompt)
- **MODIFIED**: `tether_service/protocol/orchestration/orchestrator.py` (events, logging)
- **MODIFIED**: `tether_service/providers/mlc/provider.py` (logging)
- **NEW**: `tether_service/tests/test_sliding_parser.py` (unit tests)
- **NEW**: `docs/tool_calling_fix_rca.md` (root cause analysis)

---

## Testing

```powershell
# Unit tests
pytest tether_service/tests/test_sliding_parser.py -v

# Start service
python -m tether_service.app

# Test via CLI
python tether_cli.py
# Ask: "What time is it in UTC?"
```

**Expected**: Tool invocation visible in logs and CLI output.

---

See `docs/tool_calling_fix_rca.md` for detailed analysis and full code listings.
