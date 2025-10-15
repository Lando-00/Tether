# Refactor Thinking Tools

This document outlines the incremental refactoring plan to enhance the streaming/tool-call pipeline to handle “thinking models” that emit hidden blocks (`<think>…</think>`) before invoking tools, while preserving behavior for non-thinking models.

## TODO Checklist

1. [x] Add a `RoutedChunk` dataclass and extend `HiddenBlockFilter` to **route** tokens
2. [x] Integrate `HiddenBlockFilter` into the streaming loop **before** `PrefixedToolCallDetector`
3. [x] Relax early rejection logic in `PrefixedToolCallDetector`
4. [x] Add “anywhere” regex search in `PrefixedToolCallDetector`
5. [x] Introduce a `ToolBoundaryController`
6. [x] Ensure engine can abort the current stream
 7. [x] Emit `hidden_thought` chunks with a `phase` tag: "pre_tool" or "post_tool"
 8. [x] Catch the correct timeout exception (`FuturesTimeoutError`)
9. [x] Truncate large tool results for streaming
 10. [x] Add a small test harness (`tests/test_stream_thinking_tool.py`)
11. [ ] Optional flags & docs: configurable hidden tags and feature flag

## Turn Runner Pattern

After refactoring, `ToolOrchestrator` uses a two-method pattern to handle model streaming and tool calls within a single turn:

- `_stream_one_turn`: performs exactly one `model.generate(stream=True)` invocation, feeds tokens through the `StreamParser` and `EventEmitter`, and stops immediately when a `TOOL_COMPLETE` event is encountered.
- `_run_with_stream_parser`: loops up to `max_tool_loops` invoking `_stream_one_turn`, yielding all parser events back to back and re-invoking the model after each tool result, then flushes any remaining parser events before emitting a final `done()` event.

This approach keeps single-responsibility for each method (SRP), avoids code duplication (OCP), and depends only on abstractions (DIP), enabling seamless multi-tool reasoning within one assistant turn.
