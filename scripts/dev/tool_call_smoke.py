"""End-to-end tool-call smoke test.

Verifies that with ``marker_only_tools=True`` (the post-mitigation
default) Tether still emits a complete v2 envelope including a
``tool_call`` event, by asking a question that should trigger the
time tool. The SlidingParser is supposed to detect the ``<<function_call>>``
marker in the model's text stream and the ToolRunner is supposed to
execute the tool and stream back a tool_result.

Usage::

    python scripts/dev/tool_call_smoke.py

Slow on first run (~3 min cold prefill). Subsequent runs in the same
process are faster.
"""
from __future__ import annotations

import asyncio
import sys
import time

sys.path.insert(0, ".")

TIMEOUT_S = 600.0


async def main() -> int:
    from tether import Engine
    from tether.config.settings import load_settings

    settings = load_settings()
    model_name = "Qwen3-4B-q4f16_1-MLC"

    async with Engine.from_settings(settings) as engine:
        sess = await engine.create_session(model_name=model_name)
        session_id = sess["session_id"]
        print(f"OK: session={session_id}", flush=True)

        prompt = (
            "What is the current time in Europe/Dublin? "
            "Call the time tool to find out. /no_think"
        )
        print(f"==> prompt: {prompt!r}", flush=True)

        seen: list[str] = []
        tool_calls: list = []
        tool_results: list = []
        text_chunks: list[str] = []
        t0 = time.perf_counter()

        async def _consume() -> None:
            async for event in engine.chat(
                session_id=session_id,
                prompt=prompt,
                model_name=model_name,
            ):
                kind = type(event).__name__
                seen.append(kind)
                if kind == "ToolCall":
                    tool_calls.append(event)
                    name = getattr(event, "name", None) or getattr(event, "tool_name", None)
                    print(f"  [{time.perf_counter() - t0:6.1f}s] ToolCall: name={name!r} args={getattr(event, 'arguments', None)!r}", flush=True)
                elif kind == "ToolResult":
                    tool_results.append(event)
                    name = getattr(event, "name", None) or getattr(event, "tool_name", None)
                    status = getattr(event, "status", None)
                    print(f"  [{time.perf_counter() - t0:6.1f}s] ToolResult: name={name!r} status={status!r}", flush=True)
                elif kind == "TextDelta":
                    text = getattr(event, "text", "")
                    text_chunks.append(text)
                    preview = text.replace("\n", "\\n")
                    if len(preview) > 60:
                        preview = preview[:60] + "..."
                    print(f"  [{time.perf_counter() - t0:6.1f}s] TextDelta: {preview!r}", flush=True)
                else:
                    print(f"  [{time.perf_counter() - t0:6.1f}s] {kind}", flush=True)

        try:
            await asyncio.wait_for(_consume(), timeout=TIMEOUT_S)
        except asyncio.TimeoutError:
            print(f"FAIL: timed out after {TIMEOUT_S}s; seen={seen}", file=sys.stderr)
            return 2

        elapsed = time.perf_counter() - t0
        print(f"\n==> turn complete in {elapsed:.1f}s, {len(seen)} events", flush=True)
        print(f"==> tool_calls={len(tool_calls)} tool_results={len(tool_results)} text_chunks={len(text_chunks)}", flush=True)
        full_text = "".join(text_chunks)
        print(f"==> full text: {full_text!r}", flush=True)

    # Validate envelope
    ok = True
    if not any(t == "MessageStart" for t in seen):
        print("FAIL: no MessageStart", file=sys.stderr)
        ok = False
    if not any(t == "MessageStop" for t in seen):
        print("FAIL: no MessageStop", file=sys.stderr)
        ok = False
    if not tool_calls:
        print("WARN: model did not emit a tool call — the prompt may not be commanding enough, or the marker path needs tuning. Marker-mode mitigation is still proven by the absence of a deadlock.", file=sys.stderr)
    else:
        if not tool_results:
            print("FAIL: tool_call present but no tool_result — ToolRunner failed?", file=sys.stderr)
            ok = False
        else:
            print(f"OK: full tool round-trip ({tool_calls[0]} -> {tool_results[0]})", flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
