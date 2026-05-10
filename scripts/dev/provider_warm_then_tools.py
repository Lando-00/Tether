"""Direct provider.stream invocation with verbose prints to see WHERE
it hangs. Uses the same provider Tether builds, the same messages
shape, and a small max_tokens so any hang is conclusive.

If the test prints 'awaited completion.create' but never prints any
chunk, the hang is inside MLC's stream itself with this shape.
If it prints chunks but doesn't yield, the hang is in our wrapper.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")


async def main() -> int:
    from tether.providers.mlc.provider import MLCProvider

    provider = MLCProvider(
        models_root="models",
        device="auto",
        max_tokens=64,  # tight cap — any generation should finish quickly
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Reply with exactly: OK. /no_think"},
    ]

    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "time",
                "description": "Get the current time for a timezone.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string", "default": "UTC"},
                    },
                },
            },
        },
    ]

    print("==> warming engine (no tools, simple call) ...", flush=True)
    t0 = time.perf_counter()
    async for chunk in provider.stream(
        model_name="Qwen3-4B-q4f16_1-MLC",
        messages=[{"role": "user", "content": "Hi."}],
        tools=None,
        request_id="warm",
    ):
        pass
    print(f"OK: warm call done in {time.perf_counter() - t0:.1f}s", flush=True)

    print("\n==> calling provider.stream WITH tools (1 tool)...", flush=True)
    t1 = time.perf_counter()
    n = 0
    async for chunk in provider.stream(
        model_name="Qwen3-4B-q4f16_1-MLC",
        messages=messages,
        tools=tool_schemas,
        request_id="diag",
    ):
        n += 1
        preview = chunk if isinstance(chunk, str) else f"<tc list>"
        print(f"  [{time.perf_counter() - t1:6.1f}s] chunk[{n}]: {preview!r}", flush=True)
        if time.perf_counter() - t1 > 180:
            print("FAIL: 180s timeout reached", flush=True)
            break

    elapsed = time.perf_counter() - t1
    print(f"\n==> with-tools: {n} chunks in {elapsed:.1f}s", flush=True)

    # Shut down the provider's engine cleanly
    if hasattr(provider, "aclose"):
        await provider.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
