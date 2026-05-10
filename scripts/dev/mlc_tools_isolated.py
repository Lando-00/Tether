"""Final isolation: bypass Tether entirely. Call MLC twice in ONE
process — first WITHOUT tools, then WITH tools — using max_tokens=32
to force a quick finish either way. The first run also pre-warms the
OpenCL kernel cache so cold-cache latency doesn't pollute the result.

If "no tools" finishes in <60s and "with tools" hangs, we've proved
the issue is MLC's tool-rendering path with this conv_template + DLL.

Note: After engine.terminate(), MLC's GPU memory release is best-effort
under Adreno OpenCL. To work around CL_OUT_OF_RESOURCES we reuse the
SAME engine for both calls.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

MODEL_DIR = Path("models/Qwen3-4B-q4f16_1-MLC").resolve()
MODEL_LIB = Path("models/libs/Qwen3-4B-q4f16_1-adreno.dll").resolve()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "time",
            "description": "Get the current time.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string", "default": "UTC"}},
            },
        },
    },
]

SYSTEM = "You are a helpful assistant."


async def call_engine(engine, label: str, with_tools: bool) -> None:
    kwargs = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Reply with exactly: OK. /no_think"},
        ],
        "max_tokens": 32,
        "stream": True,
        "request_id": f"diag-{label}",
    }
    if with_tools:
        kwargs["tools"] = TOOLS
        kwargs["tool_choice"] = "auto"

    print(f"\n--- {label} (with_tools={with_tools}) ---", flush=True)
    t = time.perf_counter()
    n = 0
    text = ""
    finish = None
    try:
        async for resp in await asyncio.wait_for(
            engine.chat.completions.create(**kwargs), timeout=15
        ):
            n += 1
            if n == 1:
                print(f"OK: first chunk after {time.perf_counter() - t:.2f}s", flush=True)
            delta = resp.choices[0].delta.content or ""
            text += delta
            finish = resp.choices[0].finish_reason
            if finish:
                break
            if time.perf_counter() - t > 120:
                print(f"FAIL: timeout 120s, partial text={text!r}", flush=True)
                break
    except asyncio.TimeoutError:
        print(f"FAIL: 15s timeout opening the stream (no first chunk)", flush=True)
        return
    print(
        f"OK: {n} chunks, {time.perf_counter() - t:.1f}s, "
        f"finish={finish!r}, text={text!r}",
        flush=True,
    )


async def main() -> int:
    from mlc_llm.serve.engine import AsyncMLCEngine

    print("==> loading engine ONCE...", flush=True)
    t0 = time.perf_counter()
    engine = AsyncMLCEngine(
        model=str(MODEL_DIR),
        model_lib=str(MODEL_LIB),
        device="auto",
        mode="interactive",
    )
    print(f"OK: loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    # Warm-up kernels with a no-tools call first
    await call_engine(engine, "warm-no-tools", with_tools=False)
    # Then second call with tools — kernels should be warm
    await call_engine(engine, "with-tools", with_tools=True)
    # Third call: back to no tools (regression check)
    await call_engine(engine, "no-tools-2", with_tools=False)

    engine.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
