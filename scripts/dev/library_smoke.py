"""Tiny library-mode smoke test for the post-refactor Engine.

Boots the Engine via from_settings, creates a session, streams one
short completion against the freshly-staged Qwen3-4B-q4f16_1 model,
prints each WireEvent, asserts that the v2 envelope shows up
(message_start ... message_stop).

Usage::

    python scripts/dev/library_smoke.py

This script is NOT a pytest test — it loads a real GPU model so it
takes ~30s the first time. It is intentionally minimal: it touches the
same code paths the HTTP path does, just without the FastAPI adapter.
"""
from __future__ import annotations

import asyncio
import sys
import time
import uuid

# Ensure repo root is importable
sys.path.insert(0, ".")

# Patience knob — first invocation after a cold OpenCL kernel state can
# take several minutes on Adreno; we don't want to mistake "slow first
# prefill" for "hung".
TIMEOUT_S = 600.0


async def main() -> int:
    print("==> importing tether...", flush=True)
    from tether import Engine
    from tether.config.settings import load_settings

    print("==> loading settings...", flush=True)
    settings = load_settings()

    model_name = "Qwen3-4B-q4f16_1-MLC"
    session_id = f"smoke-{uuid.uuid4().hex[:8]}"

    print(f"==> building Engine; model={model_name} session={session_id}", flush=True)
    t0 = time.perf_counter()

    async with Engine.from_settings(settings) as engine:
        boot_s = time.perf_counter() - t0
        print(f"OK: engine booted in {boot_s:.2f}s", flush=True)

        # ensure session exists (Engine.chat doesn't auto-create)
        sess = await engine.create_session(model_name=model_name)
        session_id = sess["session_id"]
        print(f"OK: session created: {session_id}", flush=True)

        print("==> streaming one short turn...", flush=True)
        seen_types: list[str] = []
        ttft: float | None = None
        t1 = time.perf_counter()

        async def _consume() -> None:
            nonlocal ttft
            async for event in engine.chat(
                session_id=session_id,
                prompt="Reply with exactly: OK. /no_think",
                model_name=model_name,
            ):
                event_type = type(event).__name__
                seen_types.append(event_type)
                if ttft is None:
                    ttft = time.perf_counter() - t1
                    print(f"OK: first event after {ttft:.2f}s ({event_type})", flush=True)

                # Compact event preview
                if hasattr(event, "text") and event.text:
                    preview = event.text.replace("\n", "\\n")
                    if len(preview) > 80:
                        preview = preview[:80] + "..."
                    print(f"  [{time.perf_counter() - t1:6.1f}s] {event_type}: {preview}", flush=True)
                else:
                    print(f"  [{time.perf_counter() - t1:6.1f}s] {event_type}", flush=True)

        try:
            await asyncio.wait_for(_consume(), timeout=TIMEOUT_S)
        except asyncio.TimeoutError:
            print(f"FAIL: timed out after {TIMEOUT_S}s; saw events: {seen_types}", file=sys.stderr)
            return 2

        total_s = time.perf_counter() - t1
        print(f"OK: turn complete in {total_s:.2f}s, {len(seen_types)} events", flush=True)

    # Validate v2 envelope
    if not seen_types:
        print("FAIL: no events received", file=sys.stderr)
        return 1
    if not any(t.lower().startswith("messagestart") for t in seen_types):
        print(f"FAIL: missing MessageStart, saw {seen_types[:5]}...", file=sys.stderr)
        return 1
    if not any(t.lower().startswith("messagestop") for t in seen_types):
        print(f"FAIL: missing MessageStop, saw ...{seen_types[-5:]}", file=sys.stderr)
        return 1
    if not any(t.lower().startswith("textdelta") for t in seen_types):
        print(
            "WARN: no TextDelta events (model produced no text)",
            file=sys.stderr,
        )

    print(f"OK: v2 envelope validated; event sequence = {seen_types}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
