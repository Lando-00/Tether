"""
Golden-stream regression test — locks current orchestrator behavior.
Synthesis §6 (regression net before behavior changes), B2 lines 160-190.

One full tool round-trip is captured and asserted against
tests/fixtures/golden_streams/one_tool_success.json.

Set TETHER_UPDATE_GOLDENS=1 to regenerate the fixture instead of
asserting against it.
"""
import json
import os
from pathlib import Path

import pytest

from tether_service.protocol.orchestration.orchestrator import orchestrate
from tether_service.protocol.parsers.sliding import SlidingParser
from tether_service.tools.time_tool import TimeTool

from tests.golden.conftest import ScriptedProvider, MinimalMemoryStore, normalize_event

# ---------------------------------------------------------------------------
# Location of the golden fixture file
# ---------------------------------------------------------------------------
GOLDEN_DIR = Path(__file__).parent.parent / "fixtures" / "golden_streams"
GOLDEN_FILE = GOLDEN_DIR / "one_tool_success.json"

# ---------------------------------------------------------------------------
# Scripted provider scripts
# Turn 1: text preamble + <<function_call>> + JSON payload in one chunk so
#         the SlidingParser can fully detect and capture it without chunk-
#         boundary complexity.
# Turn 2: a short final-answer chunk (< OVERLAP=16 chars so it stays in buf
#         and is flushed by parser.finalize()).
# ---------------------------------------------------------------------------
_TURN1_CHUNK = (
    'Here is the time '
    '<<function_call>> {"name": "time_tool", "arguments": {}}'
)
_TURN2_CHUNK = "Done."

_SCRIPTS = [
    [_TURN1_CHUNK],  # first provider call → tool call detected
    [_TURN2_CHUNK],  # second provider call → final answer
]


async def _run_orchestrator() -> list[dict]:
    """Drive orchestrate() end-to-end; return list of decoded event dicts."""
    provider = ScriptedProvider(_SCRIPTS)
    parser = SlidingParser()
    store = MinimalMemoryStore()

    # Build a minimal tools dict with just TimeTool, name injected manually
    tool = TimeTool()
    tool._registry_name = "time_tool"
    tools = {"time_tool": tool}

    events = []
    async for raw_bytes in orchestrate(
        session_id="test-session-golden",
        prompt="What time is it?",
        model_name="scripted-model",
        provider=provider,
        parser=parser,
        store=store,
        tools=tools,
        system_prompt="You are a helpful assistant.",
    ):
        line = raw_bytes.decode("utf-8").strip()
        if line:
            events.append(json.loads(line))
    return events


async def test_one_tool_success_golden():
    """Orchestrator emits the expected event sequence for a single tool call.

    Regenerate the fixture with: TETHER_UPDATE_GOLDENS=1 pytest tests/golden/
    """
    actual_raw = await _run_orchestrator()
    actual = [normalize_event(e) for e in actual_raw]

    if os.environ.get("TETHER_UPDATE_GOLDENS") == "1":
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        GOLDEN_FILE.write_text(
            json.dumps(actual, indent=2) + "\n", encoding="utf-8"
        )
        # Still pass after writing so the regen run shows green
        return

    assert GOLDEN_FILE.exists(), (
        f"Golden fixture not found: {GOLDEN_FILE}\n"
        "Run with TETHER_UPDATE_GOLDENS=1 to generate it."
    )
    golden = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    assert actual == golden, (
        "Orchestrator event sequence changed.\n"
        "If this is intentional, regenerate with TETHER_UPDATE_GOLDENS=1.\n"
        f"Expected:\n{json.dumps(golden, indent=2)}\n"
        f"Got:\n{json.dumps(actual, indent=2)}"
    )
