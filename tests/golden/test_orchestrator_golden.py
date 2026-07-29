"""
Golden-stream regression test — locks current orchestrator behavior.
Synthesis §6 (regression net before behavior changes), B2 lines 160-190.

One full tool round-trip is captured and asserted against
tests/fixtures/golden_streams/one_tool_success.json.

The test uses ChattyAgentOrchestrator.run() + transport_ndjson directly
so the fixture is v2 vocabulary (message_start / text_delta / tool_call /
tool_result / message_stop). Updated from orchestrate() + v0_compat_serialize
in p5-cutover-b-clients. Synthesis §11.3 R18; §4 Phase 5 step 54.

Set TETHER_UPDATE_GOLDENS=1 to regenerate the fixture instead of
asserting against it.
"""
import json
import os
from pathlib import Path

from tests.golden.conftest import MinimalMemoryStore, ScriptedProvider, normalize_event
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration.chatty import ChattyAgentOrchestrator
from tether.protocol.orchestration.tool_runner import ToolRunner
from tether.protocol.parsers.sliding import SlidingParser
from tether.protocol.wire.transport_ndjson import transport_ndjson
from tether.tools.time_tool import TimeTool

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
    """Drive ChattyAgentOrchestrator end-to-end via v2 transport; return list of decoded event dicts."""
    provider = ScriptedProvider(_SCRIPTS)
    parser = SlidingParser()
    store = MinimalMemoryStore()

    # Build a minimal tools dict with just TimeTool, name injected manually
    tool = TimeTool()
    tools = {"time_tool": tool}

    events = []
    orch = ChattyAgentOrchestrator(
        provider=provider,
        parser=parser,
        store=store,
        tools=tools,
        system_prompt="You are a helpful assistant.",
        config=OrchestratorConfig(
            max_tool_loops=5,
            auto_reload_on_fatal_error=False,
            save_thinking=True,
            include_thinking_in_history=False,
        ),
        tool_runner=ToolRunner(tools, timeout_sec=15),
    )
    async for raw_bytes in transport_ndjson(
        orch.run(
            session_id="test-session-golden",
            prompt="What time is it?",
            model_name="scripted-model",
        )
    ):
        line = raw_bytes.decode("utf-8").strip()
        if line:
            events.append(json.loads(line))
    return events


async def test_one_tool_success_golden():
    """Orchestrator emits the expected v2 event sequence for a single tool call.

    Regenerate the fixture with: TETHER_UPDATE_GOLDENS=1 pytest tests/golden/
    """
    actual_raw = await _run_orchestrator()
    actual = [normalize_event(e) for e in actual_raw]

    if os.environ.get("TETHER_UPDATE_GOLDENS") == "1":
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        GOLDEN_FILE.write_text(
            json.dumps(actual, indent=2) + "\n", encoding="utf-8"
        )
        # Fall through to the assertion below — the regen run must still
        # round-trip through read_golden + assert. Otherwise a bug in the
        # writer can silently corrupt the fixture and the test "passes"
        # only because it returned early. P0-G / Tribunal P0-15 (A8-F1).

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
