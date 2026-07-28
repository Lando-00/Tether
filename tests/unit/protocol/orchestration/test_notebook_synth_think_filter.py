"""Phase 9.6 I-1 — synth ``<think>...</think>`` filter.

Verifies :class:`_ThinkStripper` (a think-only streaming state machine)
strips reasoning tokens from :class:`NotebookOrchestrator`'s synth stream
without re-introducing :class:`SlidingParser` (which would violate
ADR-0020 §D1 prompt-injection defense by parsing
``<<function_call>>`` markers on untrusted snippet-derived synth output).

Unit tests exercise the stripper directly across split-marker and
edge-case scenarios; the final integration test runs a full notebook
turn with a fake provider that emits ``<think>reason</think>Final``
and asserts no ``<think>`` or reasoning text leaks into ``TextDelta``.
"""
from __future__ import annotations

from datetime import date
from typing import Any, List

import pytest

from tether.config.settings import ResearchSettings
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration.notebook import (
    NotebookOrchestrator,
    _ThinkStripper,
)
from tether.protocol.parsers.sliding import SlidingParser
from tether.protocol.wire.events import (
    MessageStop,
    TextDelta,
    ThinkingDelta,
)
from tests.fixtures.fake_research_provider import FakeResearchProvider


# ---------------------------------------------------------------------------
# _ThinkStripper unit tests
# ---------------------------------------------------------------------------


def _drive(
    chunks: List[str], *, assume_open: bool = False
) -> tuple[str, str, _ThinkStripper]:
    """Feed ``chunks`` through a fresh stripper; return concatenated
    ``(text, thinking)`` plus the stripper instance (for state asserts)."""
    stripper = _ThinkStripper(assume_open=assume_open)
    text_parts: List[str] = []
    think_parts: List[str] = []
    for chunk in chunks:
        text_part, thinking_part = stripper.feed(chunk)
        text_parts.append(text_part)
        think_parts.append(thinking_part)
    text_tail, think_tail = stripper.finalize()
    text_parts.append(text_tail)
    think_parts.append(think_tail)
    return "".join(text_parts), "".join(think_parts), stripper


def test_strip_split_open_tag_across_chunks():
    """``<thi`` + ``nk>`` must reassemble and trigger think mode."""
    text, thinking, _ = _drive(
        ["<thi", "nk>", "secret reasoning", "</think>", "visible answer"]
    )
    assert text == "visible answer"
    assert thinking == "secret reasoning"


def test_strip_split_close_tag_across_chunks():
    """``</thi`` + ``nk>`` must reassemble and exit think mode."""
    text, thinking, _ = _drive(
        ["<think>", "hidden", "</thi", "nk>", "shown answer"]
    )
    assert text == "shown answer"
    assert thinking == "hidden"


def test_strip_bare_leading_close_marker_single_chunk():
    """If ``</think>`` appears before any ``<think>`` in the first chunk
    that resolves the leading state, treat preceding content as thinking.

    The Qwen3 chat template can inject ``<think>`` out-of-band so the
    model starts mid-thinking; the stripper detects this by inspecting
    its leading-mode buffer for which marker (open vs close) appears
    first. The detection window is bounded by ``_OVERLAP`` (so that a
    pure-text stream isn't buffered indefinitely with no chance to
    stream), so the test feeds the prefix + closing marker as one chunk
    to keep the buffer below the flush threshold."""
    text, thinking, _ = _drive(
        ["pre-marker reasoning</think>", "actual user answer"]
    )
    assert text == "actual user answer"
    assert thinking == "pre-marker reasoning"
    # Sanity: no <think> opener marker should leak into either side.
    assert "<think>" not in text
    assert "</think>" not in text


def test_strip_bare_leading_close_marker_split_chunks():
    """Same semantics as above with the prefix and close marker arriving
    in separate chunks — prefix must fit within the leading overlap
    window so the buffer still holds it when ``</think>`` arrives."""
    text, thinking, _ = _drive(
        ["pre", "</think>", "actual user answer"]
    )
    assert text == "actual user answer"
    assert thinking == "pre"
    assert "<think>" not in text
    assert "</think>" not in text


def test_strip_long_leading_close_when_assume_open_think():
    """Phase 9.8: the long bare-leading preamble is solved for opted-in models.

    Handling an arbitrarily long hidden preamble before a bare-leading
    ``</think>`` generically would require buffering normal no-think output
    and regressed first-token latency. The shipped fix is instead a
    model-scoped opt-in (``synth_assume_open_think_models``): the stripper
    starts inside the hidden block, so no amount of preamble can leak.
    """
    hidden = "this is long hidden reasoning before close " * 40
    text, thinking, _ = _drive([hidden, "</think>", "visible"], assume_open=True)
    assert text == "visible"
    assert thinking == hidden
    assert "</think>" not in text


def test_assume_open_unclosed_stream_never_leaks_hidden_text():
    """An opted-in stream that never closes must yield no visible text."""
    text, thinking, stripper = _drive(
        ["hidden reasoning that never closes"], assume_open=True
    )
    assert text == ""
    assert "hidden" in thinking
    assert stripper.unclosed_assumed_block is True


def test_non_opted_long_leading_preamble_remains_a_known_limitation():
    """Documented tradeoff: without the opt-in, a long hidden preamble
    before a bare-leading close still reaches the text channel. Models with
    this template must be listed in ``synth_assume_open_think_models``."""
    hidden = "this is long hidden reasoning before close "
    text, _thinking, _ = _drive([hidden, "</think>", "visible"])
    assert "visible" in text
    assert hidden.strip() in text


def test_strip_nested_think_blocks_do_not_leak_tail_text():
    """Nested think markers should remain hidden until the outer close."""
    text, thinking, _ = _drive(
        ["<think>outer <think>inner</think> still hidden</think>visible"]
    )
    assert text == "visible"
    assert thinking == "outer inner still hidden"
    assert "</think>" not in text


def test_strip_unclosed_think_to_eos_never_leaks_to_text():
    """An unclosed ``<think>`` block at end-of-stream must NOT leak into
    text; residual flows to ``thinking_part`` so the caller can drop
    it (``save_thinking=False``) or persist it (``save_thinking=True``)."""
    text, thinking, stripper = _drive(
        ["plain prefix ", "<think>", "open block", " continues forever"]
    )
    assert text == "plain prefix "
    assert thinking == "open block continues forever"
    # No leakage of either marker.
    assert "<think>" not in text
    assert "</think>" not in text
    assert stripper._unclosed_think_count == 1  # noqa: SLF001


def test_strip_multiple_think_blocks_in_one_stream():
    """Multiple back-to-back ``<think>...</think>`` blocks all stripped."""
    text, thinking, _ = _drive(
        [
            "intro ",
            "<think>r1</think>",
            "middle ",
            "<think>r2</think>",
            "outro",
        ]
    )
    assert text == "intro middle outro"
    assert thinking == "r1r2"


def test_strip_no_think_pure_passthrough():
    """A stream with no markers must round-trip every character to
    text (no characters dropped, no thinking emitted)."""
    text, thinking, _ = _drive(
        [
            "The quick brown ",
            "fox jumps over ",
            "the lazy dog.",
        ]
    )
    assert text == "The quick brown fox jumps over the lazy dog."
    assert thinking == ""


def test_strip_empty_think_block():
    """``<think></think>`` (no body) must consume both markers and yield
    nothing to either side from the block itself."""
    text, thinking, _ = _drive(
        ["before ", "<think></think>", "after"]
    )
    assert text == "before after"
    assert thinking == ""


def test_strip_single_char_chunks_split_every_marker():
    """Worst-case: one character per chunk forces the buffer to
    reassemble every marker across many boundaries."""
    stream = "ab<think>xy</think>cd"
    text, thinking, _ = _drive(list(stream))
    assert text == "abcd"
    assert thinking == "xy"


# ---------------------------------------------------------------------------
# Orchestrator integration test
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


from tests.fixtures.recording_research_store import RecordingResearchStore


class _FakeStore(RecordingResearchStore):
    pass


class _FakeToolRegistry:
    pass


class _StubToolRunner:
    async def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "results": [
                {
                    "rank": 1,
                    "title": "Stub",
                    "url": "https://example.com/stub",
                    "snippet": "stub snippet",
                }
            ]
        }


def _orch(
    provider: FakeResearchProvider,
    *,
    save_thinking: bool,
) -> NotebookOrchestrator:
    provider.set_planner_response({"key_elements": ["q1"]})
    provider.set_extractor_responses(
        [
            {
                "facts": [{"text": "f1", "confidence": "high"}],
                "follow_up_queries": [],
            }
        ]
    )
    return NotebookOrchestrator(
        provider=provider,
        store=_FakeStore(),
        tool_registry=_FakeToolRegistry(),
        tool_runner=_StubToolRunner(),  # type: ignore[arg-type]
        parser=SlidingParser(),
        config=OrchestratorConfig(
            max_tool_loops=3,
            auto_reload_on_fatal_error=False,
            save_thinking=save_thinking,
            include_thinking_in_history=False,
        ),
        research_settings=ResearchSettings(
            max_facts=5,
            max_iterations=1,
            max_facts_per_extract=3,
        ),
        clock=lambda: date(2026, 6, 1),
    )


@pytest.mark.anyio
async def test_synth_strips_think_blocks_from_textdelta():
    """End-to-end: synth chunks containing ``<think>reason</think>Final``
    yield TextDelta(text=...) with no ``<think>`` or reasoning content,
    and MessageStop=complete."""
    provider = FakeResearchProvider()
    # Chunked to force marker reassembly across chunk boundaries.
    provider.set_synthesizer_response(
        ["<thi", "nk>", "secret reasoning", "</thi", "nk>", "Final answer."]
    )
    orch = _orch(provider, save_thinking=False)

    events: list[object] = []
    async for event in orch.run(
        session_id="s-think-strip",
        prompt="What is X?",
        model_name="dummy",
    ):
        events.append(event)

    text_events = [e for e in events if isinstance(e, TextDelta)]
    combined_text = "".join(e.text for e in text_events)
    assert combined_text == "Final answer."
    # No marker or reasoning content anywhere in TextDelta.
    assert "<think>" not in combined_text
    assert "</think>" not in combined_text
    assert "secret reasoning" not in combined_text

    # save_thinking=False → no ThinkingDelta emitted.
    thinking_events = [e for e in events if isinstance(e, ThinkingDelta)]
    assert thinking_events == []

    # MessageStop=complete (normal exhaustion, not cancelled).
    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1
    assert stops[0].stop_reason == "complete"


@pytest.mark.anyio
async def test_synth_save_thinking_true_emits_thinkingdelta():
    """With ``save_thinking=True`` the same stream yields a
    :class:`ThinkingDelta` carrying the stripped reasoning AND the
    user-facing answer still arrives via TextDelta."""
    provider = FakeResearchProvider()
    provider.set_synthesizer_response(
        ["<think>", "secret reasoning", "</think>", "Final answer."]
    )
    orch = _orch(provider, save_thinking=True)

    events: list[object] = []
    async for event in orch.run(
        session_id="s-think-keep",
        prompt="What is X?",
        model_name="dummy",
    ):
        events.append(event)

    text_events = [e for e in events if isinstance(e, TextDelta)]
    combined_text = "".join(e.text for e in text_events)
    assert combined_text == "Final answer."

    thinking_events = [e for e in events if isinstance(e, ThinkingDelta)]
    combined_thinking = "".join(e.text for e in thinking_events)
    assert combined_thinking == "secret reasoning"

    stops = [e for e in events if isinstance(e, MessageStop)]
    assert len(stops) == 1
    assert stops[0].stop_reason == "complete"


@pytest.mark.anyio
async def test_synth_save_thinking_false_drops_thinkingdelta():
    """Sibling of the above with ``save_thinking=False`` — must still
    NOT emit ThinkingDelta even when the stripper produced thinking."""
    provider = FakeResearchProvider()
    provider.set_synthesizer_response(
        ["<think>reason</think>", "answer"]
    )
    orch = _orch(provider, save_thinking=False)

    events: list[object] = []
    async for event in orch.run(
        session_id="s-think-drop",
        prompt="What is X?",
        model_name="dummy",
    ):
        events.append(event)

    thinking_events = [e for e in events if isinstance(e, ThinkingDelta)]
    assert thinking_events == []

    text_events = [e for e in events if isinstance(e, TextDelta)]
    assert "".join(e.text for e in text_events) == "answer"
