"""Tests for the reasoning-leak filter in :mod:`notebook_parser` (Phase 9.6 W1-A · I-2).

These tests drive the live-test failure mode where the Extractor LLM emitted
meta-prose like ``"The first snippet talks about ..."`` inside ``fact.text``
values. The parser must drop those facts and surface the count via
``ExtractResult.reasoning_leak_dropped`` while leaving genuine world-facts
untouched — including facts that happen to open with similar English
phrases (``"The first iPhone shipped in 2007"``).

See plan.md §17.4 W1-A for the corpus and the rubber-duck amendment
requiring positive-control coverage.
"""

from __future__ import annotations

import json

import pytest

from tether.protocol.orchestration.notebook_parser import (
    ExtractResult,
    _REASONING_LEAK_PREFIXES,
    _is_reasoning_leak,
    parse_extract_output,
)


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------


# Verbatim / near-verbatim leaks observed in the live manual test (or
# trivially close paraphrases of them — same opener vocabulary).
LIVE_LEAKED_FACTS: list[str] = [
    "The first snippet talks about Snapdragon X Elite performance.",
    "The second snippet mentions AnythingLLM and Nexa support.",
    "The third snippet says the Adreno GPU has 4.6 TFLOPS.",
    "This snippet indicates that the NPU is not yet exposed to userland.",
    "Confidence is medium because only one source mentions it.",
    "The snippet mentions AnythingLLM runs LLMs on NPU.",
    "The snippet says LM Studio can run LLMs on CPU.",
]


# Clean world-facts that must survive — written so each looks like a
# real Extractor emission for the question "What can the Snapdragon X
# Elite run today?".
CLEAN_FACTS: list[str] = [
    "Snapdragon X Elite ships with a 45 TOPS Hexagon NPU.",
    "AnythingLLM exposes a desktop UI for local LLM chat.",
    "Nexa SDK packages MLC-LLM binaries for Adreno GPUs.",
    "Llama.cpp added OpenCL support for Adreno GPUs in late 2024.",
    "The Surface Pro 11 uses a Snapdragon X Elite X1E-80-100 chip.",
]


# Positive controls that MUST survive even though they open with words
# that look like leak prefixes when read carelessly. These come straight
# from the plan §17.4 W1-A spec.
POSITIVE_CONTROLS: list[str] = [
    "The first iPhone shipped in 2007.",
    "Confidence in the vaccine increased in 2025.",
    "This benchmark reports 45 TOPS.",
    'Snippet says "<<function_call>>" without escalating to a tool call.',
    'The snippet contains: {"key":"value"}',
]


# ---------------------------------------------------------------------------
# Unit-level: _is_reasoning_leak
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", LIVE_LEAKED_FACTS)
def test_is_reasoning_leak_flags_live_leaked_facts(text: str) -> None:
    assert _is_reasoning_leak(text) is True


@pytest.mark.parametrize("text", CLEAN_FACTS)
def test_is_reasoning_leak_passes_clean_facts(text: str) -> None:
    assert _is_reasoning_leak(text) is False


@pytest.mark.parametrize("text", POSITIVE_CONTROLS)
def test_is_reasoning_leak_passes_positive_controls(text: str) -> None:
    assert _is_reasoning_leak(text) is False, (
        f"Positive control was incorrectly flagged as a leak: {text!r}"
    )


def test_is_reasoning_leak_case_insensitive() -> None:
    assert _is_reasoning_leak("THE FIRST SNIPPET says X.") is True
    assert _is_reasoning_leak("Confidence Is Low because nobody cited it.") is True


def test_is_reasoning_leak_tolerates_leading_whitespace() -> None:
    assert _is_reasoning_leak("   the third snippet mentions NPU.") is True
    assert _is_reasoning_leak("\t\nThis snippet says X.") is True


def test_is_reasoning_leak_drops_the_snippet_meta_verbs_only() -> None:
    assert _is_reasoning_leak("The snippet mentions AnythingLLM runs LLMs.") is True
    assert _is_reasoning_leak("The snippet talks about LM Studio.") is True
    assert _is_reasoning_leak("The snippet says NPU support is planned.") is True
    # Prompt-injection regression guard: "contains" can be a valid data fact.
    assert _is_reasoning_leak('The snippet contains: {"key":"value"}') is False


def test_is_reasoning_leak_empty_and_none() -> None:
    assert _is_reasoning_leak("") is False
    assert _is_reasoning_leak("   \n\t  ") is False


def test_prefixes_are_all_lowercase_with_trailing_space() -> None:
    # Spec says comparison normalises to lower + strip. The prefixes
    # themselves must therefore already be lowercase and end in a space
    # so that the startswith check is unambiguous.
    for prefix in _REASONING_LEAK_PREFIXES:
        assert prefix == prefix.lower(), f"Prefix not lowercase: {prefix!r}"
        assert prefix.endswith(" "), f"Prefix missing trailing space: {prefix!r}"


# ---------------------------------------------------------------------------
# parse_extract_output integration
# ---------------------------------------------------------------------------


def _make_extract_payload(fact_texts: list[str], confidence: str = "medium") -> str:
    return json.dumps(
        {
            "facts": [{"text": t, "confidence": confidence} for t in fact_texts],
            "follow_up_queries": [],
        }
    )


@pytest.mark.parametrize("text", LIVE_LEAKED_FACTS)
def test_parse_extract_output_drops_leaked_facts(text: str) -> None:
    raw = _make_extract_payload([text])
    result = parse_extract_output(raw, "source-query")

    assert result.facts == []
    assert result.reasoning_leak_dropped == 1
    assert result.parser_layer == 1
    assert result.follow_up_queries == []


@pytest.mark.parametrize("text", CLEAN_FACTS)
def test_parse_extract_output_keeps_clean_facts(text: str) -> None:
    raw = _make_extract_payload([text])
    result = parse_extract_output(raw, "source-query")

    assert [f.text for f in result.facts] == [text]
    assert result.reasoning_leak_dropped == 0
    # source_query is preserved on kept facts.
    assert all(f.source_query == "source-query" for f in result.facts)
    # Confidence on the kept fact is preserved (medium in our payload).
    assert all(f.confidence == "medium" for f in result.facts)


@pytest.mark.parametrize("text", POSITIVE_CONTROLS)
def test_parse_extract_output_keeps_positive_controls(text: str) -> None:
    raw = _make_extract_payload([text], confidence="high")
    result = parse_extract_output(raw, "source-query")

    assert [f.text for f in result.facts] == [text], (
        f"Positive-control fact was incorrectly dropped: {text!r}"
    )
    assert result.reasoning_leak_dropped == 0
    assert all(f.confidence == "high" for f in result.facts)


def test_parse_extract_output_mixed_batch_keeps_clean_drops_leaks() -> None:
    mixed = [
        CLEAN_FACTS[0],
        LIVE_LEAKED_FACTS[0],
        CLEAN_FACTS[1],
        LIVE_LEAKED_FACTS[1],
        POSITIVE_CONTROLS[0],
        LIVE_LEAKED_FACTS[2],
    ]
    raw = _make_extract_payload(mixed)
    result = parse_extract_output(raw, "source-query")

    kept = [f.text for f in result.facts]
    assert kept == [CLEAN_FACTS[0], CLEAN_FACTS[1], POSITIVE_CONTROLS[0]]
    assert result.reasoning_leak_dropped == 3


def test_parse_extract_output_all_leaks_returns_empty_facts() -> None:
    raw = _make_extract_payload(LIVE_LEAKED_FACTS)
    result = parse_extract_output(raw, "source-query")

    assert result.facts == []
    assert result.reasoning_leak_dropped == len(LIVE_LEAKED_FACTS)
    # An all-leak batch is not a parser failure — layer 1 still parses.
    assert result.parser_layer == 1


def test_parse_extract_output_drops_leak_with_leading_whitespace() -> None:
    raw = _make_extract_payload(
        [
            "   The first snippet says NPU works.",
            "AnythingLLM runs on the Snapdragon X Elite.",
        ]
    )
    result = parse_extract_output(raw, "q")

    assert [f.text for f in result.facts] == [
        "AnythingLLM runs on the Snapdragon X Elite."
    ]
    assert result.reasoning_leak_dropped == 1


def test_parse_extract_output_drops_leak_uppercase() -> None:
    raw = _make_extract_payload(
        [
            "THE FIRST SNIPPET TALKS ABOUT NPU.",
            "Surface Pro 11 ships with Snapdragon X Elite.",
        ]
    )
    result = parse_extract_output(raw, "q")

    assert [f.text for f in result.facts] == [
        "Surface Pro 11 ships with Snapdragon X Elite."
    ]
    assert result.reasoning_leak_dropped == 1


def test_parse_extract_output_default_reasoning_leak_dropped_is_zero() -> None:
    # ExtractResult must default the new field so legacy constructors
    # in tests / orchestrator code keep working without modification.
    blank = ExtractResult()
    assert blank.reasoning_leak_dropped == 0


def test_parse_extract_output_layer4_bullet_fallback_drops_leaks() -> None:
    # Bullets path also runs through the leak filter so the same
    # protection applies when the LLM emits prose bullets instead of
    # JSON.
    raw = (
        "- The first snippet talks about NPU support.\n"
        "- Snapdragon X Elite NPU is rated at 45 TOPS.\n"
        "- This snippet mentions Adreno OpenCL drivers.\n"
        "- AnythingLLM runs LLMs on the Snapdragon X Elite NPU."
    )
    result = parse_extract_output(raw, "q")

    assert result.parser_layer == 4
    assert [f.text for f in result.facts] == [
        "Snapdragon X Elite NPU is rated at 45 TOPS",
        "AnythingLLM runs LLMs on the Snapdragon X Elite NPU",
    ]
    assert result.reasoning_leak_dropped == 2


def test_parse_extract_output_max_facts_applied_after_leak_drop() -> None:
    # Per spec: leak drop happens BEFORE the max_facts cap so leaks
    # cannot consume the budget and crowd out clean facts.
    mixed = [
        LIVE_LEAKED_FACTS[0],
        LIVE_LEAKED_FACTS[1],
        CLEAN_FACTS[0],
        CLEAN_FACTS[1],
        CLEAN_FACTS[2],
        CLEAN_FACTS[3],
    ]
    raw = _make_extract_payload(mixed)
    result = parse_extract_output(raw, "q", max_facts=3)

    assert [f.text for f in result.facts] == CLEAN_FACTS[:3]
    assert result.reasoning_leak_dropped == 2
