"""Tests for turn triage and the direct-answer path (AutoOrchestrator).

Triage is what makes the fact-based orchestrator safe as the *default*: without
it, "hello" would be decomposed into search queries. See
``docs/design/fact-based-orchestration-default.md``.
"""
from __future__ import annotations

import pytest

from tether.protocol.intent.rules_turn_triage import RulesTurnTriage
from tether.protocol.intent.turn_triage import (
    AlwaysResearchTriage,
    TurnKind,
    TurnTriage,
)


@pytest.fixture
def triage() -> RulesTurnTriage:
    return RulesTurnTriage()


# ---------------------------------------------------------------------------
# DIRECT: must never trigger a web search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "hello",
        "Hi!",
        "hey",
        "thanks",
        "Thank you.",
        "ok",
        "bye",
        "how are you?",
        "who are you?",
        "what can you do?",
    ],
)
def test_small_talk_is_direct(triage: RulesTurnTriage, prompt: str) -> None:
    assert triage.classify(prompt) is TurnKind.DIRECT


@pytest.mark.parametrize(
    "prompt",
    [
        "Write a haiku about the sea",
        "write me a poem",
        "Compose an email to my landlord",
        "Translate this to French: good morning",
        "Rewrite the paragraph above to be shorter",
        "implement a binary search in python",
        "Draft a polite decline",
        "brainstorm names for a cat",
    ],
)
def test_creative_and_generative_work_is_direct(
    triage: RulesTurnTriage, prompt: str
) -> None:
    assert triage.classify(prompt) is TurnKind.DIRECT


@pytest.mark.parametrize(
    "prompt",
    [
        "What did I just say?",
        "what did you say about that?",
        "Repeat that again",
        "Summarise our conversation",
        "You said something earlier about the plan",
    ],
)
def test_back_references_are_direct(triage: RulesTurnTriage, prompt: str) -> None:
    """Back-references belong to the transcript, never to a web search."""
    assert triage.classify(prompt, has_history=True) is TurnKind.DIRECT


def test_empty_prompt_is_direct(triage: RulesTurnTriage) -> None:
    assert triage.classify("") is TurnKind.DIRECT
    assert triage.classify("   ") is TurnKind.DIRECT


def test_back_reference_beats_an_evidence_marker(triage: RulesTurnTriage) -> None:
    """'what did you say about the latest release' reads history, not the web."""
    assert (
        triage.classify("What did you say about the latest release?", has_history=True)
        is TurnKind.DIRECT
    )


# ---------------------------------------------------------------------------
# RESEARCH: needs external evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "What is the latest version of Python?",
        "Who won the 2022 FIFA World Cup?",
        "current price of bitcoin",
        "Any news about the election?",
        "When was Kubernetes released?",
        "What is the population of Ireland?",
        "search for tether documentation",
    ],
)
def test_evidence_seeking_questions_are_research(
    triage: RulesTurnTriage, prompt: str
) -> None:
    assert triage.classify(prompt) is TurnKind.RESEARCH


@pytest.mark.parametrize(
    "prompt",
    [
        "What time is it in Europe/Dublin?",
        "what's the time?",
        "What's the weather in Dublin?",
        "Give me the forecast for tomorrow",
        "What is the temperature right now?",
        "Is it raining in London?",
    ],
)
def test_local_tool_domains_are_direct(
    triage: RulesTurnTriage, prompt: str
) -> None:
    """Questions a registered tool answers must not go to web research.

    The research loop only knows ``web_search``; routing these there produced a
    failed turn instead of a ``time`` / ``weather`` tool call. DIRECT hands them
    to the chat orchestrator, which has the full tool loop.
    """
    assert triage.classify(prompt) is TurnKind.DIRECT


def test_evidence_marker_beats_a_creative_opener(triage: RulesTurnTriage) -> None:
    """'write a summary of today's news' still needs the news."""
    assert (
        triage.classify("Write a summary of today's news") is TurnKind.RESEARCH
    )


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_always_research_triage_never_defers() -> None:
    """Explicit mode="research" must not be downgraded to a direct answer."""
    always = AlwaysResearchTriage()
    assert always.classify("hello") is TurnKind.RESEARCH
    assert always.classify("") is TurnKind.RESEARCH


def test_rules_triage_is_a_turn_triage(triage: RulesTurnTriage) -> None:
    assert isinstance(triage, TurnTriage)


@pytest.mark.parametrize(
    "prompt",
    ["", "?", "!!!", "a", "\n\t ", "x" * 5000, "🎉🎉🎉", "SELECT * FROM t"],
)
def test_classify_never_raises(triage: RulesTurnTriage, prompt: str) -> None:
    """The contract forbids raising — it runs on every turn before any model call."""
    assert triage.classify(prompt) in (TurnKind.DIRECT, TurnKind.RESEARCH)
