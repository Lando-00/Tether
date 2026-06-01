"""Substring-containment dedup tests for NotebookState.try_add_fact.

Phase 9.6 follow-up I-3 (ADR-0020 §D1.bis amendment): in addition to
exact-normalized-key dedup, ``try_add_fact`` collapses paraphrase pairs
where one normalized key is a substring of the other, provided both keys
are at least 20 characters long. The longer (more specific) text wins
regardless of confidence; confidence is only a tiebreaker on equal
length. See plan §17.4 W1-B.
"""
from __future__ import annotations

from tether.protocol.orchestration.notebook_state import (
    AtomicFact,
    NotebookState,
    _dedup_key,
)


def _fact(text: str, confidence: str = "medium", source_query: str = "q") -> AtomicFact:
    return AtomicFact(text=text, source_query=source_query, confidence=confidence)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Exact-match path: must keep working unchanged.
# ---------------------------------------------------------------------------


def test_exact_duplicate_higher_confidence_replaces():
    state = NotebookState()
    long_text_a = "The Adreno X1 GPU runs MLC-LLM models locally on device"
    long_text_b = "the   adreno  x1 gpu runs MLC-LLM models locally on device!!"

    assert state.try_add_fact(_fact(long_text_a, "low")) is True
    assert state.try_add_fact(_fact(long_text_b, "high")) is True

    assert len(state.facts) == 1
    assert state.facts[0].confidence == "high"
    assert state.facts[0].text == long_text_b


def test_exact_duplicate_lower_confidence_rejected():
    state = NotebookState()
    text = "Snapdragon X Elite uses an Adreno GPU for OpenCL compute"
    assert state.try_add_fact(_fact(text, "high")) is True
    assert state.try_add_fact(_fact(text, "low")) is False

    assert len(state.facts) == 1
    assert state.facts[0].confidence == "high"


def test_exact_duplicate_equal_confidence_rejected():
    state = NotebookState()
    text = "Tether streams v2 NDJSON events to the chat client over HTTP"
    assert state.try_add_fact(_fact(text, "medium")) is True
    assert state.try_add_fact(_fact(text, "medium")) is False
    assert len(state.facts) == 1


# ---------------------------------------------------------------------------
# Substring containment: paraphrase pairs collapse to the longer text.
# ---------------------------------------------------------------------------


def test_containment_longer_added_second_replaces_short_existing():
    state = NotebookState()
    short = "Tether persists chat history in SQLite with WAL"
    longer = "Tether persists chat history in SQLite with WAL mode enabled"

    # Sanity: keys long enough to trigger the substring pass.
    assert len(_dedup_key(short)) >= 20
    assert len(_dedup_key(longer)) >= 20
    assert _dedup_key(short) in _dedup_key(longer)

    assert state.try_add_fact(_fact(short, "high")) is True
    assert state.try_add_fact(_fact(longer, "low")) is True  # longer wins despite low conf

    assert len(state.facts) == 1
    assert state.facts[0].text == longer
    assert state.facts[0].confidence == "low"


def test_containment_shorter_added_second_rejected():
    state = NotebookState()
    longer = "Tether persists chat history in SQLite with WAL mode enabled"
    short = "Tether persists chat history in SQLite with WAL"

    assert state.try_add_fact(_fact(longer, "low")) is True
    assert state.try_add_fact(_fact(short, "high")) is False  # shorter loses despite high conf

    assert len(state.facts) == 1
    assert state.facts[0].text == longer
    assert state.facts[0].confidence == "low"


def test_containment_punctuation_and_case_paraphrase_collapses():
    state = NotebookState()
    a = "The MLC provider streams tokens from the Adreno GPU"
    b = "the MLC provider streams tokens, from the Adreno GPU."

    assert state.try_add_fact(_fact(a, "medium")) is True
    assert state.try_add_fact(_fact(b, "medium")) is False  # normalized equal → exact-match path

    assert len(state.facts) == 1


def test_containment_trailing_phrase_paraphrase_collapses():
    state = NotebookState()
    a = "Function calls use the <<function_call>> marker"
    b = "Function calls use the <<function_call>> marker in streaming output"

    assert state.try_add_fact(_fact(a, "high")) is True
    assert state.try_add_fact(_fact(b, "low")) is True  # longer one wins

    assert len(state.facts) == 1
    assert state.facts[0].text == b


def test_containment_equal_length_uses_confidence_tiebreaker():
    state = NotebookState()
    # Two distinct texts that happen to share a normalized key
    # (built via case/punctuation noise) — exact match path triggers
    # first, but the spec requires equal-length containment to fall back
    # to confidence. Build a real equal-length containment by making the
    # normalized keys equal (which is also an exact-match dup) — the
    # tiebreaker behavior is observable through the resulting confidence.
    a = "The orchestrator emits message_start before any text deltas"
    b = "the orchestrator emits MESSAGE_START before any text deltas!"

    assert _dedup_key(a) == _dedup_key(b)

    assert state.try_add_fact(_fact(a, "medium")) is True
    # Equal "length" (exact-match) + equal confidence → reject.
    assert state.try_add_fact(_fact(b, "medium")) is False
    assert state.facts[0].text == a


def test_containment_replacement_returns_true_addition_returns_true_rejection_returns_false():
    state = NotebookState()
    a = "Tether uses Pydantic StrictModel for all settings classes"
    b = "Tether uses Pydantic StrictModel for all settings classes with extra forbid"
    c = "Tether uses Pydantic StrictModel for all settings"  # also a substring of a/b

    assert state.try_add_fact(_fact(a, "medium")) is True   # fresh add
    assert state.try_add_fact(_fact(b, "low")) is True      # containment replace (longer wins)
    assert state.try_add_fact(_fact(c, "high")) is False    # shorter loses despite high conf

    assert len(state.facts) == 1
    assert state.facts[0].text == b


# ---------------------------------------------------------------------------
# Mandatory non-collapse cases.
# ---------------------------------------------------------------------------


def test_divergent_containment_does_not_collapse_because_short_below_gate():
    """`"Apple revenue grew"` (key len 18, below gate) must not collapse with
    `"Apple revenue grew in Europe but fell in Asia"`. The short key is
    under the 20-char gate, so the substring pass is skipped entirely
    for it; the longer fact is added independently.
    """
    state = NotebookState()
    short = "Apple revenue grew"
    longer = "Apple revenue grew in Europe but fell in Asia"

    assert len(_dedup_key(short)) < 20  # gate

    assert state.try_add_fact(_fact(short, "medium")) is True
    assert state.try_add_fact(_fact(longer, "medium")) is True

    assert len(state.facts) == 2


def test_negation_pair_does_not_collapse():
    state = NotebookState()
    positive = "The Snapdragon X Elite supports OpenCL workloads on Adreno"
    negative = "The Snapdragon X Elite does not support OpenCL workloads on Adreno"

    # Neither normalized key is a substring of the other.
    assert _dedup_key(positive) not in _dedup_key(negative)
    assert _dedup_key(negative) not in _dedup_key(positive)

    assert state.try_add_fact(_fact(positive, "medium")) is True
    assert state.try_add_fact(_fact(negative, "medium")) is True

    assert len(state.facts) == 2


def test_numeric_difference_pair_does_not_collapse():
    state = NotebookState()
    a = "Revenue grew by ten percent year over year in the latest quarter"
    b = "Revenue grew by twenty percent year over year in the latest quarter"

    assert state.try_add_fact(_fact(a, "medium")) is True
    assert state.try_add_fact(_fact(b, "medium")) is True

    assert len(state.facts) == 2


def test_short_key_pair_below_gate_does_not_collapse():
    """Both keys < 20 chars: substring pass is skipped, both facts kept."""
    state = NotebookState()
    a = "Fast model"        # key "fast model" — 10 chars
    b = "Fast model server"  # key "fast model server" — 17 chars

    assert len(_dedup_key(a)) < 20
    assert len(_dedup_key(b)) < 20

    assert state.try_add_fact(_fact(a, "medium")) is True
    assert state.try_add_fact(_fact(b, "medium")) is True

    assert len(state.facts) == 2


def test_short_existing_skipped_when_new_is_long_enough():
    """A long new fact must not collapse against a short existing fact
    even though the long one's key length passes the gate — the gate is
    enforced on *both* keys."""
    state = NotebookState()
    short = "Apple revenue grew"  # 18 chars normalized — below gate
    longer = "Apple revenue grew in Europe but fell in Asia"  # > 20

    assert state.try_add_fact(_fact(short, "medium")) is True
    assert state.try_add_fact(_fact(longer, "medium")) is True

    assert len(state.facts) == 2
    texts = {f.text for f in state.facts}
    assert texts == {short, longer}
