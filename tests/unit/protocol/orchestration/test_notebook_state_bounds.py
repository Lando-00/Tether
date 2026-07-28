from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from tether.protocol.orchestration.notebook_state import (
    AtomicFact,
    NotebookState,
    _conf_rank,
    _dedup_key,
    _normalize_query,
)


def _fact(text: str, confidence: str = "medium", source_query: str = "q") -> AtomicFact:
    return AtomicFact(text=text, source_query=source_query, confidence=confidence)  # type: ignore[arg-type]


def test_atomic_fact_construction():
    fact = _fact("Tether has research mode", "high", "research mode")

    assert fact.text == "Tether has research mode"
    assert fact.source_query == "research mode"
    assert fact.confidence == "high"
    assert fact.source_kind == "web_search"
    assert fact.created_at.tzinfo is not None
    assert fact.created_at.utcoffset() == datetime.now(timezone.utc).utcoffset()
    with pytest.raises(FrozenInstanceError):
        fact.text = "changed"  # type: ignore[misc]


def test_atomic_fact_no_source_result_idx():
    sentinel = object()
    fact = _fact("No result index")

    assert getattr(fact, "source_result_idx", sentinel) is sentinel


def test_dedup_key_normalization():
    assert _dedup_key("Hello World!") == _dedup_key("HELLO  world!!!")


def test_conf_rank_ordering():
    assert _conf_rank("low") < _conf_rank("medium") < _conf_rank("high")


def test_normalize_query_strip_lower():
    assert _normalize_query("  Hello World  ") == _normalize_query("hello world")


def test_try_add_fact_first_insert_returns_true():
    state = NotebookState()

    assert state.try_add_fact(_fact("First fact")) is True
    assert len(state.facts) == 1


def test_try_add_fact_duplicate_same_conf_returns_false():
    state = NotebookState()
    assert state.try_add_fact(_fact("Hello World!", "medium")) is True

    assert state.try_add_fact(_fact("HELLO  world!!!", "medium")) is False
    assert len(state.facts) == 1


def test_try_add_fact_duplicate_higher_conf_replaces():
    state = NotebookState()
    assert state.try_add_fact(_fact("Hello World!", "low")) is True

    assert state.try_add_fact(_fact("HELLO  world!!!", "high")) is True
    assert len(state.facts) == 1
    assert state.facts[0].confidence == "high"
    assert state.facts[0].text == "HELLO  world!!!"


def test_try_add_fact_duplicate_lower_conf_returns_false():
    state = NotebookState()
    assert state.try_add_fact(_fact("Hello World!", "high")) is True

    assert state.try_add_fact(_fact("HELLO  world!!!", "low")) is False
    assert len(state.facts) == 1
    assert state.facts[0].confidence == "high"


def test_should_continue_respects_max_facts():
    state = NotebookState(max_facts=3)
    state.queue.append("next")
    for index in range(3):
        state.facts.append(_fact(f"fact {index}"))

    assert state.should_continue() is False


def test_should_continue_respects_max_iterations():
    state = NotebookState(max_iterations=2, iteration=2)
    state.queue.append("next")

    assert state.should_continue() is False


def test_should_continue_true_under_bounds():
    state = NotebookState(max_facts=3, max_iterations=2, iteration=0)
    state.queue.append("next")
    state.facts.append(_fact("one"))

    assert state.should_continue() is True


def test_limit_kind_max_facts():
    state = NotebookState(max_facts=1)
    state.facts.append(_fact("one"))

    assert state.limit_kind() == "max_facts"


def test_limit_kind_max_iterations():
    state = NotebookState(max_iterations=2, iteration=2)

    assert state.limit_kind() == "max_iterations"


def test_limit_kind_none_under_bounds():
    state = NotebookState(max_facts=3, max_iterations=2, iteration=0)
    state.facts.append(_fact("one"))

    assert state.limit_kind() is None


def test_processed_queries_dedup():
    state = NotebookState()
    state.processed_queries.add(_normalize_query("AI safety"))

    assert _normalize_query("  AI safety  ") in state.processed_queries
    assert len(state.processed_queries) == 1
