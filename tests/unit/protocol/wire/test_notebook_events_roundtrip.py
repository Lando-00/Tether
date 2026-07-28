from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from tether.protocol.wire.events import (
    NotebookClarificationRequested,
    NotebookFactAdded,
    NotebookLimitReached,
    NotebookNoFacts,
    NotebookPhaseProgress,
    NotebookPhaseStart,
    NotebookQueryAdded,
    WireEvent,
)


def _base(**overrides: Any) -> dict[str, Any]:
    payload = {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "seq": 0,
        "ts": datetime.now(timezone.utc),
    }
    payload.update(overrides)
    return payload


def test_notebook_phase_start_roundtrip_succeeds():
    event = NotebookPhaseStart(**_base(phase="plan", iteration=0))

    assert NotebookPhaseStart.model_validate(event.model_dump()) == event


def test_notebook_phase_start_rejects_invalid_iteration():
    with pytest.raises(ValidationError):
        NotebookPhaseStart(**_base(phase="plan", iteration=-1))


def test_notebook_phase_start_serializes_to_json():
    event = NotebookPhaseStart(**_base(phase="synthesize", iteration=0))

    assert NotebookPhaseStart.model_validate_json(event.model_dump_json()) == event


def test_wire_event_union_accepts_notebook_phase_start():
    event = NotebookPhaseStart(**_base(phase="explore", iteration=1))

    parsed = TypeAdapter(WireEvent).validate_python(event.model_dump())

    assert isinstance(parsed, NotebookPhaseStart)
    assert parsed == event


def test_notebook_fact_added_roundtrip_succeeds():
    event = NotebookFactAdded(
        **_base(fact_text="A fact", source_query="query", total_facts=1)
    )

    assert NotebookFactAdded.model_validate(event.model_dump()) == event


def test_notebook_fact_added_rejects_invalid_total_facts():
    with pytest.raises(ValidationError):
        NotebookFactAdded(
            **_base(fact_text="A fact", source_query="query", total_facts=0)
        )


def test_notebook_fact_added_serializes_to_json():
    event = NotebookFactAdded(
        **_base(fact_text="A fact", source_query="query", total_facts=2)
    )

    assert NotebookFactAdded.model_validate_json(event.model_dump_json()) == event


def test_wire_event_union_accepts_notebook_fact_added():
    event = NotebookFactAdded(
        **_base(fact_text="A fact", source_query="query", total_facts=1)
    )

    parsed = TypeAdapter(WireEvent).validate_python(event.model_dump())

    assert isinstance(parsed, NotebookFactAdded)
    assert parsed == event


def test_notebook_fact_added_source_kind_roundtrips():
    event = NotebookFactAdded(
        **_base(
            fact_text="2 + 2 = 4",
            source_query="2 + 2",
            source_kind="local_deterministic",
            total_facts=1,
        )
    )

    assert NotebookFactAdded.model_validate(event.model_dump()) == event


def test_notebook_clarification_requested_roundtrips_and_is_a_wire_event():
    event = NotebookClarificationRequested(
        **_base(
            reason="ambiguous_correction",
            message="Which term should I replace?",
            candidates=["Irelend"],
        )
    )

    assert TypeAdapter(WireEvent).validate_python(event.model_dump()) == event


def test_notebook_clarification_requested_enforces_bounds():
    with pytest.raises(ValidationError):
        NotebookClarificationRequested(
            **_base(reason="unsearchable_input", message="x" * 513)
        )
    with pytest.raises(ValidationError):
        NotebookClarificationRequested(
            **_base(
                reason="unsearchable_input",
                message="Need a question",
                candidates=["x"] * 6,
            )
        )
    with pytest.raises(ValidationError):
        NotebookClarificationRequested(
            **_base(
                reason="unsearchable_input",
                message="Need a question",
                candidates=["x" * 257],
            )
        )


def test_notebook_query_added_roundtrip_succeeds():
    event = NotebookQueryAdded(**_base(query="search terms", queue_depth=1))

    assert NotebookQueryAdded.model_validate(event.model_dump()) == event


def test_notebook_query_added_rejects_invalid_queue_depth():
    with pytest.raises(ValidationError):
        NotebookQueryAdded(**_base(query="search terms", queue_depth=0))


def test_notebook_query_added_serializes_to_json():
    event = NotebookQueryAdded(**_base(query="search terms", queue_depth=3))

    assert NotebookQueryAdded.model_validate_json(event.model_dump_json()) == event


def test_wire_event_union_accepts_notebook_query_added():
    event = NotebookQueryAdded(**_base(query="search terms", queue_depth=1))

    parsed = TypeAdapter(WireEvent).validate_python(event.model_dump())

    assert isinstance(parsed, NotebookQueryAdded)
    assert parsed == event


def test_notebook_limit_reached_roundtrip_succeeds():
    event = NotebookLimitReached(**_base(limit_kind="max_facts", count=40))

    assert NotebookLimitReached.model_validate(event.model_dump()) == event


def test_notebook_limit_reached_rejects_invalid_limit_kind():
    with pytest.raises(ValidationError):
        NotebookLimitReached(**_base(limit_kind="garbage", count=1))


def test_notebook_limit_reached_serializes_to_json():
    event = NotebookLimitReached(**_base(limit_kind="max_iterations", count=20))

    assert NotebookLimitReached.model_validate_json(event.model_dump_json()) == event


def test_wire_event_union_accepts_notebook_limit_reached():
    event = NotebookLimitReached(**_base(limit_kind="max_iterations", count=20))

    parsed = TypeAdapter(WireEvent).validate_python(event.model_dump())

    assert isinstance(parsed, NotebookLimitReached)
    assert parsed == event


# --- Field length bounds (Phase 9.5 Wave 1A: fu-research-event-field-bounds) ---


def test_notebook_fact_text_rejects_over_4096():
    with pytest.raises(ValidationError) as exc_info:
        NotebookFactAdded(
            **_base(fact_text="x" * 4097, source_query="query", total_facts=1)
        )

    assert "String should have at most 4096 characters" in str(exc_info.value)


def test_notebook_fact_text_accepts_4096_exactly():
    event = NotebookFactAdded(
        **_base(fact_text="x" * 4096, source_query="query", total_facts=1)
    )

    assert len(event.fact_text) == 4096


def test_notebook_fact_text_accepts_under_limit():
    event = NotebookFactAdded(
        **_base(fact_text="short fact", source_query="query", total_facts=1)
    )

    assert event.fact_text == "short fact"


def test_notebook_query_rejects_over_512():
    with pytest.raises(ValidationError) as exc_info:
        NotebookQueryAdded(**_base(query="q" * 513, queue_depth=1))

    assert "String should have at most 512 characters" in str(exc_info.value)


def test_notebook_query_accepts_512_exactly():
    event = NotebookQueryAdded(**_base(query="q" * 512, queue_depth=1))

    assert len(event.query) == 512


def test_notebook_query_accepts_under_limit():
    event = NotebookQueryAdded(**_base(query="search terms", queue_depth=1))

    assert event.query == "search terms"


# --- NotebookPhaseProgress (Phase 9.6 W1-C: nho-q-w1c-events) ---


def test_notebook_phase_progress_roundtrip_succeeds():
    event = NotebookPhaseProgress(
        **_base(phase="explore", iteration=2, elapsed_ms=1500, note="running query")
    )

    assert NotebookPhaseProgress.model_validate(event.model_dump()) == event


def test_notebook_phase_progress_wire_union_roundtrip():
    event = NotebookPhaseProgress(
        **_base(phase="synthesize", iteration=0, elapsed_ms=42)
    )

    parsed = TypeAdapter(WireEvent).validate_python(event.model_dump())

    assert isinstance(parsed, NotebookPhaseProgress)
    assert parsed == event


def test_notebook_phase_progress_json_includes_fields():
    event = NotebookPhaseProgress(
        **_base(phase="extract", iteration=1, elapsed_ms=750, note="parsing facts")
    )

    payload = event.model_dump_json()

    assert '"type":"notebook_phase_progress"' in payload
    assert '"elapsed_ms":750' in payload
    assert '"note":"parsing facts"' in payload


def test_notebook_phase_progress_rejects_negative_elapsed_ms():
    with pytest.raises(ValidationError):
        NotebookPhaseProgress(**_base(phase="plan", iteration=0, elapsed_ms=-1))


# --- NotebookNoFacts (Phase 9.7 W3-B: nho-fu-w3b-empty-signal) ---

def test_notebook_no_facts_roundtrip_succeeds():
    event = NotebookNoFacts(
        **_base(queries_attempted=3, iterations=3, note="all rate-limited")
    )

    assert NotebookNoFacts.model_validate(event.model_dump()) == event


def test_notebook_no_facts_defaults_note_to_none():
    event = NotebookNoFacts(**_base(queries_attempted=0, iterations=0))

    assert event.note is None


def test_notebook_no_facts_rejects_negative_queries_attempted():
    with pytest.raises(ValidationError):
        NotebookNoFacts(**_base(queries_attempted=-1, iterations=0))


def test_notebook_no_facts_rejects_negative_iterations():
    with pytest.raises(ValidationError):
        NotebookNoFacts(**_base(queries_attempted=0, iterations=-1))


def test_notebook_no_facts_rejects_note_over_256():
    with pytest.raises(ValidationError):
        NotebookNoFacts(
            **_base(queries_attempted=0, iterations=0, note="x" * 257)
        )


def test_notebook_no_facts_serializes_to_json():
    event = NotebookNoFacts(
        **_base(queries_attempted=2, iterations=2, note="nothing found")
    )
    payload = event.model_dump_json()

    assert '"type":"notebook_no_facts"' in payload
    assert '"queries_attempted":2' in payload
    assert '"iterations":2' in payload
    assert NotebookNoFacts.model_validate_json(payload) == event


def test_wire_event_union_accepts_notebook_no_facts():
    event = NotebookNoFacts(**_base(queries_attempted=1, iterations=1))

    parsed = TypeAdapter(WireEvent).validate_python(event.model_dump())

    assert isinstance(parsed, NotebookNoFacts)
    assert parsed == event
