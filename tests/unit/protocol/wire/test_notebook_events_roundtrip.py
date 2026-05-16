from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from tether.protocol.wire.events import (
    NotebookFactAdded,
    NotebookLimitReached,
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
