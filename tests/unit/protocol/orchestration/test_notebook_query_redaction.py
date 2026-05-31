"""Phase 9.5 fu-research-query-redaction.

Verifies that ``NotebookOrchestrator`` INFO/WARNING logs no longer carry raw
sub-query text; they carry ``query_sha256`` (8 hex chars) + ``query_length``
instead. Wire events keep emitting the verbatim query (UI contract — cap on
length is the defense there).
"""
from __future__ import annotations

import hashlib
from datetime import date
from typing import Any

import pytest
import structlog
from structlog.testing import capture_logs

from tether.config.settings import ResearchSettings
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration.notebook import NotebookOrchestrator
from tether.protocol.parsers.sliding import SlidingParser
from tests.fixtures.fake_research_provider import FakeResearchProvider


class _FakeStore:
    pass


class _FakeToolRegistry:
    pass


class _StubToolRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, args))
        return {
            "results": [
                {
                    "rank": 1,
                    "title": "stub title",
                    "url": "https://example.com/stub",
                    "snippet": "stub snippet text",
                }
            ],
            "meta": {"query": args["query"]},
        }


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _provider(question_query: str) -> FakeResearchProvider:
    provider = FakeResearchProvider()
    provider.set_planner_response({"key_elements": [question_query]})
    provider.set_extractor_responses(
        [
            {
                "facts": [{"text": "stub fact", "confidence": "high"}],
                "follow_up_queries": [],
            }
        ]
    )
    provider.set_synthesizer_response("done.")
    return provider


def _orch(provider: FakeResearchProvider) -> NotebookOrchestrator:
    return NotebookOrchestrator(
        provider=provider,
        store=_FakeStore(),
        tool_registry=_FakeToolRegistry(),
        tool_runner=_StubToolRunner(),  # type: ignore[arg-type]
        parser=SlidingParser(),
        config=OrchestratorConfig(
            max_tool_loops=3,
            auto_reload_on_fatal_error=False,
            save_thinking=False,
            include_thinking_in_history=False,
        ),
        research_settings=ResearchSettings(
            max_facts=5,
            max_iterations=1,
            max_facts_per_extract=3,
        ),
        clock=lambda: date(2026, 5, 16),
    )


async def _drain(orch: NotebookOrchestrator) -> None:
    async for _event in orch.run(
        session_id="s-redaction",
        prompt="ignored prompt",
        model_name="dummy",
    ):
        pass


def _info_records(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return INFO + WARNING records (the levels the redaction targets)."""
    return [
        rec
        for rec in logs
        if rec.get("log_level") in {"info", "warning"}
    ]


@pytest.mark.anyio
async def test_info_logs_use_query_sha256_not_raw_text():
    query_text = "What is the launch date of Project Mongoose?"
    orch = _orch(_provider(query_text))

    with capture_logs() as logs:
        await _drain(orch)

    info_records = _info_records(logs)
    # At least one INFO record from the explore phase must carry the redacted
    # query identifier rather than the raw text.
    explore_starts = [
        r for r in info_records if r.get("event") == "notebook.phase_start"
        and r.get("phase") == "explore"
    ]
    assert explore_starts, f"expected an explore phase_start INFO log, got: {info_records}"

    expected_sha = hashlib.sha256(query_text.encode("utf-8")).hexdigest()[:8]
    for rec in explore_starts:
        assert "query" not in rec, (
            f"INFO log must not carry raw 'query' field, got: {rec}"
        )
        assert rec.get("query_sha256") == expected_sha, (
            f"expected query_sha256={expected_sha!r}, got: {rec.get('query_sha256')!r}"
        )

    # And no INFO/WARNING record anywhere may carry the raw query string in
    # any of its values.
    for rec in info_records:
        for key, value in rec.items():
            if isinstance(value, str):
                assert query_text not in value, (
                    f"raw query text leaked into INFO/WARNING log field "
                    f"{key!r}: {value!r}"
                )


@pytest.mark.anyio
async def test_info_logs_include_query_length():
    query_text = "Tell me about quantum entanglement"
    orch = _orch(_provider(query_text))

    with capture_logs() as logs:
        await _drain(orch)

    info_records = _info_records(logs)
    explore_starts = [
        r for r in info_records if r.get("event") == "notebook.phase_start"
        and r.get("phase") == "explore"
    ]
    assert explore_starts, "expected at least one explore phase_start INFO log"

    for rec in explore_starts:
        assert rec.get("query_length") == len(query_text), (
            f"expected query_length={len(query_text)}, got {rec.get('query_length')!r}"
        )
        assert isinstance(rec.get("query_sha256"), str)
        assert len(rec["query_sha256"]) == 8


@pytest.mark.anyio
async def test_secret_in_query_not_leaked_to_logs():
    # A pasted API key in the chat prompt would be planner-seeded as a
    # sub-query, then leak through INFO logs without redaction.
    secret = "BRAVE_API_KEY=abc12345secretSHOULDNOTAPPEAR"
    query_text = f"Why does my code have {secret} in it?"
    orch = _orch(_provider(query_text))

    with capture_logs() as logs:
        await _drain(orch)

    # Scan EVERY captured INFO/WARNING record. No field — by key or by
    # value — may contain the secret substring.
    for rec in _info_records(logs):
        for key, value in rec.items():
            if isinstance(value, str):
                assert secret not in value, (
                    f"secret leaked into log field {key!r}: {value!r}"
                )
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, str):
                        assert secret not in item, (
                            f"secret leaked into list element of log "
                            f"field {key!r}: {item!r}"
                        )
