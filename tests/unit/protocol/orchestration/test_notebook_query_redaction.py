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


@pytest.fixture(autouse=True)
def _reset_structlog_defaults():
    """Reset structlog defaults AND replace cached module loggers.

    Required because :func:`tether.core.logging.configure_logging` —
    called by integration-test fixtures earlier in the suite — sets
    ``cache_logger_on_first_use=True`` with a stdlib-bridge factory.
    Once cached, BoundLoggers keep using stdlib regardless of
    ``capture_logs()`` swapping processors → ``capture_logs`` returns
    empty.

    Workaround: reset structlog defaults (clears global config back to
    PrintLoggerFactory, cache_logger_on_first_use=False) AND re-create
    the module-level lazy proxies in notebook + core.logging so they
    pick up the new default config on next ``logger.info`` call.

    Tracked: ``fu-notebook-tests-structlog-isolation``.
    """
    import tether.protocol.orchestration.notebook as _notebook_mod
    import tether.core.logging as _logging_mod
    structlog.reset_defaults()
    _notebook_mod.logger = structlog.get_logger(_notebook_mod.__name__)
    _logging_mod.logger = structlog.get_logger("tether")


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


# ---------------------------------------------------------------------------
# Wave 4 reconcile: L1 — verify the WARNING-path redaction
# (notebook.py explore_tool_error log site is the ACTUAL site W2's
# query-redaction fix targeted; the existing tests above only exercise
# the success-path INFO log. This test exercises the warning path.)
# ---------------------------------------------------------------------------


class _ErrorReturningToolRunner:
    """Tool runner that returns ``{"error": ...}`` instead of raising.

    Hits the ``notebook.explore_tool_error`` WARNING with ``error_type="tool_error"``
    (the dict-shaped error path at notebook.py:253-258), which is one of the
    redaction sites W2 patched. Different code path from the exception-raising
    path used in ``test_notebook_run_branches.py`` — both must be redacted.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, args))
        return {"error": "rate_limited", "retry_after": 60}


def _orch_with_error_tool(
    provider: FakeResearchProvider,
) -> NotebookOrchestrator:
    return NotebookOrchestrator(
        provider=provider,
        store=_FakeStore(),
        tool_registry=_FakeToolRegistry(),
        tool_runner=_ErrorReturningToolRunner(),  # type: ignore[arg-type]
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
        clock=lambda: date(2026, 5, 31),
    )


@pytest.mark.anyio
async def test_secret_in_warning_log_redacted():
    """The WARNING-path redaction at notebook.explore_tool_error must apply.

    Pre-W2: the warning log included ``query=<raw query>``; a secret in the
    query would leak. Post-W2: the warning logs ``query_sha256`` + ``query_length``
    only. This test exercises the dict-shaped ``{"error": ...}`` tool-result
    path (different from the exception-raising path); both should be redacted.

    Citation: nho-rev-RECONCILE.md §A3; nho-rev-xhigh Findings L1;
    nho-rev-gpt55 §4 LOW.
    """
    secret = "BRAVE_API_KEY=warningPathSecretXYZ12345"
    query_text = f"Investigate {secret} in production logs"
    provider = FakeResearchProvider()
    provider.set_planner_response({"key_elements": [query_text]})
    # Even though the tool returns error, the orchestrator may still try
    # to extract on the next iteration — but with max_iterations=1 the loop
    # exits cleanly. Provide a defensive empty extractor response.
    provider.set_extractor_responses(
        [{"facts": [], "follow_up_queries": []}]
    )
    provider.set_synthesizer_response("No data.")
    orch = _orch_with_error_tool(provider)

    with capture_logs() as logs:
        await _drain(orch)

    # The explore_tool_error WARNING must have fired exactly once (one
    # query, one tool call, one error response).
    warnings = [
        rec
        for rec in logs
        if rec.get("log_level") == "warning"
        and rec.get("event") == "notebook.explore_tool_error"
    ]
    assert warnings, (
        f"expected at least one notebook.explore_tool_error WARNING; "
        f"all logs: {[r.get('event') for r in logs]}"
    )

    expected_sha = hashlib.sha256(query_text.encode("utf-8")).hexdigest()[:8]
    for rec in warnings:
        # Must NOT carry raw query
        assert "query" not in rec, (
            f"WARNING log must not carry raw 'query' field, got: {rec}"
        )
        # MUST carry the redacted identifiers
        assert rec.get("query_sha256") == expected_sha, (
            f"expected query_sha256={expected_sha!r}, got: {rec.get('query_sha256')!r}"
        )
        assert rec.get("query_length") == len(query_text)
        # No field may contain the secret substring
        for key, value in rec.items():
            if isinstance(value, str):
                assert secret not in value, (
                    f"secret leaked into WARNING log field {key!r}: {value!r}"
                )
