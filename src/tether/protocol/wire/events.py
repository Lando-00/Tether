"""Wire-protocol typed events for Tether's NDJSON streaming API.

Phase 5 foundation. Existing v0 wire vocabulary (``text``, ``tool_started``,
``tool_completed``, ``done``, etc.) continues to ship via the legacy emitter
during the cutover. ``p5-cutover-a`` (dual-emit) will introduce content
negotiation; ``p5-cutover-c`` flips the default to v2.

``PROTOCOL_VERSION = "1.0"`` is the version handshake exposed via the
``Accept`` header during the dual-emit phase.

Synthesis §4 Phase 5 step 49; §11.3 R1 (``stop_reason`` includes
``cancelled`` + ``client_disconnect``); §11.3 R6 (capabilities-denied uses
``ToolResult`` with ``error_kind="permission"`` + ``missing_capabilities``).
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL_VERSION = "1.0"


class _Base(BaseModel):
    """Common envelope fields for every :data:`WireEvent`.

    All v2 events share these. The ``type`` discriminator is on each
    concrete subclass.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: str = PROTOCOL_VERSION
    session_id: str
    turn_id: str
    seq: int = Field(ge=0, description="Per-turn monotonically-increasing sequence number")
    ts: datetime = Field(description="Server-side emission timestamp (UTC)")


class ToolDescriptor(BaseModel):
    """Tool descriptor used in :class:`MessageStart` ``available_tools``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)


# --- WireEvent variants -----------------------------------------------------


class MessageStart(_Base):
    """Start of an assistant turn. Emitted before any text/tool events."""

    type: Literal["message_start"] = "message_start"
    available_tools: List[ToolDescriptor] = Field(default_factory=list)


class MessageStop(_Base):
    """End of an assistant turn. Always emitted last (one per turn).

    ``stop_reason`` values per synthesis §11.3 R1:

    ``complete``
        Model produced a final answer.
    ``tool_loop_exhausted``
        ``max_tool_loops`` hit (``LoopLimitPolicy``).
    ``cancelled``
        Orchestrator's ``cancel_event`` was set (or the library user
        called ``engine.aclose`` mid-stream).
    ``client_disconnect``
        HTTP client closed the connection.
    ``error``
        Recoverable error during the turn.
    """

    type: Literal["message_stop"] = "message_stop"
    stop_reason: Literal[
        "complete",
        "tool_loop_exhausted",
        "cancelled",
        "client_disconnect",
        "error",
    ]


class TextDelta(_Base):
    """Assistant text token (incremental)."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDelta(_Base):
    """Reasoning channel token (incremental). Used by models that emit
    thinking separately (e.g., R1, future Tether-supported reasoning
    models)."""

    type: Literal["thinking_delta"] = "thinking_delta"
    text: str


class ToolCall(_Base):
    """Model decides to call a tool. Followed by :class:`ToolResult` after
    execution."""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(_Base):
    """Result of a tool call. Always paired with a preceding
    :class:`ToolCall`.

    ``error_kind`` values per synthesis §11.3 R6:

    ``permission``
        Capability denied (``missing_capabilities`` populated).
    ``execution``
        Tool raised during run.
    ``timeout``
        Tool exceeded timeout budget.
    ``cancelled``
        Tool task was cancelled mid-run.
    """

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["ok", "error"]
    error_kind: Optional[Literal["permission", "execution", "timeout", "cancelled"]] = None
    missing_capabilities: List[str] = Field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class Error(_Base):
    """Non-recoverable error during the turn."""

    type: Literal["error"] = "error"
    message: str
    error_type: str
    is_fatal: bool = False


class LoopLimitReached(_Base):
    """Tool loop hit ``max_tool_loops``. Emitted by
    ``LoopLimitPolicy.EMIT_LIMIT_EVENT`` (the default per the ratified
    plan)."""

    type: Literal["loop_limit_reached"] = "loop_limit_reached"
    loops: int


class HwReset(_Base):
    """``HardwareWatchdog`` performed an ``hw_reset`` on the model.
    Informational; follow-up retry is the orchestrator's choice."""

    type: Literal["hw_reset"] = "hw_reset"
    model_name: str


# --- NotebookOrchestrator events -------------------------------------------
# Phase 9 (ADR-0020): research-mode wire events. All four are emitted
# BEFORE :class:`MessageStart`. They inherit :class:`_Base`, so they
# carry the same envelope (``session_id``, ``turn_id``, ``seq``, ``ts``,
# ``protocol_version``) as the existing v2 events. See ADR-0020 Appendix B
# for the authoritative event-schema spec.


class NotebookPhaseStart(_Base):
    """Emitted once per phase transition in the Notebook loop.

    ``phase`` values:

    ``plan``
        LLM has decomposed the user question into Key Elements / initial
        sub-queries. Emitted once, before the first Explore phase.
    ``explore``
        A sub-query has been dequeued and the tool call is about to run.
        Emitted once per loop iteration (once per sub-query consumed).
    ``extract``
        Tool result received; LLM is now extracting atomic facts from the
        raw result into the Notebook. Emitted once per loop iteration,
        after the explore tool result lands.
    ``refine``
        At least one new follow-up query was enqueued from the extract
        step. Emitted at most once per loop iteration (skipped if no new
        queries were produced by this iteration).
    ``synthesize``
        Notebook is considered complete; LLM will now synthesize the final
        answer. Followed immediately by ``MessageStart``.

    ``iteration`` is the loop iteration counter. It is ``0`` for the
    one-shot ``plan`` and ``synthesize`` phases. For ``explore``/``extract``/
    ``refine`` it is ``1``-indexed (i.e., the first dequeued sub-query
    yields ``iteration=1``) — matches the impl's increment-before-yield
    pattern in ``notebook.py`` (Wave 4 reconcile R-F2). Pydantic accepts
    ``ge=0`` to cover both shapes.
    """

    type: Literal["notebook_phase_start"] = "notebook_phase_start"
    phase: Literal["plan", "explore", "extract", "refine", "synthesize"]
    iteration: int = Field(
        default=0,
        ge=0,
        description=(
            "Loop iteration counter. 0 for one-shot plan/synthesize; "
            "1-indexed for explore/extract/refine (Wave 4 R-F2)."
        ),
    )


class NotebookPhaseProgress(_Base):
    """Heartbeat progress event emitted during a long-running Notebook phase.

    Unlike :class:`NotebookPhaseStart` (one-shot at phase entry), this event
    may be emitted multiple times within a single phase to surface elapsed
    time and an optional human-readable note. Useful for UI progress
    indicators on slow phases (e.g., long ``explore`` tool calls or
    multi-second ``extract`` / ``synthesize`` LLM passes).

    ``phase`` mirrors :class:`NotebookPhaseStart` values. ``iteration``
    follows the same indexing convention: ``0`` for one-shot
    ``plan``/``synthesize``, 1-indexed for ``explore``/``extract``/
    ``refine``. ``elapsed_ms`` is the time spent in the current phase so
    far. ``note`` is an optional short descriptor (e.g., the running
    sub-query, or ``"awaiting tool result"``).
    """

    type: Literal["notebook_phase_progress"] = "notebook_phase_progress"
    phase: Literal["plan", "explore", "extract", "refine", "synthesize"]
    iteration: int = Field(
        default=0,
        ge=0,
        description=(
            "Loop iteration counter. 0 for one-shot plan/synthesize; "
            "1-indexed for explore/extract/refine (matches NotebookPhaseStart)."
        ),
    )
    elapsed_ms: int = Field(
        ge=0,
        description="Milliseconds elapsed in the current phase at emission time.",
    )
    note: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Optional short progress note (e.g., running sub-query).",
    )


class NotebookFactAdded(_Base):
    """One atomic fact extracted into the Notebook.

    Emitted once per fact string written during the ``extract`` phase.
    ``source_query`` is the sub-query whose tool result produced this fact.
    ``total_facts`` is the running total **after** this fact is added —
    so the first fact gives ``total_facts=1``.
    """

    type: Literal["notebook_fact_added"] = "notebook_fact_added"
    fact_text: str = Field(
        max_length=4096,
        description="The atomic fact string (e.g. 'Apple CEO is Tim Cook')",
    )
    source_query: str = Field(
        description="The sub-query whose tool result produced this fact"
    )
    source_kind: Literal["web_search", "local_deterministic"] = "web_search"
    total_facts: int = Field(
        ge=1, description="Running total of Notebook facts after this addition"
    )


class NotebookClarificationRequested(_Base):
    """Research input needs a user clarification before planning can proceed."""

    type: Literal["notebook_clarification_requested"] = "notebook_clarification_requested"
    reason: Literal["ambiguous_correction", "ambiguous_entity", "unsearchable_input"]
    message: str = Field(max_length=512)
    candidates: List[Annotated[str, Field(max_length=256)]] = Field(
        default_factory=list, max_length=5
    )


class NotebookQueryAdded(_Base):
    """One sub-query enqueued for future exploration.

    Emitted when the planner (``plan`` phase) or the extractor
    (``refine`` phase) adds a new query to the work queue.
    ``queue_depth`` is the queue length **after** the enqueue.
    """

    type: Literal["notebook_query_added"] = "notebook_query_added"
    query: str = Field(max_length=512, description="The sub-query string enqueued")
    queue_depth: int = Field(ge=1, description="Queue depth after this enqueue")


class NotebookLimitReached(_Base):
    """The Notebook loop hit a configured bound and stopped early.

    ``limit_kind`` values:

    ``max_facts``
        ``len(notebook) >= max_facts`` was reached before the queue
        emptied. ``count`` is the final fact count.
    ``max_iterations``
        The loop iteration counter reached ``max_iterations`` before the
        queue emptied. ``count`` is the number of iterations completed.

    This event is always followed by
    ``NotebookPhaseStart(phase="synthesize")`` so synthesis still runs
    on the partial Notebook.
    """

    type: Literal["notebook_limit_reached"] = "notebook_limit_reached"
    limit_kind: Literal["max_facts", "max_iterations"]
    count: int = Field(
        ge=0,
        description="Final fact count (max_facts) or iteration count (max_iterations)",
    )


class NotebookNoFacts(_Base):
    """The Notebook loop ended with zero facts gathered.

    Emitted immediately before ``NotebookPhaseStart(phase="synthesize")``
    when ``len(notebook.facts) == 0`` at loop exit (and the turn was not
    cancelled). Synthesis still runs on the empty Notebook and the turn
    still terminates with ``MessageStop(stop_reason="complete")`` — this
    event is a signal, NOT an :class:`Error` and NOT a
    :class:`NotebookLimitReached`. Clients can use it to surface a
    "we couldn't find anything" affordance before the empty synthesis
    text arrives.

    ``queries_attempted`` is the number of sub-queries dequeued and sent
    to ``explore`` (including ones whose tool call errored). In the
    current single-query-per-iteration loop this equals ``iterations``
    (see ``notebook.py``: ``notebook_state.iteration`` is incremented
    once per dequeue, ratified Phase 9.7 W3-B). Both are surfaced
    independently so future multi-query iterations don't break the
    contract. Both are ``0`` when the planner produced an empty queue.
    """

    type: Literal["notebook_no_facts"] = "notebook_no_facts"
    queries_attempted: int = Field(
        ge=0,
        description=(
            "Number of sub-queries dequeued + sent to explore (includes "
            "queries whose tool call errored)."
        ),
    )
    iterations: int = Field(
        ge=0,
        description="Loop iterations completed before exiting the Notebook loop.",
    )
    note: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Optional short human-readable hint (e.g. 'empty plan').",
    )


WireEvent = Annotated[
    Union[
        MessageStart,
        MessageStop,
        TextDelta,
        ThinkingDelta,
        ToolCall,
        ToolResult,
        Error,
        LoopLimitReached,
        HwReset,
        # --- Phase 9: NotebookOrchestrator events ---
        NotebookPhaseStart,
        NotebookPhaseProgress,
        NotebookFactAdded,
        NotebookClarificationRequested,
        NotebookQueryAdded,
        NotebookLimitReached,
        NotebookNoFacts,
    ],
    Field(discriminator="type"),
]


__all__ = [
    "PROTOCOL_VERSION",
    "_Base",
    "ToolDescriptor",
    "MessageStart",
    "MessageStop",
    "TextDelta",
    "ThinkingDelta",
    "ToolCall",
    "ToolResult",
    "Error",
    "LoopLimitReached",
    "HwReset",
    "NotebookPhaseStart",
    "NotebookPhaseProgress",
    "NotebookFactAdded",
    "NotebookClarificationRequested",
    "NotebookQueryAdded",
    "NotebookLimitReached",
    "NotebookNoFacts",
    "WireEvent",
]
