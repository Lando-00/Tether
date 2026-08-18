"""NotebookOrchestrator — research-mode strategy.

Implements ADR-0020's Hanov Plan→Explore→Extract→Refine→Synthesize
research loop with an ephemeral in-memory notebook.
"""
from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import Callable
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Iterable, Literal, Optional

import structlog

from tether.config.settings import ResearchSettings
from tether.core.interfaces import (
    ModelProvider,
    Orchestrator,
    SessionStore,
    StreamParser,
)
from tether.core.types import OrchestratorConfig as ChatSettings
from tether.protocol.intent.turn_triage import TurnKind
from tether.protocol.orchestration.chatty import _AWAITER_PERSIST_BUDGET_SEC, _TOOL_CANCEL_GRACE_SEC
from tether.protocol.orchestration.notebook_input import has_entity_drift, prepare_research_input
from tether.protocol.orchestration.notebook_parser import (
    ExtractResult,
    parse_extract_output,
    parse_plan_output,
    sanitize_search_queries,
)
from tether.protocol.orchestration.notebook_prompts import (
    DIRECT_ANSWER_SYSTEM_PROMPT,
    EXTRACTOR_SYSTEM_PROMPT,
    EXTRACTOR_USER_TEMPLATE,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_TEMPLATE,
    SYNTHESIZER_SYSTEM_PROMPT,
    SYNTHESIZER_USER_TEMPLATE,
)
from tether.protocol.orchestration.notebook_state import (
    AtomicFact,
    NotebookState,
    _normalize_query,
)
from tether.protocol.wire.events import (
    Error,
    MessageStart,
    MessageStop,
    NotebookClarificationRequested,
    NotebookFactAdded,
    NotebookLimitReached,
    NotebookNoFacts,
    NotebookPhaseProgress,
    NotebookPhaseStart,
    NotebookQueryAdded,
    TextDelta,
    ThinkingDelta,
)
from tether.runtime.abandoned_tasks import get_notebook_abandoned_task_tracker

if TYPE_CHECKING:
    from tether.core.tool_registry import ToolRegistry
    from tether.protocol.intent.turn_triage import TurnTriage
    from tether.protocol.orchestration.cancel import CancelToken
    from tether.protocol.orchestration.tool_runner import ToolRunner
    from tether.protocol.wire.events import WireEvent

logger = structlog.get_logger(__name__)
_WEB_SEARCH_COUNT = 5
# Mirror Pydantic max_length constraints on NotebookQueryAdded.query
# and NotebookFactAdded.fact_text (see protocol/wire/events.py). We
# truncate at the yield sites so a real LLM emitting an overlong fact
# or query degrades gracefully (event carries truncated text) instead
# of raising ValidationError mid-stream — which would propagate past
# the outer except (only catches CancelledError) and leave the
# consumer hung without a MessageStop.
_MAX_QUERY_LENGTH = 512
_MAX_FACT_LENGTH = 4096
# Phase 9.6 I-4 (W2-B): heartbeat cadence for long research phases.
#
# Wraps the provider stream's `__anext__()` await for planner / extractor /
# synthesizer phases. If no chunk arrives within this interval, the orchestrator
# yields a `NotebookPhaseProgress` event so consumers (UI, CLI) see liveness
# during cold-load (the live 43 s silence that motivated this fix lived
# inside `_plan` -> provider.stream's first `__anext__`).
#
# 2 s is a deliberate compromise: short enough to surface real cold-loads
# and feel responsive to interactive consumers, long enough not to spam them
# during normal multi-second streaming. (Phase 9.7 W3-A lowered this from 5 s
# to 2 s after cooperative-cancel latency review.)
# Tests monkeypatch this module constant to a much smaller value
# (e.g. 0.01 s) to keep the suite fast.
_HEARTBEAT_INTERVAL_SEC = 2.0


# Cleanup tasks are retained by the bounded runtime tracker rather than an
# unbounded module-level strong-reference set.
def _abandon_cleanup_task(task: "asyncio.Task[Any]", *, kind: str) -> None:
    """Register a cleanup task without driving, cancelling, or awaiting it."""
    tracker = get_notebook_abandoned_task_tracker()
    tracker.track(task, kind=kind)

    def _on_done(t: "asyncio.Task[Any]") -> None:
        if t.cancelled():
            logger.info("notebook.abandoned_cleanup.cancelled", kind=kind)
            return
        exc = t.exception()
        if exc is not None and not isinstance(exc, (StopAsyncIteration, asyncio.CancelledError)):
            logger.warning("notebook.abandoned_cleanup.exception", kind=kind, error_type=type(exc).__name__)

    task.add_done_callback(_on_done)


async def _bounded_aclose(
    astream: "AsyncGenerator[Any, None]",
    *,
    pending_chunk: "Optional[asyncio.Task[Any]]",
    kind: str,
) -> None:
    """Best-effort: close ``astream`` within ``_TOOL_CANCEL_GRACE_SEC``.

    The caller MUST already have cancelled ``pending_chunk`` (the Task
    driving the current ``astream.__anext__()``) and waited up to
    ``_TOOL_CANCEL_GRACE_SEC`` for it.

    **Ordering guard.** If ``pending_chunk`` is still ``not done()`` after
    its grace, the asyncgen is still being advanced and calling
    ``astream.aclose()`` would raise
    ``RuntimeError("aclose(): asynchronous generator is already running")``.
    In that case we abandon ``pending_chunk`` and skip ``aclose()``
    entirely — Python's GC-time asyncgen finalizer closes it eventually.
    Calling aclose on a running asyncgen is strictly worse: it raises
    synchronously and propagates past the ``finally:`` block.

    **Aclose bound.** ``astream.aclose()`` runs inside a task wrapped in
    ``asyncio.shield`` so that ``wait_for`` does NOT cancel it on timeout —
    we want the cleanup to keep running in the background after
    abandonment so the provider's cleanup completes naturally. On
    timeout (or outer cancel), the in-flight ``aclose_task`` is
    abandoned via ``_abandon_cleanup_task`` and this helper returns
    promptly. MessageStop latency is the contract; resource cleanup
    is best-effort.
    """
    if pending_chunk is not None and not pending_chunk.done():
        _abandon_cleanup_task(pending_chunk, kind=f"{kind}.pending_anext")
        return

    aclose_task = asyncio.create_task(
        astream.aclose(), name=f"asyncgen-aclose:{kind}"
    )
    try:
        await asyncio.wait_for(
            asyncio.shield(aclose_task),
            timeout=_TOOL_CANCEL_GRACE_SEC,
        )
    except asyncio.TimeoutError:
        if not aclose_task.done():
            _abandon_cleanup_task(aclose_task, kind=f"{kind}.aclose")
    except asyncio.CancelledError:
        # Outer CancelledError arrived during cleanup. Abandon aclose
        # so this helper completes promptly, then re-raise so the
        # outer ``except CancelledError:`` block (orch.run, line ~771)
        # still runs and emits MessageStop(cancelled).
        if not aclose_task.done():
            _abandon_cleanup_task(aclose_task, kind=f"{kind}.aclose")
        raise
    except Exception:
        # Provider raised something other than CancelledError on the
        # aclose path. Don't shadow real bugs in provider cleanup; log
        # via the existing structured event and continue. (The task-
        # wrapped path normally surfaces this via the done_callback's
        # ``notebook.abandoned_cleanup.exception`` event, but a
        # synchronous raise from ``astream.aclose()`` -> coroutine
        # creation can still land here — keep the branch.)
        logger.exception("notebook.bounded_aclose_error", kind=kind)


def _query_log_fields(query: str) -> dict[str, Any]:
    """Return safe log fields for a sub-query — sha256[:8] + length, no raw text.

    INFO/WARNING-level structured logs in :mod:`notebook` use this helper instead
    of embedding the raw sub-query, because:

    * Sub-queries are user-controlled (or follow-ups derived from snippets that
      may carry user-controlled content) and can contain secrets a careless user
      pasted into chat (API keys, tokens, etc.).
    * The wire events (``NotebookQueryAdded``) still carry the verbatim query so
      the UI can render it; the size cap on that field is the defense there.
    * DEBUG logs intentionally keep the raw query for local-dev troubleshooting
      (DEBUG is not shipped to remote sinks in production).
    """
    return {
        "query_sha256": hashlib.sha256(
            query.encode("utf-8", errors="replace")
        ).hexdigest()[:8],
        "query_length": len(query),
    }


def _search_error_text(search_result: Any) -> str:
    """Short, redaction-safe reason a single explore call produced nothing."""
    if not isinstance(search_result, dict):
        return "malformed tool response"
    error = search_result.get("error")
    return str(error) if error else "no results"


def _no_facts_note(iterations: int, search_failures: list[str]) -> Optional[str]:
    """Explain an empty notebook so the client can say *why* it found nothing.

    Without this, a research turn whose search backend is unconfigured looks
    identical to one where the web genuinely had no answer: the explore phases
    flash past, ``notebook_no_facts`` carries ``note=None``, and synthesis
    reports "not enough evidence". Surfacing the backend's own error message is
    the difference between "the internet failed you" and "you have not set
    BRAVE_API_KEY".

    The message is capped to the 256-char wire limit of
    :attr:`NotebookNoFacts.note`, and the underlying text comes from the tool's
    structured ``error`` field (never raw snippets), so it is safe to surface.
    """
    if not iterations:
        return "empty plan"
    if not search_failures:
        return None
    if len(set(search_failures)) == 1:
        reason = search_failures[0]
    else:
        reason = f"{len(search_failures)} search failures, first: {search_failures[0]}"
    return f"every search failed — {reason}"[:256]


class _ThinkStripper:
    """Stateful streaming filter that splits a chunk stream into
    ``(text, thinking)`` parts by consuming ``<think>...</think>`` blocks.

    NotebookOrchestrator bypasses :class:`SlidingParser` (ADR-0020 §D1
    prompt-injection defense): tool markers MUST NOT be parsed in the
    research path because untrusted web-search snippets could otherwise
    smuggle ``<<function_call>>`` payloads through the synth turn. But
    the synth model still emits raw ``<think>...</think>`` tokens that
    leak into the user-visible text stream verbatim if we just pass
    chunks through. This class is the minimum bookkeeping needed to
    strip those blocks without re-introducing tool-marker parsing.

    Design (mirrors SlidingParser's think sub-state-machine, NO tool
    marker handling):

    * Three states: ``"leading"`` (start of stream, ambiguous), ``"text"``
      (passthrough), ``"think"`` (inside a ``<think>`` block).
    * Retained overlap = ``len(THINK_CLOSE) - 1`` so a marker split across
      chunk boundaries (``"<thi"`` + ``"nk>"`` or ``"</thi"`` + ``"nk>"``)
      is detected.
    * Leading state handles the *short* "bare-leading ``</think>``" case:
      Qwen sometimes starts mid-thinking because the chat template injects
      ``<think>`` out-of-band. The detection window is intentionally bounded
      to the marker-overlap tail so no-think streams still begin streaming
      promptly; long hidden preambles before a bare close are tracked as a
      follow-up because fixing them requires trading off first-token latency.
    * Nested think blocks are treated conservatively with a depth counter.
      Hidden text is not released until the matching outer close marker.
    * On unclosed ``<think>`` at end-of-stream, :meth:`finalize` returns
      the residual as thinking (never as text). The caller decides
      whether to surface it as a :class:`ThinkingDelta` based on
      ``save_thinking``.
    """

    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"
    _OVERLAP = max(len(THINK_OPEN), len(THINK_CLOSE)) - 1
    def __init__(self, assume_open: bool = False) -> None:
        self._assume_open = assume_open
        self._saw_assumed_close = False
        self._mode: str = "think" if assume_open else "leading"
        self._buf: str = ""
        self._unclosed_think_count: int = 0
        self._think_depth: int = 1 if assume_open else 0

    def feed(self, chunk: str) -> tuple[str, str]:
        """Consume one chunk; return ``(text_part, thinking_part)``.

        Either part may be ``""``. Multiple ``<think>`` blocks in a single
        chunk are all handled; split markers across chunks are buffered
        via the retained overlap.
        """
        if not chunk:
            return "", ""

        self._buf += chunk
        text_out: list[str] = []
        think_out: list[str] = []

        while True:
            if self._mode == "leading":
                # Ambiguous start: a leading bare ``</think>`` means the
                # model began inside a thinking block. Disambiguate by
                # which marker appears first.
                lowered = self._buf.lower()
                idx_open = lowered.find(self.THINK_OPEN)
                idx_close = lowered.find(self.THINK_CLOSE)
                first_pos = -1
                marker_kind: Optional[str] = None
                if idx_open != -1:
                    first_pos = idx_open
                    marker_kind = "open"
                if idx_close != -1 and (first_pos == -1 or idx_close < first_pos):
                    first_pos = idx_close
                    marker_kind = "close"

                if marker_kind == "open":
                    text_out.append(self._buf[:first_pos])
                    self._buf = self._buf[first_pos + len(self.THINK_OPEN):]
                    self._mode = "think"
                    self._think_depth = 1
                    continue
                if marker_kind == "close":
                    think_out.append(self._buf[:first_pos])
                    self._buf = self._buf[first_pos + len(self.THINK_CLOSE):]
                    self._mode = "text"
                    self._think_depth = 0
                    continue

                # No marker yet. Hold up to OVERLAP chars in case a marker
                # is split across this and the next chunk; flush the rest as
                # text and transition out of leading mode. This preserves
                # first-token streaming for normal no-think synth output.
                if len(self._buf) > self._OVERLAP:
                    emit = self._buf[: -self._OVERLAP]
                    if emit:
                        text_out.append(emit)
                    self._buf = self._buf[-self._OVERLAP:]
                    self._mode = "text"
                break

            if self._mode == "text":
                idx_open = self._buf.lower().find(self.THINK_OPEN)
                if idx_open != -1:
                    text_out.append(self._buf[:idx_open])
                    self._buf = self._buf[idx_open + len(self.THINK_OPEN):]
                    self._mode = "think"
                    self._think_depth = 1
                    continue
                if len(self._buf) > self._OVERLAP:
                    emit = self._buf[: -self._OVERLAP]
                    if emit:
                        text_out.append(emit)
                    self._buf = self._buf[-self._OVERLAP:]
                break

            # self._mode == "think"
            lowered = self._buf.lower()
            idx_open = lowered.find(self.THINK_OPEN)
            idx_close = lowered.find(self.THINK_CLOSE)
            if idx_open != -1 and (idx_close == -1 or idx_open < idx_close):
                think_out.append(self._buf[:idx_open])
                self._buf = self._buf[idx_open + len(self.THINK_OPEN):]
                # An opted-in model can visibly repeat the template open marker.
                # It confirms the assumed block; it does not nest it.
                if self._assume_open and not self._saw_assumed_close and self._think_depth == 1:
                    continue
                self._think_depth += 1
                continue
            if idx_close != -1:
                think_out.append(self._buf[:idx_close])
                self._buf = self._buf[idx_close + len(self.THINK_CLOSE):]
                self._think_depth = max(0, self._think_depth - 1)
                if self._think_depth == 0:
                    self._mode = "text"
                    if self._assume_open:
                        self._saw_assumed_close = True
                continue
            if len(self._buf) > self._OVERLAP:
                emit = self._buf[: -self._OVERLAP]
                if emit:
                    think_out.append(emit)
                self._buf = self._buf[-self._OVERLAP:]
            break

        return "".join(text_out), "".join(think_out)

    @property
    def unclosed_assumed_block(self) -> bool:
        return self._assume_open and not self._saw_assumed_close

    def finalize(self) -> tuple[str, str]:
        """Flush any residual buffered state at end-of-stream.

        Returns ``(text_part, thinking_part)``. An unclosed ``<think>``
        block is returned as thinking (never as text) so it can't leak
        into the user-visible answer; the caller drops it unless
        ``save_thinking`` is true. Increments an internal debug counter
        on unclosed-block flush.
        """
        if not self._buf:
            return "", ""

        residual = self._buf
        self._buf = ""

        if self._mode == "think":
            self._unclosed_think_count += 1
            self._think_depth = 0
            logger.debug(
                "notebook.synth.unclosed_think",
                residual_length=len(residual),
                unclosed_count=self._unclosed_think_count,
            )
            return "", residual
        # "leading" with no marker ever seen, or "text" with trailing
        # overlap — both flush as text.
        return residual, ""


class NotebookOrchestrator(Orchestrator):
    """Research-mode orchestration.

    Constructor is pinned by ADR-0020 §D5 so Engine.chat() can thread
    research settings via inspect.signature.
    """

    is_implemented: bool = True

    def __init__(
        self,
        *,
        # Inherited ABC kwargs (engine.py threads via inspect.signature):
        provider: "ModelProvider",
        store: "SessionStore",
        tool_registry: "ToolRegistry",
        tool_runner: "ToolRunner",
        parser: "StreamParser",
        config: "ChatSettings",
        # Notebook-specific (engine.py adds these when mode="research"):
        research_settings: Optional["ResearchSettings"] = None,
        clock: Callable[[], date] = lambda: date.today(),
        triage: Optional["TurnTriage"] = None,
        excluded_tools: Optional[Iterable[str]] = None,
    ) -> None:
        # The parser is accepted for ABC compatibility but intentionally
        # ignored by research mode (ADR-0020 §D1 prompt-injection defense).
        self.provider = provider
        self.store = store
        self.tool_registry = tool_registry
        self.tool_runner = tool_runner
        self.parser = parser
        self.config = config
        # Default to a stock ResearchSettings() when None — library callers
        # constructing Engine directly (without from_settings) get sensible
        # defaults instead of an AttributeError mid-stream. The from_settings
        # path always threads a real instance from Settings.orchestrator.research.
        if research_settings is None:
            from tether.config.settings import ResearchSettings as _RS
            research_settings = _RS()
        self.research_settings = research_settings
        self.clock = clock
        # Disabled tools are pruned from the model-facing history (see
        # Engine.set_tool_enabled). Research mode only calls web_search, but the
        # direct-answer path replays history and must honour the same rule.
        self.excluded_tools: set[str] = set(excluded_tools or ())
        # Default: research every turn, i.e. the pre-triage behaviour. An
        # explicit `mode="research"` request means the caller *wants* the
        # research loop, so it must not be second-guessed. The triaged variant
        # used for the default mode is :class:`AutoOrchestrator` below.
        if triage is None:
            from tether.protocol.intent.turn_triage import AlwaysResearchTriage
            triage = AlwaysResearchTriage()
        self.triage = triage

    async def _history(self, session_id: str) -> list[dict[str, Any]]:
        """Prior history, minus any disabled tools' calls/results.

        ``exclude_tools`` is only passed when something is actually excluded so
        session stores predating the parameter keep working on the common path.
        """
        if self.excluded_tools:
            return await self.store.get_history(
                session_id,
                include_thinking=False,
                exclude_tools=self.excluded_tools,
            )
        return await self.store.get_history(session_id, include_thinking=False)

    async def run(
        self, *, session_id: str, prompt: str, model_name: str,
        cancel_token: Optional["CancelToken"] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncIterator["WireEvent"]:
        """Run research with a normal persisted turn lifecycle."""
        turn_id = uuid.uuid4().hex[:12]
        seq = 0
        message_started = terminal_emitted = turn_started = False
        cancelled = failed = False
        answer_parts: list[str] = []
        thinking_parts: list[str] = []

        def _envelope() -> dict[str, Any]:
            nonlocal seq
            value = {"session_id": session_id, "turn_id": turn_id, "seq": seq, "ts": datetime.now(timezone.utc)}
            structlog.contextvars.bind_contextvars(seq=seq)
            seq += 1
            return value

        def _is_cancelled() -> bool:
            return cancel_token is not None and cancel_token.cancelled()

        async def _start() -> MessageStart:
            nonlocal message_started
            message_started = True
            return MessageStart(**_envelope(), available_tools=[])

        async def _clarify(
            reason: Literal[
                "ambiguous_correction", "ambiguous_entity", "unsearchable_input"
            ],
            message: str,
            candidates: tuple[str, ...] = (),
        ) -> AsyncIterator["WireEvent"]:
            nonlocal terminal_emitted
            yield await _start()
            yield NotebookClarificationRequested(
                **_envelope(),
                reason=reason,
                message=message[:512],
                # Truncate at the yield site: an overlong candidate would
                # otherwise raise ValidationError and abort the stream with
                # no MessageStop.
                candidates=[item[:256] for item in candidates[:5]],
            )
            answer_parts.append(message[:512])
            terminal_emitted = True
            yield MessageStop(**_envelope(), stop_reason="complete")

        try:
            structlog.contextvars.bind_contextvars(turn_id=turn_id)
            await self.store.start_turn(session_id, turn_id, model_name=model_name)
            turn_started = True
            # Snapshot precedes add_user so a correction cannot match itself.
            prior_history = await self._history(session_id)
            await self.store.add_user(session_id, prompt, turn_id=turn_id)
            prepared = prepare_research_input(prompt, prior_history)
            if prepared.clarification is not None:
                async for event in _clarify(
                    prepared.clarification.reason,
                    prepared.clarification.message,
                    prepared.clarification.candidates,
                ):
                    yield event
                return

            # ``question`` drives planning/search (arithmetic removed);
            # ``full_question`` is what Extract and Synthesize see so a
            # locally answered sub-question is never hidden from the answer.
            question = prepared.effective_question
            full_question = prepared.resolved_question or question
            today_iso = self.clock().isoformat()
            # Triage decides whether this turn needs external evidence at all.
            # Arithmetic already resolved locally is still a research-shaped
            # turn (it has facts to synthesize), so only consult triage when
            # there is a residual question left to research.
            direct_answer = bool(question) and not prepared.local_facts and (
                self.triage.classify(prompt, has_history=bool(prior_history))
                is TurnKind.DIRECT
            )
            logger.info(
                "notebook.triage",
                turn_kind="direct" if direct_answer else "research",
                has_history=bool(prior_history),
            )
            planner_model = self.research_settings.planner_model or model_name
            extractor_model = self.research_settings.extractor_model or model_name
            synthesizer_model = self.research_settings.synthesizer_model or model_name
            state = NotebookState(
                max_facts=self.research_settings.max_facts,
                max_iterations=self.research_settings.max_iterations,
                max_facts_per_extract=self.research_settings.max_facts_per_extract,
            )
            # Reasons the search backend rejected an explore call. Collected so
            # a turn that gathered nothing can say *why* instead of reporting a
            # bare "no facts" that reads like "the web had no answer".
            search_failures: list[str] = []

            # Local facts are first-class cited notebook entries, never Brave input.
            for fact in prepared.local_facts:
                if state.try_add_fact(fact):
                    yield NotebookFactAdded(
                        **_envelope(),
                        fact_text=fact.text[:_MAX_FACT_LENGTH],
                        source_query=fact.source_query,
                        source_kind=fact.source_kind,
                        total_facts=len(state.facts),
                    )

            if _is_cancelled():
                cancelled = True
            elif question and not direct_answer:
                yield NotebookPhaseStart(**_envelope(), phase="plan", iteration=0)
                plan_queries: list[str] = []
                async for kind, payload in self._plan(
                    model_name=planner_model,
                    question=question,
                    today_iso=today_iso,
                    cancel_token=cancel_token,
                    reasoning_effort=reasoning_effort,
                ):
                    if kind == "heartbeat":
                        yield NotebookPhaseProgress(**_envelope(), phase="plan", iteration=0, elapsed_ms=payload)
                    else:
                        plan_queries = payload
                if any(has_entity_drift(query, question) for query in plan_queries):
                    async for event in _clarify(
                        "ambiguous_entity",
                        "Please clarify the entity you want me to research.",
                    ):
                        yield event
                    return
                if not plan_queries:
                    plan_queries = sanitize_search_queries([question], max_queries=1)
                    if not plan_queries:
                        async for event in _clarify(
                            "unsearchable_input",
                            "Please provide a specific question I can safely research.",
                        ):
                            yield event
                        return
                for query in plan_queries:
                    state.queue.append(query)
                    state.processed_queries.add(_normalize_query(query))
                    yield NotebookQueryAdded(
                        **_envelope(),
                        query=query[:_MAX_QUERY_LENGTH],
                        queue_depth=len(state.queue),
                    )

            while not cancelled and state.should_continue():
                if _is_cancelled():
                    cancelled = True
                    break
                state.iteration += 1
                iteration = state.iteration
                query = state.queue.popleft()
                yield NotebookPhaseStart(**_envelope(), phase="explore", iteration=iteration)
                logger.info(
                    "notebook.phase_start",
                    phase="explore",
                    iteration=iteration,
                    **_query_log_fields(query),
                )
                task = asyncio.create_task(
                    self.tool_runner.run(
                        "web_search", {"query": query, "count": _WEB_SEARCH_COUNT}
                    )
                )
                try:
                    while not task.done() and not _is_cancelled():
                        await asyncio.wait({task}, timeout=0.01)
                    if _is_cancelled():
                        cancelled = True
                        break
                    search_result = await task
                except Exception as exc:
                    search_failures.append(type(exc).__name__)
                    logger.warning(
                        "notebook.explore_tool_error",
                        iteration=iteration,
                        error_type=type(exc).__name__,
                        **_query_log_fields(query),
                    )
                    continue
                finally:
                    # External CancelledError would otherwise leave the
                    # in-flight web_search running (mirrors chatty F3).
                    if not task.done():
                        task.cancel()
                        try:
                            await asyncio.wait_for(task, timeout=_TOOL_CANCEL_GRACE_SEC)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass
                if not isinstance(search_result, dict) or search_result.get("error"):
                    search_failures.append(
                        _search_error_text(search_result)
                    )
                    logger.warning(
                        "notebook.explore_tool_error",
                        iteration=iteration,
                        error_type="tool_error",
                        **_query_log_fields(query),
                    )
                    continue
                logger.info(
                    "notebook.phase_complete",
                    phase="explore",
                    iteration=iteration,
                    results=len(search_result.get("results", []) or []),
                )
                yield NotebookPhaseStart(**_envelope(), phase="extract", iteration=iteration)
                extracted: Optional[ExtractResult] = None
                async for kind, payload in self._extract(
                    model_name=extractor_model,
                    question=full_question,
                    source_query=query,
                    snippets=search_result.get("results", []),
                    facts=state.facts,
                    today_iso=today_iso,
                    cancel_token=cancel_token,
                    reasoning_effort=reasoning_effort,
                ):
                    if kind == "heartbeat":
                        yield NotebookPhaseProgress(
                            **_envelope(),
                            phase="extract",
                            iteration=iteration,
                            elapsed_ms=payload,
                        )
                    else:
                        extracted = payload
                if _is_cancelled():
                    cancelled = True
                    break
                # ``_extract`` always yields a final ("result", ExtractResult);
                # the assert pins that invariant for future refactors.
                assert extracted is not None
                logger.info(
                    "notebook.phase_complete",
                    phase="extract",
                    iteration=iteration,
                    facts=len(extracted.facts),
                    follow_ups=len(extracted.follow_up_queries),
                    parser_layer=extracted.parser_layer,
                    raw_length=extracted.raw_length,
                )
                for fact in extracted.facts:
                    if state.try_add_fact(fact):
                        yield NotebookFactAdded(
                            **_envelope(),
                            fact_text=fact.text[:_MAX_FACT_LENGTH],
                            source_query=fact.source_query,
                            source_kind=fact.source_kind,
                            total_facts=len(state.facts),
                        )
                    if state.limit_kind() == "max_facts":
                        break
                if state.limit_kind() is not None:
                    break
                followups = [
                    q
                    for q in extracted.follow_up_queries
                    if _normalize_query(q) not in state.processed_queries
                ]
                if followups:
                    logger.info(
                        "notebook.phase_start", phase="refine", iteration=iteration
                    )
                    yield NotebookPhaseStart(**_envelope(), phase="refine", iteration=iteration)
                    for followup in followups:
                        state.processed_queries.add(_normalize_query(followup))
                        state.queue.append(followup)
                        yield NotebookQueryAdded(
                            **_envelope(),
                            query=followup[:_MAX_QUERY_LENGTH],
                            queue_depth=len(state.queue),
                        )
                    logger.info(
                        "notebook.phase_complete",
                        phase="refine",
                        iteration=iteration,
                        queries=len(followups),
                    )

            if cancelled:
                if not message_started:
                    yield await _start()
                terminal_emitted = True
                yield MessageStop(**_envelope(), stop_reason="cancelled")
                return
            limit_kind = state.limit_kind()
            if limit_kind is not None and state.queue:
                count = len(state.facts) if limit_kind == "max_facts" else state.iteration
                yield NotebookLimitReached(
                    **_envelope(), limit_kind=limit_kind, count=count
                )
            if not state.facts and not direct_answer:
                yield NotebookNoFacts(
                    **_envelope(),
                    queries_attempted=state.iteration,
                    iterations=state.iteration,
                    note=_no_facts_note(state.iteration, search_failures),
                )
            if not direct_answer:
                yield NotebookPhaseStart(**_envelope(), phase="synthesize", iteration=0)
            yield await _start()
            stripper = _ThinkStripper(
                assume_open=synthesizer_model
                in self.research_settings.synth_assume_open_think_models
            )
            astream = (
                self._direct_answer_stream(
                    model_name=synthesizer_model,
                    prompt=prompt,
                    history=prior_history,
                    today_iso=today_iso,
                    reasoning_effort=reasoning_effort,
                )
                if direct_answer
                else self._synthesize_stream(
                    model_name=synthesizer_model,
                    question=full_question,
                    facts=state.facts,
                    today_iso=today_iso,
                    reasoning_effort=reasoning_effort,
                )
            )
            synth_loop = asyncio.get_running_loop()
            synth_started = synth_loop.time()
            pending: Optional[asyncio.Task[Any]] = None
            try:
                pending = asyncio.create_task(astream.__anext__())
                while True:
                    if _is_cancelled():
                        cancelled = True
                        break
                    done, _ = await asyncio.wait({pending}, timeout=_HEARTBEAT_INTERVAL_SEC)
                    if pending not in done:
                        yield NotebookPhaseProgress(
                            **_envelope(),
                            phase="synthesize",
                            iteration=0,
                            elapsed_ms=int((synth_loop.time() - synth_started) * 1000),
                        )
                        continue
                    try:
                        chunk = pending.result()
                    except StopAsyncIteration:
                        pending = None
                        break
                    text, thinking = stripper.feed(chunk)
                    # The post-yield cancel checks are load-bearing: the
                    # stripper holds up to OVERLAP chars, so the chunk that
                    # triggered this delta is one ahead of the chunk the
                    # consumer reacted to.
                    if text:
                        answer_parts.append(text)
                        yield TextDelta(**_envelope(), text=text)
                        if _is_cancelled():
                            cancelled = True
                            break
                    if thinking:
                        thinking_parts.append(thinking)
                        if self.config.save_thinking:
                            yield ThinkingDelta(**_envelope(), text=thinking)
                            if _is_cancelled():
                                cancelled = True
                                break
                    pending = asyncio.create_task(astream.__anext__())
                if not cancelled:
                    tail, thought = stripper.finalize()
                    if tail:
                        answer_parts.append(tail)
                        yield TextDelta(**_envelope(), text=tail)
                    if thought:
                        thinking_parts.append(thought)
                    if stripper.unclosed_assumed_block:
                        # Fail closed: hidden content is never reclassified as
                        # answer text, and the turn must not report success.
                        failed = True
                        logger.warning("notebook.synth.unclosed_assumed_think")
                        terminal_emitted = True
                        yield Error(
                            **_envelope(),
                            message="Synthesis ended before its thinking block closed.",
                            error_type="UnclosedThinkBlock",
                            is_fatal=False,
                        )
                        yield MessageStop(**_envelope(), stop_reason="error")
                        return
                    if not "".join(answer_parts).strip():
                        # Research ran but the synthesizer produced no visible
                        # answer. Reporting ``complete`` here would be a silent
                        # empty success.
                        failed = True
                        logger.warning("notebook.synth.empty_answer")
                        terminal_emitted = True
                        yield Error(
                            **_envelope(),
                            message="Research finished without producing an answer.",
                            error_type="EmptySynthesis",
                            is_fatal=False,
                        )
                        yield MessageStop(**_envelope(), stop_reason="error")
                        return
            finally:
                # Cancel the in-flight ``__anext__`` first (bounded by the
                # cancel grace), then hand cleanup to ``_bounded_aclose``,
                # which abandons uncooperative cleanup rather than pinning
                # the request past the grace.
                if pending is not None and not pending.done():
                    pending.cancel()
                    try:
                        await asyncio.wait_for(pending, timeout=_TOOL_CANCEL_GRACE_SEC)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                    except StopAsyncIteration:
                        pass
                await _bounded_aclose(astream, pending_chunk=pending, kind="synth")
            terminal_emitted = True
            yield MessageStop(**_envelope(), stop_reason="cancelled" if cancelled else "complete")
        except asyncio.CancelledError:
            cancelled = True
            if not terminal_emitted:
                try:
                    if not message_started:
                        yield await _start()
                    terminal_emitted = True
                    yield MessageStop(**_envelope(), stop_reason="cancelled")
                except BaseException:
                    pass
            raise
        except Exception as exc:
            failed = True
            logger.exception("notebook.run_error", error_type=type(exc).__name__)
            if not terminal_emitted:
                if not message_started:
                    yield await _start()
                terminal_emitted = True
                yield Error(
                    **_envelope(),
                    message="Research could not be completed.",
                    error_type=type(exc).__name__,
                    is_fatal=False,
                )
                yield MessageStop(**_envelope(), stop_reason="error")
        finally:
            if turn_started:
                status = "cancelled" if cancelled else "failed" if failed else "completed"
                try:
                    await asyncio.wait_for(
                        self.store.add_assistant_text(
                            session_id,
                            "".join(answer_parts),
                            thinking_text="".join(thinking_parts),
                            save_thinking=self.config.save_thinking,
                            turn_id=turn_id,
                        ),
                        timeout=_AWAITER_PERSIST_BUDGET_SEC,
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.warning("notebook.persist_assistant_timeout")
                except Exception:
                    logger.warning("notebook.persist_assistant_error", exc_info=True)
                try:
                    await asyncio.wait_for(
                        self.store.complete_turn(
                            turn_id,
                            status=status,
                            stop_reason=(
                                "cancelled"
                                if cancelled
                                else "error"
                                if failed
                                else "complete"
                            ),
                        ),
                        timeout=_AWAITER_PERSIST_BUDGET_SEC,
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    logger.warning("notebook.complete_turn_timeout")
                except Exception:
                    logger.warning("notebook.complete_turn_error", exc_info=True)
            structlog.contextvars.unbind_contextvars("turn_id", "seq")

    async def _plan(
        self,
        *,
        model_name: str,
        question: str,
        today_iso: str,
        cancel_token: Optional["CancelToken"] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Run the Planner as raw text, then parse seed queries.

        Async generator: yields ``("heartbeat", elapsed_ms)`` items while
        the planner provider stream is idle, then a final ``("result",
        queries)`` item (``queries: list[str]``). The orchestrator wraps
        heartbeats into :class:`NotebookPhaseProgress` events.
        """
        with structlog.contextvars.bound_contextvars(phase="plan"):
            logger.info("notebook.phase_start", phase="plan")
            raw = ""
            async for item in self._collect_stream_text(
                model_name=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": PLANNER_SYSTEM_PROMPT.format(today_iso=today_iso),
                    },
                    {
                        "role": "user",
                        "content": PLANNER_USER_TEMPLATE.format(question=question),
                    },
                ],
                cancel_token=cancel_token,
                reasoning_effort=reasoning_effort,
            ):
                kind, payload = item
                if kind == "heartbeat":
                    yield item
                else:
                    raw = payload
            queries = parse_plan_output(raw, max_queries=5)
            logger.info(
                "notebook.phase_complete",
                phase="plan",
                queries=len(queries),
                raw_length=len(raw),
                parser_layer=None,
            )
            yield ("result", queries)

    async def _extract(
        self,
        *,
        model_name: str,
        question: str,
        source_query: str,
        snippets: Any,
        facts: list[AtomicFact],
        today_iso: str,
        cancel_token: Optional["CancelToken"] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Run the Extractor as raw text, then parse facts and follow-ups.

        Async generator: yields ``("heartbeat", elapsed_ms)`` items while
        the extractor provider stream is idle, then a final ``("result",
        ExtractResult)`` item. The orchestrator wraps heartbeats into
        :class:`NotebookPhaseProgress` events tagged with the current
        iteration counter.
        """
        with structlog.contextvars.bound_contextvars(phase="extract"):
            logger.info("notebook.phase_start", phase="extract")
            raw = ""
            async for item in self._collect_stream_text(
                model_name=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": EXTRACTOR_SYSTEM_PROMPT.format(
                            today_iso=today_iso,
                            max_facts=self.research_settings.max_facts_per_extract,
                        ),
                    },
                    {
                        "role": "user",
                        "content": EXTRACTOR_USER_TEMPLATE.format(
                            question=question,
                            sub_query=source_query,
                            notebook_count=len(facts),
                            notebook_block=_format_notebook_block(facts),
                            n=len(snippets) if isinstance(snippets, list) else 0,
                            results_block=_format_results_block(snippets),
                        ),
                    },
                ],
                cancel_token=cancel_token,
                reasoning_effort=reasoning_effort,
            ):
                kind, payload = item
                if kind == "heartbeat":
                    yield item
                else:
                    raw = payload
            extract_result = parse_extract_output(
                raw,
                source_query,
                max_facts=self.research_settings.max_facts_per_extract,
            )
            yield ("result", extract_result)

    async def _synthesize_stream(
        self,
        *,
        model_name: str,
        question: str,
        facts: list[AtomicFact],
        today_iso: str,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream synthesizer text directly; never route through SlidingParser."""
        with structlog.contextvars.bound_contextvars(phase="synthesize"):
            logger.info("notebook.phase_start", phase="synthesize")
            stream_kwargs: dict[str, Any] = {
                "model_name": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": SYNTHESIZER_SYSTEM_PROMPT.format(today_iso=today_iso),
                    },
                    {
                        "role": "user",
                        "content": SYNTHESIZER_USER_TEMPLATE.format(
                            question=question,
                            notebook_block=_format_notebook_block(facts),
                        ),
                    },
                ],
                "tools": None,
            }
            if reasoning_effort is not None:
                stream_kwargs["reasoning_effort"] = reasoning_effort
            async for chunk in self.provider.stream(**stream_kwargs):
                if isinstance(chunk, str):
                    yield chunk
                else:
                    logger.warning("notebook.non_text_provider_chunk", phase="synthesize")

    async def _direct_answer_stream(
        self,
        *,
        model_name: str,
        prompt: str,
        history: list[dict[str, Any]],
        today_iso: str,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a plain conversational answer — no plan, no search, no tools.

        Used when :class:`~tether.protocol.intent.turn_triage.TurnTriage` routes
        a turn to ``DIRECT``. Unlike synthesis, this *does* see prior history, so
        back-references ("what did I just say?") resolve. Like synthesis, it
        never routes through :class:`SlidingParser`, so a model that emits a
        stray ``<<function_call>>`` marker yields harmless text rather than
        executing anything (ADR-0020 §D1).
        """
        with structlog.contextvars.bound_contextvars(phase="direct"):
            logger.info("notebook.phase_start", phase="direct")
            messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": DIRECT_ANSWER_SYSTEM_PROMPT.format(today_iso=today_iso),
                },
                *history,
                {"role": "user", "content": prompt},
            ]
            stream_kwargs: dict[str, Any] = {
                "model_name": model_name,
                "messages": messages,
                "tools": None,
            }
            if reasoning_effort is not None:
                stream_kwargs["reasoning_effort"] = reasoning_effort
            async for chunk in self.provider.stream(**stream_kwargs):
                if isinstance(chunk, str):
                    yield chunk
                else:
                    logger.warning("notebook.non_text_provider_chunk", phase="direct")

    async def _collect_stream_text(
        self,
        *,
        model_name: str,
        messages: list[dict[str, Any]],
        cancel_token: Optional["CancelToken"] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Stream provider text with heartbeat sentinels.

        Yields:

        ``("heartbeat", elapsed_ms)``
            Emitted each time no chunk arrives within
            ``_HEARTBEAT_INTERVAL_SEC``. ``elapsed_ms`` is the int
            milliseconds since the helper started. The caller wraps these
            into a :class:`NotebookPhaseProgress` event so ``seq`` stays
            monotonic (we never spawn a side-channel emitter task).

        ``("text", accumulated_text)``
            Final sentinel, emitted exactly once after the stream is
            exhausted (or cooperatively cancelled). The caller treats
            ``accumulated_text`` as the raw planner / extractor output and
            feeds it to the corresponding parser.

        Implementation: single-consumer ``asyncio.wait({pending}, timeout=...)``
        pattern. The pending task awaits one ``astream.__anext__()`` at a
        time; on timeout we yield a heartbeat and re-enter ``wait``. On
        cancellation or exception, the pending task is cancelled and
        grace-waited (``_TOOL_CANCEL_GRACE_SEC``), then ``astream.aclose()``
        is bounded by the same grace. ``StopAsyncIteration`` resolution of
        ``pending`` ends the loop normally ? note the bare ``except`` for
        ``StopAsyncIteration`` is required because it is *not* an
        ``Exception`` subclass on the result-side of a task.
        """
        chunks: list[str] = []
        stream_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "messages": messages,
            "tools": None,
        }
        if reasoning_effort is not None:
            stream_kwargs["reasoning_effort"] = reasoning_effort
        astream = self.provider.stream(**stream_kwargs)
        loop = asyncio.get_running_loop()
        started = loop.time()
        pending: Optional[asyncio.Task[Any]] = None
        try:
            pending = asyncio.create_task(astream.__anext__())
            while True:
                if cancel_token is not None and cancel_token.cancelled():
                    break
                done, _still_pending = await asyncio.wait(
                    {pending}, timeout=_HEARTBEAT_INTERVAL_SEC
                )
                if pending in done:
                    try:
                        chunk = pending.result()
                    except StopAsyncIteration:
                        pending = None
                        break
                    if isinstance(chunk, str):
                        chunks.append(chunk)
                    else:
                        logger.warning("notebook.non_text_provider_chunk")
                    pending = asyncio.create_task(astream.__anext__())
                else:
                    elapsed_ms = int((loop.time() - started) * 1000)
                    yield ("heartbeat", elapsed_ms)
        finally:
            # Phase 9.7 W4: bounded cleanup matches the synth-loop path —
            # cancel the in-flight ``__anext__`` task (within grace), then
            # call ``_bounded_aclose`` which handles the uncooperative-
            # cleanup + ordering invariants (see helper docstring and the
            # synth-loop finally block for the full rationale).
            if pending is not None and not pending.done():
                pending.cancel()
                try:
                    await asyncio.wait_for(
                        pending, timeout=_TOOL_CANCEL_GRACE_SEC
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                except StopAsyncIteration:
                    # Provider exhausted right as we cancelled ? benign.
                    pass
            await _bounded_aclose(
                astream, pending_chunk=pending, kind="collect_stream"
            )

        yield ("text", "".join(chunks))


def _format_notebook_block(facts: list[AtomicFact]) -> str:
    if not facts:
        return "(none)"
    return "\n".join(
        f"{index}. "
        f"{'[local calculation] ' if fact.source_kind == 'local_deterministic' else ''}"
        f"{fact.text} [{fact.confidence}]"
        for index, fact in enumerate(facts, start=1)
    )


def _format_results_block(snippets: Any) -> str:
    if not isinstance(snippets, list) or not snippets:
        return "(none)"

    blocks: list[str] = []
    for fallback_rank, item in enumerate(snippets, start=1):
        if not isinstance(item, dict):
            continue
        rank = item.get("rank") or fallback_rank
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        blocks.append(
            f"[{rank}] TITLE: {title}\n"
            f"    URL: {url}\n"
            f"    SNIPPET: {snippet}"
        )
    return "\n\n".join(blocks) if blocks else "(none)"


class AutoOrchestrator(NotebookOrchestrator):
    """Fact-based orchestration with per-turn triage — the default mode.

    Identical to :class:`NotebookOrchestrator` except that it asks a
    :class:`~tether.protocol.intent.turn_triage.TurnTriage` whether each turn
    actually needs external evidence:

    * ``DIRECT``   → hand the turn to :class:`ChattyAgentOrchestrator`, which
      answers conversationally *and* can call the registered tools.
    * ``RESEARCH`` → the full Plan → Explore → Extract → Refine → Synthesize loop.

    This is what makes the fact-based loop safe as the *default* orchestrator.
    Without triage, "hello" would be decomposed into search queries and sent to
    a web search backend. With it, small talk, creative work and back-references
    stay cheap, and only evidence-seeking questions pay for research.

    **Why DIRECT delegates instead of answering inline.** The research loop only
    knows one tool, ``web_search``. A bare "answer from the model" path therefore
    has *no* tools at all, which breaks every question a local tool exists to
    serve: "what time is it in Europe/Dublin?" would be sent to a web search
    (and fail without a search backend) instead of calling the ``time`` tool.
    Delegating DIRECT turns to the chat orchestrator gives them the full tool
    loop, so the split is "needs the open web" vs "everything else" rather than
    "research" vs "no tools".

    ``mode="research"`` keeps using :class:`NotebookOrchestrator` directly, so an
    explicit research request is never downgraded.
    """

    def __init__(
        self,
        *,
        provider: "ModelProvider",
        store: "SessionStore",
        tool_registry: "ToolRegistry",
        tool_runner: "ToolRunner",
        parser: "StreamParser",
        config: "ChatSettings",
        research_settings: Optional["ResearchSettings"] = None,
        clock: Callable[[], date] = lambda: date.today(),
        triage: Optional["TurnTriage"] = None,
        excluded_tools: Optional[Iterable[str]] = None,
        # Threaded by Engine for the delegated chat path.
        tools: Optional[dict[str, Any]] = None,
        system_prompt: str = "",
        hw_watchdog: Optional[Any] = None,
        provider_id: Optional[str] = None,
        audit_store_args: bool = False,
        confirm_intent_classifier: Optional[Any] = None,
    ) -> None:
        # The signature is spelled out rather than forwarded via **kwargs
        # because Engine threads constructor arguments by inspecting the
        # signature (ADR-0020 §D5); a **kwargs-only constructor advertises no
        # parameters and would be called with nothing.
        if triage is None:
            from tether.protocol.intent.rules_turn_triage import RulesTurnTriage
            triage = RulesTurnTriage()
        super().__init__(
            provider=provider,
            store=store,
            tool_registry=tool_registry,
            tool_runner=tool_runner,
            parser=parser,
            config=config,
            research_settings=research_settings,
            clock=clock,
            triage=triage,
            excluded_tools=excluded_tools,
        )
        from tether.protocol.orchestration.chatty import ChattyAgentOrchestrator

        self._chat = ChattyAgentOrchestrator(
            provider=provider,
            parser=parser,
            store=store,
            tools=tools if tools is not None else {},
            system_prompt=system_prompt,
            config=config,
            tool_runner=tool_runner,
            hw_watchdog=hw_watchdog,
            provider_id=provider_id,
            confirm_intent_classifier=confirm_intent_classifier,
            audit_store_args=audit_store_args,
            excluded_tools=excluded_tools,
        )

    async def run(
        self, *, session_id: str, prompt: str, model_name: str,
        cancel_token: Optional["CancelToken"] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncIterator["WireEvent"]:
        """Triage the turn, then run it through the right orchestrator."""
        prior_history = await self._history(session_id)
        kind = self.triage.classify(prompt, has_history=bool(prior_history))
        logger.info("auto.triage", turn_kind=kind.value)
        if kind is TurnKind.DIRECT:
            async for event in self._chat.run(
                session_id=session_id,
                prompt=prompt,
                model_name=model_name,
                cancel_token=cancel_token,
                reasoning_effort=reasoning_effort,
            ):
                yield event
            return
        async for event in super().run(
            session_id=session_id,
            prompt=prompt,
            model_name=model_name,
            cancel_token=cancel_token,
            reasoning_effort=reasoning_effort,
        ):
            yield event


__all__ = ["NotebookOrchestrator", "AutoOrchestrator"]
