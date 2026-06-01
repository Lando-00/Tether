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
from typing import Any, AsyncIterator, ClassVar, Optional, TYPE_CHECKING

import structlog

from tether.config.settings import ResearchSettings
from tether.core.interfaces import (
    ModelProvider,
    Orchestrator,
    SessionStore,
    StreamParser,
)
from tether.core.types import OrchestratorConfig as ChatSettings
from tether.protocol.orchestration.chatty import _TOOL_CANCEL_GRACE_SEC
from tether.protocol.orchestration.notebook_parser import (
    ExtractResult,
    parse_extract_output,
    parse_plan_output,
)
from tether.protocol.orchestration.notebook_prompts import (
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
    MessageStart,
    MessageStop,
    NotebookFactAdded,
    NotebookLimitReached,
    NotebookNoFacts,
    NotebookPhaseProgress,
    NotebookPhaseStart,
    NotebookQueryAdded,
    TextDelta,
    ThinkingDelta,
)

if TYPE_CHECKING:
    from tether.core.tool_registry import ToolRegistry
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
    def __init__(self) -> None:
        self._mode: str = "leading"
        self._buf: str = ""
        self._unclosed_think_count: int = 0
        self._think_depth: int = 0

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
                self._think_depth += 1
                continue
            if idx_close != -1:
                think_out.append(self._buf[:idx_close])
                self._buf = self._buf[idx_close + len(self.THINK_CLOSE):]
                self._think_depth = max(0, self._think_depth - 1)
                if self._think_depth == 0:
                    self._mode = "text"
                continue
            if len(self._buf) > self._OVERLAP:
                emit = self._buf[: -self._OVERLAP]
                if emit:
                    think_out.append(emit)
                self._buf = self._buf[-self._OVERLAP:]
            break

        return "".join(text_out), "".join(think_out)

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

    is_implemented: ClassVar[bool] = True

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

    async def run(
        self,
        *,
        session_id: str,
        prompt: str,
        model_name: str,
        cancel_token: Optional["CancelToken"] = None,
    ) -> AsyncIterator["WireEvent"]:
        """Run the Hanov Plan→Explore→Extract→Refine→Synthesize loop."""
        turn_id = uuid.uuid4().hex[:12]
        seq = 0
        message_started = False
        cancelled = False

        def _next_seq() -> int:
            nonlocal seq
            n = seq
            seq += 1
            structlog.contextvars.bind_contextvars(seq=n)
            return n

        def _envelope() -> dict[str, Any]:
            return {
                "session_id": session_id,
                "turn_id": turn_id,
                "seq": _next_seq(),
                "ts": datetime.now(timezone.utc),
            }

        def _is_cancelled() -> bool:
            return cancel_token is not None and cancel_token.cancelled()

        async def _emit_message_start() -> MessageStart:
            nonlocal message_started
            message_started = True
            return MessageStart(**_envelope(), available_tools=[])

        try:
            structlog.contextvars.bind_contextvars(turn_id=turn_id)
            today_iso = self.clock().isoformat()
            planner_model = self.research_settings.planner_model or model_name
            extractor_model = self.research_settings.extractor_model or model_name
            synthesizer_model = self.research_settings.synthesizer_model or model_name

            notebook_state = NotebookState(
                max_facts=self.research_settings.max_facts,
                max_iterations=self.research_settings.max_iterations,
                max_facts_per_extract=self.research_settings.max_facts_per_extract,
            )

            if _is_cancelled():
                cancelled = True
            else:
                yield NotebookPhaseStart(
                    **_envelope(),
                    phase="plan",
                    iteration=0,
                )
                plan_queries: list[str] = []
                async for item in self._plan(
                    model_name=planner_model,
                    question=prompt,
                    today_iso=today_iso,
                    cancel_token=cancel_token,
                ):
                    kind, payload = item
                    if kind == "heartbeat":
                        yield NotebookPhaseProgress(
                            **_envelope(),
                            phase="plan",
                            iteration=0,
                            elapsed_ms=payload,
                        )
                    else:
                        plan_queries = payload
                logger.info("notebook.phase_complete", phase="plan", queries=len(plan_queries))

                for query in plan_queries:
                    notebook_state.queue.append(query)
                    notebook_state.processed_queries.add(_normalize_query(query))
                    yield NotebookQueryAdded(
                        **_envelope(),
                        query=query[:_MAX_QUERY_LENGTH],
                        queue_depth=len(notebook_state.queue),
                    )

            while not cancelled and notebook_state.should_continue():
                if _is_cancelled():
                    cancelled = True
                    break

                notebook_state.iteration += 1
                iteration = notebook_state.iteration
                query = notebook_state.queue.popleft()

                yield NotebookPhaseStart(
                    **_envelope(),
                    phase="explore",
                    iteration=iteration,
                )
                logger.info(
                    "notebook.phase_start",
                    phase="explore",
                    iteration=iteration,
                    **_query_log_fields(query),
                )

                tool_task = asyncio.create_task(
                    self.tool_runner.run(
                        "web_search",
                        {"query": query, "count": _WEB_SEARCH_COUNT},
                    )
                )
                try:
                    while not tool_task.done():
                        if _is_cancelled():
                            cancelled = True
                            break
                        await asyncio.wait({tool_task}, timeout=0.01)
                    if cancelled:
                        # Cooperative cancel: the ``finally`` block below
                        # cancels + grace-waits the tool task. We break out
                        # of the outer ``while not cancelled`` loop after
                        # the finally runs.
                        break

                    try:
                        search_result = await tool_task
                    except Exception as exc:
                        logger.warning(
                            "notebook.explore_tool_error",
                            iteration=iteration,
                            error_type=type(exc).__name__,
                            exc_info=True,
                            **_query_log_fields(query),
                        )
                        continue

                    if not isinstance(search_result, dict) or search_result.get("error"):
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
                        results=len(search_result.get("results", [])),
                    )
                finally:
                    # Phase 9.5 fu-research-external-cancel-pattern
                    # (mirrors chatty.py:1292-1322 F3 pattern).
                    #
                    # External ``asyncio.CancelledError`` propagating from
                    # ``asyncio.wait({tool_task}, timeout=0.01)`` would
                    # otherwise unwind without cancelling ``tool_task``,
                    # leaking the in-flight web_search call. This finally
                    # runs on:
                    #   1. Normal completion (tool_task done → no-op).
                    #   2. Cooperative cancel (_is_cancelled() True →
                    #      cancel + grace-wait here, then break above).
                    #   3. External CancelledError (tool_task still
                    #      pending → cancel + grace-wait, then re-raise).
                    #   4. ``continue`` from the error paths above
                    #      (tool_task already done → no-op).
                    if not tool_task.done():
                        tool_task.cancel()
                        try:
                            await asyncio.wait_for(
                                tool_task,
                                timeout=_TOOL_CANCEL_GRACE_SEC,
                            )
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            # Tool either over-ran the grace or honored
                            # the cancel. Either way, we're done with it.
                            # Don't add ``Exception`` to the tuple —
                            # let real bugs surface to the logger.
                            pass

                if _is_cancelled():
                    cancelled = True
                    break

                yield NotebookPhaseStart(
                    **_envelope(),
                    phase="extract",
                    iteration=iteration,
                )
                extract_result: Optional[ExtractResult] = None
                async for item in self._extract(
                    model_name=extractor_model,
                    question=prompt,
                    source_query=query,
                    snippets=search_result.get("results", []),
                    facts=notebook_state.facts,
                    today_iso=today_iso,
                    cancel_token=cancel_token,
                ):
                    kind, payload = item
                    if kind == "heartbeat":
                        yield NotebookPhaseProgress(
                            **_envelope(),
                            phase="extract",
                            iteration=iteration,
                            elapsed_ms=payload,
                        )
                    else:
                        extract_result = payload
                # ``_extract`` always yields a final ("result", ExtractResult).
                # The assert pins that invariant for type-checkers and surfaces
                # any future regression where the helper exits before yielding
                # the result sentinel (which would otherwise NoneType-crash
                # later when we iterate over .facts).
                assert extract_result is not None
                logger.info(
                    "notebook.phase_complete",
                    phase="extract",
                    iteration=iteration,
                    facts=len(extract_result.facts),
                    follow_ups=len(extract_result.follow_up_queries),
                    parser_layer=extract_result.parser_layer,
                    raw_length=extract_result.raw_length,
                )

                if _is_cancelled():
                    cancelled = True
                    break

                for fact in extract_result.facts:
                    # Order: mutate first, then yield. Pydantic ge=1 on total_facts
                    # catches emit-before-append off-by-one bugs.
                    if notebook_state.try_add_fact(fact):
                        yield NotebookFactAdded(
                            **_envelope(),
                            fact_text=fact.text[:_MAX_FACT_LENGTH],
                            source_query=fact.source_query,
                            total_facts=len(notebook_state.facts),
                        )
                    if notebook_state.limit_kind() == "max_facts":
                        break

                if notebook_state.limit_kind() is not None:
                    break

                deduped_follow_ups: list[str] = []
                for follow_up in extract_result.follow_up_queries:
                    normalized = _normalize_query(follow_up)
                    if normalized in notebook_state.processed_queries:
                        continue
                    deduped_follow_ups.append(follow_up)
                    notebook_state.processed_queries.add(normalized)

                if deduped_follow_ups:
                    logger.info(
                        "notebook.phase_start",
                        phase="refine",
                        iteration=iteration,
                    )
                    yield NotebookPhaseStart(
                        **_envelope(),
                        phase="refine",
                        iteration=iteration,
                    )
                    for follow_up in deduped_follow_ups:
                        notebook_state.queue.append(follow_up)
                        yield NotebookQueryAdded(
                            **_envelope(),
                            query=follow_up[:_MAX_QUERY_LENGTH],
                            queue_depth=len(notebook_state.queue),
                        )
                    logger.info(
                        "notebook.phase_complete",
                        phase="refine",
                        iteration=iteration,
                        queries=len(deduped_follow_ups),
                    )

            if not cancelled and not notebook_state.should_continue():
                limit_kind = notebook_state.limit_kind()
                if limit_kind is not None and notebook_state.queue:
                    count = (
                        len(notebook_state.facts)
                        if limit_kind == "max_facts"
                        else notebook_state.iteration
                    )
                    yield NotebookLimitReached(
                        **_envelope(),
                        limit_kind=limit_kind,
                        count=count,
                    )

            if cancelled:
                if not message_started:
                    yield await _emit_message_start()
                yield MessageStop(**_envelope(), stop_reason="cancelled")
                return

            if not notebook_state.facts:
                # Phase 9.7 W3-B (nho-fu-w3b-empty-signal): surface an
                # empty-Notebook signal BEFORE synthesize so clients can
                # distinguish "we ran the loop but found nothing" from
                # "we found something and are synthesizing". This is
                # NOT an Error and NOT a NotebookLimitReached — synthesis
                # still runs on the empty Notebook and MessageStop is
                # still ``complete``.
                #
                # ``queries_attempted`` and ``iterations`` are both sourced
                # from ``notebook_state.iteration``: the counter is
                # incremented once per dequeue+explore (notebook.py:413),
                # so in the current single-query-per-iteration loop they
                # coincide. Both are surfaced independently so the wire
                # contract survives future multi-query iterations.
                queries_attempted = notebook_state.iteration
                iterations = notebook_state.iteration
                note = "empty plan" if queries_attempted == 0 else None
                yield NotebookNoFacts(
                    **_envelope(),
                    queries_attempted=queries_attempted,
                    iterations=iterations,
                    note=note,
                )

            yield NotebookPhaseStart(
                **_envelope(),
                phase="synthesize",
                iteration=0,
            )
            yield await _emit_message_start()
            astream = self._synthesize_stream(
                model_name=synthesizer_model,
                question=prompt,
                facts=notebook_state.facts,
                today_iso=today_iso,
            )
            # Phase 9.6 I-1 (HIGH-A): strip <think>...</think> blocks
            # from the synth stream. NotebookOrchestrator bypasses
            # SlidingParser (ADR-0020 §D1 prompt-injection defense), so
            # without this filter Qwen3's thinking tokens bleed verbatim
            # into TextDelta. _ThinkStripper is a think-only state
            # machine — it does NOT detect <<function_call>> markers.
            #
            # The post-yield ``_is_cancelled()`` checks are load-bearing:
            # the stripper holds up to OVERLAP chars before emitting, so
            # the chunk that triggered TextDelta is one ahead of the
            # chunk the consumer's cancel reacted to. Without rechecking
            # after each yield, the next ``__anext__()`` can park on a
            # blocking await (provider sleep) and we'd miss the cancel
            # until the bounded ``aclose()`` grace fires.
            #
            # Phase 9.6 I-4 (W2-B): drive ``astream.__anext__()`` from a
            # single-consumer ``asyncio.wait({pending}, timeout=interval)``
            # so we can emit ``NotebookPhaseProgress`` heartbeats during
            # cold-load idle without losing ``seq`` monotonicity (all
            # yields still flow through ``_envelope()``).
            stripper = _ThinkStripper()
            synth_loop = asyncio.get_running_loop()
            synth_started = synth_loop.time()
            pending_chunk: Optional[asyncio.Task[Any]] = None
            try:
                pending_chunk = asyncio.create_task(astream.__anext__())
                while True:
                    if _is_cancelled():
                        cancelled = True
                        break
                    done, _still = await asyncio.wait(
                        {pending_chunk}, timeout=_HEARTBEAT_INTERVAL_SEC
                    )
                    if pending_chunk in done:
                        try:
                            chunk = pending_chunk.result()
                        except StopAsyncIteration:
                            pending_chunk = None
                            break
                        text_part, thinking_part = stripper.feed(chunk)
                        if text_part:
                            yield TextDelta(**_envelope(), text=text_part)
                            if _is_cancelled():
                                cancelled = True
                                break
                        if thinking_part and self.config.save_thinking:
                            yield ThinkingDelta(**_envelope(), text=thinking_part)
                            if _is_cancelled():
                                cancelled = True
                                break
                        pending_chunk = asyncio.create_task(astream.__anext__())
                    else:
                        elapsed_ms = int(
                            (synth_loop.time() - synth_started) * 1000
                        )
                        yield NotebookPhaseProgress(
                            **_envelope(),
                            phase="synthesize",
                            iteration=0,
                            elapsed_ms=elapsed_ms,
                        )
                # Normal exhaustion only. On cancel we drop the residual
                # rather than emit stale text after the cancel signal.
                if not cancelled and not _is_cancelled():
                    text_tail, thinking_tail = stripper.finalize()
                    if text_tail:
                        yield TextDelta(**_envelope(), text=text_tail)
                    if thinking_tail and self.config.save_thinking:
                        yield ThinkingDelta(**_envelope(), text=thinking_tail)
            finally:
                # Phase 9.6 I-4: cancel the in-flight __anext__ task first,
                # bounded by the cancel grace, so an external CancelledError
                # arriving during an idle wait can't leak the task. Mirrors
                # the explore-phase tool_task pattern at lines 446-458.
                if pending_chunk is not None and not pending_chunk.done():
                    pending_chunk.cancel()
                    try:
                        await asyncio.wait_for(
                            pending_chunk, timeout=_TOOL_CANCEL_GRACE_SEC
                        )
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                    except StopAsyncIteration:
                        pass
                # Phase 9.5 fu-research-synth-cancel-grace
                # (mirrors chatty.py:601-612).
                #
                # Bound the synth iterator's ``aclose()`` so an unresponsive
                # provider (or one that re-suspends inside its own ``finally``)
                # can't keep the request alive past the cancellation grace.
                # On a normally-exhausted generator ``aclose()`` is a no-op.
                try:
                    await asyncio.wait_for(
                        astream.aclose(),
                        timeout=_TOOL_CANCEL_GRACE_SEC,
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    # Either the provider over-ran the grace or an outer
                    # cancel arrived during cleanup. Don't shadow the
                    # in-flight exception (if any) — finally completes
                    # cleanly and the original exception keeps propagating.
                    pass
                except Exception:
                    # Don't shadow real bugs in provider cleanup paths.
                    # Log and continue so MessageStop still emits below.
                    logger.exception("notebook.synth_aclose_error")

            yield MessageStop(
                **_envelope(),
                stop_reason="cancelled" if cancelled else "complete",
            )
        except asyncio.CancelledError:
            cancelled = True
            try:
                if not message_started:
                    yield await _emit_message_start()
                yield MessageStop(**_envelope(), stop_reason="cancelled")
            except BaseException:
                pass
            raise
        finally:
            structlog.contextvars.unbind_contextvars("turn_id", "seq")

    async def _plan(
        self,
        *,
        model_name: str,
        question: str,
        today_iso: str,
        cancel_token: Optional["CancelToken"] = None,
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
    ) -> AsyncIterator[str]:
        """Stream synthesizer text directly; never route through SlidingParser."""
        with structlog.contextvars.bound_contextvars(phase="synthesize"):
            logger.info("notebook.phase_start", phase="synthesize")
            async for chunk in self.provider.stream(
                model_name=model_name,
                messages=[
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
                tools=None,
            ):
                if isinstance(chunk, str):
                    yield chunk
                else:
                    logger.warning("notebook.non_text_provider_chunk", phase="synthesize")

    async def _collect_stream_text(
        self,
        *,
        model_name: str,
        messages: list[dict[str, Any]],
        cancel_token: Optional["CancelToken"] = None,
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
        astream = self.provider.stream(
            model_name=model_name,
            messages=messages,
            tools=None,
        )
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
            # Bounded cleanup mirrors the synth-cancel pattern below and
            # chatty.py F3 (lines 1292-1322): cancel any in-flight
            # __anext__() task and wait for it within the grace, then
            # bound astream.aclose() the same way. Without these bounds
            # an external CancelledError could leak the in-flight
            # provider call and an unresponsive aclose() could keep the
            # request alive past the cancellation grace.
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
            try:
                await asyncio.wait_for(
                    astream.aclose(),
                    timeout=_TOOL_CANCEL_GRACE_SEC,
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception:
                logger.exception("notebook.collect_stream_aclose_error")

        yield ("text", "".join(chunks))


def _format_notebook_block(facts: list[AtomicFact]) -> str:
    if not facts:
        return "(none)"
    return "\n".join(
        f"{index}. {fact.text} [{fact.confidence}]"
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


__all__ = ["NotebookOrchestrator"]
