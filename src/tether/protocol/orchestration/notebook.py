"""NotebookOrchestrator — research-mode strategy.

Implements ADR-0020's Hanov Plan→Explore→Extract→Refine→Synthesize
research loop with an ephemeral in-memory notebook.
"""
from __future__ import annotations

import asyncio
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
    NotebookPhaseStart,
    NotebookQueryAdded,
    TextDelta,
)

if TYPE_CHECKING:
    from tether.core.tool_registry import ToolRegistry
    from tether.protocol.orchestration.cancel import CancelToken
    from tether.protocol.orchestration.tool_runner import ToolRunner
    from tether.protocol.wire.events import WireEvent

logger = structlog.get_logger(__name__)
_WEB_SEARCH_COUNT = 5


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
                plan_queries = await self._plan(
                    model_name=planner_model,
                    question=prompt,
                    today_iso=today_iso,
                    cancel_token=cancel_token,
                )
                logger.info("notebook.phase_complete", phase="plan", queries=len(plan_queries))

                for query in plan_queries:
                    notebook_state.queue.append(query)
                    notebook_state.processed_queries.add(_normalize_query(query))
                    yield NotebookQueryAdded(
                        **_envelope(),
                        query=query,
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
                )

                tool_task = asyncio.create_task(
                    self.tool_runner.run(
                        "web_search",
                        {"query": query, "count": _WEB_SEARCH_COUNT},
                    )
                )
                while not tool_task.done():
                    if _is_cancelled():
                        cancelled = True
                        tool_task.cancel()
                        try:
                            await asyncio.wait_for(
                                tool_task,
                                timeout=_TOOL_CANCEL_GRACE_SEC,
                            )
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass
                        break
                    await asyncio.wait({tool_task}, timeout=0.01)
                if cancelled:
                    break

                try:
                    search_result = await tool_task
                except Exception as exc:
                    logger.warning(
                        "notebook.explore_tool_error",
                        iteration=iteration,
                        query=query,
                        error_type=type(exc).__name__,
                        exc_info=True,
                    )
                    continue

                if not isinstance(search_result, dict) or search_result.get("error"):
                    logger.warning(
                        "notebook.explore_tool_error",
                        iteration=iteration,
                        query=query,
                        error_type="tool_error",
                    )
                    continue
                logger.info(
                    "notebook.phase_complete",
                    phase="explore",
                    iteration=iteration,
                    results=len(search_result.get("results", [])),
                )

                if _is_cancelled():
                    cancelled = True
                    break

                yield NotebookPhaseStart(
                    **_envelope(),
                    phase="extract",
                    iteration=iteration,
                )
                extract_result = await self._extract(
                    model_name=extractor_model,
                    question=prompt,
                    source_query=query,
                    snippets=search_result.get("results", []),
                    facts=notebook_state.facts,
                    today_iso=today_iso,
                    cancel_token=cancel_token,
                )
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
                            fact_text=fact.text,
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
                            query=follow_up,
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

            yield NotebookPhaseStart(
                **_envelope(),
                phase="synthesize",
                iteration=0,
            )
            yield await _emit_message_start()
            async for chunk in self._synthesize_stream(
                model_name=synthesizer_model,
                question=prompt,
                facts=notebook_state.facts,
                today_iso=today_iso,
            ):
                if _is_cancelled():
                    cancelled = True
                    break
                yield TextDelta(**_envelope(), text=chunk)

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
    ) -> list[str]:
        """Run the Planner as raw text, then parse seed queries."""
        with structlog.contextvars.bound_contextvars(phase="plan"):
            logger.info("notebook.phase_start", phase="plan")
            raw = await self._collect_stream_text(
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
            )
            queries = parse_plan_output(raw, max_queries=5)
            logger.info(
                "notebook.phase_complete",
                phase="plan",
                queries=len(queries),
                raw_length=len(raw),
                parser_layer=None,
            )
            return queries

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
    ) -> ExtractResult:
        """Run the Extractor as raw text, then parse facts and follow-ups."""
        with structlog.contextvars.bound_contextvars(phase="extract"):
            logger.info("notebook.phase_start", phase="extract")
            raw = await self._collect_stream_text(
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
            )
            return parse_extract_output(
                raw,
                source_query,
                max_facts=self.research_settings.max_facts_per_extract,
            )

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
    ) -> str:
        chunks: list[str] = []
        async for chunk in self.provider.stream(
            model_name=model_name,
            messages=messages,
            tools=None,
        ):
            # Research-mode LLM phases are raw text only. Do not route them
            # through SlidingParser; untrusted snippets may contain tool markers.
            if cancel_token is not None and cancel_token.cancelled():
                break
            if isinstance(chunk, str):
                chunks.append(chunk)
            else:
                logger.warning("notebook.non_text_provider_chunk")
        return "".join(chunks)


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
