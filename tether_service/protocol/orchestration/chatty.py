"""ChattyAgentOrchestrator — the standard tool-loop agent.

Implements the Orchestrator ABC (tether_service.core.interfaces).
The "chatty agent" mode is the default Tether experience: a user
prompt enters; the model reasons, optionally calls tools, and produces
a final answer; events stream to the caller.

Briefing §2 Seam B: this is the concrete impl that historically lived
as a free function (orchestrate()) and then as the unnamed Orchestrator
class. Renamed in p5-orchestrator-abc-strategy to make the
orchestration strategy a first-class architectural seam.

Other impls under the same ABC:
  - NotebookOrchestrator (notebook.py) — research mode, stubbed.

Stateful per-turn but thread-safe across turns: the class instance can
be reused (``Engine.chat`` constructs one per :class:`Engine`, calls
:meth:`ChattyAgentOrchestrator.run` per turn). State carried during a
turn lives in the ``run()`` async generator's closure, not on ``self``.

Public API:

  ``ChattyAgentOrchestrator(*, provider, parser, store, tools,
                             system_prompt, config, tool_runner,
                             hw_watchdog=None)``

  ``async def run(*, session_id, prompt, model_name, cancel_token=None)
       -> AsyncIterator[WireEvent]``

Yielded events are typed :data:`WireEvent` instances. HTTP transports
serialize each via :func:`v0_compat_serialize` (legacy v0 vocabulary)
or :class:`NdjsonEmitter` (v2 vocabulary; ``p5-cutover-a-dual-emit``
chooses based on ``Accept`` header).

Cancellation contract (synthesis §3.5):

  1. Provider stream wrapped in ``aclosing()`` — ``async for`` exit
     triggers the generator's ``finally`` block immediately.
  2. In-flight tool task cancelled with **250 ms grace**
     (``asyncio.wait_for(task, 0.25)``).
  3. Partial assistant text persisted with **200 ms write timeout**
     (``asyncio.wait_for(store.add_assistant_text(...), 0.20)``).
  4. Parser ``finalize()`` called.
  5. ONE :class:`MessageStop` with ``stop_reason='cancelled'`` emitted
     (in v2 vocabulary, ``MessageStop`` IS the "done" event).

Tool-error policy (default ``FEED_BACK_TO_MODEL``): tool errors no
longer break the loop — a :class:`ToolResult` row with
``status='error'`` is persisted and the loop continues so the model
can recover. ``BREAK_LOOP`` (legacy behaviour) keeps the single-turn
shape for deterministic tests.

Loop-limit policy (default ``EMIT_LIMIT_EVENT``): exhausting
``max_tool_loops`` yields :class:`LoopLimitReached` plus
:class:`MessageStop(stop_reason='tool_loop_exhausted')` and exits
cleanly. ``RAISE`` raises :class:`tether_service.core.errors.LoopLimitReached`.

Synthesis §3.4 (Engine.chat returns AsyncIterator[WireEvent]),
§3.5 (cancellation + policy contracts), §11.3 R1 / R3 / R6 / R7 / R10.
Synthesis §3.5; briefing §2 item 4.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from contextlib import aclosing

import structlog
from datetime import datetime, timezone
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
)

from tether_service.core.errors import (
    LoopLimitReached as LoopLimitReachedError,
    TransientProviderError,
)
from tether_service.core.interfaces import (
    ModelProvider,
    Orchestrator as OrchestratorABC,
    SessionStore,
    StreamParser,
    Tool,
)
from tether_service.core.logging import logger
from tether_service.core.types import OrchestratorConfig, ToolExecutionContext
from tether_service.protocol.orchestration.cancel import CancelToken
from tether_service.protocol.orchestration.policies import (
    LoopLimitPolicy,
    ToolErrorPolicy,
)
from tether_service.protocol.orchestration.tool_runner import ToolRunner
from tether_service.protocol.parsers.events import (
    PParseError,
    PStreamEnd,
    PText,
    PThink,
    PToolCallDetected,
    PToolCallParsed,
)
from tether_service.protocol.wire.events import (
    Error,
    HwReset,
    LoopLimitReached as LoopLimitReachedWire,
    MessageStart,
    MessageStop,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolDescriptor,
    ToolResult,
    WireEvent,
)

if TYPE_CHECKING:
    from tether_service.runtime.hw_watchdog import HardwareWatchdog


# Synthesis §3.5: cancellation contract bounds.
_TOOL_CANCEL_GRACE_SEC = 0.25  # in-flight tool task gets 250 ms after cancel
_PARTIAL_PERSIST_TIMEOUT_SEC = 0.20  # partial-text write budget on cancel


def _redact(payload: Any, max_len: int = 120) -> str:
    """Truncate ``repr(payload)`` to ``max_len`` chars for safe DEBUG logging.

    §13 R5: prompts / args / results may contain PII; never log them
    raw. Phase 7 will swap this for structlog field-level redaction.
    """
    s = repr(payload)
    return s if len(s) <= max_len else s[:max_len] + "...[+%dB]" % (len(s) - max_len)


def _args_sha256(args: Dict[str, Any]) -> str:
    """Stable SHA-256 of tool args for audit log (synthesis §11.3 R3).

    JSON-encode with sorted keys + ``default=str`` so non-JSON-native
    objects (datetime, etc.) hash deterministically.
    """
    encoded = json.dumps(args, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ChattyAgentOrchestrator(OrchestratorABC):
    """Thin async orchestrator with named seams.

    Implements :class:`tether_service.core.interfaces.Orchestrator`.
    Synthesis §3.5. Construct once per :class:`Engine`; call
    :meth:`run` per turn. Stateful per-turn but thread-safe across
    turns: per-turn state lives in the async generator's closure, not
    on ``self``.

    The named seams are:

      - :meth:`_seed_history`        — system prompt + add user message
      - :meth:`_run_one_turn_until_tool_or_end` — drive provider + parser
      - :meth:`_dispatch_tools`      — execute pending tool call
      - :meth:`_persist_partial`     — bounded partial-text write
      - :meth:`_classify_outcome`    — final ``stop_reason``
      - :meth:`_wire`                — translate ParserEvent → WireEvent
    """

    def __init__(
        self,
        *,
        provider: ModelProvider,
        parser: StreamParser,
        store: SessionStore,
        tools: Dict[str, Tool],
        system_prompt: str,
        config: OrchestratorConfig,
        tool_runner: ToolRunner,
        hw_watchdog: Optional["HardwareWatchdog"] = None,
    ):
        self.provider = provider
        self.parser = parser
        self.store = store
        self.tools = tools
        self.system_prompt = system_prompt
        self.config = config
        self.tool_runner = tool_runner
        self.hw_watchdog = hw_watchdog

    # --- Public entry point ------------------------------------------------

    async def run(
        self,
        *,
        session_id: str,
        prompt: str,
        model_name: str,
        cancel_token: Optional[CancelToken] = None,
    ) -> AsyncIterator[WireEvent]:
        """Run one turn of the model→parser→tool-execution loop.

        Yields typed :data:`WireEvent` objects. HTTP transports
        serialize via :func:`v0_compat_serialize` (legacy) or
        :class:`NdjsonEmitter` (v2). Library consumers use
        :meth:`Engine.chat` to iterate :data:`WireEvent` directly.
        """
        turn_id = uuid.uuid4().hex[:12]
        # Phase 7 step 69: bind turn_id to structlog contextvars so every
        # log line emitted during this turn includes it automatically.
        # Cleanup is in the existing finally block below.
        structlog.contextvars.bind_contextvars(turn_id=turn_id)
        seq = 0

        def _next_seq() -> int:
            nonlocal seq
            n = seq
            seq += 1
            # Phase 7 step 69: update seq contextvar so logs between yields
            # carry the seq of the most recently emitted event.
            structlog.contextvars.bind_contextvars(seq=n)
            return n

        def _envelope() -> Dict[str, Any]:
            return {
                "session_id": session_id,
                "turn_id": turn_id,
                "seq": _next_seq(),
                "ts": datetime.now(timezone.utc),
            }

        cancelled = False
        last_response_text = ""
        last_thinking_text = ""
        text_persisted = False
        active_tool_task: Optional[asyncio.Task] = None
        # Final stop_reason classified at the end of the body. None
        # means "no terminal event yet" — the finally block decides.
        final_stop_reason: Optional[str] = None

        # Seed history first; if this raises we never even get to
        # MessageStart.
        await self._seed_history(session_id, prompt)

        # v2 turn lifecycle: open the turn row before the loop so all
        # add_* calls below can link their v2 rows to this turn_id.
        # complete_turn is called in the finally block. Synthesis §3.6.
        await self.store.start_turn(
            session_id, turn_id, model_name=model_name
        )

        try:
            # Yield message_start with available tools (synthesis §3.4).
            yield MessageStart(
                **_envelope(),
                available_tools=self._tool_descriptors(),
            )

            for loop_num in range(self.config.max_tool_loops):
                logger.info(
                    f"Tool loop {loop_num+1}/{self.config.max_tool_loops} "
                    f"for session_id={session_id}"
                )

                if cancel_token is not None and cancel_token.cancelled():
                    cancelled = True
                    break

                # Per-iteration accumulators. Mirror into the carry-over
                # variables so the finally block can persist whichever
                # iteration last produced text. Synthesis §4 Phase 2 step 20.
                full_response_text = ""
                full_thinking_text = ""
                tool_call_to_run: Optional[PToolCallParsed] = None
                text_persisted = False

                # Mutable per-turn state shared with the streaming
                # seam. The seam mutates this dict instead of
                # returning a struct so it can stay an async generator
                # that yields WireEvents during streaming.
                turn_state: Dict[str, Any] = {
                    "full_response_text": "",
                    "full_thinking_text": "",
                    "tool_call": None,
                    "stream_error": None,
                    "cancelled": False,
                }

                async for wire in self._run_one_turn_until_tool_or_end(
                    session_id=session_id,
                    model_name=model_name,
                    cancel_token=cancel_token,
                    envelope_factory=_envelope,
                    turn_state=turn_state,
                ):
                    yield wire
                    # Mirror accumulators into the carry-over variables
                    # so the finally block has the right text on cancel.
                    last_response_text = turn_state["full_response_text"]
                    last_thinking_text = turn_state["full_thinking_text"]

                full_response_text = turn_state["full_response_text"]
                full_thinking_text = turn_state["full_thinking_text"]
                tool_call_to_run = turn_state["tool_call"]
                if turn_state["cancelled"]:
                    cancelled = True
                stream_error = turn_state["stream_error"]

                if stream_error is not None:
                    # Dispatch error path (HwReset + Error event +
                    # partial-text persist) — handled inline so we can
                    # yield from this generator.
                    if isinstance(stream_error, TransientProviderError):
                        # Synthesis §11.3 R10: provider unload race.
                        # Watchdog doesn't try to reset for transient.
                        logger.warning(
                            f"Transient provider error in loop "
                            f"{loop_num+1}: {stream_error}"
                        )
                        yield Error(
                            **_envelope(),
                            message=(
                                f"Model streaming failed: {stream_error}"
                            ),
                            error_type=type(stream_error).__name__,
                            is_fatal=False,
                        )
                    else:
                        error_msg = str(stream_error)
                        error_type = type(stream_error).__name__
                        logger.error(
                            f"Model streaming error in loop "
                            f"{loop_num+1}: {error_msg}",
                            exc_info=(type(stream_error), stream_error, None),
                        )
                        is_fatal = False
                        if (
                            self.hw_watchdog is not None
                            and self.config.auto_reload_on_fatal_error
                        ):
                            try:
                                recovered = await self.hw_watchdog.reset_after(
                                    stream_error, model_name=model_name
                                )
                            except Exception as wd_err:
                                logger.exception(
                                    "HardwareWatchdog.reset_after raised: %s",
                                    wd_err,
                                )
                                recovered = False
                            if recovered:
                                is_fatal = True
                                yield HwReset(
                                    **_envelope(),
                                    model_name=model_name,
                                )
                        yield Error(
                            **_envelope(),
                            message=f"Model streaming failed: {error_msg}",
                            error_type=error_type,
                            is_fatal=is_fatal,
                        )

                    if full_response_text or full_thinking_text:
                        await self.store.add_assistant_text(
                            session_id,
                            full_response_text,
                            thinking_text=full_thinking_text,
                            save_thinking=self.config.save_thinking,
                        )
                        text_persisted = True
                        logger.info(
                            "Partial assistant text persisted before "
                            "error: session_id=%s, text_length=%s, "
                            "thinking_length=%s",
                            session_id,
                            len(full_response_text),
                            len(full_thinking_text),
                        )

                    final_stop_reason = "error"
                    break

                if cancelled:
                    break

                if tool_call_to_run is not None:
                    # Dispatch the tool. _dispatch_tools is an async
                    # generator yielding WireEvents and updating
                    # state via the dispatch_state dict.
                    dispatch_state: Dict[str, Any] = {
                        "active_task_holder": [None],
                        "should_break": False,
                        "cancelled": False,
                    }
                    async for wire in self._dispatch_tools(
                        tool_call=tool_call_to_run,
                        session_id=session_id,
                        turn_id=turn_id,
                        prompt=prompt,
                        envelope_factory=_envelope,
                        cancel_token=cancel_token,
                        dispatch_state=dispatch_state,
                    ):
                        yield wire
                        if (
                            cancel_token is not None
                            and cancel_token.cancelled()
                        ):
                            cancelled = True
                    active_tool_task = dispatch_state["active_task_holder"][0]
                    if dispatch_state.get("cancelled"):
                        cancelled = True
                        break
                    if dispatch_state["should_break"]:
                        # BREAK_LOOP policy fired (or unrecoverable).
                        final_stop_reason = "error"
                        break
                    # FEED_BACK_TO_MODEL: continue the loop.
                    continue
                else:
                    # No tool call: model produced a final answer.
                    if full_response_text or full_thinking_text:
                        await self.store.add_assistant_text(
                            session_id,
                            full_response_text,
                            thinking_text=full_thinking_text,
                            save_thinking=self.config.save_thinking,
                        )
                        text_persisted = True
                        logger.info(
                            "Assistant text persisted: session_id=%s, "
                            "text_length=%s, thinking_length=%s",
                            session_id,
                            len(full_response_text),
                            len(full_thinking_text),
                        )
                    final_stop_reason = "complete"
                    break
            else:
                # for-loop fell through: max_tool_loops exhausted with
                # the model still wanting to call tools (synthesis §3.5).
                if self.config.loop_limit_policy is LoopLimitPolicy.RAISE:
                    # Phase 5 followups F5: emit MessageStop BEFORE
                    # raising — async generators cannot yield once an
                    # exception is propagating through ``finally``, so
                    # the post-finally yield below never runs on this
                    # path. Mirrors the F2 fix for outer CancelledError.
                    # Synthesis §3.5: every terminal path emits one
                    # MessageStop.
                    try:
                        yield MessageStop(
                            **_envelope(),
                            stop_reason="tool_loop_exhausted",
                        )
                    except BaseException:
                        pass
                    raise LoopLimitReachedError(
                        f"max_tool_loops={self.config.max_tool_loops} reached"
                    )
                yield LoopLimitReachedWire(
                    **_envelope(),
                    loops=self.config.max_tool_loops,
                )
                final_stop_reason = "tool_loop_exhausted"

            # Flush parser residue (text/think/parse-errors pending).
            for parser_evt in self.parser.finalize() or []:
                logger.debug(f"Parser finalize event: {parser_evt}")
                wire = self._wire(parser_evt, _envelope())
                if wire is not None:
                    yield wire

        except asyncio.CancelledError:
            # Loop-level cancellation: yield ONE terminal MessageStop
            # before re-raising. Async generators cannot yield once an
            # exception is propagating through ``finally``, so the
            # post-finally ``yield MessageStop`` below never runs on
            # this path. Synthesis §3.5: cancellation contract requires
            # exactly one terminal MessageStop on every cancel path
            # (including outer ``task.cancel()`` from FastAPI's response-
            # generator teardown or library callers using ``aclosing``).
            #
            # Phase 5 followups F2 (rubber-duck review by xhigh): wrap
            # in try/except so a consumer that already aclose()'d the
            # generator (GeneratorExit during yield) doesn't mask the
            # original CancelledError that follows. ``BaseException``
            # is intentionally broad — we're already in cleanup mode
            # and the bare ``raise`` below re-raises the ORIGINAL
            # outer CancelledError currently being handled.
            cancelled = True
            try:
                yield MessageStop(
                    **_envelope(),
                    stop_reason="cancelled",
                )
            except BaseException:
                pass
            raise
        except LoopLimitReachedError:
            # RAISE policy — propagate to caller without further wire events.
            raise
        except Exception as e:
            logger.exception(
                f"Exception in orchestrate: session_id={session_id}, error={e}"
            )
            yield Error(
                **_envelope(),
                message=str(e),
                error_type=type(e).__name__,
                is_fatal=False,
            )
            if final_stop_reason is None:
                final_stop_reason = "error"
        finally:
            # Cancellation contract step 2: cancel in-flight tool task
            # with 250 ms grace.
            if active_tool_task is not None and not active_tool_task.done():
                active_tool_task.cancel()
                try:
                    await asyncio.wait_for(
                        active_tool_task, timeout=_TOOL_CANCEL_GRACE_SEC
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    # Phase 5 followups F8: dropped ``Exception`` from
                    # the tuple — let unexpected exceptions surface to
                    # logs rather than silently swallowing real bugs.
                    pass

            # Cancellation contract step 3: persist partial assistant text
            # with 200 ms write timeout.
            if not text_persisted and (last_response_text or last_thinking_text):
                try:
                    await self._persist_partial(
                        session_id=session_id,
                        text=last_response_text,
                        thinking_text=last_thinking_text,
                    )
                    text_persisted = True
                    logger.info(
                        "Partial assistant text persisted in finally: "
                        "session_id=%s, text_length=%s, thinking_length=%s",
                        session_id,
                        len(last_response_text),
                        len(last_thinking_text),
                    )
                except (asyncio.TimeoutError, Exception) as fin_exc:
                    logger.exception(
                        f"Failed to persist partial text in finally: {fin_exc}"
                    )

            # v2 turn lifecycle: close the turn row opened above.
            # Map the internal stop_reason to the turns.status CHECK values:
            #   cancelled → cancelled
            #   error / tool_loop_exhausted → failed
            #   complete / None → completed
            # Synthesis §3.6.
            _turn_status_map = {
                "cancelled": "cancelled",
                "error": "failed",
                "tool_loop_exhausted": "failed",
            }
            _final_turn_status = _turn_status_map.get(
                final_stop_reason or "", "completed"
            )
            if cancelled:
                _final_turn_status = "cancelled"
            try:
                await self.store.complete_turn(
                    turn_id,
                    status=_final_turn_status,
                    stop_reason=final_stop_reason or ("cancelled" if cancelled else "complete"),
                )
            except Exception as ct_exc:
                logger.warning(
                    "complete_turn failed (non-fatal): turn_id=%s error=%s",
                    turn_id, ct_exc,
                )

            # Phase 7 step 69: unbind turn_id (and seq) from structlog
            # contextvars to prevent leaking across turns on the same
            # event-loop task. Must be last in finally so turn_id is
            # present in all cleanup log lines above.
            structlog.contextvars.unbind_contextvars("turn_id", "seq")

        # Outside the try/finally: emit terminal MessageStop. The
        # CancelledError path doesn't reach here (it re-raised), so
        # this only runs on the normal / handled-error paths.
        stop_reason = self._classify_outcome(
            cancelled=cancelled,
            final_stop_reason=final_stop_reason,
        )
        yield MessageStop(**_envelope(), stop_reason=stop_reason)
        logger.info(
            f"Orchestration complete: session_id={session_id}, "
            f"stop_reason={stop_reason}"
        )

    # --- Named seams -------------------------------------------------------

    async def _seed_history(self, session_id: str, prompt: str) -> None:
        """Ensure system prompt + record user message before the model
        sees history.
        """
        await self.store.ensure_system_prompt(session_id, self.system_prompt)
        await self.store.add_user(session_id, prompt)
        # §13 R5: prompt is PII — redact before logging, demote to DEBUG.
        logger.debug(
            f"User message added: session_id={session_id}, prompt={_redact(prompt)}"
        )

    def _wire(
        self, parser_event: Any, envelope: Dict[str, Any]
    ) -> Optional[WireEvent]:
        """Translate a single :class:`ParserEvent` to a :class:`WireEvent`.

        Returns ``None`` for parser events that don't map to a wire
        event (currently :class:`PToolCallDetected` and
        :class:`PStreamEnd`). The caller skips the yield.

        This is the ONLY translation site between parser-internal
        events and the wire-facing typed event vocabulary (synthesis
        §3.5).
        """
        if isinstance(parser_event, PText):
            return TextDelta(**envelope, text=parser_event.text)
        if isinstance(parser_event, PThink):
            return ThinkingDelta(**envelope, text=parser_event.text)
        if isinstance(parser_event, PParseError):
            return Error(
                **envelope,
                message=parser_event.message,
                error_type="ParseError",
                is_fatal=False,
            )
        # PToolCallDetected, PStreamEnd, PToolCallParsed: handled by the
        # main loop — no wire translation here.
        return None

    async def _run_one_turn_until_tool_or_end(
        self,
        *,
        session_id: str,
        model_name: str,
        cancel_token: Optional[CancelToken],
        envelope_factory,
        turn_state: Dict[str, Any],
    ) -> AsyncIterator[WireEvent]:
        """Drive one provider stream until a tool call is detected, the
        stream completes, an error fires, or cancellation arrives.

        Yields :class:`TextDelta` / :class:`ThinkingDelta` /
        :class:`Error` (parse errors) wire events as the parser produces
        them. Mutates ``turn_state`` to communicate the outcome:

          - ``full_response_text`` / ``full_thinking_text``:
            accumulated text so the outer loop can persist on success.
          - ``tool_call``: :class:`PToolCallParsed` if the model
            decided to call a tool; ``None`` otherwise.
          - ``stream_error``: any :class:`Exception` raised inside the
            provider stream; ``None`` on clean exit.
          - ``cancelled``: True if ``cancel_token`` flipped True
            mid-stream.

        Synthesis §3.5 — this is the streaming half of the seam pair
        (the other half is :meth:`_dispatch_tools`).
        """
        messages = await self.store.get_history(
            session_id,
            include_thinking=self.config.include_thinking_in_history,
        )
        tool_schemas = [tool.schema for tool in self.tools.values()]

        full_response_text = ""
        full_thinking_text = ""
        tool_call_to_run: Optional[PToolCallParsed] = None

        try:
            async with aclosing(
                self.provider.stream(
                    model_name=model_name,
                    messages=messages,
                    tools=tool_schemas,
                )
            ) as provider_stream:
                async for chunk in provider_stream:
                    logger.debug(f"Provider stream chunk: {chunk}")
                    for parser_evt in self.parser.feed(chunk):
                        logger.debug(f"Parser event: {parser_evt}")

                        if isinstance(parser_evt, PText):
                            if parser_evt.text:
                                full_response_text += parser_evt.text
                                turn_state["full_response_text"] = (
                                    full_response_text
                                )
                                wire = self._wire(parser_evt, envelope_factory())
                                if wire is not None:
                                    yield wire

                        elif isinstance(parser_evt, PThink):
                            if parser_evt.text:
                                full_thinking_text += parser_evt.text
                                turn_state["full_thinking_text"] = (
                                    full_thinking_text
                                )
                                wire = self._wire(parser_evt, envelope_factory())
                                if wire is not None:
                                    yield wire

                        elif isinstance(parser_evt, PToolCallDetected):
                            # Parser-internal marker; no v2 wire
                            # equivalent. The v0 ``tool_marker_detected``
                            # event has been dropped.
                            logger.info(
                                f"Tool call marker detected for "
                                f"session_id={session_id}"
                            )

                        elif isinstance(parser_evt, PToolCallParsed):
                            tool_call_to_run = parser_evt
                            logger.info(
                                f"Tool call detected: name={parser_evt.name}, "
                                f"id={parser_evt.tool_call_id}"
                            )
                            break  # exit parser-event loop

                        elif isinstance(parser_evt, PParseError):
                            logger.error(
                                f"Parser error: {parser_evt.message}, "
                                f"raw={(parser_evt.raw or '')[:100]}"
                            )
                            wire = self._wire(parser_evt, envelope_factory())
                            if wire is not None:
                                yield wire

                        elif isinstance(parser_evt, PStreamEnd):
                            # SlidingParser doesn't emit this today;
                            # future-proofing.
                            pass

                    if tool_call_to_run is not None:
                        break

                    if cancel_token is not None and cancel_token.cancelled():
                        logger.info(
                            f"Cancellation requested mid-stream for "
                            f"session_id={session_id}"
                        )
                        turn_state["cancelled"] = True
                        break
        except Exception as stream_error:
            turn_state["stream_error"] = stream_error
            return

        turn_state["tool_call"] = tool_call_to_run
        turn_state["full_response_text"] = full_response_text
        turn_state["full_thinking_text"] = full_thinking_text

    async def _dispatch_tools(
        self,
        *,
        tool_call: PToolCallParsed,
        session_id: str,
        turn_id: str,
        prompt: str,
        envelope_factory,
        cancel_token: Optional[CancelToken],
        dispatch_state: Dict[str, Any],
    ) -> AsyncIterator[WireEvent]:
        """Execute a tool, yielding :class:`ToolCall` + :class:`ToolResult`
        wire events.

        Tool-error policy (synthesis §3.5):

          - ``FEED_BACK_TO_MODEL`` (default): persist error as
            ``tool_result``, yield ``ToolResult(status='error')``,
            continue.
          - ``BREAK_LOOP``: same, but signal the outer loop to break.

        Cancellation: if ``cancel_token`` flips True during execution,
        cancel the tool task with 250 ms grace and yield
        ``ToolResult(status='error', error_kind='cancelled')``.
        """
        tool_name = tool_call.name
        tool_args = tool_call.arguments
        tool_call_id = tool_call.tool_call_id

        # Persist the assistant's tool-call intent before yielding the
        # ToolCall wire event, so even if the consumer hangs up after
        # tool_started the call is in history. Mirrors legacy ordering.
        # Thread v2 IDs so SqliteSessionStore writes a tool_calls row.
        # Synthesis §3.6.
        await self.store.add_assistant_toolcall(
            session_id, tool_name, tool_args,
            turn_id=turn_id, tool_call_id=tool_call_id,
        )
        logger.debug(
            f"Assistant tool call persisted: session_id={session_id}, "
            f"tool_name={tool_name}, tool_args={_redact(tool_args)}"
        )

        yield ToolCall(
            **envelope_factory(),
            tool_call_id=tool_call_id,
            name=tool_name,
            arguments=dict(tool_args),
        )

        tool_ctx = ToolExecutionContext(
            session_id=session_id,
            turn_id=turn_id,
            last_user_message=prompt,
            user_confirmed_send=False,
        )

        # Wrap the tool execution in a Task so we can cancel it with
        # bounded grace if cancel_token flips True. ToolRunner already
        # imposes its own timeout; we're adding the cancel-grace layer.
        # Phase 7 step 71: measure wall time for _audit_tool_call(duration_ms).
        _tool_dispatch_start = time.monotonic()
        task = asyncio.create_task(
            self.tool_runner.run(
                tool_name, tool_args, context=tool_ctx,
                tool_call_id=tool_call_id,  # Phase 7 step 71: span correlation
            ),
            name=f"tool:{tool_name}",
        )
        dispatch_state["active_task_holder"][0] = task

        sha = _args_sha256(tool_args)

        try:
            # Poll cancel_token while the tool runs. Don't busy-spin —
            # use asyncio.wait with a small timeout so we yield the
            # event loop.
            while True:
                done, _ = await asyncio.wait({task}, timeout=0.05)
                if done:
                    break
                if cancel_token is not None and cancel_token.cancelled():
                    task.cancel()
                    try:
                        await asyncio.wait_for(
                            task, timeout=_TOOL_CANCEL_GRACE_SEC
                        )
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                    except Exception:
                        # Tool may raise during cancel cleanup — swallow,
                        # we're emitting cancelled below.
                        pass
                    error_msg = f"Tool '{tool_name}' cancelled by client"
                    await self.store.add_tool_result(
                        session_id, tool_name, {"error": error_msg},
                        turn_id=turn_id, tool_call_id=tool_call_id,
                        status="cancelled", error=error_msg,
                    )
                    yield ToolResult(
                        **envelope_factory(),
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        status="error",
                        error_kind="cancelled",
                        error=error_msg,
                    )
                    await self._audit_tool_call(
                        session_id=session_id,
                        turn_id=turn_id,
                        tool_name=tool_name,
                        args_sha256=sha,
                        status="cancelled",
                        error_kind="cancelled",
                        duration_ms=int((time.monotonic() - _tool_dispatch_start) * 1000),
                    )
                    dispatch_state["cancelled"] = True
                    dispatch_state["should_break"] = True
                    return

            # Task finished naturally: success, raised, or timed out.
            try:
                result = task.result()
            except asyncio.TimeoutError:
                error_msg = f"Tool '{tool_name}' timed out"
                logger.error(error_msg)
                await self.store.add_tool_result(
                    session_id, tool_name, {"error": error_msg},
                    turn_id=turn_id, tool_call_id=tool_call_id,
                    status="error", error=error_msg,
                )
                yield ToolResult(
                    **envelope_factory(),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    status="error",
                    error_kind="timeout",
                    error=error_msg,
                )
                await self._audit_tool_call(
                    session_id=session_id,
                    turn_id=turn_id,
                    tool_name=tool_name,
                    args_sha256=sha,
                    status="error",
                    error_kind="timeout",
                    duration_ms=int((time.monotonic() - _tool_dispatch_start) * 1000),
                )
                if (
                    self.config.tool_error_policy
                    is ToolErrorPolicy.BREAK_LOOP
                ):
                    dispatch_state["should_break"] = True
                return
            except asyncio.CancelledError:
                # Should only happen if outer cancellation cancelled us.
                dispatch_state["cancelled"] = True
                dispatch_state["should_break"] = True
                raise
            except Exception as e:
                error_msg = f"Error running tool {tool_name}: {e}"
                logger.exception(error_msg)
                await self.store.add_tool_result(
                    session_id, tool_name, {"error": error_msg},
                    turn_id=turn_id, tool_call_id=tool_call_id,
                    status="error", error=error_msg,
                )
                yield ToolResult(
                    **envelope_factory(),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    status="error",
                    error_kind="execution",
                    error=error_msg,
                )
                await self._audit_tool_call(
                    session_id=session_id,
                    turn_id=turn_id,
                    tool_name=tool_name,
                    args_sha256=sha,
                    status="error",
                    error_kind="execution",
                    duration_ms=int((time.monotonic() - _tool_dispatch_start) * 1000),
                )
                if (
                    self.config.tool_error_policy
                    is ToolErrorPolicy.BREAK_LOOP
                ):
                    dispatch_state["should_break"] = True
                return

            # Success path.
            logger.debug(
                f"Tool executed: {tool_name}, result={_redact(result)}"
            )

            # Phase 7 step 77: ToolRunner returns a structured error dict for
            # oversized results rather than raising. Route it through the
            # execution-error branch so the wire event carries status="error"
            # and the model receives it via FEED_BACK_TO_MODEL (synthesis §3.5).
            if (
                isinstance(result, dict)
                and result.get("error") == "tool_result_oversized"
            ):
                error_msg = (
                    f"Tool '{tool_name}' result too large: "
                    f"{result.get('size_bytes')} bytes "
                    f"(limit {result.get('limit_bytes')} bytes)"
                )
                logger.warning(error_msg)
                await self.store.add_tool_result(
                    session_id, tool_name, result,
                    turn_id=turn_id, tool_call_id=tool_call_id,
                    status="error", error=error_msg,
                )
                yield ToolResult(
                    **envelope_factory(),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    status="error",
                    error_kind="execution",
                    error=error_msg,
                )
                await self._audit_tool_call(
                    session_id=session_id,
                    turn_id=turn_id,
                    tool_name=tool_name,
                    args_sha256=sha,
                    status="error",
                    error_kind="execution",
                    duration_ms=int((time.monotonic() - _tool_dispatch_start) * 1000),
                )
                if self.config.tool_error_policy is ToolErrorPolicy.BREAK_LOOP:
                    dispatch_state["should_break"] = True
                return

            await self.store.add_tool_result(
                session_id, tool_name, result,
                turn_id=turn_id, tool_call_id=tool_call_id, status="ok",
            )
            # ``result`` may be any JSON-able value; ToolResult.result is
            # Optional[Dict] so wrap non-dicts in {"result": ...} per the
            # legacy v0 shape.
            wire_result = (
                result if isinstance(result, dict) else {"result": result}
            )
            yield ToolResult(
                **envelope_factory(),
                tool_call_id=tool_call_id,
                name=tool_name,
                status="ok",
                result=wire_result,
            )
            await self._audit_tool_call(
                session_id=session_id,
                turn_id=turn_id,
                tool_name=tool_name,
                args_sha256=sha,
                status="ok",
                duration_ms=int((time.monotonic() - _tool_dispatch_start) * 1000),
            )
        finally:
            # Phase 5 followups F3 (rubber-duck review by xhigh + gpt-5.5):
            # cancel any still-running tool task before clearing the holder.
            # When outer cancellation arrives while we're in the 50ms-poll
            # ``asyncio.wait`` loop above, ``CancelledError`` propagates out
            # WITHOUT cancelling the awaited tool task, AND the orchestrator's
            # outer ``finally`` doesn't see the task because the
            # ``async for wire in self._dispatch_tools(...)`` was interrupted
            # mid-iteration (so ``active_tool_task`` was never assigned in
            # the outer scope). Result: tool task leaks unbounded after
            # outer cancel.
            #
            # Fix: this ``finally`` always runs (whether normal exit, a
            # ``return``, or outer cancellation), so it's the right place
            # to enforce the cancellation contract on the tool task.
            # Synthesis §3.5: 250 ms grace bounds the tool's
            # CancelledError handler.
            pending = dispatch_state["active_task_holder"][0]
            dispatch_state["active_task_holder"][0] = None
            if pending is not None and not pending.done():
                pending.cancel()
                try:
                    await asyncio.wait_for(
                        pending, timeout=_TOOL_CANCEL_GRACE_SEC
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    # Tool either over-ran the grace or honored the
                    # cancel. Either way, we're done with it. Don't
                    # add ``Exception`` to the tuple (Phase 5 followups
                    # F8) — let real bugs surface to the logger.
                    pass

    async def _persist_partial(
        self,
        *,
        session_id: str,
        text: str,
        thinking_text: str,
    ) -> None:
        """Persist accumulated partial text with a 200 ms write budget
        (synthesis §3.5 cancellation contract step 3).

        Raises :class:`asyncio.TimeoutError` if the store can't write
        in time; the caller's ``finally`` swallows it (logged).
        """
        await asyncio.wait_for(
            self.store.add_assistant_text(
                session_id,
                text,
                thinking_text=thinking_text,
                save_thinking=self.config.save_thinking,
            ),
            timeout=_PARTIAL_PERSIST_TIMEOUT_SEC,
        )

    def _classify_outcome(
        self,
        *,
        cancelled: bool,
        final_stop_reason: Optional[str],
    ) -> str:
        """Choose the terminal :class:`MessageStop.stop_reason`.

        Precedence (synthesis §11.3 R1):

          1. ``cancelled`` flag wins over everything (a partial
             ``complete`` set inside the loop is overridden if cancel
             arrived).
          2. Otherwise use the loop's classification (``complete``,
             ``error``, ``tool_loop_exhausted``).
          3. Default to ``complete`` if nothing was set (e.g., the
             model produced no text and no tool call).
        """
        if cancelled:
            return "cancelled"
        if final_stop_reason is not None:
            return final_stop_reason
        return "complete"

    def _tool_descriptors(self) -> List[ToolDescriptor]:
        """Build the list of :class:`ToolDescriptor` for
        :class:`MessageStart`. Defensive against tools whose schema
        doesn't quite match (legacy schema format vs flat name)."""
        descriptors: List[ToolDescriptor] = []
        for name, tool in self.tools.items():
            schema = tool.schema or {}
            desc_name = schema.get("name") if isinstance(schema, dict) else None
            # Legacy schemas wrap as {"type": "function", "function": {...}}
            if (
                not desc_name
                and isinstance(schema, dict)
                and isinstance(schema.get("function"), dict)
            ):
                desc_name = schema["function"].get("name")
            if not desc_name:
                desc_name = name
            description = ""
            parameters: Dict[str, Any] = {}
            if isinstance(schema, dict):
                description = schema.get("description", "") or ""
                params = schema.get("parameters")
                if isinstance(params, dict):
                    parameters = params
                elif isinstance(schema.get("function"), dict):
                    fn = schema["function"]
                    description = description or fn.get("description", "")
                    fn_params = fn.get("parameters")
                    if isinstance(fn_params, dict):
                        parameters = fn_params
            descriptors.append(
                ToolDescriptor(
                    name=desc_name,
                    description=description,
                    parameters=parameters,
                )
            )
        return descriptors

    async def _audit_tool_call(
        self,
        *,
        session_id: str,
        turn_id: str,
        tool_name: str,
        args_sha256: str,
        status: str,
        error_kind: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        """Audit-log hook (no-op until Phase 7 step 73 ships ``tool_audit``).

        Synthesis §11.3 R3 + Phase 7 step 73-74. The orchestrator
        already calls this at success / error / timeout / cancel
        sites; Phase 7 swaps the body for an actual SQL ``INSERT INTO
        tool_audit``. Do NOT add the table here — that's strictly
        Phase 7's column ownership (synthesis B5).
        """
        # TODO(phase7-step73): implement real audit log row insert.
        return None


__all__ = ["ChattyAgentOrchestrator"]
