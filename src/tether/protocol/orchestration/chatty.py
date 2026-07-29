"""ChattyAgentOrchestrator — the standard tool-loop agent.

Implements the Orchestrator ABC (tether.core.interfaces).
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
     (``asyncio.wait_for(task, 0.25)``).  Note: this bounds the
     AWAITER only; tasks holding native handles may still run on the
     background loop.  P0-C / Tribunal P0-06.
  3. Partial assistant text persisted with **200 ms awaiter budget**
     (``asyncio.wait_for(store.add_assistant_text(...), 0.20)``).
     Same awaiter-only caveat: the underlying aiosqlite worker thread
     keeps executing SQL even after ``wait_for`` times out.
  4. Parser ``finalize()`` called from the ``finally:`` block — runs
     on every exit path (success, cancel, exception, loop-limit
     raise).  P0-C / Tribunal P0-05.
  5. ONE :class:`MessageStop` with the appropriate ``stop_reason``
     emitted (in v2 vocabulary, ``MessageStop`` IS the "done" event).
     On the cancel path the ``MessageStop`` is yielded from the
     ``except CancelledError`` branch BEFORE re-raise.

Tool-error policy (default ``FEED_BACK_TO_MODEL``): tool errors no
longer break the loop — a :class:`ToolResult` row with
``status='error'`` is persisted and the loop continues so the model
can recover. ``BREAK_LOOP`` (legacy behaviour) keeps the single-turn
shape for deterministic tests.

Loop-limit policy (default ``EMIT_LIMIT_EVENT``): exhausting
``max_tool_loops`` yields :class:`LoopLimitReached` plus
:class:`MessageStop(stop_reason='tool_loop_exhausted')` and exits
cleanly. ``RAISE`` raises :class:`tether.core.errors.LoopLimitReached`.

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
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    cast,
)

import structlog

from tether.core.errors import (
    LoopLimitReached as LoopLimitReachedError,
)
from tether.core.errors import (
    TransientProviderError,
)
from tether.core.interfaces import (
    ModelProvider,
    SessionStore,
    StreamParser,
    Tool,
)
from tether.core.interfaces import (
    Orchestrator as OrchestratorABC,
)
from tether.core.logging import logger
from tether.core.types import OrchestratorConfig, ToolExecutionContext
from tether.protocol.orchestration.cancel import CancelToken
from tether.protocol.orchestration.policies import (
    LoopLimitPolicy,
    ToolErrorPolicy,
)
from tether.protocol.orchestration.tool_runner import ToolRunner
from tether.protocol.parsers.events import (
    PParseError,
    PStreamEnd,
    PText,
    PThink,
    PToolCallDetected,
    PToolCallParsed,
)
from tether.protocol.wire.events import (
    Error,
    HwReset,
    MessageStart,
    MessageStop,
    StopReason,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolDescriptor,
    ToolResult,
    WireEvent,
)
from tether.protocol.wire.events import (
    LoopLimitReached as LoopLimitReachedWire,
)

if TYPE_CHECKING:
    from tether.protocol.intent.classifier import ConfirmIntentClassifier
    from tether.runtime.hw_watchdog import HardwareWatchdog


# Synthesis §3.5: cancellation contract bounds.
#
# P0-C / Tribunal P0-06: both timeouts below bound the AWAITER (the
# coroutine that calls ``await``), NOT the underlying worker.  For
# ``_TOOL_CANCEL_GRACE_SEC`` a tool that holds a native handle (HTTP
# socket, GPU buffer, file descriptor) can keep running on the
# background loop after the awaiter gives up.  For the persist budget
# (renamed to ``_AWAITER_PERSIST_BUDGET_SEC``), the aiosqlite worker
# thread keeps executing SQL even after ``wait_for`` times out.  True
# work-bounded cancellation requires a shared cancel event consulted
# inside the store / tool implementation; see backlog fu-* items.
_TOOL_CANCEL_GRACE_SEC = 0.25  # awaiter-only: 250 ms after cancel
_AWAITER_PERSIST_BUDGET_SEC = 0.20  # awaiter-only: partial-text write budget


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

    Implements :class:`tether.core.interfaces.Orchestrator`.
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
        provider_id: Optional[str] = None,
        confirm_intent_classifier: "ConfirmIntentClassifier | None" = None,
        audit_store_args: bool = False,
    ):
        self.provider = provider
        self.parser = parser
        self.store = store
        self.tools = tools
        self.system_prompt = system_prompt
        self.config = config
        self.tool_runner = tool_runner
        self.hw_watchdog = hw_watchdog
        self.provider_id = provider_id
        # Phase 7 step 74: when True, raw args_json is stored in tool_audit
        # alongside the SHA-256 hash. Default False (privacy-preserving).
        # Synthesis §3.6 + B5 step 7.
        self._audit_store_args = audit_store_args
        if confirm_intent_classifier is None:
            from tether.protocol.intent.classifier import NullConfirmIntentClassifier

            confirm_intent_classifier = NullConfirmIntentClassifier()
        self._confirm_intent_classifier = confirm_intent_classifier

    # --- Public entry point ------------------------------------------------

    async def run(
        self,
        *,
        session_id: str,
        prompt: str,
        model_name: str,
        cancel_token: Optional[CancelToken] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncIterator[WireEvent]:
        """Run one turn of the model→parser→tool-execution loop.

        Yields typed :data:`WireEvent` objects. HTTP transports
        serialize via :func:`v0_compat_serialize` (legacy) or
        :class:`NdjsonEmitter` (v2). Library consumers use
        :meth:`Engine.chat` to iterate :data:`WireEvent` directly.
        """
        turn_id = uuid.uuid4().hex[:12]
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
        # P0-C / Tribunal P0-05: track whether an exception is propagating
        # so the parser-finalize residue loop in ``finally`` knows not to
        # yield (async generators cannot yield through a finally that's
        # serving an exception).
        exception_in_flight = False
        last_response_text = ""
        last_thinking_text = ""
        text_persisted = False
        active_tool_task: Optional[asyncio.Task] = None
        # Final stop_reason classified at the end of the body. None
        # means "no terminal event yet" — the finally block decides.
        final_stop_reason: Optional[str] = None

        # Phase 7 step 69 + RD followup (FIX 4): bind turn_id INSIDE the try
        # block so the matching unbind in `finally` always runs even when
        # `_seed_history` or `store.start_turn` raises. Bind FIRST (before
        # any awaitable that can fail) so error logs from those calls still
        # carry turn_id for forensics. The previous arrangement bound BEFORE
        # the try, which leaked the contextvar onto the calling task if
        # _seed_history / start_turn raised.
        try:
            structlog.contextvars.bind_contextvars(turn_id=turn_id)

            # Seed history first; if this raises we never even get to
            # MessageStart but the bind/unbind is still symmetric.
            await self._seed_history(session_id, prompt)

            # v2 turn lifecycle: open the turn row before the loop so all
            # add_* calls below can link their v2 rows to this turn_id.
            # complete_turn is called in the finally block. Synthesis §3.6.
            await self.store.start_turn(
                session_id, turn_id, model_name=model_name
            )

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
                    reasoning_effort=reasoning_effort,
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
                                    stream_error,
                                    model_name=model_name,
                                    provider_id=self.provider_id,
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
            exception_in_flight = True
            raise
        except Exception as e:
            exception_in_flight = True
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
            # P0-C / Tribunal P0-05 / Synthesis §3.5 cancel-contract step 4:
            # parser.finalize() is invoked on EVERY exit path (success,
            # cancel, exception, loop-limit raise, outer ``aclose()``).
            # Guarded with try/except so a misbehaving parser cannot mask
            # the original exception.
            #
            # We do NOT yield from the parser-finalize residue when an
            # outer exception is already propagating — async generators
            # cannot yield through a ``finally`` that's serving an
            # exception (Python raises ``RuntimeError: async generator
            # ignored GeneratorExit``).  We detect that case three ways:
            #
            #   * ``cancelled`` is set in the ``except CancelledError``
            #     branch before the bare ``raise``;
            #   * ``exception_in_flight`` is set in the
            #     ``except LoopLimitReachedError`` / ``except Exception``
            #     branches;
            #   * ``sys.exc_info()`` catches the leftover case of an
            #     un-caught exception flowing into ``finally`` —
            #     in particular ``GeneratorExit`` from
            #     ``aclose()`` / ``aclosing(...)``.
            import sys as _sys
            _active_exc_type = _sys.exc_info()[0]
            try:
                _residue = self.parser.finalize() or []
            except BaseException as _fin_exc:  # noqa: BLE001
                logger.exception(
                    "parser.finalize_raised",
                    error=str(_fin_exc),
                )
                _residue = []
            if (
                not cancelled
                and not exception_in_flight
                and _active_exc_type is None
            ):
                for parser_evt in _residue:
                    logger.debug(f"Parser finalize event: {parser_evt}")
                    residue_wire = self._wire(parser_evt, _envelope())
                    if residue_wire is not None:
                        yield residue_wire

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
                # Phase 7 RD followup (FIX 5): bound complete_turn with the
                # same _AWAITER_PERSIST_BUDGET_SEC budget used for partial-text
                # persistence. The whole `finally` block runs on cancel paths,
                # so a slow store can stall outer cancel — symmetric to the
                # _persist_partial budget above.
                await asyncio.wait_for(
                    self.store.complete_turn(
                        turn_id,
                        status=_final_turn_status,
                        stop_reason=final_stop_reason or ("cancelled" if cancelled else "complete"),
                    ),
                    timeout=_AWAITER_PERSIST_BUDGET_SEC,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "turn.complete_timeout",
                    turn_id=turn_id,
                    timeout_sec=_AWAITER_PERSIST_BUDGET_SEC,
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
        reasoning_effort: Optional[str],
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

        # Phase 7 step 72: provider streaming spans.
        # Pull request_id from contextvars (bound by RequestIdMiddleware;
        # merge_contextvars processor makes it appear in all structlog events
        # automatically, but we also forward it to provider.stream() for
        # provider-internal log correlation). Synthesis §3 (observability).
        _plog = structlog.get_logger(__name__)
        _caller_rid: Optional[str] = structlog.contextvars.get_contextvars().get("request_id")
        _chunk_sample: int = self.config.provider_chunk_log_sample
        _stream_start = time.monotonic()
        _stream_chunks = 0
        _plog.info("provider.stream.start", model_id=model_name)

        try:
            stream_kwargs: Dict[str, Any] = {
                "model_name": model_name,
                "messages": messages,
                "tools": tool_schemas,
                "request_id": _caller_rid,
            }
            if reasoning_effort is not None:
                stream_kwargs["reasoning_effort"] = reasoning_effort
            async with aclosing(
                self.provider.stream(**stream_kwargs)
            ) as provider_stream:
                async for chunk in provider_stream:
                    _stream_chunks += 1
                    if _chunk_sample and (
                        _stream_chunks == 1 or _stream_chunks % _chunk_sample == 0
                    ):
                        _size = (
                            len(chunk.encode("utf-8"))
                            if isinstance(chunk, str)
                            else len(repr(chunk))
                        )
                        _plog.info(
                            "provider.stream.chunk",
                            chunk_index=_stream_chunks,
                            size_bytes=_size,
                        )
                    logger.debug(f"Provider stream chunk: {chunk}")

                    # P0-E / Tribunal §3 P0-10 (A11-F1, A1-F4): the
                    # provider stream contract (core/interfaces.py:28)
                    # is ``str | List[Dict[str, Any]]``. String chunks
                    # carry text + ``<<function_call>>`` markers; list
                    # chunks carry MLC-native ``delta.tool_calls`` deltas
                    # (provider.py:792-795). ``SlidingParser.feed`` does
                    # ``self.buf += chunk`` and TypeErrors on a list.
                    # Today ``marker_only_tools=true`` suppresses the
                    # list shape, but a single config flip would crash
                    # the orchestrator. Dispatch the list shape directly
                    # here, reusing the same ``_dispatch_tools`` path
                    # the marker parser feeds into. The full
                    # ``provider.stream_typed()`` cutover (Phase-5
                    # step 52) is still the long-term plan.
                    if isinstance(chunk, list):
                        tool_call_to_run = self._native_tool_call_from_chunk(
                            chunk
                        )
                        if tool_call_to_run is not None:
                            logger.info(
                                "Native tool_call from list-shaped chunk: "
                                f"name={tool_call_to_run.name}, "
                                f"id={tool_call_to_run.tool_call_id}"
                            )
                            break
                        # Empty / malformed list — skip, continue stream.
                        continue

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

            # Stream exited cleanly (exhausted, tool-call break, or
            # soft-cancel break). Emit the end span.
            _plog.info(
                "provider.stream.end",
                model_id=model_name,
                duration_ms=int((time.monotonic() - _stream_start) * 1000),
                chunks_emitted=_stream_chunks,
            )
        except asyncio.CancelledError:
            # Hard task cancellation (e.g., HTTP client disconnect or
            # outer asyncio.Task.cancel()). Log the error span and re-raise
            # so the outer CancelledError handler in run() fires correctly.
            # Synthesis §3.5 cancellation contract.
            _plog.warning(
                "provider.stream.error",
                model_id=model_name,
                error_kind="cancelled",
                duration_ms=int((time.monotonic() - _stream_start) * 1000),
                chunks_emitted=_stream_chunks,
            )
            raise
        except Exception as stream_error:
            _plog.error(
                "provider.stream.error",
                model_id=model_name,
                error_kind="provider_error",
                error_class=type(stream_error).__name__,
                duration_ms=int((time.monotonic() - _stream_start) * 1000),
                chunks_emitted=_stream_chunks,
            )
            turn_state["stream_error"] = stream_error
            return

        turn_state["tool_call"] = tool_call_to_run
        turn_state["full_response_text"] = full_response_text
        turn_state["full_thinking_text"] = full_thinking_text

    @staticmethod
    def _native_tool_call_from_chunk(
        chunk: List[Dict[str, Any]],
    ) -> Optional[PToolCallParsed]:
        """Adapt an MLC-native ``delta.tool_calls`` chunk to a
        :class:`PToolCallParsed`.

        P0-E / Tribunal §3 P0-10 (A11-F1, A1-F4). Only the first entry
        is honoured — the orchestrator's loop is single-call-per-turn
        (matches the marker-parser path which also exits the
        provider-stream loop on the first :class:`PToolCallParsed`).

        Each ``tc`` is the ``delta.tool_calls[i].model_dump()`` shape
        produced by :mod:`tether.providers.mlc.provider` (see line 794
        of that module): ``{"id"?, "type", "function":
        {"name", "arguments": str | dict}}``. The ``arguments`` field
        is a JSON string in the OpenAI-style protocol; we parse it.
        Falls back to ``{"_raw": ...}`` on malformed JSON so the model
        still sees something deterministic.

        Returns ``None`` if ``chunk`` is empty or the first entry has
        no resolvable name (treated as a dropped delta).
        """
        if not chunk:
            return None
        tc = chunk[0]
        if not isinstance(tc, dict):
            return None
        fn = tc.get("function")
        if not isinstance(fn, dict):
            fn = {}
        name = fn.get("name") or tc.get("name")
        if not name:
            return None
        raw_args = fn.get("arguments")
        if raw_args is None:
            raw_args = tc.get("arguments", {})
        if isinstance(raw_args, str):
            if raw_args:
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
            else:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {"_raw": raw_args}
        if not isinstance(args, dict):
            args = {"_raw": args}
        tool_call_id = tc.get("id") or f"call-{uuid.uuid4().hex[:12]}"
        return PToolCallParsed(
            tool_call_id=str(tool_call_id),
            name=str(name),
            arguments=args,
        )

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
            user_confirmed_send=self._confirm_intent_classifier.classify(prompt or ""),
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
        # Canonical JSON for args_json column (populated only when
        # audit_store_args=True; same encoding as sha input so the hash
        # is always verifiable against the stored JSON). Phase 7 step 74.
        _args_json_str = json.dumps(tool_args, sort_keys=True, default=str)

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
                    # Phase 7 RD followup (FIX 3): cancel-path _audit_tool_call
                    # bounded with the same 200ms budget as _persist_partial /
                    # complete_turn. The success / exception / timeout paths
                    # below run on normal time and don't need the budget — only
                    # the soft-cancel branch is reached during outer cancel and
                    # must respect the cancellation deadline.
                    try:
                        await asyncio.wait_for(
                            self._audit_tool_call(
                                session_id=session_id,
                                turn_id=turn_id,
                                tool_call_id=tool_call_id,
                                tool_name=tool_name,
                                args_json=_args_json_str,
                                args_sha256=sha,
                                status="cancelled",
                                error_kind="cancelled",
                                duration_ms=int((time.monotonic() - _tool_dispatch_start) * 1000),
                            ),
                            timeout=_AWAITER_PERSIST_BUDGET_SEC,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "tool_audit.cancel_path_timeout",
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            timeout_sec=_AWAITER_PERSIST_BUDGET_SEC,
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
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        args_json=_args_json_str,
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
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        args_json=_args_json_str,
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
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    args_json=_args_json_str,
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
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                args_json=_args_json_str,
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
            timeout=_AWAITER_PERSIST_BUDGET_SEC,
        )

    def _classify_outcome(
        self,
        *,
        cancelled: bool,
        final_stop_reason: Optional[str],
    ) -> StopReason:
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
            # The loop only ever assigns members of StopReason; cast rather
            # than re-validating a value we just produced ourselves.
            return cast(StopReason, final_stop_reason)
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
        tool_call_id: Optional[str] = None,
        tool_name: str,
        args_json: Optional[str] = None,
        args_sha256: str,
        status: str,
        error_kind: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        """Write a tool_audit row through the SessionStore. Phase 7 step 74.

        ``correlation_id`` prefers the ``request_id`` structlog contextvar
        (set by the HTTP request middleware for each incoming HTTP request)
        and falls back to ``turn_id`` for library-mode callers that run
        without HTTP middleware. Synthesis §3.6 + B5 step 7.

        The call is wrapped in try/except so an audit write failure never
        breaks the orchestrator — tool execution continues regardless.
        """
        ctx = structlog.contextvars.get_contextvars()
        correlation_id: str = ctx.get("request_id") or turn_id

        try:
            await self.store.audit_tool_call(
                correlation_id=correlation_id,
                session_id=session_id,
                turn_id=turn_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                args_sha256=args_sha256,
                args_json=args_json if self._audit_store_args else None,
                status=status,
                error_kind=error_kind,
                duration_ms=duration_ms,
            )
        except Exception as e:
            # Audit write must never break the orchestrator. Log and continue.
            logger.warning(
                "tool_audit.write_failed",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                error=str(e)[:200],
            )


__all__ = ["ChattyAgentOrchestrator"]
