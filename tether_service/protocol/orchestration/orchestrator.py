"""Core orchestration loop.

Drives the model → parser → tool-execution multi-turn loop and emits NDJSON
events. Receives all configuration via dependency injection (no module-level
config reads); per _synthesis.md §4 Phase 2 step 23. The outer ``try/finally``
(added in p2-cleanup) guarantees that any partial assistant text is persisted
on cancellation, client disconnect, or an unexpected exception path. Per A5
orchestrator investigation.

Phase 3 step 36 (this PR, ``p3-lifespan-slim``): the post-stream-error
recovery path delegates classification + reset to
:class:`tether_service.runtime.hw_watchdog.HardwareWatchdog`, replacing
the substring-grep ``is_fatal`` pattern. Synthesis §6 row 13 / §11.3 R21.
"""
from __future__ import annotations

import asyncio
from contextlib import aclosing
from typing import Any, AsyncGenerator, Dict, Optional, TYPE_CHECKING

from tether_service.core.interfaces import (
    ModelProvider,
    SessionStore,
    StreamParser,
    Tool,
)
from tether_service.core.logging import logger
from tether_service.core.types import OrchestratorConfig, StreamEvent
from tether_service.protocol.orchestration.emitter import NdjsonEmitter
from tether_service.protocol.orchestration.tool_runner import ToolRunner

if TYPE_CHECKING:
    from tether_service.runtime.hw_watchdog import HardwareWatchdog


# §13 R5: Interim redaction helper for prompt/args/result payloads.
# Phase 7 will replace this with structlog field-level redaction; keep it
# simple for now — one function, no class hierarchy (R6 anti-overengineering).
def _redact(payload, max_len: int = 120) -> str:
    """Truncate repr(payload) to max_len chars for safe DEBUG logging."""
    s = repr(payload)
    return s if len(s) <= max_len else s[:max_len] + "...[+%dB]" % (len(s) - max_len)


async def orchestrate(
    *,
    session_id: str,
    prompt: str,
    model_name: str,
    provider: ModelProvider,
    parser: StreamParser,
    store: SessionStore,
    tools: Dict[str, Tool],
    system_prompt: str,
    config: OrchestratorConfig,
    tool_runner: ToolRunner,
    cancel_event: Optional[asyncio.Event] = None,
    hw_watchdog: Optional["HardwareWatchdog"] = None,
) -> AsyncGenerator[bytes, None]:
    """Core orchestration: history → provider stream → parser events → store → NDJSON.

    Args:
        config: Typed slice of Settings (max_tool_loops, save_thinking, etc.).
            Built once by ``Engine.from_settings`` and reused across calls.
        tool_runner: Pre-constructed ``ToolRunner``. Engine builds one per
            instance with the configured ``tool_timeout_sec``.
        cancel_event: Optional ``asyncio.Event`` checked after each provider
            chunk. If set, the loop exits early and the ``finally`` block
            persists any in-progress assistant text. Phase 5 will replace
            this with a richer ``CancelToken`` (per connector spec §4).
        hw_watchdog: Optional :class:`HardwareWatchdog`. When the model
            stream raises mid-flight, the watchdog classifies + recovers.
            ``None`` (e.g., direct test invocations) skips recovery —
            the error event is still emitted. Synthesis §4 Phase 3 step 36.
    """
    emitter = NdjsonEmitter()

    # Deliberate cross-iteration carry-over (synthesis §4 Phase 2 step 20):
    # `last_response_text` / `last_thinking_text` mirror the per-iteration
    # accumulators and are NOT reset between tool-loop iterations. If iteration N
    # emits text/thinking and iteration N+1 is cancelled before any deltas
    # arrive, the finally block persists iteration N's text. This is the
    # "rescue any text the user saw on-wire that history would otherwise drop"
    # semantic — internally consistent with
    # test_orchestrator_finally_runs_even_on_unexpected_exception.
    # Future contributors: do NOT move these resets inside the for-loop.
    last_response_text = ""
    last_thinking_text = ""
    text_persisted = False

    try:
        logger.debug(
            f"Orchestration started: session_id={session_id}, model_name={model_name}"
        )
        # 1. Ensure the system prompt is set for the session
        await store.ensure_system_prompt(session_id, system_prompt)
        logger.debug(f"System prompt ensured for session_id={session_id}")

        # 2. Record the user's message
        await store.add_user(session_id, prompt)
        # §13 R5: prompt is PII — redact before logging, demote to DEBUG
        logger.debug(
            f"User message added: session_id={session_id}, prompt={_redact(prompt)}"
        )

        cancelled = False

        # 3. Main loop for multi-turn tool use
        for loop_num in range(config.max_tool_loops):
            logger.info(
                f"Tool loop {loop_num+1}/{config.max_tool_loops} for session_id={session_id}"
            )
            # 4. Get latest history and available tool schemas
            messages = await store.get_history(
                session_id, include_thinking=config.include_thinking_in_history
            )
            tool_schemas = [tool.schema for tool in tools.values()]

            # 5. Stream raw text from the model provider
            full_response_text = ""
            full_thinking_text = ""
            tool_call_to_run = None
            tool_started_notified = False
            # New iteration → reset persisted flag; finally only fires if this
            # iteration didn't reach a successful inner persist site.
            text_persisted = False

            try:
                async with aclosing(
                    provider.stream(
                        model_name=model_name, messages=messages, tools=tool_schemas
                    )
                ) as provider_stream:
                    async for chunk in provider_stream:
                        logger.debug(f"Provider stream chunk: {chunk}")
                        events = parser.feed(chunk)
                        for evt in events:
                            logger.debug(f"Parser event: {evt}")
                            evt_type = evt.get("type")
                            evt_data = evt.get("data", {})

                            if evt_type == StreamEvent.TEXT:
                                delta = evt_data.get("delta", "")
                                if delta:
                                    full_response_text += delta
                                    last_response_text = full_response_text
                                    yield emitter.emit({
                                        "type": "text",
                                        "session_id": session_id,
                                        "data": {"delta": delta},
                                    })

                            elif evt_type == StreamEvent.THINK:
                                delta = evt_data.get("delta", "")
                                if delta:
                                    full_thinking_text += delta
                                    last_thinking_text = full_thinking_text
                                    yield emitter.emit({
                                        "type": "think",
                                        "session_id": session_id,
                                        "data": {"delta": delta},
                                    })

                            elif evt_type == StreamEvent.TOOL_STARTED:
                                # Parser detected <<function_call>> marker
                                logger.info(
                                    f"Tool call marker detected for session_id={session_id}"
                                )
                                if not tool_started_notified:
                                    yield emitter.emit({
                                        "type": "tool_marker_detected",
                                        "session_id": session_id,
                                        "data": {},
                                    })
                                    tool_started_notified = True

                            elif evt_type == StreamEvent.TOOL_COMPLETE:
                                # A tool call has been fully parsed.
                                tool_call_to_run = evt_data
                                logger.info(f"Tool call detected: {tool_call_to_run}")
                                # We break the inner loop to proceed with execution.
                                break

                            elif evt_type == StreamEvent.ERROR:
                                logger.error(f"Parser error: {evt_data}")
                                yield emitter.emit({
                                    "type": "error",
                                    "session_id": session_id,
                                    "data": evt_data,
                                })

                        if tool_call_to_run:
                            break

                        # Cancellation check — granular at the chunk boundary, not
                        # the parser-event boundary (R6 anti-overengineering: too
                        # frequent a check buys nothing on small chunks).
                        if cancel_event is not None and cancel_event.is_set():
                            logger.info(
                                f"Cancellation requested mid-stream for session_id={session_id}"
                            )
                            cancelled = True
                            break

            except Exception as stream_error:
                # Handle model streaming errors (e.g., TVM/OpenCL errors)
                error_msg = str(stream_error)
                error_type = type(stream_error).__name__
                logger.error(
                    f"Model streaming error in loop {loop_num+1}: {error_msg}",
                    exc_info=True,
                )

                # Phase 3 step 36 (synthesis §6 row 13 / §11.3 R21): defer
                # classification + recovery to HardwareWatchdog. The legacy
                # error-message substring grep is gone — providers now own
                # classification via HwErrorClass through the
                # HardwareLifecycle Protocol. is_fatal in the wire event is
                # True iff a recovery (hw_reset) actually fired.
                is_fatal = False
                if hw_watchdog is not None and config.auto_reload_on_fatal_error:
                    try:
                        recovered = await hw_watchdog.reset_after(
                            stream_error, model_name=model_name
                        )
                    except Exception as wd_err:
                        logger.exception(
                            "HardwareWatchdog.reset_after raised: %s", wd_err
                        )
                        recovered = False
                    if recovered:
                        is_fatal = True
                        yield emitter.emit({
                            "type": "info",
                            "session_id": session_id,
                            "data": {
                                "message": (
                                    f"Model '{model_name}' was reset by "
                                    "HardwareWatchdog after fatal error"
                                )
                            },
                        })

                # Send error event to client
                yield emitter.emit({
                    "type": "error",
                    "session_id": session_id,
                    "data": {
                        "message": f"Model streaming failed: {error_msg}",
                        "error_type": error_type,
                        "is_fatal": is_fatal,
                        "recoverable": False,
                    },
                })

                # If we have partial text response, save it now so the outer
                # finally doesn't double-persist.
                if full_response_text or full_thinking_text:
                    await store.add_assistant_text(
                        session_id,
                        full_response_text,
                        thinking_text=full_thinking_text,
                        save_thinking=config.save_thinking,
                    )
                    text_persisted = True
                    logger.info(
                        "Partial assistant text persisted before error: session_id=%s, text_length=%s, thinking_length=%s",
                        session_id,
                        len(full_response_text),
                        len(full_thinking_text),
                    )

                # Exit the tool loop - don't retry on streaming errors
                break

            # Cancellation check at the outer-loop level too; finally handles
            # persistence below.
            if cancelled:
                break

            # 6. After the stream, check if a tool needs to be run
            if tool_call_to_run:
                tool_name = tool_call_to_run.get("tool_name")
                tool_args = tool_call_to_run.get("tool_args", {})

                # Persist the assistant's intent to call the tool
                await store.add_assistant_toolcall(session_id, tool_name, tool_args)
                # §13 R5: tool_args may contain PII — redact before logging, demote to DEBUG
                logger.debug(
                    f"Assistant tool call persisted: session_id={session_id}, tool_name={tool_name}, tool_args={_redact(tool_args)}"
                )
                yield emitter.emit({
                    "type": "tool_started",
                    "session_id": session_id,
                    "data": {"tool_name": tool_name, "tool_args": tool_args},
                })

                # Execute the tool
                try:
                    result = await tool_runner.run(tool_name, tool_args)
                    # §13 R5: result may contain PII — redact before logging, demote to DEBUG
                    logger.debug(f"Tool executed: {tool_name}, result={_redact(result)}")
                    await store.add_tool_result(session_id, tool_name, result)
                    yield emitter.emit({
                        "type": "tool_completed",
                        "session_id": session_id,
                        "data": {"tool_name": tool_name, "tool_result": result},
                    })
                    # Continue the loop to let the model process the tool result
                    continue
                except Exception as e:
                    error_message = f"Error running tool {tool_name}: {e}"
                    logger.exception(error_message)
                    await store.add_tool_result(
                        session_id, tool_name, {"error": error_message}
                    )
                    yield emitter.emit({
                        "type": "tool_error",
                        "session_id": session_id,
                        "data": {"tool_name": tool_name, "error": error_message},
                    })
                    # Break the loop on tool error
                    break
            else:
                # No tool call was made, so persist the final text and exit the loop
                if full_response_text or full_thinking_text:
                    await store.add_assistant_text(
                        session_id,
                        full_response_text,
                        thinking_text=full_thinking_text,
                        save_thinking=config.save_thinking,
                    )
                    text_persisted = True
                    logger.info(
                        "Assistant text persisted: session_id=%s, text_length=%s, thinking_length=%s",
                        session_id,
                        len(full_response_text),
                        len(full_thinking_text),
                    )
                break

        # 7. Finalize the stream
        for evt in parser.finalize() or []:
            logger.debug(f"Parser finalize event: {evt}")
            yield emitter.emit({
                "type": evt.get("type", "text"),
                "session_id": session_id,
                "data": evt.get("data", {}),
            })

    except Exception as e:
        logger.exception(
            f"Exception in orchestrate: session_id={session_id}, error={e}"
        )
        yield emitter.emit({
            "type": "error",
            "session_id": session_id,
            "data": {"message": str(e)},
        })
    finally:
        # Persist any partial text not yet stored. Covers client-disconnect,
        # cancellation, and unexpected exit paths (per A5 orchestrator design).
        # Re-entry guarded by ``text_persisted`` so the inner success / error
        # persist sites don't get double-written.
        if not text_persisted and (last_response_text or last_thinking_text):
            try:
                await store.add_assistant_text(
                    session_id,
                    last_response_text,
                    thinking_text=last_thinking_text,
                    save_thinking=config.save_thinking,
                )
                text_persisted = True
                logger.info(
                    "Partial assistant text persisted in finally: session_id=%s, text_length=%s, thinking_length=%s",
                    session_id,
                    len(last_response_text),
                    len(last_thinking_text),
                )
            except Exception as fin_exc:
                logger.exception(
                    f"Failed to persist partial text in finally: {fin_exc}"
                )

    # Always signal completion (existing wire-protocol guarantee — line 224
    # of the pre-cleanup orchestrator). Stays outside try/finally so a single
    # ``done`` event is emitted regardless of how the body exited.
    logger.info(f"Orchestration complete: session_id={session_id}")
    yield emitter.emit({"type": "done", "session_id": session_id, "data": {}})
