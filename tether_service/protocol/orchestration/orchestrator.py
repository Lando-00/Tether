from typing import Dict, Any, AsyncGenerator


from tether_service.core.interfaces import ModelProvider, StreamParser, SessionStore, Tool
from tether_service.protocol.orchestration.emitter import NdjsonEmitter
from tether_service.protocol.orchestration.tool_runner import ToolRunner
from tether_service.core.config import load_settings_legacy
from tether_service.core.types import StreamEvent
from tether_service.core.logging import logger


# §13 R5: Interim redaction helper for prompt/args/result payloads.
# Phase 7 will replace this with structlog field-level redaction; keep it
# simple for now — one function, no class hierarchy (R6 anti-overengineering).
def _redact(payload, max_len: int = 120) -> str:
    """Truncate repr(payload) to max_len chars for safe DEBUG logging."""
    s = repr(payload)
    return s if len(s) <= max_len else s[:max_len] + "...[+%dB]" % (len(s) - max_len)


async def orchestrate(
    session_id: str,
    prompt: str,
    model_name: str,
    provider: ModelProvider,
    parser: StreamParser,
    store: SessionStore,
    tools: Dict[str, Tool],
    system_prompt: str,
) -> AsyncGenerator[bytes, None]:
    """
    Core orchestration: history -> provider stream -> parser events -> store -> NDJSON.
    Manages a multi-turn loop for tool execution.
    """
    emitter = NdjsonEmitter()
    settings = load_settings_legacy()
    limits = settings.get("limits", {})
    max_tool_loops = limits.get("max_tool_loops", 3)
    auto_reload_on_fatal_error = limits.get("auto_reload_on_fatal_error", False)
    context_settings = settings.get("context", {})
    save_thinking = context_settings.get("save_thinking", True)
    include_thinking_in_history = context_settings.get(
        "include_thinking_in_history", False
    )

    # The tool_runner is now created with the tools dict directly.
    tool_runner = ToolRunner(tools)

    try:
        logger.debug(f"Orchestration started: session_id={session_id}, model_name={model_name}")
        # 1. Ensure the system prompt is set for the session
        await store.ensure_system_prompt(session_id, system_prompt)
        logger.debug(f"System prompt ensured for session_id={session_id}")

        # 2. Record the user's message
        await store.add_user(session_id, prompt)
        # §13 R5: prompt is PII — redact before logging, demote to DEBUG
        logger.debug(f"User message added: session_id={session_id}, prompt={_redact(prompt)}")

        # 3. Main loop for multi-turn tool use
        for loop_num in range(max_tool_loops):
            logger.info(f"Tool loop {loop_num+1}/{max_tool_loops} for session_id={session_id}")
            # 4. Get latest history and available tool schemas
            messages = await store.get_history(
                session_id, include_thinking=include_thinking_in_history
            )
            tool_schemas = [tool.schema for tool in tools.values()]

            # 5. Stream raw text from the model provider
            full_response_text = ""
            full_thinking_text = ""
            tool_call_to_run = None
            tool_started_notified = False

            try:
                async for chunk in provider.stream(model_name=model_name, messages=messages, tools=tool_schemas):
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
                                yield emitter.emit({"type": "text", "session_id": session_id, "data": {"delta": delta}})
                        
                        elif evt_type == StreamEvent.THINK:
                            delta = evt_data.get("delta", "")
                            if delta:
                                full_thinking_text += delta
                                yield emitter.emit({"type": "think", "session_id": session_id, "data": {"delta": delta}})
                        
                        elif evt_type == StreamEvent.TOOL_STARTED:
                            # Parser detected <<function_call>> marker
                            logger.info(f"Tool call marker detected for session_id={session_id}")
                            if not tool_started_notified:
                                yield emitter.emit({"type": "tool_marker_detected", "session_id": session_id, "data": {}})
                                tool_started_notified = True

                        elif evt_type == StreamEvent.TOOL_COMPLETE:
                            # A tool call has been fully parsed.
                            tool_call_to_run = evt_data
                            logger.info(f"Tool call detected: {tool_call_to_run}")
                            # We break the inner loop to proceed with execution.
                            break
                        
                        elif evt_type == StreamEvent.ERROR:
                            logger.error(f"Parser error: {evt_data}")
                            yield emitter.emit({"type": "error", "session_id": session_id, "data": evt_data})
                    
                    if tool_call_to_run:
                        break
            
            except Exception as stream_error:
                # Handle model streaming errors (e.g., TVM/OpenCL errors)
                error_msg = str(stream_error)
                error_type = type(stream_error).__name__
                logger.error(f"Model streaming error in loop {loop_num+1}: {error_msg}", exc_info=True)
                
                # Check if this is a fatal error (OpenCL/TVM)
                is_fatal = "TVMError" in error_type or "CLML" in error_msg or "CL_" in error_msg
                
                # Attempt model recovery if enabled and it's a fatal error
                if is_fatal and auto_reload_on_fatal_error and hasattr(provider, 'unload_model'):
                    logger.warning(f"Fatal error detected, attempting to unload model {model_name} for recovery")
                    try:
                        provider.unload_model(model_name)
                        yield emitter.emit({
                            "type": "info",
                            "session_id": session_id,
                            "data": {"message": "Model unloaded due to fatal error. It will be reloaded on next request."}
                        })
                    except Exception as unload_err:
                        logger.error(f"Failed to unload model: {unload_err}")
                
                # Send error event to client
                yield emitter.emit({
                    "type": "error",
                    "session_id": session_id,
                    "data": {
                        "message": f"Model streaming failed: {error_msg}",
                        "error_type": error_type,
                        "is_fatal": is_fatal,
                        "recoverable": False
                    }
                })
                
                # If we have partial text response, save it
                if full_response_text or full_thinking_text:
                    await store.add_assistant_text(
                        session_id,
                        full_response_text,
                        thinking_text=full_thinking_text,
                        save_thinking=save_thinking,
                    )
                    logger.info(
                        "Partial assistant text persisted before error: session_id=%s, text_length=%s, thinking_length=%s",
                        session_id,
                        len(full_response_text),
                        len(full_thinking_text),
                    )
                
                # Exit the tool loop - don't retry on streaming errors
                break
            
            # 6. After the stream, check if a tool needs to be run
            if tool_call_to_run:
                tool_name = tool_call_to_run.get("tool_name")
                tool_args = tool_call_to_run.get("tool_args", {})

                # Persist the assistant's intent to call the tool
                await store.add_assistant_toolcall(session_id, tool_name, tool_args)
                # §13 R5: tool_args may contain PII — redact before logging, demote to DEBUG
                logger.debug(f"Assistant tool call persisted: session_id={session_id}, tool_name={tool_name}, tool_args={_redact(tool_args)}")
                yield emitter.emit({"type": "tool_started", "session_id": session_id, "data": {"tool_name": tool_name, "tool_args": tool_args}})

                # Execute the tool
                try:
                    result = await tool_runner.run(tool_name, tool_args)
                    # §13 R5: result may contain PII — redact before logging, demote to DEBUG
                    logger.debug(f"Tool executed: {tool_name}, result={_redact(result)}")
                    await store.add_tool_result(session_id, tool_name, result)
                    yield emitter.emit({"type": "tool_completed", "session_id": session_id, "data": {"tool_name": tool_name, "tool_result": result}})
                    # Continue the loop to let the model process the tool result
                    continue
                except Exception as e:
                    error_message = f"Error running tool {tool_name}: {e}"
                    logger.exception(error_message)
                    await store.add_tool_result(session_id, tool_name, {"error": error_message})
                    yield emitter.emit({"type": "tool_error", "session_id": session_id, "data": {"tool_name": tool_name, "error": error_message}})
                    # Break the loop on tool error
                    break
            else:
                # No tool call was made, so persist the final text and exit the loop
                if full_response_text or full_thinking_text:
                    await store.add_assistant_text(
                        session_id,
                        full_response_text,
                        thinking_text=full_thinking_text,
                        save_thinking=save_thinking,
                    )
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
            yield emitter.emit({"type": evt.get("type", "text"), "session_id": session_id, "data": evt.get("data", {})})

    except Exception as e:
        logger.exception(f"Exception in orchestrate: session_id={session_id}, error={e}")
        yield emitter.emit({"type": "error", "session_id": session_id, "data": {"message": str(e)}})

    # Always signal completion
    logger.info(f"Orchestration complete: session_id={session_id}")
    yield emitter.emit({"type": "done", "session_id": session_id, "data": {}})
