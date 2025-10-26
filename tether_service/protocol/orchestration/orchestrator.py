from typing import Dict, Any, AsyncGenerator


from tether_service.core.interfaces import ModelProvider, StreamParser, SessionStore, Tool
from tether_service.protocol.orchestration.emitter import NdjsonEmitter
from tether_service.protocol.orchestration.tool_runner import ToolRunner
from tether_service.core.config import load_settings
from tether_service.core.types import StreamEvent
from tether_service.core.logging import logger


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
    settings = load_settings()
    limits = settings.get("limits", {})
    max_tool_loops = limits.get("max_tool_loops", 3)

    # The tool_runner is now created with the tools dict directly.
    tool_runner = ToolRunner(tools)

    try:
        logger.info(f"Orchestration started: session_id={session_id}, model_name={model_name}")
        # 1. Ensure the system prompt is set for the session
        await store.ensure_system_prompt(session_id, system_prompt)
        logger.info(f"System prompt ensured for session_id={session_id}")

        # 2. Record the user's message
        await store.add_user(session_id, prompt)
        logger.info(f"User message added: session_id={session_id}, prompt={prompt}")

        # 3. Main loop for multi-turn tool use
        for loop_num in range(max_tool_loops):
            logger.info(f"Tool loop {loop_num+1}/{max_tool_loops} for session_id={session_id}")
            # 4. Get latest history and available tool schemas
            messages = await store.get_history(session_id)
            tool_schemas = [tool.schema for tool in tools.values()]

            # 5. Stream raw text from the model provider
            full_response_text = ""
            tool_call_to_run = None
            tool_started_notified = False

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
            
            # 6. After the stream, check if a tool needs to be run
            if tool_call_to_run:
                tool_name = tool_call_to_run.get("tool_name")
                tool_args = tool_call_to_run.get("tool_args", {})

                # Persist the assistant's intent to call the tool
                await store.add_assistant_toolcall(session_id, tool_name, tool_args)
                logger.info(f"Assistant tool call persisted: session_id={session_id}, tool_name={tool_name}, tool_args={tool_args}")
                yield emitter.emit({"type": "tool_started", "session_id": session_id, "data": {"tool_name": tool_name, "tool_args": tool_args}})

                # Execute the tool
                try:
                    result = await tool_runner.run(tool_name, tool_args)
                    logger.info(f"Tool executed: {tool_name}, result={result}")
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
                if full_response_text:
                    await store.add_assistant_text(session_id, full_response_text)
                    logger.info(f"Assistant text persisted: session_id={session_id}, text_length={len(full_response_text)}")
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
