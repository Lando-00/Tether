
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal
import asyncio
import json
from tether_service.core.logging import logger

router = APIRouter(prefix="/chat", tags=["chat"])

class StreamRequest(BaseModel):
    session_id: str = Field(
        ...,
        description="The unique identifier for the session.",
        pattern=r"^[A-Za-z0-9_-]{1,128}$",
    )
    prompt: str = Field(
        ...,
        description="The user's prompt.",
        min_length=1,
        max_length=32768,
    )
    model_name: str = Field(
        ...,
        description="The name of the model to use for this generation.",
        pattern=r"^[A-Za-z0-9._-]{1,128}$",
    )
    mode: Literal["chat", "research"] = Field(
        default="chat",
        description=(
            "Orchestrator mode. 'chat' uses ChattyAgentOrchestrator; "
            "'research' uses NotebookOrchestrator (currently 501 Not Implemented). "
            "Honored on both Accept: text/event-stream (SSE) and "
            "application/x-ndjson (NDJSON back-compat) responses."
        ),
    )


@router.post("/stream")
async def stream(request: Request, body: StreamRequest):
    """Stream chat events. Content negotiation via Accept header:

        text/event-stream     -> SSE-framed v2 events (typed WireEvent)
        application/x-ndjson  -> NDJSON v0 dict events (default; back-compat)
        (absent)              -> NDJSON v0 dict events (default)

    Response carries X-Tether-Protocol-Version: 1.0 on every path.

    Mode dispatch: body.mode selects the Orchestrator via the registry
    (settings.orchestrator.registry). Unimplemented modes (is_implemented=False)
    return 501 before any streaming begins. Unknown modes rejected by
    Pydantic Literal with 422. Briefing §2 Seam B item 4; synthesis §3.5.

    Synthesis §4 Phase 5 step 53.
    """
    accept = request.headers.get("accept", "")
    use_sse = "text/event-stream" in accept.lower()

    logger.info(
        f"/chat/stream called: session_id={body.session_id}, "
        f"model_name={body.model_name}, mode={body.mode}, sse={use_sse}"
    )

    engine = request.app.state.gen_svc
    headers = {"X-Tether-Protocol-Version": "1.0"}

    # Eagerly resolve the Orchestrator class to return 501 before streaming
    # begins if the mode is a stub (is_implemented=False). Pydantic's Literal
    # already rejects unknown modes with 422, so this branch handles only the
    # known-but-unimplemented case. Briefing §2 Seam B item 4.
    from tether_service.protocol.orchestration.registry import (
        UnknownOrchestratorMode,
        resolve_orchestrator_class,
    )
    try:
        orchestrator_cls = resolve_orchestrator_class(
            body.mode, engine._orchestrator_registry
        )
    except UnknownOrchestratorMode as exc:
        # Defensive: Pydantic Literal rejects unknowns at validation time,
        # so this branch is reached only when Pydantic is bypassed (library
        # mode). 400 is appropriate here.
        raise HTTPException(status_code=400, detail=str(exc))

    if not orchestrator_cls.is_implemented:
        raise HTTPException(
            status_code=501,
            detail="research mode tracked in docs/research/06_context_strategies.md",
        )

    if use_sse:
        # SSE path: Engine.chat() yields typed WireEvents; transport_sse frames them.
        # Lazy import keeps tether_service importable without triggering the full
        # transport module graph (library-first invariant). Synthesis §3.4.
        from tether_service.protocol.orchestration.cancel import AsyncEventCancelToken
        from tether_service.protocol.wire.transport_sse import transport_sse

        async def sse_generator():
            cancel_token = AsyncEventCancelToken()
            try:
                async def cancellable_chat():
                    async for event in engine.chat(
                        session_id=body.session_id,
                        prompt=body.prompt,
                        model_name=body.model_name,
                        mode=body.mode,
                        cancel_token=cancel_token,
                    ):
                        if await request.is_disconnected():
                            logger.info(
                                f"Client disconnected (SSE): session_id={body.session_id}"
                            )
                            cancel_token.set()
                            break
                        yield event

                async for chunk in transport_sse(cancellable_chat()):
                    yield chunk
            except Exception as e:
                logger.exception(f"Exception in /chat/stream (SSE): {e}")
                error_payload = {
                    "type": "error",
                    "message": f"Streaming error: {str(e)}",
                    "error_type": type(e).__name__,
                }
                yield (
                    f"event: error\ndata: {json.dumps(error_payload)}\n\n"
                ).encode("utf-8")

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers=headers,
        )

    # Default: NDJSON v0 dict bytes (back-compat; p5-cutover-a-dual-emit flips this).
    async def ndjson_generator():
        cancel_event = asyncio.Event()
        try:
            async for chunk in engine.stream(
                session_id=body.session_id,
                prompt=body.prompt,
                model_name=body.model_name,
                mode=body.mode,
                cancel_event=cancel_event,
            ):
                if await request.is_disconnected():
                    logger.info(
                        f"Client disconnected (NDJSON): session_id={body.session_id}"
                    )
                    cancel_event.set()
                    break
                yield chunk
        except Exception as e:
            logger.exception(f"Exception in /chat/stream (NDJSON): {e}")
            error_event = {
                "type": "error",
                "session_id": body.session_id,
                "data": {
                    "message": f"Streaming error: {str(e)}",
                    "error_type": type(e).__name__,
                },
                "ts": None,
            }
            yield (json.dumps(error_event) + "\n").encode("utf-8")

    return StreamingResponse(
        ndjson_generator(),
        media_type="application/x-ndjson",
        headers=headers,
    )
