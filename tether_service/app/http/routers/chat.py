
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
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


@router.post("/stream")
async def stream(request: Request, body: StreamRequest):
    logger.info(f"/chat/stream called: session_id={body.session_id}, model_name={body.model_name}")
    gen_service = request.app.state.gen_svc

    async def event_generator():
        try:
            async for chunk in gen_service.stream(
                session_id=body.session_id,
                prompt=body.prompt,
                model_name=body.model_name,
            ):
                # Check disconnect BEFORE yielding
                if await request.is_disconnected():
                    logger.info(f"Client disconnected: session_id={body.session_id}")
                    break
                yield chunk
        except Exception as e:
            # Log and send error event to client
            logger.exception(f"Exception in /chat/stream for session {body.session_id}: {e}")
            
            # Send error as NDJSON event
            error_event = {
                "type": "error",
                "session_id": body.session_id,
                "data": {
                    "message": f"Streaming error: {str(e)}",
                    "error_type": type(e).__name__
                },
                "ts": None  # Will be added by emitter if needed
            }
            yield (json.dumps(error_event) + "\n").encode("utf-8")
            
            # Don't re-raise - we've handled it gracefully

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
