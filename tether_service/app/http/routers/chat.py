
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from tether_service.core.logging import logger

router = APIRouter(prefix="/chat", tags=["chat"])

class StreamRequest(BaseModel):
    session_id: str = Field(..., description="The unique identifier for the session.")
    prompt: str = Field(..., description="The user's prompt.")
    model_name: str = Field(..., description="The name of the model to use for this generation.")


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
            logger.exception(f"Exception in /chat/stream: {e}")
            raise

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
