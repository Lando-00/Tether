from typing import List
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/sessions", tags=["sessions"])


class Session(BaseModel):
    session_id: str = Field(..., description="The unique identifier for the session.")
    created_at: str = Field(..., description="The timestamp when the session was created.")


class Message(BaseModel):
    role: str
    content: str


@router.post("", response_model=Session)
async def create_session(request: Request):
    """Create a new session."""
    gen_svc = request.app.state.gen_svc
    session_details = await gen_svc.create_session()
    return session_details


@router.get("", response_model=List[Session])
async def list_sessions(request: Request):
    """List all sessions."""
    gen_svc = request.app.state.gen_svc
    sessions = await gen_svc.list_sessions()
    return sessions


@router.get("/{session_id}/messages", response_model=List[Message])
async def get_session_messages(session_id: str, request: Request):
    """Get messages for a session."""
    gen_svc = request.app.state.gen_svc
    messages = await gen_svc.get_session_messages(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Session not found or has no messages")
    return messages


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request):
    """Delete a session by ID."""
    gen_svc = request.app.state.gen_svc
    success = await gen_svc.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {}


@router.delete("", status_code=200)
async def delete_all_sessions(request: Request):
    """Delete all sessions."""
    gen_svc = request.app.state.gen_svc
    count = await gen_svc.delete_all_sessions()
    return {"deleted_count": count}
