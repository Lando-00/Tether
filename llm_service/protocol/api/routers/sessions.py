"""
Router for session-related endpoints.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException

from llm_service.protocol.api.schemas import CreateSessionResponse, Message
from llm_service.protocol.service.protocol_service import ProtocolService


def get_sessions_router(protocol_service_provider):
    """
    Creates a router for session-related endpoints.
    
    Args:
        protocol_service_provider: Callable that returns a ProtocolService instance
        
    Returns:
        FastAPI APIRouter for session-related endpoints
    """
    router = APIRouter(prefix="/sessions", tags=["sessions"])
    
    @router.post("", response_model=CreateSessionResponse)
    def create_session(
        protocol: ProtocolService = Depends(protocol_service_provider)
    ):
        """Create a new session."""
        result = protocol.create_session()
        return CreateSessionResponse(
            session_id=result["session_id"],
            created_at=result["created_at"]
        )
    
    @router.get("")
    def list_sessions(
        protocol: ProtocolService = Depends(protocol_service_provider)
    ):
        """List all sessions."""
        return protocol.list_sessions()
    
    @router.delete("/{session_id}")
    def delete_session(
        session_id: str,
        protocol: ProtocolService = Depends(protocol_service_provider)
    ):
        """Delete a session by ID."""
        if protocol.delete_session(session_id):
            return {"detail": "Session deleted"}
        raise HTTPException(status_code=404, detail="Session not found")
    
    @router.delete("")
    def delete_all_sessions(
        protocol: ProtocolService = Depends(protocol_service_provider)
    ):
        """Delete all sessions."""
        count = protocol.delete_all_sessions()
        return {"detail": f"Deleted {count} sessions."}
    
    @router.get("/{session_id}/messages", response_model=List[Message])
    def get_session_messages(
        session_id: str,
        protocol: ProtocolService = Depends(protocol_service_provider)
    ):
        """Get messages for a session."""
        messages = protocol.get_session_messages(session_id)
        if not messages:
            raise HTTPException(status_code=404, detail="Session not found")
        return [
            Message(
                id=m["id"],
                role=m["role"],
                content=m["content"],
                created_at=m["created_at"]
            ) for m in messages
        ]
    
    return router