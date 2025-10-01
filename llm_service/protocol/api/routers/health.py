"""
Router for health check endpoints.
"""
from fastapi import APIRouter


def get_health_router():
    """
    Creates a router for health check endpoints.
    
    Returns:
        FastAPI APIRouter for health check endpoints
    """
    router = APIRouter(tags=["health"])
    
    @router.get("/healthz")
    def health_check():
        """Simple health check endpoint."""
        return {"status": "ok"}
    
    return router