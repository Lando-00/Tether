from typing import List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from tether.providers.types import ModelDetails

router = APIRouter(prefix="/models", tags=["models"])


class ModelList(BaseModel):
    models: List[str]


class UnloadRequest(BaseModel):
    model_name: str = Field(..., description="The name of the model to unload.")


@router.get("", response_model=List[str])
def list_models(request: Request):
    """List available model identifiers.

    Returns a plain ``list[str]`` for backward compatibility with existing
    CLI / clients. Richer per-model metadata (provider kind, source,
    context window, reasoning capability) lives at ``/models/details``.
    """
    gen_svc = request.app.state.gen_svc
    models = gen_svc.list_models()
    return models


@router.get("/details", response_model=List[ModelDetails])
def list_model_details(request: Request):
    """List per-model capability metadata.

    Companion to ``GET /api/v1/models``: callers that need to know which
    models support reasoning effort or thinking output, what context
    window to expect, or whether the model is hosted remotely should hit
    this endpoint. Backward-compatible: existing clients keep using
    ``GET /models`` (``list[str]``) until they opt into the richer shape.
    """
    gen_svc = request.app.state.gen_svc
    return gen_svc.list_model_info()


@router.post("/unload")
def unload_model(body: UnloadRequest, request: Request):
    """Unload a model from memory."""
    gen_svc = request.app.state.gen_svc
    success = gen_svc.unload_model(body.model_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{body.model_name}' not found or could not be unloaded.")
    return {"success": True, "model_name": body.model_name}
