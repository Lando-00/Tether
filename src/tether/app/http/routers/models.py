from typing import List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/models", tags=["models"])


class ModelList(BaseModel):
    models: List[str]


class UnloadRequest(BaseModel):
    model_name: str = Field(..., description="The name of the model to unload.")


@router.get("", response_model=List[str])
def list_models(request: Request):
    """List available models."""
    gen_svc = request.app.state.gen_svc
    models = gen_svc.list_models()
    return models


@router.get("/details")
def list_model_details(request: Request):
    """List rich model metadata from all healthy providers.

    Returns a JSON array of ModelDetails objects, each carrying
    ``provider_id`` set to the registry key (ADR-0021).
    """
    gen_svc = request.app.state.gen_svc
    try:
        infos = gen_svc.list_model_info()
    except Exception:
        return []
    return [info.model_dump() for info in infos]


@router.post("/unload")
def unload_model(body: UnloadRequest, request: Request):
    """Unload a model from memory."""
    gen_svc = request.app.state.gen_svc
    success = gen_svc.unload_model(body.model_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{body.model_name}' not found or could not be unloaded.")
    return {"success": True, "model_name": body.model_name}
