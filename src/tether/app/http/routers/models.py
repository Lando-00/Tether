from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from tether.core.errors import (
    AmbiguousModelError,
    ProviderUnhealthyError,
    UnknownModelError,
    UnknownProviderError,
)
from tether.core.provider_ids import PROVIDER_ID_PATTERN

router = APIRouter(prefix="/models", tags=["models"])


class ModelList(BaseModel):
    models: List[str]


class UnloadRequest(BaseModel):
    model_name: str = Field(..., description="The name of the model to unload.")
    provider_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional explicit provider routing key. Required when multiple providers advertise the same model."
        ),
        pattern=PROVIDER_ID_PATTERN,
    )


def _raise_routing_http_error(exc: Exception) -> None:
    if isinstance(exc, ProviderUnhealthyError):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Provider '{exc.provider_id}' is currently unavailable. "
                "Query /api/v1/readyz for the per-provider health map."
            ),
        ) from exc
    if isinstance(exc, UnknownProviderError):
        raise HTTPException(
            status_code=422,
            detail=f"Unknown provider_id '{exc.provider_id}'.",
        ) from exc
    if isinstance(exc, UnknownModelError):
        suffix = f" on provider '{exc.provider_id}'" if exc.provider_id is not None else ""
        raise HTTPException(
            status_code=422,
            detail=f"Model '{exc.model_name}' is not available{suffix}.",
        ) from exc
    if isinstance(exc, AmbiguousModelError):
        raise HTTPException(
            status_code=422,
            detail=(f"Model '{exc.model_name}' is available from multiple providers; specify provider_id."),
        ) from exc
    raise exc


@router.get("", response_model=List[str])
def list_models(request: Request):
    """List de-duplicated raw model IDs from healthy providers.

    Use ``/models/details`` to pair an ID with its provider when the same
    model is available from multiple providers.
    """
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
    try:
        if body.provider_id is None:
            success = gen_svc.unload_model(body.model_name)
        else:
            success = gen_svc.unload_model(
                body.model_name,
                provider_id=body.provider_id,
            )
    except (
        AmbiguousModelError,
        ProviderUnhealthyError,
        UnknownModelError,
        UnknownProviderError,
    ) as exc:
        _raise_routing_http_error(exc)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{body.model_name}' not found or could not be unloaded.")
    return {
        "success": True,
        "model_name": body.model_name,
        "provider_id": body.provider_id,
    }
