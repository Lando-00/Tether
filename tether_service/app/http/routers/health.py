from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz():
    return {"ok": True}


@router.get("/readyz")
async def readyz(request: Request):
    """
    Readiness probe: verifies store and provider are functional.

    Synthesis §6 row 2 / B6 §1.2 #4:
    - Store: can read history for a sentinel session (exercises DB connectivity).
    - Provider: list_models() returns at least one model name (cheap; no inference).

    Phase 3 will replace the provider check with HardwareWatchdog.hw_health()
    once the watchdog is available. We avoid a streaming probe here because MLC
    engines may take 5-60s on cold cache, which always exceeds a 1-second timeout.
    """
    svc = request.app.state.gen_svc
    # Store check
    try:
        _ = await svc.store.get_history("_readiness")
    except Exception as e:
        return {"ready": False, "store": False, "provider": None, "error": str(e)}

    # Provider check: list_models() is synchronous and cheap — no model loading.
    # Synthesis §6 row 2: use list_models() instead of streaming probe.
    try:
        models = svc.provider.list_models()
        if not models:
            return {"ready": False, "store": True, "provider": False, "error": "no models available"}
        return {"ready": True, "store": True, "provider": True, "models_available": len(models)}
    except Exception as e:
        return {"ready": False, "store": True, "provider": False, "error": str(e)}
