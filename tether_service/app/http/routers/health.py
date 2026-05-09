from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz():
    return {"ok": True}


@router.get("/readyz")
async def readyz(request: Request):
    """Readiness probe: verifies store and provider are functional.

    Phase 3 step 37 (synthesis §6 row 2 / B6 §1.2 #4 / §4 Phase 3): the
    provider check is now driven by ``HardwareWatchdog.health_summary()``,
    which aggregates ``HardwareLifecycle.hw_health()`` across providers.

    Behavior:
      - Store: read history for a sentinel session (exercises DB connectivity).
      - Provider (HW path): when ``engine.hw_watchdog`` is present (always
        true for engines built via ``Engine.from_settings``), aggregate health
        across HW providers. ``error`` → ready=false. ``healthy`` /
        ``degraded`` → ready=true (cold-cache MLC providers report
        ``degraded`` until a model is loaded; that's normal).
      - Provider (fallback): no watchdog (engine constructed directly without
        ``from_settings``) → fall back to ``list_models()``.

    A streaming probe is still avoided: MLC engines may take 5–60s on cold
    cache. Health is reported by counting cached entries, not loading.
    """
    svc = request.app.state.gen_svc
    try:
        _ = await svc.store.get_history("_readiness")
    except Exception as e:
        return {"ready": False, "store": False, "provider": None, "error": str(e)}

    try:
        if getattr(svc, "hw_watchdog", None) is not None:
            health = await svc.hw_watchdog.health_summary()
            if health["overall"] == "error":
                return {
                    "ready": False,
                    "store": True,
                    "provider": False,
                    "error": "hw_health: error",
                    "hw_health": health,
                }
            return {
                "ready": True,
                "store": True,
                "provider": True,
                "hw_health": health,
            }

        models = svc.provider.list_models()
        if not models:
            return {
                "ready": False,
                "store": True,
                "provider": False,
                "error": "no models available",
            }
        return {
            "ready": True,
            "store": True,
            "provider": True,
            "models_available": len(models),
        }
    except Exception as e:
        return {"ready": False, "store": True, "provider": False, "error": str(e)}
