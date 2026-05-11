from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz():
    return {"ok": True}


async def _connector_health_block(svc) -> list:
    """Build the ``connectors`` array for /readyz.

    Phase 4.5 step 47e (synthesis §4): inline iteration over
    ``connector_registry.all()`` instead of adding a new method to the
    registry — keeps the registry surface tight (R6 anti-overengineering).
    Each connector's ``health()`` is contractually cheap (no network calls
    per connector spec §3.1); we still defensively catch so one bad
    connector cannot 500 readyz.
    """
    registry = getattr(svc, "connector_registry", None)
    if registry is None:
        return []
    out = []
    for conn in registry.all():
        try:
            h = await conn.health()
            out.append(
                {
                    "id": conn.id,
                    "state": h.state.value,
                    "detail": h.detail,
                }
            )
        except Exception as exc:  # noqa: BLE001 - per-connector defensive
            out.append(
                {"id": conn.id, "state": "error", "detail": str(exc)}
            )
    return out


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
      - Connectors (Phase 4.5 step 47e): when ``engine.connector_registry``
        is present, append a ``connectors`` array with each connector's
        ``{id, state, detail}`` snapshot. Connector failures do NOT flip
        ``ready`` — connectors in UNCONFIGURED / LOGGED_OUT / ERROR are an
        expected steady state until the user runs the login flow (connector
        spec §3.3).
      - Connector start failures (P0-F / Tribunal P0-07 / A2-F2): when a
        connector that was READY at config time raised from ``start()``
        during ``Engine.__aenter__``, the failing id is included in the
        ``connector_start_failures`` array and ``ready`` is set to false
        so process supervisors can take action. The response remains 200.

    A streaming probe is still avoided: MLC engines may take 5–60s on cold
    cache. Health is reported by counting cached entries, not loading.
    """
    svc = request.app.state.gen_svc
    try:
        _ = await svc.store.get_history("_readiness")
    except Exception as e:
        return {"ready": False, "store": False, "provider": None, "error": str(e)}

    connectors_block = await _connector_health_block(svc)

    # P0-F / Tribunal P0-07 (A2-F2): if any connector that was READY at
    # config time failed to start, Engine.__aenter__ will have already
    # removed it from the registry and recorded its id here. Surface as
    # ready=false on /readyz so process supervisors can take action.
    connector_start_failures = list(
        getattr(svc, "_connector_start_failures", []) or []
    )

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
                    "connectors": connectors_block,
                    "connector_start_failures": connector_start_failures,
                }
            return {
                "ready": not connector_start_failures,
                "store": True,
                "provider": True,
                "hw_health": health,
                "connectors": connectors_block,
                "connector_start_failures": connector_start_failures,
            }

        models = svc.provider.list_models()
        if not models:
            return {
                "ready": False,
                "store": True,
                "provider": False,
                "error": "no models available",
                "connectors": connectors_block,
                "connector_start_failures": connector_start_failures,
            }
        return {
            "ready": not connector_start_failures,
            "store": True,
            "provider": True,
            "models_available": len(models),
            "connectors": connectors_block,
            "connector_start_failures": connector_start_failures,
        }
    except Exception as e:
        return {"ready": False, "store": True, "provider": False, "error": str(e)}
