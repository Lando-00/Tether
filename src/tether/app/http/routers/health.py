from typing import Any, Dict, Optional

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
    """Readiness probe: verifies store and provider(s) are functional.

    Phase 3 step 37 (synthesis §6 row 2 / B6 §1.2 #4 / §4 Phase 3): the
    HW-provider health check is driven by ``HardwareWatchdog.health_summary()``,
    which aggregates ``HardwareLifecycle.hw_health()`` across providers.

    ADR-0021 P2.A: response is additive. New top-level keys:

      - ``providers``: ``{pid: {healthy, kind, source, error}}`` from
        ``engine.list_provider_health()``. Includes BOTH healthy entries
        and ids in ``_provider_start_failures``.
      - ``default_provider_id``: the engine's current default.

    Legacy keys preserved verbatim:

      - ``ready`` — overall flag; now also requires ≥1 healthy provider.
      - ``store`` — DB connectivity.
      - ``provider`` — True iff ≥1 healthy provider (supervisors that
        key on this bool keep working).
      - ``hw_health`` — HW watchdog summary (or absent when no watchdog).
      - ``connectors`` / ``connector_start_failures`` — connector layer.
    """
    svc = request.app.state.gen_svc
    try:
        _ = await svc.store.get_history("_readiness")
    except Exception as e:
        return {"ready": False, "store": False, "provider": None, "error": str(e)}

    connectors_block = await _connector_health_block(svc)
    connector_start_failures = list(
        getattr(svc, "_connector_start_failures", []) or []
    )

    # ADR-0021 P2.A: per-provider health block. ``list_provider_health``
    # is sync, cheap, and never raises (defensive in the engine).
    # Engines built via the legacy singular shim expose a single
    # ``{"default": {...}}`` entry, so single-provider deployments stay
    # visually consistent with their CLI tables.
    providers_block: Dict[str, Dict[str, Any]] = {}
    default_provider_id: Optional[str] = None
    if hasattr(svc, "list_provider_health"):
        try:
            providers_block = svc.list_provider_health()
        except Exception:  # noqa: BLE001 - defensive
            providers_block = {}
        default_provider_id = getattr(svc, "default_provider_id", None)

    any_healthy_provider = any(
        p.get("healthy") for p in providers_block.values()
    ) if providers_block else True  # back-compat: legacy engines (no helper)

    try:
        if getattr(svc, "hw_watchdog", None) is not None:
            health = await svc.hw_watchdog.health_summary()
            if health["overall"] == "error":
                body: Dict[str, Any] = {
                    "ready": False,
                    "store": True,
                    "provider": False,
                    "error": "hw_health: error",
                    "hw_health": health,
                    "connectors": connectors_block,
                    "connector_start_failures": connector_start_failures,
                }
                if providers_block:
                    body["providers"] = providers_block
                if default_provider_id is not None:
                    body["default_provider_id"] = default_provider_id
                return body
            ready = (
                any_healthy_provider
                and not connector_start_failures
            )
            body = {
                "ready": ready,
                "store": True,
                "provider": any_healthy_provider,
                "hw_health": health,
                "connectors": connectors_block,
                "connector_start_failures": connector_start_failures,
            }
            if providers_block:
                body["providers"] = providers_block
            if default_provider_id is not None:
                body["default_provider_id"] = default_provider_id
            return body

        models = svc.provider.list_models()
        if not models:
            body = {
                "ready": False,
                "store": True,
                "provider": False,
                "error": "no models available",
                "connectors": connectors_block,
                "connector_start_failures": connector_start_failures,
            }
            if providers_block:
                body["providers"] = providers_block
            if default_provider_id is not None:
                body["default_provider_id"] = default_provider_id
            return body
        body = {
            "ready": (
                (any_healthy_provider if providers_block else True)
                and not connector_start_failures
            ),
            "store": True,
            "provider": (
                any_healthy_provider if providers_block else True
            ),
            "models_available": len(models),
            "connectors": connectors_block,
            "connector_start_failures": connector_start_failures,
        }
        if providers_block:
            body["providers"] = providers_block
        if default_provider_id is not None:
            body["default_provider_id"] = default_provider_id
        return body
    except Exception as e:
        return {"ready": False, "store": True, "provider": False, "error": str(e)}
