from typing import Any, Dict, Optional

from fastapi import APIRouter, Request

from tether.runtime.abandoned_tasks import get_notebook_abandoned_task_tracker

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


def _operational_health_block() -> dict:
    """Return informational process-health state without affecting readiness."""
    return {
        "notebook_cleanup": get_notebook_abandoned_task_tracker()
        .snapshot()
        .to_dict()
    }


def _apply_hardware_health(
    providers_block: Dict[str, Dict[str, Any]],
    hardware_health: Dict[str, Any],
) -> None:
    """Overlay provider-ID-aware hardware failures onto Engine health state."""
    for entry in hardware_health.get("providers", []):
        provider_id = entry.get("provider_id")
        if provider_id not in providers_block:
            continue
        if entry.get("status") == "error":
            providers_block[provider_id]["healthy"] = False
            providers_block[provider_id]["error"] = "hardware health: error"


@router.get("/readyz")
async def readyz(request: Request):
    """Readiness probe: verifies store and provider(s) are functional.

    ADR-0021 P2.A: response is additive. New top-level keys:

      - ``providers``: ``{pid: {healthy, kind, source, error}}`` from
        ``engine.list_provider_health()``.
      - ``default_provider_id``: the engine's current default.

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
      - Operational health: ``operational_health.notebook_cleanup`` reports
        bounded abandoned-task tracking for diagnostics only. Its state never
        changes readiness or initiates recovery.

    Legacy keys preserved on every response path:

      - ``ready`` — overall flag; now also requires ≥1 healthy provider.
      - ``store`` — DB connectivity.
      - ``provider`` — True iff ≥1 healthy provider.
      - ``hw_health`` — HW watchdog summary (or absent when no watchdog).
      - ``connectors`` / ``connector_start_failures`` — connector layer.
    """
    svc = request.app.state.gen_svc

    # ADR-0021 P2.A: per-provider health block.
    providers_block: Dict[str, Dict[str, Any]] = {}
    default_provider_id: Optional[str] = None
    if hasattr(svc, "list_provider_health"):
        try:
            providers_block = svc.list_provider_health()
        except Exception:  # noqa: BLE001 - defensive
            providers_block = {}
        default_provider_id = getattr(svc, "default_provider_id", None)

    any_healthy_provider = (
        any(p.get("healthy") for p in providers_block.values())
        if providers_block
        else True  # back-compat: legacy engines (no helper)
    )

    body: Dict[str, Any] = {
        "ready": False,
        "store": False,
        "provider": any_healthy_provider,
        "connectors": [],
        "connector_start_failures": [],
        # Informational only — never gates readiness (Phase 9.8 cleanup policy).
        "operational_health": _operational_health_block(),
    }
    if providers_block:
        body["providers"] = providers_block
    if default_provider_id is not None:
        body["default_provider_id"] = default_provider_id

    # Store probe.
    try:
        _ = await svc.store.get_history("_readiness")
    except Exception as e:
        body["ready"] = False
        body["store"] = False
        body["error"] = str(e)
        return body
    body["store"] = True

    # Connectors (best-effort; never 500s readyz).
    connectors_block = await _connector_health_block(svc)
    connector_start_failures = list(
        getattr(svc, "_connector_start_failures", []) or []
    )
    body["connectors"] = connectors_block
    body["connector_start_failures"] = connector_start_failures

    try:
        if getattr(svc, "hw_watchdog", None) is not None:
            health = await svc.hw_watchdog.health_summary()
            body["hw_health"] = health
            has_provider_ids = any(
                entry.get("provider_id") is not None
                for entry in health.get("providers", [])
            )
            if providers_block and has_provider_ids:
                _apply_hardware_health(providers_block, health)
                any_healthy_provider = any(
                    provider.get("healthy")
                    for provider in providers_block.values()
                )
                body["providers"] = providers_block
                body["provider"] = any_healthy_provider
                body["ready"] = (
                    any_healthy_provider and not connector_start_failures
                )
                return body
            if health["overall"] == "error":
                body["ready"] = False
                body["provider"] = any_healthy_provider
                body["error"] = "hw_health: error"
                return body
            body["ready"] = (
                any_healthy_provider
                and not connector_start_failures
            )
            body["provider"] = any_healthy_provider
            return body

        models = svc.provider.list_models()
        if not models:
            body["ready"] = False
            body["provider"] = False
            body["error"] = "no models available"
            return body
        body["ready"] = (
            (any_healthy_provider if providers_block else True)
            and not connector_start_failures
        )
        body["provider"] = (
            any_healthy_provider if providers_block else True
        )
        body["models_available"] = len(models)
        return body
    except Exception as e:
        body["ready"] = False
        body["provider"] = False
        body["error"] = str(e)
        return body
