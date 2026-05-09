"""Connector lifecycle HTTP routes (per connector spec §3.8).

Mounted under ``/api/v1/connectors`` in :func:`create_app`; six routes
matching the spec table verbatim:

* ``GET    /``                    — list every connector + state.
* ``GET    /{id}/inbox``          — 501 stub (Phase 6.5 lands SqliteInbox).
* ``POST   /{id}/login/begin``    — :meth:`Connector.begin_login`.
* ``POST   /{id}/login/complete`` — :meth:`Connector.complete_login`;
                                    on READY, ``start_connector(id)``.
* ``GET    /{id}/oauth/callback`` — OAuth redirect target; consumes
                                    ``state`` from registry.oauth_state,
                                    forwards to ``complete_login``;
                                    on READY, ``start_connector(id)``.
* ``POST   /{id}/logout``         — :meth:`Connector.logout`.

The OAuth callback route returns a JSON body for now (matching the
``login/complete`` shape) — Phase 2a/2b WhatsApp/Gmail sessions will
refine this to a friendly redirect to a UI page.

References: connector spec §3.8; synthesis §4 Phase 4.5 step 47e.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from tether_service.connectors.base import Connector
from tether_service.connectors.types import (
    AuthStatus,
    ConnectorState,
    HealthStatus,
    LoginContinueResult,
    LoginPrompt,
)
from tether_service.core.connector_registry import ConnectorRegistry

router = APIRouter(prefix="/connectors", tags=["connectors"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_registry(request: Request) -> ConnectorRegistry:
    """Pull the ConnectorRegistry off ``app.state.gen_svc``.

    503 when the engine was constructed without a registry (e.g. a test
    using the legacy direct-constructor path); 503 not 500 because the
    HTTP surface is structurally not configured for connectors.
    """
    svc = request.app.state.gen_svc
    registry: Optional[ConnectorRegistry] = getattr(svc, "connector_registry", None)
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="Connector registry not configured on this engine",
        )
    return registry


def _resolve(registry: ConnectorRegistry, connector_id: str) -> Connector:
    """Translate :meth:`ConnectorRegistry.get` ``KeyError`` into a 404."""
    try:
        return registry.get(connector_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Connector not found: {connector_id!r}",
        )


def _serialize_health(h: HealthStatus) -> Dict[str, Any]:
    return {
        "state": h.state.value,
        "detail": h.detail,
        "last_success": h.last_success.isoformat() if h.last_success else None,
        "last_error": h.last_error.isoformat() if h.last_error else None,
        "last_error_message": h.last_error_message,
    }


def _serialize_auth(a: AuthStatus) -> Dict[str, Any]:
    return {
        "state": a.state.value,
        "user_id": a.user_id,
        "expires_at": a.expires_at.isoformat() if a.expires_at else None,
        "detail": a.detail,
    }


def _serialize_prompt(p: LoginPrompt) -> Dict[str, Any]:
    return {
        "kind": p.kind,
        "payload": p.payload,
        "expires_at": p.expires_at.isoformat() if p.expires_at else None,
        "extra": p.extra,
    }


def _serialize_result(r: LoginContinueResult) -> Dict[str, Any]:
    return {
        "state": r.state.value,
        "detail": r.detail,
        "next_prompt": _serialize_prompt(r.next_prompt) if r.next_prompt else None,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class LoginCompleteBody(BaseModel):
    """Request body for ``POST /{id}/login/complete``.

    ``payload`` is connector-specific (QR-scan confirmation, OAuth
    authorization code, password, MFA code) per connector spec §3.5;
    we intentionally accept an opaque dict here so the schema stays
    additive when new connectors land.
    """

    payload: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("")
async def list_connectors(request: Request) -> List[Dict[str, Any]]:
    """List every registered connector and its current health/auth.

    Per connector spec §3.8 row 1. ``health()`` and ``auth_status()``
    are contractually cheap (no network); we still defensively catch
    so one bad connector doesn't 500 the whole listing.
    """
    registry = _get_registry(request)
    out: List[Dict[str, Any]] = []
    for conn in registry.all():
        try:
            health = await conn.health()
            auth = await conn.auth_status()
            out.append(
                {
                    "id": conn.id,
                    "health": _serialize_health(health),
                    "auth": _serialize_auth(auth),
                }
            )
        except Exception as exc:  # noqa: BLE001 - per-connector defensive
            out.append({"id": conn.id, "error": str(exc)})
    return out


@router.get("/{connector_id}/inbox")
async def get_inbox(connector_id: str, request: Request) -> Dict[str, Any]:
    """Return inbox events for ``connector_id``.

    Phase 6.5 lands :class:`tether_service.context.inbox_store.SqliteInbox`
    along with the connector inbound-stream drain task (connector spec
    §3.4); for now this route returns 501 NOT IMPLEMENTED with a 404
    pre-check so unknown ids still surface clearly.
    """
    registry = _get_registry(request)
    _resolve(registry, connector_id)
    raise HTTPException(
        status_code=501,
        detail=(
            "Inbox routes are not implemented yet (Phase 6.5 lands "
            "SqliteInbox + drain task per connector spec §3.4)."
        ),
    )


@router.post("/{connector_id}/login/begin")
async def login_begin(connector_id: str, request: Request) -> Dict[str, Any]:
    """Initiate the login flow for ``connector_id``.

    Returns a serialized :class:`LoginPrompt` (QR-code data URL, OAuth
    URL, password instructions, etc.) per connector spec §3.5.
    """
    registry = _get_registry(request)
    conn = _resolve(registry, connector_id)
    prompt = await conn.begin_login()
    return _serialize_prompt(prompt)


@router.post("/{connector_id}/login/complete")
async def login_complete(
    connector_id: str,
    body: LoginCompleteBody,
    request: Request,
) -> Dict[str, Any]:
    """Submit user-provided login data; on READY, start the connector.

    Per connector spec §3.5 + §3.3 step 7 (registry.start_connector(id)
    runs after a successful complete_login so the user gets a working
    connector without restarting the process).
    """
    registry = _get_registry(request)
    conn = _resolve(registry, connector_id)
    result = await conn.complete_login(payload=body.payload)
    if result.state is ConnectorState.READY:
        await registry.start_connector(connector_id)
    return _serialize_result(result)


@router.get("/{connector_id}/oauth/callback")
async def oauth_callback(
    connector_id: str,
    request: Request,
    state: str = Query(..., description="OAuth state token (CSRF guard)."),
    code: str = Query(..., description="OAuth authorization code."),
) -> Dict[str, Any]:
    """OAuth redirect target (Gmail-style flows).

    Per connector spec §3.8 row 3: verifies ``state`` against the
    registry's TTL cache (populated by ``begin_login`` on the connector
    side), then forwards ``{state, code, state_payload}`` to
    :meth:`Connector.complete_login`. Mirrors ``login/complete`` for
    the start-on-READY behavior.

    Returns JSON for now; Phase 2a/2b sessions may swap this for a
    redirect to a friendly success page once the desktop UI exists.
    Missing/expired state → 400 (CSRF protection).
    """
    registry = _get_registry(request)
    conn = _resolve(registry, connector_id)

    state_payload = registry.oauth_state.pop(state)
    if state_payload is None:
        raise HTTPException(
            status_code=400,
            detail="OAuth state token missing or expired",
        )

    result = await conn.complete_login(
        payload={
            "state": state,
            "code": code,
            "state_payload": state_payload,
        }
    )
    if result.state is ConnectorState.READY:
        await registry.start_connector(connector_id)
    return _serialize_result(result)


@router.post("/{connector_id}/logout")
async def logout(connector_id: str, request: Request) -> Dict[str, Any]:
    """Delete persisted creds + transition to LOGGED_OUT.

    The Connector instance is preserved (re-login can re-use it). Per
    connector spec §3.1, ``logout()`` is responsible for any internal
    ``stop()``; the registry does not auto-stop here.
    """
    registry = _get_registry(request)
    conn = _resolve(registry, connector_id)
    await conn.logout()
    return {"ok": True, "id": connector_id}


__all__ = ["router"]
