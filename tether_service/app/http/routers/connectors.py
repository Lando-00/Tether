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

    Phase 4.5 follow-up (rubber-duck consensus, 1m CONCERN): if the
    connector emits an ``oauth_state`` token in ``LoginPrompt.extra``,
    persist it in the registry's TTL cache so a subsequent
    ``/oauth/callback?state=...`` can validate the round-trip
    (CSRF guard). Without this hop, Phase 2b Gmail's OAuth flow could
    never succeed end-to-end — ``oauth_callback`` would always 400 on
    missing state. Connectors that don't use OAuth (e.g. EchoConnector,
    WhatsApp's QR flow) simply omit ``oauth_state`` and this branch
    is a no-op.
    """
    registry = _get_registry(request)
    conn = _resolve(registry, connector_id)
    prompt = await conn.begin_login()

    state_token = (
        prompt.extra.get("oauth_state") if prompt.extra else None
    )
    if state_token:
        # Forward the full ``extra`` dict to ``complete_login`` on
        # callback so connectors can attach arbitrary connector-defined
        # data (PKCE verifier, scope hints, etc.) without changing this
        # route's signature.
        registry.oauth_state.set(
            state_token,
            {
                "connector_id": connector_id,
                "extra": dict(prompt.extra),
            },
        )

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
    """Delete persisted creds + stop the connector + transition to LOGGED_OUT.

    Per connector spec §3.8: ``logout()`` is responsible for credential
    deletion; the registry is responsible for stopping background tasks
    / connections owned by the connector. Both must run.

    Phase 4.5 follow-up (rubber-duck consensus, gpt-5.5 BLOCKING #2):
    previously this route only called ``conn.logout()`` and explicitly
    skipped ``registry.stop_connector()``. That left
    ``inbound_stream`` consumers, scheduled tasks, and any open
    connections alive after credentials were deleted — a use-after-
    logout hazard that future Gmail/WhatsApp connectors would hit.

    Order: credentials first (so a slow stop can't keep using them),
    then ``stop_connector`` with the registry's 2 s cooperative budget
    (spec §3.3 step 6). ``stop_connector`` is idempotent and swallows
    its own exceptions so a flaky stop never bubbles up here.
    """
    registry = _get_registry(request)
    conn = _resolve(registry, connector_id)

    # 1. Delete credentials (transitions to LOGGED_OUT for spec-compliant
    #    connectors).
    await conn.logout()

    # 2. Stop background tasks / connections under the 2 s spec §3.3
    #    cooperative budget. Idempotent + safe even when ``logout()``
    #    already stopped internal tasks.
    await registry.stop_connector(connector_id)

    return {"ok": True, "id": connector_id, "state": ConnectorState.LOGGED_OUT.value}


__all__ = ["router"]
