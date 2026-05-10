"""Connector framework types.

Per connector spec §3.2 + §3.5. Types only; no behavior. Used by the
:class:`tether.connectors.base.Connector` ABC, the future
``ConnectorRegistry`` (``p4_5-connector-registry``), and the future HTTP
routes (``p4_5-engine-wiring-routes``).

All dataclasses are ``frozen=True`` so connector code cannot mutate a status
snapshot in place — health/auth reporting is expected to construct a fresh
object each time. This keeps the registry's view consistent.

Citations:
    - Connector spec §3.2 (InboundEvent, ConnectorHealth shapes).
    - Connector spec §3.5 (LoginPrompt / LoginContinueResult flow for
      begin_login / complete_login).
    - Synthesis §4 Phase 4.5 step 47a.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Literal, Optional


_LOGIN_PROMPT_KINDS = ("qr_code", "url", "password", "code")


class ConnectorState(Enum):
    """Lifecycle state of a connector instance.

    Per connector spec §3.2 ``ConnectorHealth.state`` literal values plus
    ``AUTHENTICATING`` for the in-flight ``begin_login`` / ``complete_login``
    window. Phase 4.5 step 47a fixes these to a single enum so registry,
    HTTP routes, and the inbox can share the same vocabulary.

    Values:
        UNCONFIGURED   — no credentials yet; tool methods raise
                         ``ConnectorNotConfiguredError``.
        AUTHENTICATING — login flow in progress (between ``begin_login``
                         and ``complete_login``).
        READY          — credentials persisted; tools are callable;
                         ``inbound_stream`` may be live.
        DEGRADED       — credentials present but recent operations failing
                         (transient backend issues, rate limits).
        ERROR          — fatal error; manual intervention required.
        LOGGED_OUT     — explicit ``logout()`` was called; creds deleted;
                         tool methods raise ``ConnectorNotConfiguredError``
                         until ``begin_login``/``complete_login`` succeeds.
    """

    UNCONFIGURED = "unconfigured"
    AUTHENTICATING = "authenticating"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"
    LOGGED_OUT = "logged_out"


@dataclass(frozen=True)
class HealthStatus:
    """Connector health snapshot.

    Returned by :meth:`Connector.health`; surfaced by the future
    ``/api/v1/connectors/<id>/health`` route. Cheap to compute — concrete
    connectors MUST NOT make network calls inside ``health()`` per
    connector spec §3.1.
    """

    state: ConnectorState
    detail: Optional[str] = None
    last_success: Optional[datetime] = None
    last_error: Optional[datetime] = None
    last_error_message: Optional[str] = None


@dataclass(frozen=True)
class AuthStatus:
    """Auth status snapshot.

    Returned by :meth:`Connector.auth_status`. ``user_id`` carries the
    account identifier (email for Gmail, phone JID for WhatsApp);
    ``expires_at`` is set if the underlying token has a known expiry.
    """

    state: ConnectorState
    user_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    detail: Optional[str] = None


@dataclass(frozen=True)
class LoginPrompt:
    """Returned by :meth:`Connector.begin_login` — describes what the user
    must do to complete authentication.

    ``kind`` selects the UI affordance:

    - ``qr_code``: ``payload`` is the data to encode (e.g. WhatsApp Web QR).
    - ``url``: ``payload`` is the URL to open (e.g. OAuth consent URL).
    - ``password``: ``payload`` is human-readable instruction text.
    - ``code``: ``payload`` is human-readable instruction text; the user
      sends back a code (e.g. SMS / email verification code) via
      ``complete_login``.

    Per connector spec §3.5.
    """

    kind: Literal["qr_code", "url", "password", "code"]
    payload: str
    expires_at: Optional[datetime] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _LOGIN_PROMPT_KINDS:
            raise ValueError(
                f"LoginPrompt.kind={self.kind!r} must be one of "
                f"{_LOGIN_PROMPT_KINDS}"
            )


@dataclass(frozen=True)
class LoginContinueResult:
    """Returned by :meth:`Connector.complete_login`.

    On success ``state == ConnectorState.READY``; if more steps are needed
    (e.g. multi-factor auth) ``state == AUTHENTICATING`` and
    ``next_prompt`` carries the follow-up :class:`LoginPrompt`. On failure
    ``state == ERROR`` with ``detail`` populated.
    """

    state: ConnectorState
    detail: Optional[str] = None
    next_prompt: Optional[LoginPrompt] = None


@dataclass(frozen=True)
class InboundEvent:
    """A single inbound event from a connector's inbound stream.

    Persisted to ``SqliteInbox`` in Phase 6.5; emitted on the wire as
    ``InboundEvent`` wire events in Phase 5.

    ``event_id`` is the connector-supplied idempotency key — registry
    de-duplicates on ``(connector_id, event_id)``. ``payload`` is
    connector-specific JSON (size-capped per ``inbox.max_payload_bytes``,
    connector spec §3.4 + §3.7); ``summary`` is the short human-readable
    preview surfaced by ``/api/v1/inbox`` listings.

    Per connector spec §3.2 + §3.5.
    """

    event_id: str
    connector_id: str
    kind: str
    received_at: datetime
    payload: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None


__all__ = [
    "ConnectorState",
    "HealthStatus",
    "AuthStatus",
    "LoginPrompt",
    "LoginContinueResult",
    "InboundEvent",
]
