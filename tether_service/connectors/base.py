"""Connector ABC.

Per connector spec §3.1 (locked design at
``C:/Users/lovan/.copilot/session-state/5c8a15fc-11c0-4eef-98e1-cf5cd5f6a520/plan.md``).
Read the full spec for behavior contracts; this module captures the type
signatures only.

A connector is a long-lived plugin that:

1. Has its own auth lifecycle (``begin_login`` / ``complete_login`` /
   ``logout``) — the registry never authenticates on the connector's behalf.
2. Exposes one or more :class:`tether_service.core.interfaces.Tool` instances
   whose names MUST be prefixed with ``f"{id}_"`` to avoid colliding with
   bundled tools or other connectors. The future ``ConnectorRegistry``
   (``p4_5-connector-registry``) enforces this at boot.
3. Maintains a background inbound stream of :class:`InboundEvent` values.
   Pure-outbound connectors (e.g. a future calendar-write connector)
   implement an empty async generator.
4. Reports health + auth_status synchronously without making network calls.

Tool methods raise :class:`tether_service.core.errors.ConnectorNotConfiguredError`
when the connector is in ``UNCONFIGURED`` or ``LOGGED_OUT`` state; failed
logins raise :class:`tether_service.core.errors.ConnectorAuthError`.

Citations:
    - Connector spec §3.1 (Connector ABC), §3.2 (state types), §3.5
      (login prompts).
    - Synthesis §4 Phase 4.5 step 47a.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, ClassVar, Dict

from tether_service.connectors.types import (
    AuthStatus,
    HealthStatus,
    InboundEvent,
    LoginContinueResult,
    LoginPrompt,
)
from tether_service.core.interfaces import Tool


_CONNECTOR_ID_RE = re.compile(r"^[a-z0-9_]+$")


class Connector(ABC):
    """Connector lifecycle ABC.

    Subclasses MUST override the ``id`` :class:`ClassVar` with a string
    matching ``r"^[a-z0-9_]+$"``. The pattern is checked in
    ``__init_subclass__`` so a misnamed connector fails at class-definition
    time, well before ``ConnectorRegistry`` instantiates it.

    Subclasses' tool names MUST start with ``f"{id}_"`` — enforced at
    registry boot time by ``ConnectorRegistry`` (``p4_5-connector-registry``).

    See module docstring for the high-level contract; see connector spec §3.1
    for the full behavioral contract.
    """

    #: Stable identifier for this connector (e.g. ``"whatsapp"``, ``"gmail"``,
    #: ``"echo"``). Subclasses MUST override; pattern enforced in
    #: ``__init_subclass__``.
    id: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # ``id`` is intentionally an empty string on Connector itself; only
        # validate when a concrete subclass actually sets it.
        if cls.id and not _CONNECTOR_ID_RE.match(cls.id):
            raise ValueError(
                f"Connector.id={cls.id!r} must match {_CONNECTOR_ID_RE.pattern} "
                f"(class {cls.__module__}.{cls.__qualname__})"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def start(self) -> None:
        """Begin operation. Idempotent.

        Loads creds via :class:`tether_service.core.secrets.SecretsProvider`;
        if no creds, transitions to ``UNCONFIGURED`` and returns. If creds
        present, opens connections and transitions to ``READY`` (or
        ``DEGRADED`` / ``ERROR`` on issues).

        Per connector spec §3.1: idempotent — calling twice without an
        intervening ``stop()`` is a no-op.
        """

    @abstractmethod
    async def stop(self) -> None:
        """Clean shutdown. Idempotent.

        Closes connections, cancels the inbound stream, and preserves
        creds. Use :meth:`logout` to delete creds.

        Per connector spec §3.1: must complete within 2s for cooperative
        async paths (the registry enforces this with ``asyncio.wait_for``);
        connectors with potentially blocking native cleanup MUST themselves
        use the daemon-thread + force-exit pattern that
        ``shutdown_provider_with_timeout()`` uses for the MLC provider.
        """

    @abstractmethod
    async def logout(self) -> None:
        """Delete persisted creds + transition to ``LOGGED_OUT``.

        After logout, all tool methods raise ``ConnectorNotConfiguredError``
        until ``begin_login`` + ``complete_login`` succeed again.

        Per connector spec §3.1.
        """

    # ------------------------------------------------------------------
    # Status — must be cheap; do NOT make network calls
    # ------------------------------------------------------------------

    @abstractmethod
    async def health(self) -> HealthStatus:
        """Return current health snapshot. Cheap; do NOT make network calls.

        Surfaced by ``/api/v1/connectors/<id>/health`` (Phase 4.5 step
        ``p4_5-engine-wiring-routes``) and the future doctor command.
        """

    @abstractmethod
    async def auth_status(self) -> AuthStatus:
        """Return current auth status snapshot. Cheap; do NOT make
        network calls.
        """

    # ------------------------------------------------------------------
    # Login flow
    # ------------------------------------------------------------------

    @abstractmethod
    async def begin_login(self) -> LoginPrompt:
        """Initiate the login flow.

        Returns the first :class:`LoginPrompt` (QR code, OAuth URL,
        password prompt, etc.). Sets state to ``AUTHENTICATING``.

        Per connector spec §3.5.
        """

    @abstractmethod
    async def complete_login(
        self, *, payload: Dict[str, Any]
    ) -> LoginContinueResult:
        """Submit user-provided login data.

        ``payload`` is connector-specific (QR-scan confirmation, OAuth
        authorization code, password, MFA code). On success state becomes
        ``READY``; if more steps are needed the result carries a
        ``next_prompt``; on failure
        :class:`tether_service.core.errors.ConnectorAuthError` may be
        raised, OR the result may report ``state=ERROR`` with detail.

        Per connector spec §3.5.
        """

    # ------------------------------------------------------------------
    # Outbound + inbound
    # ------------------------------------------------------------------

    @abstractmethod
    def tools(self) -> Dict[str, Tool]:
        """Return the tools this connector exposes, keyed by tool name.

        Names MUST be prefixed with ``f"{self.id}_"``. ``ConnectorRegistry``
        validates this at boot and raises a clear error on collision (per
        connector spec §3.3).
        """

    @abstractmethod
    async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
        """Async generator yielding :class:`InboundEvent` values as they
        arrive.

        Long-lived; ``ConnectorRegistry`` consumes this and persists to
        ``SqliteInbox`` (Phase 6.5). Pure-outbound connectors implement
        an empty async generator (``if False: yield`` body).

        The iterator MUST be cancellable via standard task cancellation;
        the registry owns the drain task and may cancel it at any time
        (e.g. during ``stop()`` / ``logout()`` / reload).

        Defined as ``async def`` with a ``yield``-bearing body so the ABC
        itself is an async generator function — subclass overrides MUST
        also be async generator functions (use ``yield``). Callers iterate
        with ``async for event in connector.inbound_stream(): ...``.

        Per connector spec §3.1.
        """
        # The ``if False: yield`` body makes this an async generator function
        # at the ABC level; ``@abstractmethod`` ensures subclasses must
        # override regardless. Keeping the body lets static analyzers and
        # the abc machinery agree the return type is an async iterator.
        if False:  # pragma: no cover - never executed
            yield  # type: ignore[unreachable]


__all__ = ["Connector"]
