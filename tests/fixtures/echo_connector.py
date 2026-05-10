"""EchoConnector fixture — Phase 4.5 spec §8.3 validation harness.

Implements the full :class:`tether.connectors.base.Connector`
contract (spec §3.1) without any real network or external dependency.
Used by ``tests/integration/test_connectors_lifecycle.py`` and
``tests/unit/connectors/test_echo_schema.py`` to exercise the spec §8.3
acceptance scenarios.

Surfaces four tools that together exercise every BaseTool schema branch
and the ``user_confirmed_send`` draft+confirm pattern:

* ``echo_send(message: str)`` — validates plain ``str`` schema + outbox
  side-effect.
* ``echo_mark_seen(event_ids: list[str])`` — validates ``list[str]`` →
  ``"array of string"`` schema (spec §8.1 acceptance).
* ``echo_with_optional(text: str, label: Optional[str] = None)`` —
  validates ``Optional[T]`` → ``nullable: true`` schema (spec §8.1).
* ``echo_confirm_send(draft_id: str)`` — validates the
  :attr:`ToolExecutionContext.user_confirmed_send` draft+confirm
  pattern (spec §4 footer; synthesis §10.8 #4).

NOT a production connector. Lives under ``tests/fixtures/`` so it never
ships in the wheel. Citations in the docstrings reference the connector
spec at
``C:/Users/lovan/.copilot/session-state/5c8a15fc-11c0-4eef-98e1-cf5cd5f6a520/plan.md``
and ``_synthesis.md`` Phase 4.5.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable, ClassVar, Dict, List, Optional, Set

from tether.connectors.base import Connector
from tether.connectors.types import (
    AuthStatus,
    ConnectorState,
    HealthStatus,
    InboundEvent,
    LoginContinueResult,
    LoginPrompt,
)
from tether.core.errors import ConnectorNotConfiguredError
from tether.core.interfaces import Tool
from tether.core.types import ToolExecutionContext
from tether.tools.base import BaseTool


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
#
# Each tool subclass sets ``_tether_tool_registered_name`` directly on the
# class body. This is the same marker the @tool decorator installs (per
# tether.tools.registration._TOOL_MARKER_ATTR) but bypasses the
# global ``_DECORATED_TOOLS`` registry — connector tools are wired through
# the connector's own ``tools()`` dict, not the in-tree decorator registry,
# so polluting the global registry would actively confuse other tests.

StateProvider = Callable[[], ConnectorState]


def _require_ready(state_provider: StateProvider, label: str) -> None:
    """Raise :class:`ConnectorNotConfiguredError` unless state is READY.

    Per connector spec §3.1: tool methods MUST raise
    ``ConnectorNotConfiguredError`` when the connector is in
    ``UNCONFIGURED`` or ``LOGGED_OUT``. We extend the same check to every
    non-READY state so the harness tests can also exercise the error path
    after ``logout()`` and during ``AUTHENTICATING``.
    """
    state = state_provider()
    if state is not ConnectorState.READY:
        raise ConnectorNotConfiguredError(
            f"echo connector tool {label!r} requires READY state "
            f"(current: {state.value})"
        )


class EchoSendTool(BaseTool):
    """Echo a message; records it in the connector's in-memory outbox.

    Validates that a plain ``str`` parameter produces ``{"type": "string"}``
    in the OpenAI-style schema (spec §8.1) and that an outbound side-effect
    is observable from the test (spec §8.3 outbound assertion).
    """

    _tether_tool_registered_name: ClassVar[str] = "echo_send"

    def __init__(self, *, outbox: List[str], state_provider: StateProvider) -> None:
        super().__init__()
        self._outbox = outbox
        self._state_provider = state_provider

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(self, message: str) -> dict:
        """Append ``message`` to the outbox.

        Args:
            message: Arbitrary text to record. Stored verbatim.
        """
        _require_ready(self._state_provider, "echo_send")
        self._outbox.append(message)
        return {"sent": True, "outbox_size": len(self._outbox)}


class EchoMarkSeenTool(BaseTool):
    """Mark event ids as seen.

    Validates that a ``list[str]`` parameter produces an ``array`` schema
    with ``items.type == "string"`` (spec §8.1 acceptance) and that the
    schema marks the parameter as required when no default is provided.
    """

    _tether_tool_registered_name: ClassVar[str] = "echo_mark_seen"

    def __init__(self, *, seen: Set[str], state_provider: StateProvider) -> None:
        super().__init__()
        self._seen = seen
        self._state_provider = state_provider

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(self, event_ids: List[str]) -> dict:
        """Mark the supplied ids as seen.

        Args:
            event_ids: Event ids to record. Each id is added to an
                in-memory set; duplicates are silently ignored.
        """
        _require_ready(self._state_provider, "echo_mark_seen")
        for eid in event_ids:
            self._seen.add(eid)
        return {"marked": list(event_ids), "total_seen": len(self._seen)}


class EchoWithOptionalTool(BaseTool):
    """Log a (text, label) pair where label may be omitted.

    Validates ``Optional[T]`` → ``nullable: true`` schema generation
    (spec §8.1) and that an Optional-typed parameter is excluded from the
    ``required`` list because it carries a default.
    """

    _tether_tool_registered_name: ClassVar[str] = "echo_with_optional"

    def __init__(
        self, *, log: List[Dict[str, Any]], state_provider: StateProvider
    ) -> None:
        super().__init__()
        self._log = log
        self._state_provider = state_provider

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(self, text: str, label: Optional[str] = None) -> dict:
        """Record a labelled (or unlabelled) entry.

        Args:
            text: The text to log.
            label: Optional category tag; ``None`` records the entry
                without a label.
        """
        _require_ready(self._state_provider, "echo_with_optional")
        entry = {"text": text, "label": label}
        self._log.append(entry)
        return {"logged": entry}


class EchoConfirmSendTool(BaseTool):
    """Send a draft only when the user has explicitly confirmed.

    Reads :attr:`ToolExecutionContext.user_confirmed_send`; refuses
    unless the flag is ``True``. Validates the connector spec §4 footer
    draft+confirm pattern: WhatsApp/Gmail will use the same discipline
    in subsequent sessions. In this refactor the regex classifier that
    flips ``user_confirmed_send`` is intentionally deferred (synthesis
    §10.8 #4), so the orchestrator path never sets it; the positive case
    therefore has to be exercised by manually constructing a context.
    """

    _tether_tool_registered_name: ClassVar[str] = "echo_confirm_send"

    def __init__(
        self,
        *,
        confirmed_drafts: List[str],
        state_provider: StateProvider,
    ) -> None:
        super().__init__()
        self._confirmed_drafts = confirmed_drafts
        self._state_provider = state_provider

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(
        self,
        draft_id: str,
        *,
        context: Optional[ToolExecutionContext] = None,
    ) -> dict:
        """Confirm-send a draft.

        Args:
            draft_id: Identifier of the draft to send.
        """
        _require_ready(self._state_provider, "echo_confirm_send")
        if context is None or not context.user_confirmed_send:
            return {
                "confirmed": False,
                "reason": (
                    "user_confirmed_send is False; draft requires "
                    "explicit confirmation"
                ),
            }
        self._confirmed_drafts.append(draft_id)
        return {"confirmed": True, "draft_id": draft_id}


# ---------------------------------------------------------------------------
# The Connector itself
# ---------------------------------------------------------------------------


class EchoConnector(Connector):
    """No-op connector for spec §8.3 acceptance tests.

    State machine matches the spec §3.2 contract end-to-end:

    * ``UNCONFIGURED`` (initial) — no creds; tool methods raise
      ``ConnectorNotConfiguredError``.
    * ``AUTHENTICATING`` — set by :meth:`begin_login`; cleared by
      :meth:`complete_login` (READY on success, stays AUTHENTICATING on
      bad code).
    * ``READY`` — credentials present; tools succeed.
    * ``LOGGED_OUT`` — set by :meth:`logout`; tool methods raise
      ``ConnectorNotConfiguredError`` until the user logs in again.

    Side-effect storage (``outbox``, ``seen``, ``optional_log``,
    ``confirmed_drafts``) is held on the instance so tests can inspect
    or mutate it without going through the tool dict.

    ``_stop_delay_sec`` lets the slow-stop test
    (``test_aclose_within_2s_with_slow_stop``) force ``stop()`` to exceed
    the registry's 2 s cooperative budget per spec §3.3 step 6.
    """

    id: ClassVar[str] = "echo"

    #: Code accepted by :meth:`complete_login`. Tests can override at the
    #: instance level if they want to simulate a different secret.
    expected_code: str = "ok"

    def __init__(self) -> None:
        self._state: ConnectorState = ConnectorState.UNCONFIGURED
        self._user_id: Optional[str] = None

        self.outbox: List[str] = []
        self.seen: Set[str] = set()
        self.optional_log: List[Dict[str, Any]] = []
        self.confirmed_drafts: List[str] = []

        # Slow-stop knob — see :meth:`stop` and the lifecycle test.
        self._stop_delay_sec: float = 0.0

        # Per-instance tool dict so the side-effect storage is shared
        # between the connector and its tools without singletons.
        self._tools: Dict[str, Tool] = {
            "echo_send": EchoSendTool(
                outbox=self.outbox, state_provider=self._state_provider
            ),
            "echo_mark_seen": EchoMarkSeenTool(
                seen=self.seen, state_provider=self._state_provider
            ),
            "echo_with_optional": EchoWithOptionalTool(
                log=self.optional_log, state_provider=self._state_provider
            ),
            "echo_confirm_send": EchoConfirmSendTool(
                confirmed_drafts=self.confirmed_drafts,
                state_provider=self._state_provider,
            ),
        }

    # ------------------------------------------------------------------
    # State helper
    # ------------------------------------------------------------------

    def _state_provider(self) -> ConnectorState:
        return self._state

    @property
    def state(self) -> ConnectorState:
        """Current state — convenience accessor for tests."""
        return self._state

    # ------------------------------------------------------------------
    # Connector ABC: lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Idempotent start.

        UNCONFIGURED / LOGGED_OUT / AUTHENTICATING — no-op (we need
        creds first or are mid-login).
        Otherwise transition to READY iff a user is set; preserves
        DEGRADED / ERROR if the test pre-set them.
        """
        if self._state in (
            ConnectorState.UNCONFIGURED,
            ConnectorState.AUTHENTICATING,
            ConnectorState.LOGGED_OUT,
        ):
            return
        if self._user_id is not None and self._state is not ConnectorState.READY:
            self._state = ConnectorState.READY

    async def stop(self) -> None:
        """Idempotent stop. Optionally slow per ``_stop_delay_sec``.

        State is preserved across stop/start (per spec §3.1: ``stop()``
        keeps creds; only :meth:`logout` deletes them).
        """
        if self._stop_delay_sec > 0:
            await asyncio.sleep(self._stop_delay_sec)

    async def logout(self) -> None:
        """Delete creds + transition to ``LOGGED_OUT`` (spec §3.1)."""
        self._user_id = None
        self._state = ConnectorState.LOGGED_OUT

    # ------------------------------------------------------------------
    # Connector ABC: status (cheap; no network)
    # ------------------------------------------------------------------

    async def health(self) -> HealthStatus:
        return HealthStatus(state=self._state, detail="echo connector")

    async def auth_status(self) -> AuthStatus:
        return AuthStatus(state=self._state, user_id=self._user_id)

    # ------------------------------------------------------------------
    # Connector ABC: login flow (spec §3.5)
    # ------------------------------------------------------------------

    async def begin_login(self) -> LoginPrompt:
        self._state = ConnectorState.AUTHENTICATING
        return LoginPrompt(
            kind="code",
            payload=f"echo://login (provide code={self.expected_code!r} to complete)",
        )

    async def complete_login(self, *, payload: Dict[str, Any]) -> LoginContinueResult:
        code = payload.get("code")
        if code != self.expected_code:
            # Stay in AUTHENTICATING so the user can retry without
            # re-issuing begin_login (spec §3.5: multi-step flows may
            # carry next_prompt; the simple echo case just preserves
            # state).
            return LoginContinueResult(
                state=ConnectorState.AUTHENTICATING,
                detail=f"invalid code (expected {self.expected_code!r})",
            )
        self._user_id = payload.get("user_id", "echo_user")
        self._state = ConnectorState.READY
        return LoginContinueResult(state=ConnectorState.READY)

    # ------------------------------------------------------------------
    # Connector ABC: outbound + inbound
    # ------------------------------------------------------------------

    def tools(self) -> Dict[str, Tool]:
        # Per spec §3.3: keys MUST start with f"{cid}_" — registry
        # validates this at boot. Returning a fresh shallow copy keeps
        # callers from mutating the canonical dict.
        return dict(self._tools)

    async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
        """Empty inbound stream.

        Phase 6.5 will land real inbound-event production + the SqliteInbox
        drain task (spec §3.4). The Phase 4.5 harness only needs the
        async-generator shape so registry wiring + lifecycle tests run
        end-to-end without producing events.
        """
        if False:  # pragma: no cover - empty stream
            yield  # type: ignore[unreachable]


__all__ = [
    "EchoConnector",
    "EchoSendTool",
    "EchoMarkSeenTool",
    "EchoWithOptionalTool",
    "EchoConfirmSendTool",
]
