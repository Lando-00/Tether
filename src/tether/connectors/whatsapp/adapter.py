"""WhatsAppClientAdapter ABC — the swappable seam over neonize.

Phase 2b · WhatsApp connector. Per ADR-0018 D2 + wa-D-synthesis §D1 / §D9.
The concrete neonize-backed body lives in
:class:`tether.connectors.whatsapp.neonize_adapter.NeonizeWhatsAppClientAdapter`
in Wave 2 (track IMP-A); tests substitute a fake/mock adapter that satisfies
this same surface (wa-D-neonize.md §11).

No neonize types leak through this contract:

* JIDs are plain ``str`` (``"447911…@s.whatsapp.net"``); neonize's protobuf
  ``JID`` is converted at the boundary via ``build_jid()`` in the concrete
  impl.
* Contacts are :class:`Contact` dataclasses, not neonize's
  ``ContactInfo`` protobuf.
* Send results are :class:`SendResult` dataclasses, not neonize's
  ``SendResponse`` protobuf.
* Inbound events are Tether's :class:`tether.connectors.types.InboundEvent`,
  mapped from neonize's ``MessageEv`` / ``HistorySyncEv`` in
  ``inbound_events.py`` (Wave 2).

This isolation is what makes the connector test-fakable without ever
importing neonize (see wa-D-neonize.md §11 leak table).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import AsyncIterator

from tether.connectors.types import InboundEvent

# ---------------------------------------------------------------------------
# Pure-data value types (frozen; safe to share across threads/queues)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterHealthSnapshot:
    """Cheap-to-construct view of the adapter's last-known state.

    Returned by :meth:`WhatsAppClientAdapter.health`. The owning
    :class:`tether.connectors.whatsapp.connector.WhatsAppConnector` uses
    this to compose its own :class:`tether.connectors.types.HealthStatus`
    without making any neonize call — connector spec §3.1 forbids network
    I/O inside ``health()``.

    Fields:
        state: A short string tag for the adapter's current lifecycle
            phase (e.g. ``"stopped"``, ``"connecting"``, ``"connected"``,
            ``"reconnecting"``, ``"logged_out"``, ``"error"``). The
            connector maps this onto :class:`ConnectorState` per
            wa-D-synthesis §D3.
        last_event_at: UTC timestamp of the most recent inbound event the
            adapter observed (live ``MessageEv`` or ``HistorySyncEv``).
            ``None`` until the first event arrives. Drives the
            watchdog/staleness heuristic per wa-D-openclaw.md §1.
        last_error: Short human-readable description of the most recent
            adapter-level failure, or ``None`` if no failure has been
            recorded. Mirrors OpenClaw's ``formatError`` output shape
            (wa-D-openclaw.md §7).
    """

    state: str
    last_event_at: datetime | None
    last_error: str | None


@dataclass(frozen=True)
class Contact:
    """A WhatsApp contact in Tether's normalised shape.

    Returned in batches by :meth:`WhatsAppClientAdapter.get_contacts`;
    consumed by ``target_resolver.resolve_target`` (Wave 2) for the
    display-name → JID lookup path per wa-D-synthesis §D5.

    Fields:
        jid: Bare WhatsApp JID (``"447911…@s.whatsapp.net"``,
            ``"…@lid"``, ``"…@g.us"`` for groups, etc.). Never carries a
            ``:device`` suffix at this layer — the adapter strips it.
        name: Push name / saved contact name; may be ``None`` for
            contacts WhatsApp has not yet resolved.
        e164: Best-effort E.164 phone string (``"+447911…"``) parsed from
            the JID local part via ``phonenumbers``; ``None`` for JID
            types that have no phone (group, newsletter, LID without a
            known phone).
        is_business: ``True`` iff the contact is a WhatsApp Business
            account. Surfaced for UX hints only; tools do not behave
            differently based on this flag.
    """

    jid: str
    name: str | None
    e164: str | None
    is_business: bool


@dataclass(frozen=True)
class SendResult:
    """Outcome of a successful outbound send.

    Returned by :meth:`WhatsAppClientAdapter.send_text` and
    :meth:`WhatsAppClientAdapter.send_media`. Tools surface
    ``message_id`` to the model so subsequent
    ``whatsapp_mark_platform_read`` / ``whatsapp_get_thread`` calls can
    correlate.

    Fields:
        message_id: Server-assigned ID extracted from neonize's
            ``SendResponse.ID`` at the adapter boundary. Stable across
            retries (neonize-side dedup).
        to_jid: The bare JID the message was sent to (echo of the
            adapter input, with any ``:device`` suffix stripped).
    """

    message_id: str
    to_jid: str


class PairStatus(Enum):
    """Result of :meth:`WhatsAppClientAdapter.await_paired`.

    Per wa-D-synthesis §D4 (two-phase QR login from OpenClaw
    ``login-qr.ts``). The connector inspects this to decide whether to
    transition to :class:`ConnectorState.READY`, surface a refreshed QR
    via ``LoginContinueResult.next_prompt``, or report failure.

    Values:
        PAIRED      — neonize ``ConnectedEv`` after a successful pair.
                      Connector flips to READY.
        QR_ROTATED  — neonize emitted a new ``QREv`` before the user
                      scanned. Connector returns a refreshed
                      ``LoginPrompt`` so the UI can re-render.
        LOGGED_OUT  — neonize emitted ``LoggedOutEv`` during the pair
                      window (forced unpair while scanning). Connector
                      returns LOGGED_OUT with
                      ``detail="logged_out_during_pair"``.
        FAILED      — neonize emitted a terminal pair failure such as
                      ``TemporaryBanEv`` or ``ClientOutdatedEv``.
                      Connector returns ERROR with the adapter detail.
        TIMEOUT     — the long-poll budget elapsed with neither outcome.
                      Caller decides whether to retry.
    """

    PAIRED = "paired"
    QR_ROTATED = "qr_rotated"
    LOGGED_OUT = "logged_out"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# The adapter ABC
# ---------------------------------------------------------------------------


class WhatsAppClientAdapter(abc.ABC):
    """The swappable seam over neonize (and any future WhatsApp Web lib).

    Per ADR-0018 + wa-D-synthesis §D1 / §D9. Implementations:

    * ``NeonizeWhatsAppClientAdapter`` — production; wraps
      ``neonize.aioze.NewAClient`` (Wave 2, track IMP-A).
    * ``MockWhatsAppClientAdapter`` — tests; satisfies the same surface
      with an in-memory queue (Wave 3 fixtures).

    The lifecycle is owned by
    :class:`tether.connectors.whatsapp.connector.WhatsAppConnector` —
    ``start()`` is called from ``Connector.start()`` /
    ``Connector.complete_login()``; ``stop()`` and ``logout()`` from the
    matching ``Connector`` methods. The connector spec's 2 s
    ``stop()`` budget is honoured by the concrete impl per
    wa-D-synthesis §D1 (daemon-thread + force-exit pattern lifted from
    ``shutdown_provider_with_timeout``).

    Every method is async EXCEPT :meth:`subscribe_inbound` (sync factory
    that returns an async iterator) and :meth:`health` (cheap synchronous
    snapshot per connector spec §3.1).
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def start(self, auth_dir: Path) -> None:
        """Open the neonize session against creds under ``auth_dir``.

        Idempotent (guard on an internal ``_started`` flag). The flag is
        reset in :meth:`stop` so a later ``start()`` re-runs
        initialisation. Concrete impl MUST guard the Go-side
        ``connect()`` call with ``if self._started: return`` and MUST:

        1. Re-apply the platform-machine patch defensively (the package
           ``__init__`` already did it, but a defensive second call is
           cheap and survives reload paths).
        2. Reset ``neonize.aioze.events.event_global_loop = None`` before
           the first ``client.connect()`` so a restart re-registers the
           loop (wa-D-neonize.md BLOCKER-2).
        3. Register event handlers (``MessageEv``, ``QREv``,
           ``ConnectedEv``, ``DisconnectedEv``, ``LoggedOutEv``, …)
           BEFORE calling ``connect()`` so no events are dropped.
        4. Call ``client.connect()`` and return — the long-lived
           ``connect_task`` runs on neonize's side; the adapter merely
           supervises it.
        5. Register ``connect_task.add_done_callback(...)``. If the
           long-lived Go task raises while the connector reports READY,
           the callback flips the cached health snapshot to
           ``AdapterHealthSnapshot.state == "error"`` and records
           ``last_error`` so the connector can map to
           ``ConnectorState.ERROR``.

        ``auth_dir`` is the connector's own data dir
        (``<data_dir>/connectors/whatsapp``); neonize creates its session
        SQLite under this path. Per wa-D-synthesis §D1.
        """

    @abc.abstractmethod
    async def stop(self) -> None:
        """Close the neonize session. Bounded to ~1.8 s.

        Concrete impl uses ``asyncio.wait_for(client.stop(),
        timeout=1.8)`` so the owning connector's 2 s budget (connector
        spec §3.1) is never blown; if it fires the daemon-thread +
        force-exit pattern from ``provider.py::shutdown_provider_with_timeout``
        takes over.

        MUST reset ``neonize.aioze.events.event_global_loop = None``
        before returning — wa-D-neonize.md BLOCKER-2: leaving it set
        causes the next ``start()`` to silently drop callbacks because
        ``aioze/events.py:52-60`` only assigns the loop if the global
        is ``None``.

        MUST reset ``_started = False`` BEFORE the daemon-thread
        fallback is abandoned so a concurrent restart attempt does not
        silently no-op.
        """

    @abc.abstractmethod
    async def logout(self) -> None:
        """Tell the WhatsApp server to drop the linked-device session,
        then delete the local neonize session DB.

        Distinct from :meth:`stop`: ``stop()`` preserves creds for a
        later restart; ``logout()`` makes the next ``start()`` need a
        fresh QR pair. Per connector spec §3.1.

        Delete creds and the neonize SQLite session DB unconditionally,
        even if ``stop()`` or ``client.logout()`` raises. The 2 s budget
        bounds the server/session stop path only; ``logout()`` MUST NOT
        honour that budget for the credential-delete step.
        """

    # ------------------------------------------------------------------
    # Login (two-phase QR flow per wa-D-synthesis §D4)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def pair_qr(self) -> bytes:
        """Wait for the first neonize ``QREv`` and return raw QR text bytes.

        Called from :class:`WhatsAppConnector.begin_login`. The connector
        decodes the UTF-8 raw QR string into ``LoginPrompt.payload``; the CLI
        or route renders that text to PNG when the optional ``qrcode`` package
        is available. Already-rendered PNG bytes are treated only as a
        defensive fallback and are exposed through prompt ``extra``.

        Subsequent ``QREv`` events from neonize (WhatsApp rotates the QR
        every ~60 s) MUST update an internal cache so
        :meth:`await_paired` can surface the freshest QR via
        :class:`PairStatus.QR_ROTATED`.
        """

    @abc.abstractmethod
    async def await_paired(self, timeout_sec: float) -> PairStatus:
        """Long-poll for pair success, QR rotation, failure, or timeout.

        Called from :class:`WhatsAppConnector.complete_login`. Returns
        as soon as any of these is observed:

        * ``ConnectedEv`` after the user scans the QR → :attr:`PairStatus.PAIRED`.
        * A newer ``QREv`` than the cached one (i.e. QR rotation while
          the user was scanning) → :attr:`PairStatus.QR_ROTATED`. The
          connector then re-renders and returns the new QR via
          :attr:`LoginContinueResult.next_prompt`.
        * ``LoggedOutEv`` during the pair window → :attr:`PairStatus.LOGGED_OUT`.
          The connector reports LOGGED_OUT instead of staying stuck in
          AUTHENTICATING.
        * Terminal pair errors (``TemporaryBanEv`` /
          ``ClientOutdatedEv``) → :attr:`PairStatus.FAILED`. The
          connector reports ERROR with adapter-provided detail.
        * ``timeout_sec`` elapsed with no pairing or rotation →
          :attr:`PairStatus.TIMEOUT`. Caller decides next step
          (typically: re-poll, or abandon if the 3-minute active-login
          TTL (wa-D-synthesis §D4) is exhausted).
        """

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def send_text(self, to_jid: str, text: str) -> SendResult:
        """Send a plain text message; return the server-assigned id.

        ``to_jid`` MUST be a bare WhatsApp JID — the adapter does NOT
        normalise display names or phone numbers. That is
        ``target_resolver.resolve_target``'s job in the tool layer
        (wa-D-synthesis §D5).

        Raises :class:`tether.core.errors.ConnectorAuthError` if the
        underlying client is not logged in; transient network errors
        propagate as plain ``ConnectionError`` for the connector's
        reconnect/backoff loop to handle.
        """

    @abc.abstractmethod
    async def send_media(
        self,
        to_jid: str,
        body: bytes,
        mime: str,
        filename: str,
        caption: str | None = None,
    ) -> SendResult:
        """Send a media message (image / video / audio / document).

        Dispatches on ``mime`` prefix at the adapter boundary
        (``image/*`` → ``send_image``, ``video/*`` → ``send_video``,
        ``audio/*`` → ``send_audio``, everything else → ``send_document``)
        per wa-D-synthesis §D9.

        ``body`` is the raw bytes — NOT a path. The connector tool layer
        is responsible for reading files (and for enforcing any per-tool
        max-size cap from settings).
        """

    @abc.abstractmethod
    async def send_read_receipt(
        self, jid: str, message_ids: list[str]
    ) -> None:
        """Mark one or more messages as read on the server.

        Wraps ``client.mark_read(*message_ids, chat=jid, sender=jid,
        receipt=ReceiptType.READ)``. ``ReceiptType`` is a neonize enum
        that does NOT leak through the adapter surface — it is hardcoded
        to ``READ`` per wa-D-neonize.md §11 leak table.
        """

    # ------------------------------------------------------------------
    # Read paths
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def get_contacts(self) -> list[Contact]:
        """Return the user's contact list as Tether :class:`Contact`s.

        Concrete impl pulls neonize's contact list (protobuf) and maps
        each entry to a :class:`Contact` at the boundary — no protobuf
        types are exposed (wa-D-neonize.md §11).
        """

    @abc.abstractmethod
    async def fetch_history_sync(self) -> None:
        """Drive the one-shot historical sync that fires on first connect.

        WhatsApp Web (whatsmeow) exposes no arbitrary history fetch
        (wa-D-synthesis §D2). History arrives during initial connect via
        ``HistorySyncEv``; the adapter ingests those events into the
        same inbound queue that live ``MessageEv``s flow through, and
        Tether's :class:`SqliteInbox` becomes the local message archive
        that ``whatsapp_get_thread`` queries.

        This method is the explicit hook for "kick off / await the
        initial historical drain". On subsequent connects it is a no-op
        because neonize will not re-emit ``HistorySyncEv``.

        """

    # ------------------------------------------------------------------
    # Inbound event stream
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def subscribe_inbound(self) -> AsyncIterator[InboundEvent]:
        """Return the async iterator the connector drains for inbound
        events.

        Bridges neonize's callback-based event dispatch (sync callbacks
        invoked from Go threads via ``run_coroutine_threadsafe``,
        wa-D-neonize.md §12 #4) onto an :class:`asyncio.Queue` and
        exposes that queue as a cancellable async generator. The
        connector's :meth:`tether.connectors.base.Connector.inbound_stream`
        forwards from this iterator.

        Implementation contract (Wave 2):

        * Yield :class:`InboundEvent` values shaped per
          wa-D-synthesis §D7 (``payload`` keys: ``from``, ``from_name``,
          ``is_group``, ``group_name``, ``sender_e164``,
          ``text_snippet``, ``media_ref``, ``reply_to_id``,
          ``reply_to_snippet``, ``is_edit``, ``is_delete``).
        * ``event_id = f"{chat_jid}:{message_id}"`` (connector spec §4).
        * Filter ``fromMe == True`` messages — we do not ingest our own
          outgoing as inbound (single-user concern per wa-D-synthesis §D7).
        * Cancellation: the standard asyncio cancellation of the
          consuming task MUST drain cleanly (close the queue).

        NOT marked ``async def`` — mirrors the production adapter's
        intended factory shape: a sync method that returns an async
        iterator the connector can ``async for`` over without going
        through an extra ``await``.
        """

    # ------------------------------------------------------------------
    # Health snapshot (cheap; no network)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def health(self) -> AdapterHealthSnapshot:
        """Return the cached :class:`AdapterHealthSnapshot`.

        Cheap and synchronous — the adapter maintains the snapshot as
        side-effects of event handling (``ConnectedEv`` flips state to
        ``"connected"`` and stamps ``last_event_at``,
        ``DisconnectedEv`` flips to ``"reconnecting"``, etc.). The
        connector composes its own :class:`HealthStatus` from this
        snapshot per connector spec §3.1 (no network calls inside
        ``health()``).
        """


__all__ = [
    "WhatsAppClientAdapter",
    "AdapterHealthSnapshot",
    "Contact",
    "SendResult",
    "PairStatus",
]
