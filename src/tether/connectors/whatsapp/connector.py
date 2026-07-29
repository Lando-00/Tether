"""WhatsApp connector lifecycle implementation.

Wires the WhatsApp adapter, draft store, contacts cache, target resolver,
connector tools, and inbound stream behind the generic Connector ABC.
"""
from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, ClassVar, cast

import structlog

from tether.connectors.base import Connector
from tether.connectors.types import (
    AuthStatus,
    ConnectorState,
    HealthStatus,
    InboundEvent,
    LoginContinueResult,
    LoginPrompt,
)
from tether.connectors.whatsapp.adapter import (
    Contact,
    PairStatus,
    WhatsAppClientAdapter,
)
from tether.connectors.whatsapp.draft_store import DraftStore
from tether.core.errors import ConnectorNotConfiguredError
from tether.core.interfaces import Tool
from tether.core.secrets import EnvFileSecretsProvider, SecretsProvider

if TYPE_CHECKING:
    from tether.context.inbox_store import InboundInbox

logger = structlog.get_logger(__name__)


def _adapter_state_to_connector_state(state: str) -> ConnectorState:
    """Map adapter health string tags onto ConnectorState values."""
    normalized = (state or "").strip().lower()
    if normalized in {"ready", "connected"}:
        return ConnectorState.READY
    if normalized in {"connecting", "degraded", "reconnecting", "disconnected"}:
        return ConnectorState.DEGRADED
    if normalized in {"error", "failed"}:
        return ConnectorState.ERROR
    if normalized in {"logged_out", "logged-out", "logout"}:
        return ConnectorState.LOGGED_OUT
    if normalized in {"", "stopped", "unconfigured"}:
        return ConnectorState.UNCONFIGURED
    return ConnectorState.DEGRADED


def _qr_prompt_from_bytes(
    qr_bytes: bytes,
    *,
    expires_at: datetime | None = None,
) -> LoginPrompt:
    """Build a QR LoginPrompt from adapter bytes.

    Neonize callbacks should provide the raw WhatsApp QR string as UTF-8 bytes.
    If defensive code receives already-rendered PNG bytes, the PNG is exposed
    only via ``extra["png_b64"]`` and ``payload`` stays empty.
    """
    extra: dict[str, Any] = {}
    payload = qr_bytes.decode("utf-8", errors="ignore")
    if qr_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        png_b64 = base64.b64encode(qr_bytes).decode("ascii")
        extra["png_b64"] = png_b64
        payload = ""
    elif not payload:
        extra["bytes_b64"] = base64.b64encode(qr_bytes).decode("ascii")

    return LoginPrompt(
        kind="qr_code",
        payload=payload,
        expires_at=expires_at,
        extra=extra,
    )


class _InboundInboxProxy:
    """Delegates inbox calls to the connector's current injected inbox.

    Structurally an :class:`~tether.context.inbox_store.InboundInbox` — it
    implements every abstract method — but deliberately does not inherit it.
    ``InboundInbox`` is a TYPE_CHECKING-only import here so that importing
    the connector package does not drag in the context/store layer (R8 lazy
    imports). Call sites therefore pass it through ``_as_inbox``.
    """

    def __init__(self, owner: "WhatsAppConnector") -> None:
        self._owner = owner

    def _target(self) -> "InboundInbox":
        inbox = self._owner._inbox
        if inbox is None:
            raise ConnectorNotConfiguredError(
                "WhatsApp inbox not injected; ConnectorRegistry should call "
                "accept_inbox()"
            )
        return inbox

    async def append_many(self, events: list[InboundEvent]) -> int:
        return await self._target().append_many(events)

    async def list_unread(
        self,
        connector_id: str,
        limit: int = 50,
    ) -> list[InboundEvent]:
        return await self._target().list_unread(connector_id, limit)

    async def list_recent(
        self,
        connector_id: str,
        limit: int = 50,
    ) -> list[InboundEvent]:
        return await self._target().list_recent(connector_id, limit)

    async def mark_seen(self, connector_id: str, event_ids: list[str]) -> int:
        return await self._target().mark_seen(connector_id, event_ids)

    async def prune_older_than(self, retention_days: int) -> int:
        return await self._target().prune_older_than(retention_days)


def _as_inbox(proxy: _InboundInboxProxy) -> "InboundInbox":
    """Narrow the structural proxy to the nominal InboundInbox type.

    See :class:`_InboundInboxProxy` for why it cannot simply inherit.
    """
    return cast("InboundInbox", proxy)


class WhatsAppConnector(Connector):
    """WhatsApp Web connector — long-lived link to a single user account."""

    id: ClassVar[str] = "whatsapp"

    def __init__(
        self,
        *,
        secrets_provider: SecretsProvider | None = None,
        data_dir: Path | None = None,
        adapter_factory: Callable[[], WhatsAppClientAdapter] | None = None,
        inbox: "InboundInbox | None" = None,
        uuid: str = "tether-wa",
    ) -> None:
        self._secrets_provider: SecretsProvider | None = secrets_provider
        self._data_dir: Path | None = data_dir
        self._adapter_factory: Callable[[], WhatsAppClientAdapter] | None = (
            adapter_factory or self._default_adapter_factory
        )
        self._inbox: InboundInbox | None = inbox
        self._uuid = uuid

        self._draft_store = DraftStore()
        self._contacts_cache: list[Contact] = []
        self._state: ConnectorState = ConnectorState.UNCONFIGURED
        self._last_error: str | None = None
        self._auth_user_id: str | None = None
        self._state_sync_task: asyncio.Task | None = None
        self._inbound_task_handle: AsyncIterator[InboundEvent] | None = None
        self._adapter_started = False
        self._inbox_proxy = _InboundInboxProxy(self)
        self._adapter: WhatsAppClientAdapter | None = self._build_adapter()
        self._tools: dict[str, Tool] = self._build_tools()

    @property
    def state(self) -> ConnectorState:
        """Current connector state; primarily useful in tests."""
        return self._state

    # ------------------------------------------------------------------
    # Registry injection seam
    # ------------------------------------------------------------------

    def accept_inbox(self, inbox: "InboundInbox") -> None:
        """Inject the shared inbox after construction.

        ConnectorRegistry owns SqliteInbox construction and cannot pass Python
        object references through YAML connector args. Tools receive an inbox
        proxy at construction, so updating ``self._inbox`` here is enough for
        already-aggregated tools to see the injected inbox.
        """
        self._inbox = inbox

    # ------------------------------------------------------------------
    # Connector ABC: lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start from on-disk credentials, if present."""
        if self._adapter is not None and self._adapter_started and (
            self._state is not ConnectorState.UNCONFIGURED
        ):
            return

        self._ensure_secrets_provider()
        auth_dir = self._resolve_data_dir() / "auth"
        creds_present = (auth_dir / "neonize.db").exists()
        if not creds_present:
            self._state = ConnectorState.UNCONFIGURED
            self._last_error = None
            return

        if self._adapter is None:
            self._adapter = self._build_adapter()
            self._tools = self._build_tools()

        self._state = ConnectorState.DEGRADED
        self._last_error = "connecting_on_resume"
        try:
            await self._adapter.start(auth_dir)
            self._adapter_started = True
        except Exception as exc:
            self._state = ConnectorState.ERROR
            self._last_error = str(exc)
            raise

        try:
            self._set_contacts(await self._adapter.get_contacts())
        except Exception as exc:  # noqa: BLE001 - cache refresh is best effort
            logger.warning(
                "whatsapp.contacts_refresh_failed",
                error=str(exc),
                connector_id=self.id,
            )
            self._set_contacts([])

        if self._state_sync_task is None or self._state_sync_task.done():
            self._state_sync_task = asyncio.create_task(self._poll_adapter_health())

    async def stop(self) -> None:
        """Stop the adapter and background state poller; keep credentials."""
        if self._state_sync_task and not self._state_sync_task.done():
            self._state_sync_task.cancel()
            try:
                await self._state_sync_task
            except asyncio.CancelledError:
                pass
        self._state_sync_task = None
        self._inbound_task_handle = None

        if self._adapter is not None:
            try:
                await self._adapter.stop()
            except Exception as exc:  # noqa: BLE001 - stop is best effort
                logger.warning(
                    "whatsapp.adapter_stop_error",
                    error=str(exc),
                    connector_id=self.id,
                )
            finally:
                self._adapter_started = False

        if self._state is ConnectorState.READY:
            self._state = ConnectorState.UNCONFIGURED

    async def logout(self) -> None:
        """Logout server-side where possible, delete creds, and mark logged out."""
        if self._adapter is not None:
            try:
                await self._adapter.logout()
            except Exception as exc:  # noqa: BLE001 - still delete local creds
                logger.warning(
                    "whatsapp.logout_error",
                    error=str(exc),
                    connector_id=self.id,
                )

        try:
            await self.stop()
        except Exception:  # noqa: BLE001 - logout must still delete creds
            pass

        creds_db = self._resolve_data_dir() / "auth" / "neonize.db"
        if creds_db.exists():
            try:
                creds_db.unlink()
            except OSError as exc:
                logger.warning(
                    "whatsapp.creds_delete_failed",
                    error=str(exc),
                    connector_id=self.id,
                )

        self._auth_user_id = None
        self._last_error = None
        self._adapter_started = False
        self._state = ConnectorState.LOGGED_OUT

    # ------------------------------------------------------------------
    # Connector ABC: status (cheap; no network)
    # ------------------------------------------------------------------

    async def health(self) -> HealthStatus:
        """Return a cheap connector health snapshot."""
        snap = self._adapter.health() if self._adapter else None
        detail = (snap.last_error if snap and snap.last_error else None) or self._last_error
        return HealthStatus(
            state=self._state,
            detail=detail,
            last_success=None,
            last_error=snap.last_event_at if snap and snap.last_error else None,
            last_error_message=snap.last_error if snap else None,
        )

    async def auth_status(self) -> AuthStatus:
        """Return auth status without importing or touching neonize pre-start."""
        if not self._adapter_started:
            if self._state is ConnectorState.LOGGED_OUT:
                return AuthStatus(state=ConnectorState.LOGGED_OUT)
            creds_path = self._resolve_data_dir() / "auth" / "neonize.db"
            if creds_path.exists():
                return AuthStatus(
                    state=ConnectorState.READY,
                    user_id=None,
                    detail="creds present; first connect pending",
                )
            return AuthStatus(state=ConnectorState.UNCONFIGURED)

        return AuthStatus(
            state=self._state,
            user_id=self._auth_user_id,
            detail=self._last_error,
        )

    # ------------------------------------------------------------------
    # Connector ABC: login flow
    # ------------------------------------------------------------------

    async def begin_login(self) -> LoginPrompt:
        """Start the QR-pair flow and return the first QR prompt."""
        if self._adapter is None:
            self._adapter = self._build_adapter()
            self._tools = self._build_tools()

        auth_dir = self._resolve_data_dir() / "auth"
        auth_dir.mkdir(parents=True, exist_ok=True)
        self._state = ConnectorState.AUTHENTICATING
        self._last_error = None
        try:
            await self._adapter.start(auth_dir)
            self._adapter_started = True
            qr_bytes = await self._adapter.pair_qr()
        except Exception as exc:
            self._state = ConnectorState.ERROR
            self._last_error = str(exc)
            raise

        return _qr_prompt_from_bytes(
            qr_bytes,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=3),
        )

    async def complete_login(
        self,
        *,
        payload: dict[str, Any],
    ) -> LoginContinueResult:
        """Long-poll for pair completion, QR rotation, or terminal failure."""
        if self._adapter is None:
            self._state = ConnectorState.ERROR
            self._last_error = "adapter not initialised"
            return LoginContinueResult(
                state=ConnectorState.ERROR,
                detail="adapter not initialised",
            )

        timeout_sec = float(payload.get("timeout_sec", 60.0))
        status = await self._adapter.await_paired(timeout_sec)

        if status is PairStatus.PAIRED:
            try:
                self._set_contacts(await self._adapter.get_contacts())
            except Exception:  # noqa: BLE001 - first-pair refresh is best effort
                self._set_contacts([])
            if self._state_sync_task is None or self._state_sync_task.done():
                self._state_sync_task = asyncio.create_task(
                    self._poll_adapter_health()
                )
            self._state = ConnectorState.READY
            self._last_error = None
            return LoginContinueResult(state=ConnectorState.READY)

        if status is PairStatus.QR_ROTATED:
            self._state = ConnectorState.AUTHENTICATING
            qr_bytes = await self._adapter.pair_qr()
            return LoginContinueResult(
                state=ConnectorState.AUTHENTICATING,
                next_prompt=_qr_prompt_from_bytes(
                    qr_bytes,
                    expires_at=datetime.now(timezone.utc) + timedelta(minutes=3),
                ),
            )

        if status is PairStatus.LOGGED_OUT:
            self._state = ConnectorState.LOGGED_OUT
            self._last_error = "logged_out_during_pair"
            return LoginContinueResult(
                state=ConnectorState.LOGGED_OUT,
                detail="logged_out_during_pair",
            )

        if status is PairStatus.FAILED:
            snap = self._adapter.health()
            detail = snap.last_error or "auth_failed_during_pair"
            self._state = ConnectorState.ERROR
            self._last_error = detail
            return LoginContinueResult(state=ConnectorState.ERROR, detail=detail)

        self._state = ConnectorState.AUTHENTICATING
        return LoginContinueResult(
            state=ConnectorState.AUTHENTICATING,
            detail="qr_scan_timeout",
        )

    # ------------------------------------------------------------------
    # Connector ABC: outbound + inbound
    # ------------------------------------------------------------------

    def tools(self) -> dict[str, Tool]:
        """Return the 9 WhatsApp tools, keyed by canonical names."""
        return dict(self._tools)

    async def inbound_stream(self) -> AsyncIterator[InboundEvent]:
        """Forward inbound events from the adapter's async iterator."""
        if self._adapter is None:
            return

        iterator = self._adapter.subscribe_inbound()
        self._inbound_task_handle = iterator
        try:
            async for event in iterator:
                yield event
        except asyncio.CancelledError:
            raise
        finally:
            self._inbound_task_handle = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_adapter(self) -> WhatsAppClientAdapter:
        if self._adapter_factory is None:
            self._adapter_factory = self._default_adapter_factory
        return self._adapter_factory()

    def _default_adapter_factory(self) -> WhatsAppClientAdapter:
        from tether.connectors.whatsapp.neonize_adapter import (
            NeonizeWhatsAppClientAdapter,
        )

        return NeonizeWhatsAppClientAdapter(uuid=self._uuid)

    def _build_tools(self) -> dict[str, Tool]:
        from tether.connectors.whatsapp.target_resolver import resolve_target
        from tether.connectors.whatsapp.tools import (
            WhatsAppConfirmSendTool,
            WhatsAppGetContactsTool,
            WhatsAppGetThreadTool,
            WhatsAppInboxMarkSeenTool,
            WhatsAppListUnreadTool,
            WhatsAppMarkPlatformReadTool,
            WhatsAppPrepareSendTool,
            WhatsAppResolveContactTool,
            WhatsAppSendMediaTool,
        )

        if self._adapter is None:
            raise ConnectorNotConfiguredError("WhatsApp adapter not initialised")

        def _resolver_fn(value: str) -> str | None:
            return resolve_target(value, self._contacts_cache)

        def _state_provider() -> ConnectorState:
            return self._state

        return {
            "whatsapp_prepare_send": WhatsAppPrepareSendTool(
                adapter=self._adapter,
                draft_store=self._draft_store,
                target_resolver_fn=_resolver_fn,
                state_provider=_state_provider,
            ),
            "whatsapp_confirm_send": WhatsAppConfirmSendTool(
                adapter=self._adapter,
                draft_store=self._draft_store,
                state_provider=_state_provider,
            ),
            "whatsapp_list_unread": WhatsAppListUnreadTool(
                inbox=_as_inbox(self._inbox_proxy),
                state_provider=_state_provider,
            ),
            "whatsapp_get_thread": WhatsAppGetThreadTool(
                inbox=_as_inbox(self._inbox_proxy),
                target_resolver_fn=_resolver_fn,
                state_provider=_state_provider,
            ),
            "whatsapp_inbox_mark_seen": WhatsAppInboxMarkSeenTool(
                inbox=_as_inbox(self._inbox_proxy),
                state_provider=_state_provider,
            ),
            "whatsapp_mark_platform_read": WhatsAppMarkPlatformReadTool(
                adapter=self._adapter,
                state_provider=_state_provider,
            ),
            "whatsapp_send_media": WhatsAppSendMediaTool(
                adapter=self._adapter,
                draft_store=self._draft_store,
                target_resolver_fn=_resolver_fn,
                state_provider=_state_provider,
            ),
            "whatsapp_get_contacts": WhatsAppGetContactsTool(
                adapter=self._adapter,
                contacts_cache=self._contacts_cache,
                state_provider=_state_provider,
            ),
            "whatsapp_resolve_contact": WhatsAppResolveContactTool(
                adapter=self._adapter,
                contacts_cache=self._contacts_cache,
                state_provider=_state_provider,
            ),
        }

    def _resolve_data_dir(self) -> Path:
        if self._data_dir is not None:
            return self._data_dir
        try:
            from platformdirs import user_data_dir  # type: ignore[import-not-found]

            self._data_dir = Path(user_data_dir("Tether")) / "connectors" / self.id
        except ImportError:
            self._data_dir = Path("data") / "connectors" / self.id
        return self._data_dir

    def _ensure_secrets_provider(self) -> SecretsProvider:
        if self._secrets_provider is None:
            self._secrets_provider = EnvFileSecretsProvider(
                data_dir=self._resolve_data_dir()
            )
        return self._secrets_provider

    def _set_contacts(self, contacts: list[Contact]) -> None:
        self._contacts_cache.clear()
        self._contacts_cache.extend(contacts)

    async def _poll_adapter_health(self) -> None:
        try:
            while True:
                await asyncio.sleep(1.0)
                try:
                    if self._adapter is None:
                        continue
                    snap = self._adapter.health()
                    new_state = _adapter_state_to_connector_state(snap.state)
                    self._state = new_state
                    if snap.last_error:
                        self._last_error = snap.last_error
                    elif new_state is ConnectorState.DEGRADED and (
                        snap.state or ""
                    ).strip().lower() == "connecting":
                        self._last_error = "connecting_on_resume"
                    else:
                        self._last_error = None
                except Exception as exc:
                    logger.warning(
                        "whatsapp.poller_error",
                        error=str(exc),
                        connector_id=self.id,
                    )
                    continue
        except asyncio.CancelledError:
            return


__all__ = ["WhatsAppConnector"]
