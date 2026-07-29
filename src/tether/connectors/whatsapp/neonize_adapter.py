"""neonize-backed implementation of :mod:`tether.connectors.whatsapp.adapter`.

The module is intentionally importable without neonize installed. All neonize
imports are deferred until methods that actually touch the WhatsApp client.
"""
from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import AsyncIterator, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from tether.connectors.types import InboundEvent
from tether.connectors.whatsapp.adapter import (
    AdapterHealthSnapshot,
    Contact,
    PairStatus,
    SendResult,
    WhatsAppClientAdapter,
)

logger = structlog.get_logger("tether.whatsapp.adapter")

_LOG_BASE = {"connector_id": "whatsapp", "adapter": "neonize"}


class NeonizeWhatsAppClientAdapter(WhatsAppClientAdapter):
    """Concrete WhatsApp adapter backed by ``neonize.aioze``.

    After :meth:`start` returns, state is ``"connecting"`` until neonize emits
    ``ConnectedEv``. The owning connector maps that snapshot to
    ``DEGRADED`` with detail ``"connecting_on_resume"``.
    """

    def __init__(self, *, uuid: str = "tether-wa") -> None:
        self._uuid = uuid
        self._auth_dir: Path | None = None
        # neonize is an optional extra and ships no type information, so the
        # client/factory are Any. Without the annotation mypy infers the
        # declared type as None from this initialiser and then rejects every
        # attribute access on them after they are assigned in start().
        self._client: Any = None
        self._factory: Any = None
        self._started = False
        self._connect_task: asyncio.Task | None = None
        self._qr_future: asyncio.Future[bytes] | None = None
        self._latest_qr_bytes: bytes | None = None
        self._pair_future: asyncio.Future[PairStatus] | None = None
        self._pending_pair_outcome: PairStatus | None = None
        self._inbound_queue: asyncio.Queue[InboundEvent] = asyncio.Queue(
            maxsize=10000
        )
        self._last_event_at: datetime | None = None
        self._last_error: str | None = None
        self._state: str = "stopped"
        self._lock = asyncio.Lock()

    async def start(self, auth_dir: Path) -> None:
        """Start the neonize session and enter ``connecting`` state."""

        async with self._lock:
            if self._started:
                return
            self._auth_dir = auth_dir
            self._auth_dir = auth_dir
            self._last_error = None

            try:
                import neonize.aioze.events as _ev

                _ev.event_global_loop = None
                from neonize.aioze.client import ClientFactory
                from neonize.aioze.events import (
                    ClientOutdatedEv,
                    ConnectedEv,
                    ConnectFailureEv,
                    DisconnectedEv,
                    HistorySyncEv,
                    KeepAliveRestoredEv,
                    KeepAliveTimeoutEv,
                    LoggedOutEv,
                    MessageEv,
                    PairStatusEv,
                    QREv,
                    ReceiptEv,
                    StreamReplacedEv,
                    TemporaryBanEv,
                )
            except ImportError as exc:
                raise _neonize_missing() from exc

            auth_dir.mkdir(parents=True, exist_ok=True)
            _chmod_session_db(auth_dir)
            self._factory = ClientFactory(str(auth_dir / "neonize.db"))
            self._client = self._factory.new_client(uuid=self._uuid)
            client = self._client

            @client.event(QREv)
            async def _on_qr(_client: Any, event: Any) -> None:
                self._record_event("QREv")
                qr_bytes = _extract_qr_bytes(event)
                self._latest_qr_bytes = qr_bytes
                if self._qr_future and not self._qr_future.done():
                    self._qr_future.set_result(qr_bytes)

            @client.event(PairStatusEv)
            async def _on_pair_status(_client: Any, _event: Any) -> None:
                self._record_event("PairStatusEv")

            @client.event(ConnectedEv)
            async def _on_connected(_client: Any, _event: Any) -> None:
                self._record_event("ConnectedEv", state="ready")
                self._set_pair_outcome(PairStatus.PAIRED)

            @client.event(KeepAliveTimeoutEv)
            async def _on_keepalive_timeout(_client: Any, _event: Any) -> None:
                self._record_event("KeepAliveTimeoutEv", state="degraded")

            @client.event(KeepAliveRestoredEv)
            async def _on_keepalive_restored(_client: Any, _event: Any) -> None:
                self._record_event("KeepAliveRestoredEv", state="ready")

            @client.event(LoggedOutEv)
            async def _on_logged_out(_client: Any, _event: Any) -> None:
                self._record_event("LoggedOutEv", state="logged_out")
                self._set_pair_outcome(PairStatus.LOGGED_OUT)

            @client.event(DisconnectedEv)
            async def _on_disconnected(_client: Any, _event: Any) -> None:
                self._record_event("DisconnectedEv", state="degraded")

            @client.event(ConnectFailureEv)
            async def _on_connect_failure(_client: Any, event: Any) -> None:
                self._record_event(
                    "ConnectFailureEv",
                    state="degraded",
                    error=_event_error(event),
                )

            @client.event(TemporaryBanEv)
            async def _on_temporary_ban(_client: Any, event: Any) -> None:
                self._record_event(
                    "TemporaryBanEv",
                    state="error",
                    error=_event_error(event) or "temporary_ban",
                )
                self._set_pair_outcome(PairStatus.FAILED)

            @client.event(ClientOutdatedEv)
            async def _on_client_outdated(_client: Any, event: Any) -> None:
                self._record_event(
                    "ClientOutdatedEv",
                    state="error",
                    error=_event_error(event) or "client_outdated",
                )
                self._set_pair_outcome(PairStatus.FAILED)

            @client.event(StreamReplacedEv)
            async def _on_stream_replaced(_client: Any, _event: Any) -> None:
                self._record_event(
                    "StreamReplacedEv",
                    state="error",
                    error="session_conflict",
                )
                self._set_pair_outcome(PairStatus.FAILED)

            @client.event(MessageEv)
            async def _on_message(_client: Any, event: Any) -> None:
                self._record_event("MessageEv")
                inbound = _message_ev_to_inbound_event(event)
                if inbound is not None:
                    await self._enqueue_inbound(inbound, event_kind="MessageEv")

            @client.event(ReceiptEv)
            async def _on_receipt(_client: Any, event: Any) -> None:
                self._record_event("ReceiptEv")
                inbound = _receipt_ev_to_inbound_event(event)
                await self._enqueue_inbound(inbound, event_kind="ReceiptEv")

            @client.event(HistorySyncEv)
            async def _on_history_sync(_client: Any, event: Any) -> None:
                self._record_event("HistorySyncEv")
                for message in _history_sync_messages(event):
                    inbound = _message_ev_to_inbound_event(message)
                    if inbound is not None:
                        await self._enqueue_inbound(
                            inbound,
                            event_kind="HistorySyncEv",
                        )

            self._state = "connecting"
            self._connect_task = asyncio.create_task(client.connect())
            self._connect_task.add_done_callback(self._on_connect_done)
            self._started = True

    async def stop(self) -> None:
        if not self._started:
            return

        self._started = False
        self._state = "stopped"

        if self._connect_task and not self._connect_task.done():
            self._connect_task.cancel()

        try:
            await asyncio.wait_for(self._do_disconnect(), timeout=1.5)
        except TimeoutError:
            logger.warning(
                "whatsapp.adapter.stop_timeout",
                **_LOG_BASE,
            )

        try:
            import neonize.aioze.events as _ev

            _ev.event_global_loop = None
        except ImportError:
            pass

        self._connect_task = None
        self._qr_future = None
        self._pair_future = None
        self._pending_pair_outcome = None
        self._latest_qr_bytes = None
        self._client = None
        self._factory = None

    async def logout(self) -> None:
        try:
            if self._client:
                await self._client.logout()
        except Exception as exc:
            logger.warning(
                "whatsapp.adapter.logout_failed",
                **_LOG_BASE,
                error=str(exc),
            )
        finally:
            try:
                await self.stop()
            except Exception:
                pass
            if self._auth_dir and (self._auth_dir / "neonize.db").exists():
                try:
                    (self._auth_dir / "neonize.db").unlink()
                except OSError as exc:
                    logger.warning(
                        "whatsapp.adapter.db_delete_failed",
                        **_LOG_BASE,
                        error=str(exc),
                    )
            self._state = "logged_out"

    async def pair_qr(self) -> bytes:
        if self._latest_qr_bytes is not None:
            return self._latest_qr_bytes
        if self._qr_future is None or self._qr_future.done():
            self._qr_future = asyncio.get_running_loop().create_future()
        return await self._qr_future

    async def await_paired(self, timeout_sec: float) -> PairStatus:
        if self._pending_pair_outcome is not None:
            outcome = self._pending_pair_outcome
            self._pending_pair_outcome = None
            return outcome

        self._pair_future = asyncio.get_running_loop().create_future()
        qr_seen_before = self._latest_qr_bytes
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_sec

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return PairStatus.TIMEOUT
            try:
                outcome = await asyncio.wait_for(
                    asyncio.shield(self._pair_future),
                    timeout=min(remaining, 1.0),
                )
                if self._pending_pair_outcome is outcome:
                    self._pending_pair_outcome = None
                return outcome
            except TimeoutError:
                if (
                    self._latest_qr_bytes is not None
                    and self._latest_qr_bytes != qr_seen_before
                ):
                    return PairStatus.QR_ROTATED

    async def send_text(self, to_jid: str, text: str) -> SendResult:
        client = self._require_client()
        try:
            from neonize.exc import SendMessageError
        except ImportError as exc:
            raise _neonize_missing() from exc

        try:
            response = await client.send_message(to_jid, text)
        except SendMessageError as exc:
            raise RuntimeError(f"send_text failed: {exc}") from exc

        message_id = str(response.ID)
        logger.info(
            "whatsapp.adapter.send_text",
            **_LOG_BASE,
            chat_jid=to_jid,
            message_id=message_id,
        )
        return SendResult(message_id=message_id, to_jid=to_jid)

    async def send_media(
        self,
        to_jid: str,
        body: bytes,
        mime: str,
        filename: str,
        caption: str | None = None,
    ) -> SendResult:
        client = self._require_client()
        try:
            from neonize.exc import SendMessageError
        except ImportError as exc:
            raise _neonize_missing() from exc

        try:
            if mime.startswith("image/"):
                response = await client.send_image(to_jid, body, caption=caption)
            elif mime.startswith("video/"):
                response = await client.send_video(to_jid, body, caption=caption)
            elif mime.startswith("audio/"):
                response = await client.send_audio(to_jid, body)
            else:
                response = await client.send_document(
                    to_jid,
                    body,
                    caption=caption,
                    filename=filename,
                )
        except SendMessageError as exc:
            raise RuntimeError(f"send_media failed: {exc}") from exc

        message_id = str(response.ID)
        logger.info(
            "whatsapp.adapter.send_media",
            **_LOG_BASE,
            chat_jid=to_jid,
            message_id=message_id,
        )
        return SendResult(message_id=message_id, to_jid=to_jid)

    async def send_read_receipt(
        self,
        jid: str,
        message_ids: list[str],
    ) -> None:
        client = self._require_client()
        try:
            from neonize.exc import MarkReadError
            from neonize.utils.enum import ReceiptType
        except ImportError as exc:
            raise _neonize_missing() from exc

        try:
            await client.mark_read(
                *message_ids,
                chat=jid,
                sender=jid,
                receipt=ReceiptType.READ,
            )
        except MarkReadError as exc:
            raise RuntimeError(f"send_read_receipt failed: {exc}") from exc

        logger.info(
            "whatsapp.adapter.send_read_receipt",
            **_LOG_BASE,
            chat_jid=jid,
            message_ids=message_ids,
        )

    async def get_contacts(self) -> list[Contact]:
        client = self._require_client()
        try:
            from neonize.utils.jid import Jid2String
        except ImportError as exc:
            raise _neonize_missing() from exc

        raw = await client.contact.get_all_contacts()
        return [
            Contact(
                jid=Jid2String(c.JID),
                name=c.PushName or c.BusinessName or None,
                e164=_jid_to_e164(c.JID),
                is_business=bool(c.BusinessName),
            )
            for c in raw
        ]

    async def fetch_history_sync(self) -> None:
        return None

    def subscribe_inbound(self) -> AsyncIterator[InboundEvent]:
        async def _gen() -> AsyncIterator[InboundEvent]:
            while True:
                try:
                    event = await self._inbound_queue.get()
                except asyncio.CancelledError:
                    break
                yield event

        return _gen()

    def health(self) -> AdapterHealthSnapshot:
        return AdapterHealthSnapshot(
            state=self._state,
            last_event_at=self._last_event_at,
            last_error=self._last_error,
        )

    def _record_event(
        self,
        event_kind: str,
        *,
        state: str | None = None,
        error: str | None = None,
    ) -> None:
        self._last_event_at = datetime.now(timezone.utc)
        if state is not None:
            self._state = state
        if error is not None:
            self._last_error = error
        logger.debug(
            "whatsapp.adapter.event",
            **_LOG_BASE,
            event_kind=event_kind,
        )

    async def _enqueue_inbound(
        self,
        event: InboundEvent,
        *,
        event_kind: str,
    ) -> None:
        if self._inbound_queue.full():
            logger.warning(
                "whatsapp.adapter.inbound_queue_full",
                **_LOG_BASE,
                event_kind=event_kind,
                message_id=event.event_id,
                chat_jid=event.payload.get("from"),
            )
            return
        await self._inbound_queue.put(event)

    async def _do_disconnect(self) -> None:
        client = self._client
        if client is None:
            return
        try:
            await client.stop()
        except Exception as exc:
            logger.warning(
                "whatsapp.adapter.stop_failed",
                **_LOG_BASE,
                error=str(exc),
            )

    def _on_connect_done(self, task: asyncio.Task) -> None:
        if self._auth_dir is not None:
            _chmod_session_db(self._auth_dir)
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is None:
            return
        logger.warning(
            "whatsapp.adapter.connect_task_failed",
            **_LOG_BASE,
            error=str(exc),
        )
        if self._state not in ("stopped", "logged_out"):
            self._state = "error"
            self._last_error = str(exc)

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("WhatsApp adapter is not started")
        return self._client

    def _set_pair_outcome(self, outcome: PairStatus) -> None:
        self._pending_pair_outcome = outcome
        if self._pair_future and not self._pair_future.done():
            self._pair_future.set_result(outcome)


def _neonize_missing() -> ImportError:
    return ImportError("neonize is not installed; pip install tether[whatsapp]")


def _chmod_session_db(auth_dir: Path) -> None:
    if os.name == "nt":
        return
    db_path = auth_dir / "neonize.db"
    if not db_path.exists():
        return
    try:
        db_path.chmod(0o600)
    except OSError:
        pass


def _extract_qr_bytes(event: Any) -> bytes:
    if isinstance(event, bytes):
        return event
    if isinstance(event, bytearray):
        return bytes(event)

    for attr in (
        "qr",
        "QR",
        "qr_code",
        "QRCode",
        "qr_bytes",
        "code",
        "Code",
        "Bytes",
        "Data",
        "data",
    ):
        value = getattr(event, attr, None)
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        if isinstance(value, str):
            return value.encode("utf-8")

    text = str(event)
    return text.encode("utf-8")


def _message_ev_to_inbound_event(ev: Any) -> InboundEvent | None:
    try:
        info = ev.Info
        msg = ev.Message
        source = info.MessageSource
        if bool(source.IsFromMe):
            return None

        chat_jid = _jid_to_string(source.Chat)
        msg_id = _safe_str(getattr(info, "ID", ""))
        is_group = bool(source.IsGroup)
        sender_jid = _jid_to_string(source.Sender)
        sender_e164 = _jid_to_e164(source.Sender)
        text = _extract_text(msg)
        text_snippet = text[:4000]
        is_edit = bool(getattr(ev, "IsEdit", False))
        is_delete = bool(
            _has_field(msg, "protocolMessage")
            and getattr(msg.protocolMessage, "type", None) == 0
        )
        media_ref = _extract_media_ref(msg)
        if media_ref:
            capped_media_ref = {
                "kind": media_ref["kind"],
                "mimetype": _cap(media_ref.get("mimetype", ""), 128) or "",
            }
            filename = _cap(media_ref.get("filename"), 256)
            if filename is not None:
                capped_media_ref["filename"] = filename
            media_ref = capped_media_ref
        reply_to_id, reply_to_snippet = _extract_reply(msg)
        push_name = _cap(_safe_str(getattr(info, "PushName", ""))) or None
        received_at = _message_received_at(info)
        summary = _cap(
            f"WhatsApp from {_cap(push_name, 64) or sender_e164 or sender_jid}: "
            f"{text_snippet[:80]}",
            300,
        )
        payload = {
            "from": chat_jid,
            "from_name": push_name,
            "is_group": is_group,
            "group_name": None,
            "sender_e164": sender_e164,
            "text_snippet": text_snippet,
            "media_ref": media_ref,
            "reply_to_id": reply_to_id,
            "reply_to_snippet": _cap(reply_to_snippet, 200),
            "is_edit": is_edit,
            "is_delete": is_delete,
        }
        return InboundEvent(
            event_id=f"{chat_jid}:{msg_id}",
            connector_id="whatsapp",
            kind="message",
            received_at=received_at,
            payload=payload,
            summary=summary,
        )
    except Exception as exc:
        logger.warning(
            "whatsapp.adapter.message_mapping_failed",
            **_LOG_BASE,
            event_kind="MessageEv",
            error=str(exc),
        )
        return None


def _extract_text(msg: Any) -> str:
    conversation = _safe_str(getattr(msg, "conversation", ""))
    if conversation:
        return conversation
    if _has_field(msg, "extendedTextMessage"):
        return _safe_str(getattr(msg.extendedTextMessage, "text", ""))
    return ""


def _extract_media_ref(msg: Any) -> dict[str, str] | None:
    for kind, attr in (
        ("image", "imageMessage"),
        ("video", "videoMessage"),
        ("audio", "audioMessage"),
        ("document", "documentMessage"),
        ("sticker", "stickerMessage"),
    ):
        if not _has_field(msg, attr):
            continue
        media_msg = getattr(msg, attr)
        media_ref = {
            "kind": kind,
            "mimetype": _safe_str(getattr(media_msg, "mimetype", "")),
            "filename": (
                _safe_str(getattr(media_msg, "fileName", ""))
                if attr == "documentMessage"
                else None
            ),
        }
        return {k: v for k, v in media_ref.items() if v is not None}
    return None


def _extract_reply(msg: Any) -> tuple[str | None, str | None]:
    if not _has_field(msg, "extendedTextMessage"):
        return None, None

    ctx = getattr(msg.extendedTextMessage, "contextInfo", None)
    if ctx is None:
        return None, None

    stanza_id = _safe_str(getattr(ctx, "stanzaId", ""))
    if not stanza_id:
        return None, None

    reply_to_snippet = None
    if _has_field(ctx, "quotedMessage"):
        quoted = getattr(ctx, "quotedMessage", None)
        reply_to_snippet = _safe_str(getattr(quoted, "conversation", ""))[:200]
    return stanza_id, reply_to_snippet


def _message_received_at(info: Any) -> datetime:
    timestamp = getattr(info, "Timestamp", None)
    if isinstance(timestamp, int | float) and timestamp > 0:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return datetime.now(timezone.utc)


def _receipt_ev_to_inbound_event(ev: Any) -> InboundEvent:
    chat_jid = _jid_to_string(
        getattr(ev, "Chat", None)
        or getattr(ev, "chat", None)
        or getattr(ev, "JID", None)
        or getattr(ev, "jid", "")
    )
    ids = _normalise_message_ids(
        getattr(ev, "MessageIDs", None)
        or getattr(ev, "message_ids", None)
        or getattr(ev, "IDs", None)
        or getattr(ev, "ids", None)
    )
    received_at = datetime.now(timezone.utc)
    event_id = f"{chat_jid}:receipt:{','.join(ids) or received_at.timestamp()}"
    payload = {
        "from": chat_jid,
        "message_ids": ids,
        "receipt_type": _safe_str(
            getattr(ev, "Type", None) or getattr(ev, "type", "")
        ),
    }
    return InboundEvent(
        event_id=event_id,
        connector_id="whatsapp",
        kind="receipt",
        received_at=received_at,
        payload=payload,
        summary=f"WhatsApp receipt from {chat_jid}",
    )


def _history_sync_messages(ev: Any) -> Iterable[Any]:
    yield from _walk_history(ev, seen=set())


def _walk_history(obj: Any, *, seen: set[int]) -> Iterable[Any]:
    if obj is None:
        return
    ident = id(obj)
    if ident in seen:
        return
    seen.add(ident)

    if hasattr(obj, "Info") and hasattr(obj, "Message"):
        yield obj
        return

    if isinstance(obj, Iterable) and not isinstance(obj, str | bytes | bytearray):
        for item in obj:
            yield from _walk_history(item, seen=seen)
        return

    for attr in (
        "Data",
        "data",
        "HistorySync",
        "historySync",
        "historySyncMessage",
        "Conversations",
        "conversations",
        "Conversation",
        "conversation",
        "Messages",
        "messages",
        "Message",
        "message",
    ):
        value = getattr(obj, attr, None)
        if value is not None:
            yield from _walk_history(value, seen=seen)


def _normalise_message_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def _jid_to_e164(jid: Any) -> str | None:
    jid_str = _jid_to_string(jid)
    if "@" not in jid_str:
        return None
    local, domain = jid_str.split("@", 1)
    local = local.split(":", 1)[0].removeprefix("+")
    if domain not in {"s.whatsapp.net", "c.us"}:
        return None
    if not local.isdigit():
        return None
    return f"+{local}"


def _jid_to_string(jid: Any) -> str:
    if isinstance(jid, str):
        return jid
    try:
        from neonize.utils.jid import Jid2String
    except ImportError:
        return str(jid)
    try:
        return Jid2String(jid)
    except Exception:
        return str(jid)


def _safe_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if inspect.isroutine(value):
        return ""
    module_name = type(value).__module__
    if module_name.startswith("unittest.mock"):
        return ""
    return str(value)


def _cap(s: str | None, n: int = 256) -> str | None:
    if s is None:
        return None
    return s[:n]


def _has_field(obj: Any, name: str) -> bool:
    has_field = getattr(obj, "HasField", None)
    if callable(has_field):
        try:
            result = has_field(name)
            if isinstance(result, bool):
                return result
        except (TypeError, ValueError):
            pass
    value = getattr(obj, name, None)
    if value is None:
        return False
    return not type(value).__module__.startswith("unittest.mock")


def _event_error(event: Any) -> str | None:
    for attr in ("Error", "error", "Reason", "reason", "Message", "message"):
        value = _safe_str(getattr(event, attr, ""))
        if value:
            return value
    return None


__all__ = ["NeonizeWhatsAppClientAdapter"]
