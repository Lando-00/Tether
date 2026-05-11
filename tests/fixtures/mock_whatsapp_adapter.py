"""In-memory WhatsApp adapter fixture for connector and tool tests."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

from tether.connectors.types import InboundEvent
from tether.connectors.whatsapp.adapter import (
    AdapterHealthSnapshot,
    Contact,
    PairStatus,
    SendResult,
    WhatsAppClientAdapter,
)


class MockWhatsAppClientAdapter(WhatsAppClientAdapter):
    """In-memory mock adapter for WhatsApp connector tests.

    Mirrors the spirit of tests/fixtures/echo_connector.py: side-effect
    storage is held on the instance so tests can inspect/mutate it without
    going through the ABC contract.
    """

    def __init__(self) -> None:
        self.outbox: list[SendResult] = []
        self.read_receipts: list[tuple[str, list[str]]] = []
        self.contacts: list[Contact] = []
        self.pending_inbound: asyncio.Queue[InboundEvent] = asyncio.Queue()
        self.next_pair_status: PairStatus = PairStatus.PAIRED
        self.qr_bytes: bytes = b"FAKE_QR_PNG_BYTES"
        self.sent_texts: list[tuple[str, str, SendResult]] = []
        self.sent_media: list[dict[str, object]] = []
        self._started = False
        self._state = "stopped"
        self._last_event_at: datetime | None = None
        self._last_error: str | None = None
        self._auth_dir: Path | None = None

    async def start(self, auth_dir: Path) -> None:
        self._auth_dir = auth_dir
        self._started = True
        self._state = "connecting"

    async def stop(self) -> None:
        self._started = False
        self._state = "stopped"

    async def logout(self) -> None:
        self._started = False
        self._state = "logged_out"

    async def pair_qr(self) -> bytes:
        return self.qr_bytes

    async def await_paired(self, timeout_sec: float) -> PairStatus:
        return self.next_pair_status

    async def send_text(self, to_jid: str, text: str) -> SendResult:
        result = SendResult(
            message_id=f"mock-text-{len(self.outbox) + 1}",
            to_jid=to_jid,
        )
        self.outbox.append(result)
        self.sent_texts.append((to_jid, text, result))
        return result

    async def send_media(
        self,
        to_jid: str,
        body: bytes,
        mime: str,
        filename: str,
        caption: str | None = None,
    ) -> SendResult:
        kind = mime.split("/", 1)[0] if "/" in mime else "document"
        if kind not in {"image", "video", "audio"}:
            kind = "document"
        result = SendResult(
            message_id=f"mock-media-{kind}-{len(self.outbox) + 1}",
            to_jid=to_jid,
        )
        self.outbox.append(result)
        self.sent_media.append(
            {
                "to_jid": to_jid,
                "body": body,
                "mime": mime,
                "filename": filename,
                "caption": caption,
                "kind": kind,
                "result": result,
            }
        )
        return result

    async def send_read_receipt(
        self,
        jid: str,
        message_ids: list[str],
    ) -> None:
        self.read_receipts.append((jid, list(message_ids)))

    async def get_contacts(self) -> list[Contact]:
        return list(self.contacts)

    async def fetch_history_sync(self) -> None:
        return None

    def subscribe_inbound(self) -> AsyncIterator[InboundEvent]:
        async def _gen() -> AsyncIterator[InboundEvent]:
            while True:
                try:
                    event = await self.pending_inbound.get()
                except asyncio.CancelledError:
                    break
                self._last_event_at = datetime.now(timezone.utc)
                yield event

        return _gen()

    def health(self) -> AdapterHealthSnapshot:
        return AdapterHealthSnapshot(
            state=self._state,
            last_event_at=self._last_event_at,
            last_error=self._last_error,
        )


__all__ = ["MockWhatsAppClientAdapter"]
