from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.fixtures.mock_whatsapp_adapter import MockWhatsAppClientAdapter
from tether.connectors.types import InboundEvent
from tether.connectors.whatsapp.adapter import (
    Contact,
    PairStatus,
    WhatsAppClientAdapter,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_whatsappclientadapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        WhatsAppClientAdapter()

    expected = {
        "start",
        "stop",
        "logout",
        "pair_qr",
        "await_paired",
        "send_text",
        "send_media",
        "send_read_receipt",
        "get_contacts",
        "fetch_history_sync",
        "subscribe_inbound",
        "health",
    }
    assert expected <= WhatsAppClientAdapter.__abstractmethods__


def test_pair_status_enum_has_five_members() -> None:
    assert {member.name for member in PairStatus} == {
        "PAIRED",
        "QR_ROTATED",
        "TIMEOUT",
        "LOGGED_OUT",
        "FAILED",
    }


@pytest.mark.anyio
async def test_mock_adapter_outbox_records_send_text() -> None:
    mock = MockWhatsAppClientAdapter()

    result = await mock.send_text("+1@s.whatsapp.net", "hi")

    assert result.to_jid == "+1@s.whatsapp.net"
    assert result.message_id == "mock-text-1"
    assert mock.outbox == [result]
    assert mock.sent_texts == [("+1@s.whatsapp.net", "hi", result)]


@pytest.mark.anyio
async def test_mock_adapter_outbox_records_send_media() -> None:
    mock = MockWhatsAppClientAdapter()

    image = await mock.send_media("jid", b"i", "image/jpeg", "i.jpg")
    video = await mock.send_media("jid", b"v", "video/mp4", "v.mp4")
    audio = await mock.send_media("jid", b"a", "audio/ogg", "a.ogg")
    doc = await mock.send_media("jid", b"d", "application/pdf", "d.pdf")

    assert mock.outbox == [image, video, audio, doc]
    assert [entry["kind"] for entry in mock.sent_media] == [
        "image",
        "video",
        "audio",
        "document",
    ]


@pytest.mark.anyio
async def test_mock_adapter_read_receipts_recorded() -> None:
    mock = MockWhatsAppClientAdapter()

    await mock.send_read_receipt("chat@s.whatsapp.net", ["m1", "m2"])

    assert mock.read_receipts == [("chat@s.whatsapp.net", ["m1", "m2"])]


@pytest.mark.anyio
async def test_mock_adapter_get_contacts_returns_populated_list() -> None:
    mock = MockWhatsAppClientAdapter()
    contacts = [
        Contact(
            jid="15551234567@s.whatsapp.net",
            name="Ada",
            e164="+15551234567",
            is_business=False,
        )
    ]
    mock.contacts.extend(contacts)

    assert await mock.get_contacts() == contacts


@pytest.mark.anyio
async def test_mock_adapter_subscribe_inbound_yields_pending_events() -> None:
    mock = MockWhatsAppClientAdapter()
    first = _inbound("1")
    second = _inbound("2")
    await mock.pending_inbound.put(first)
    await mock.pending_inbound.put(second)

    seen = []
    async for event in mock.subscribe_inbound():
        seen.append(event)
        if len(seen) == 2:
            break

    assert seen == [first, second]


@pytest.mark.anyio
async def test_mock_adapter_await_paired_returns_configured_status() -> None:
    mock = MockWhatsAppClientAdapter()

    for status in PairStatus:
        mock.next_pair_status = status
        assert await mock.await_paired(timeout_sec=0.1) is status


@pytest.mark.anyio
async def test_mock_adapter_health_state_transitions() -> None:
    mock = MockWhatsAppClientAdapter()

    await mock.start(Path("auth"))
    assert mock.health().state == "connecting"
    mock._state = "ready"
    assert mock.health().state == "ready"
    await mock.stop()
    assert mock.health().state == "stopped"
    await mock.logout()
    assert mock.health().state == "logged_out"


def test_neonize_adapter_class_conforms() -> None:
    from tether.connectors.whatsapp.neonize_adapter import (
        NeonizeWhatsAppClientAdapter,
    )

    assert issubclass(NeonizeWhatsAppClientAdapter, WhatsAppClientAdapter)
    assert NeonizeWhatsAppClientAdapter.__abstractmethods__ == frozenset()


def test_message_ev_to_inbound_event_drops_self_echo() -> None:
    from tether.connectors.whatsapp.neonize_adapter import (
        _message_ev_to_inbound_event,
    )

    event = _message_event(is_from_me=True)

    assert _message_ev_to_inbound_event(event) is None


def test_message_ev_to_inbound_event_group_message() -> None:
    from tether.connectors.whatsapp.neonize_adapter import (
        _message_ev_to_inbound_event,
    )

    event = _message_event(
        chat="120363000000000000@g.us",
        sender="15551234567@s.whatsapp.net",
        is_group=True,
        text="hello group",
    )

    inbound = _message_ev_to_inbound_event(event)

    assert inbound is not None
    assert inbound.event_id == "120363000000000000@g.us:msg-1"
    assert inbound.payload["is_group"] is True
    assert inbound.payload["from"] == "120363000000000000@g.us"
    assert inbound.payload["sender_e164"] == "+15551234567"


def test_message_ev_to_inbound_event_media_image() -> None:
    from tether.connectors.whatsapp.neonize_adapter import (
        _message_ev_to_inbound_event,
    )

    event = _message_event(text="", fields={"imageMessage"})
    event.Message.imageMessage = MagicMock()
    event.Message.imageMessage.mimetype = "image/jpeg"

    inbound = _message_ev_to_inbound_event(event)

    assert inbound is not None
    assert inbound.payload["media_ref"] == {
        "kind": "image",
        "mimetype": "image/jpeg",
    }


def test_message_ev_to_inbound_event_reply() -> None:
    from tether.connectors.whatsapp.neonize_adapter import (
        _message_ev_to_inbound_event,
    )

    event = _message_event(text="", fields={"extendedTextMessage"})
    ext = event.Message.extendedTextMessage
    ext.text = "replying"
    ext.contextInfo = MagicMock()
    ext.contextInfo.stanzaId = "quoted-id"
    ext.contextInfo.HasField.side_effect = (
        lambda name: name in {"stanzaId", "quotedMessage"}
    )
    ext.contextInfo.quotedMessage = MagicMock()
    ext.contextInfo.quotedMessage.conversation = "quoted text"

    inbound = _message_ev_to_inbound_event(event)

    assert inbound is not None
    assert inbound.payload["reply_to_id"] == "quoted-id"
    assert inbound.payload["reply_to_snippet"] == "quoted text"


def test_platform_machine_patch_applied_on_windows() -> None:
    import os
    import platform

    if os.name != "nt":
        pytest.skip("Windows-only patch")
    import tether.connectors.whatsapp

    assert tether.connectors.whatsapp is not None
    assert platform.machine() == "AMD64"


def test_neonize_adapter_module_imports_without_neonize() -> None:
    from tether.connectors.whatsapp.neonize_adapter import (
        NeonizeWhatsAppClientAdapter,
    )

    adapter = NeonizeWhatsAppClientAdapter(uuid="test")
    snap = adapter.health()

    assert snap.state == "stopped"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "status",
    [PairStatus.PAIRED, PairStatus.LOGGED_OUT, PairStatus.FAILED],
)
async def test_await_paired_consumes_pre_existing_outcome(
    status: PairStatus,
) -> None:
    from tether.connectors.whatsapp.neonize_adapter import (
        NeonizeWhatsAppClientAdapter,
    )

    adapter = NeonizeWhatsAppClientAdapter(uuid="test")
    adapter._pending_pair_outcome = status

    result = await adapter.await_paired(timeout_sec=0.5)

    assert result is status
    assert adapter._pending_pair_outcome is None


@pytest.mark.anyio
async def test_on_connect_done_flips_error_on_initial_connect_failure() -> None:
    from tether.connectors.whatsapp.neonize_adapter import (
        NeonizeWhatsAppClientAdapter,
    )

    async def fail_connect() -> None:
        raise RuntimeError("connect boom")

    adapter = NeonizeWhatsAppClientAdapter(uuid="test")
    adapter._state = "connecting"
    task = asyncio.create_task(fail_connect())
    await asyncio.sleep(0)

    adapter._on_connect_done(task)

    snap = adapter.health()
    assert snap.state == "error"
    assert snap.last_error == "connect boom"


@pytest.mark.anyio
async def test_stop_resets_started_immediately() -> None:
    from tether.connectors.whatsapp.neonize_adapter import (
        NeonizeWhatsAppClientAdapter,
    )

    adapter = NeonizeWhatsAppClientAdapter(uuid="test")
    adapter._started = True
    adapter._state = "ready"
    disconnect_started = asyncio.Event()
    release_disconnect = asyncio.Event()

    async def slow_disconnect() -> None:
        disconnect_started.set()
        await release_disconnect.wait()

    adapter._do_disconnect = slow_disconnect  # type: ignore[method-assign]
    stop_task = asyncio.create_task(adapter.stop())
    await asyncio.wait_for(disconnect_started.wait(), timeout=0.5)

    assert adapter._started is False
    assert adapter.health().state == "stopped"

    release_disconnect.set()
    await stop_task


def _inbound(event_id: str) -> InboundEvent:
    return InboundEvent(
        event_id=event_id,
        connector_id="whatsapp",
        kind="message",
        received_at=datetime.now(timezone.utc),
        payload={},
        summary=None,
    )


def _message_event(
    *,
    chat: str = "15551234567@s.whatsapp.net",
    sender: str = "15551234567@s.whatsapp.net",
    is_group: bool = False,
    is_from_me: bool = False,
    text: str = "hello",
    fields: set[str] | None = None,
) -> MagicMock:
    field_set = fields or set()
    event = MagicMock()
    event.IsEdit = False
    event.Info = MagicMock()
    event.Info.ID = "msg-1"
    event.Info.PushName = "Ada"
    event.Info.Timestamp = 1_700_000_000
    event.Info.MessageSource = MagicMock()
    event.Info.MessageSource.IsFromMe = is_from_me
    event.Info.MessageSource.Chat = chat
    event.Info.MessageSource.Sender = sender
    event.Info.MessageSource.IsGroup = is_group
    event.Message = MagicMock()
    event.Message.conversation = text
    event.Message.HasField.side_effect = lambda name: name in field_set
    event.Message.extendedTextMessage = MagicMock()
    event.Message.extendedTextMessage.text = ""
    event.Message.extendedTextMessage.contextInfo = MagicMock()
    event.Message.extendedTextMessage.contextInfo.HasField.return_value = False
    return event
