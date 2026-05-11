from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.fixtures.mock_whatsapp_adapter import MockWhatsAppClientAdapter
from tether.connectors.types import ConnectorState, InboundEvent
from tether.connectors.whatsapp.adapter import Contact
from tether.connectors.whatsapp.target_resolver import resolve_target
from tether.connectors.whatsapp.tools import (
    WhatsAppGetThreadTool,
    WhatsAppInboxMarkSeenTool,
    WhatsAppListUnreadTool,
    WhatsAppMarkPlatformReadTool,
    _serialize_event,
)
from tether.context.inbox_store import InboundInbox
from tether.core.errors import ConnectorNotConfiguredError


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class InMemoryInbox(InboundInbox):
    def __init__(self, events: list[InboundEvent]) -> None:
        self.events = list(events)
        self.seen: set[str] = set()

    async def append_many(self, events: list[InboundEvent]) -> int:
        before = {event.event_id for event in self.events}
        added = [event for event in events if event.event_id not in before]
        self.events.extend(added)
        return len(added)

    async def list_unread(
        self,
        connector_id: str,
        limit: int = 50,
    ) -> list[InboundEvent]:
        return [
            event
            for event in self.events
            if event.connector_id == connector_id and event.event_id not in self.seen
        ][:limit]

    async def list_recent(
        self,
        connector_id: str,
        limit: int = 50,
    ) -> list[InboundEvent]:
        return [event for event in self.events if event.connector_id == connector_id][
            :limit
        ]

    async def mark_seen(self, connector_id: str, event_ids: list[str]) -> int:
        event_set = {
            event.event_id
            for event in self.events
            if event.connector_id == connector_id
        }
        affected = 0
        for event_id in event_ids:
            if event_id in event_set and event_id not in self.seen:
                self.seen.add(event_id)
                affected += 1
        return affected

    async def prune_older_than(self, retention_days: int) -> int:
        return 0


def _ready() -> ConnectorState:
    return ConnectorState.READY


def _unconfigured() -> ConnectorState:
    return ConnectorState.UNCONFIGURED


def _event(
    event_id: str,
    *,
    from_jid: str = "15551234567@s.whatsapp.net",
    connector_id: str = "whatsapp",
) -> InboundEvent:
    return InboundEvent(
        event_id=event_id,
        connector_id=connector_id,
        kind="message",
        received_at=datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc),
        payload={"from": from_jid, "text_snippet": f"message {event_id}"},
        summary=f"summary {event_id}",
    )


def _resolver(value: str) -> str | None:
    contacts = [
        Contact(
            jid="15551234567@s.whatsapp.net",
            name="Ada",
            e164="+15551234567",
            is_business=False,
        )
    ]
    return resolve_target(value, contacts)


@pytest.mark.anyio
async def test_list_unread_happy_path_serializes_events() -> None:
    inbox = InMemoryInbox([_event("e1"), _event("e2")])
    tool = WhatsAppListUnreadTool(inbox=inbox, state_provider=_ready)

    result = await tool.run(limit=1)

    assert result["count"] == 1
    assert result["events"][0]["event_id"] == "e1"
    assert result["events"][0]["received_at_iso"] == "2026-05-11T12:00:00+00:00"


@pytest.mark.anyio
async def test_list_unread_refuses_when_not_ready() -> None:
    tool = WhatsAppListUnreadTool(
        inbox=InMemoryInbox([]),
        state_provider=_unconfigured,
    )

    with pytest.raises(ConnectorNotConfiguredError):
        await tool.run()


@pytest.mark.anyio
@pytest.mark.parametrize("limit", [0, 501])
async def test_list_unread_rejects_invalid_limit(limit: int) -> None:
    tool = WhatsAppListUnreadTool(inbox=InMemoryInbox([]), state_provider=_ready)

    result = await tool.run(limit=limit)

    assert result == {"error": f"limit must be in [1, 500], got {limit}"}


@pytest.mark.anyio
async def test_get_thread_happy_path_filters_by_peer() -> None:
    peer = "15551234567@s.whatsapp.net"
    inbox = InMemoryInbox(
        [
            _event("e1", from_jid=peer),
            _event("e2", from_jid="15557654321@s.whatsapp.net"),
            _event("e3", from_jid=peer),
        ]
    )
    tool = WhatsAppGetThreadTool(
        inbox=inbox,
        target_resolver_fn=_resolver,
        state_provider=_ready,
    )

    result = await tool.run(peer="Ada", limit=1)

    assert result["peer_jid"] == peer
    assert result["count"] == 1
    assert result["events"][0]["event_id"] == "e1"
    assert result["scanned"] == 3


@pytest.mark.anyio
async def test_get_thread_refuses_when_not_ready() -> None:
    tool = WhatsAppGetThreadTool(
        inbox=InMemoryInbox([]),
        target_resolver_fn=_resolver,
        state_provider=_unconfigured,
    )

    with pytest.raises(ConnectorNotConfiguredError):
        await tool.run(peer="Ada")


@pytest.mark.anyio
async def test_get_thread_rejects_invalid_limit() -> None:
    tool = WhatsAppGetThreadTool(
        inbox=InMemoryInbox([]),
        target_resolver_fn=_resolver,
        state_provider=_ready,
    )

    result = await tool.run(peer="Ada", limit=-1)

    assert result == {"error": "limit must be in [1, 500], got -1"}


@pytest.mark.anyio
async def test_get_thread_rejects_unresolved_peer() -> None:
    tool = WhatsAppGetThreadTool(
        inbox=InMemoryInbox([]),
        target_resolver_fn=_resolver,
        state_provider=_ready,
    )

    result = await tool.run(peer="nobody")

    assert result == {"error": "could not resolve peer 'nobody' to a JID"}


@pytest.mark.anyio
async def test_inbox_mark_seen_happy_path() -> None:
    inbox = InMemoryInbox([_event("e1"), _event("e2")])
    tool = WhatsAppInboxMarkSeenTool(inbox=inbox, state_provider=_ready)

    result = await tool.run(event_ids=["e1", "e2"])

    assert result == {"affected": 2, "requested": 2}


@pytest.mark.anyio
async def test_inbox_mark_seen_second_call_is_idempotent() -> None:
    inbox = InMemoryInbox([_event("e1")])
    tool = WhatsAppInboxMarkSeenTool(inbox=inbox, state_provider=_ready)

    first = await tool.run(event_ids=["e1"])
    second = await tool.run(event_ids=["e1"])

    assert first == {"affected": 1, "requested": 1}
    assert second == {"affected": 0, "requested": 1}


@pytest.mark.anyio
async def test_inbox_mark_seen_refuses_when_not_ready() -> None:
    tool = WhatsAppInboxMarkSeenTool(
        inbox=InMemoryInbox([]),
        state_provider=_unconfigured,
    )

    with pytest.raises(ConnectorNotConfiguredError):
        await tool.run(event_ids=["e1"])


@pytest.mark.anyio
async def test_inbox_mark_seen_rejects_too_many_ids() -> None:
    tool = WhatsAppInboxMarkSeenTool(inbox=InMemoryInbox([]), state_provider=_ready)

    result = await tool.run(event_ids=[str(i) for i in range(901)])

    assert result == {"error": "max 900 event_ids per call, got 901"}


@pytest.mark.anyio
async def test_mark_platform_read_happy_path_normalizes_jid() -> None:
    adapter = MockWhatsAppClientAdapter()
    tool = WhatsAppMarkPlatformReadTool(adapter=adapter, state_provider=_ready)

    result = await tool.run(jid="+15551234567", message_ids=["m1", "m2"])

    assert result == {"marked": 2, "jid": "15551234567@s.whatsapp.net"}
    assert adapter.read_receipts == [
        ("15551234567@s.whatsapp.net", ["m1", "m2"])
    ]


@pytest.mark.anyio
async def test_mark_platform_read_refuses_when_not_ready() -> None:
    tool = WhatsAppMarkPlatformReadTool(
        adapter=MockWhatsAppClientAdapter(),
        state_provider=_unconfigured,
    )

    with pytest.raises(ConnectorNotConfiguredError):
        await tool.run(jid="+15551234567", message_ids=["m1"])


@pytest.mark.anyio
async def test_mark_platform_read_rejects_invalid_jid() -> None:
    tool = WhatsAppMarkPlatformReadTool(
        adapter=MockWhatsAppClientAdapter(),
        state_provider=_ready,
    )

    result = await tool.run(jid="not-a-jid", message_ids=["m1"])

    assert result == {"error": "could not resolve jid 'not-a-jid'"}


@pytest.mark.anyio
async def test_mark_platform_read_rejects_empty_message_ids() -> None:
    tool = WhatsAppMarkPlatformReadTool(
        adapter=MockWhatsAppClientAdapter(),
        state_provider=_ready,
    )

    result = await tool.run(jid="+15551234567", message_ids=[])

    assert result == {"error": "message_ids must be non-empty"}


def test_serialize_event_round_trip() -> None:
    event = _event("round-trip")

    serialized = _serialize_event(event)
    reconstructed = InboundEvent(
        event_id=serialized["event_id"],
        connector_id="whatsapp",
        kind=serialized["kind"],
        received_at=datetime.fromisoformat(serialized["received_at_iso"]),
        payload=serialized["payload"],
        summary=serialized["summary"],
    )

    assert reconstructed == event
