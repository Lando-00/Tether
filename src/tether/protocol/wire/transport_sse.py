"""Server-Sent Events (SSE) transport for v2 WireEvent streams.

Thin wrapper that takes ``AsyncIterator[WireEvent]`` and yields SSE-
framed bytes per the W3C/WHATWG spec:

    id: {seq}
    event: {type}
    data: {one-line-json}
    {blank line}

The blank line terminates the event. Multiple events are separated by
their respective blank-line terminators.

Synthesis §3.4 (transport, two formatters); §4 Phase 5 step 53.

Reachable via ``Accept: text/event-stream`` on ``/api/v1/chat/stream``.
"""
from __future__ import annotations

from typing import AsyncIterable, AsyncIterator

from tether.protocol.wire.events import WireEvent


async def transport_sse(
    events: AsyncIterable[WireEvent],
) -> AsyncIterator[bytes]:
    """Serialize each WireEvent as an SSE event block.

    Format per event::

        id: <seq>\\n
        event: <type>\\n
        data: <json>\\n
        \\n

    The trailing blank line is the W3C SSE event terminator (LF only,
    not CRLF). ``model_dump_json()`` produces compact single-line JSON
    so the ``data:`` field never contains embedded newlines.
    """
    async for event in events:
        # Use the discriminator value as the SSE event type (e.g.,
        # "text_delta", "message_stop", "tool_call"). model_dump_json()
        # produces compact single-line JSON.
        block = (
            f"id: {event.seq}\n"
            f"event: {event.type}\n"
            f"data: {event.model_dump_json()}\n"
            "\n"
        )
        yield block.encode("utf-8")


__all__ = ["transport_sse"]
