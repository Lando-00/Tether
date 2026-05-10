"""NDJSON transport for v2 WireEvent streams.

Thin wrapper that takes ``AsyncIterator[WireEvent]`` and yields NDJSON
bytes (one JSON object per line, terminated by ``\\n``). Each line is
the v2 vocabulary directly via ``WireEvent.model_dump_json()``.

Synthesis §3.4 (HTTP transport is a 6-line wrapper); §4 Phase 5 step 53.

NOT used by the default ``/api/v1/chat/stream`` path yet — see chat.py
content-negotiation for routing. ``p5-cutover-a-dual-emit`` will wire
this onto ``Accept: application/x-ndjson;version=2`` opt-in.
"""
from __future__ import annotations

from typing import AsyncIterable, AsyncIterator

from tether.protocol.wire.events import WireEvent


async def transport_ndjson(
    events: AsyncIterable[WireEvent],
) -> AsyncIterator[bytes]:
    """Serialize each WireEvent as a single NDJSON line."""
    async for event in events:
        yield (event.model_dump_json() + "\n").encode("utf-8")


__all__ = ["transport_ndjson"]
