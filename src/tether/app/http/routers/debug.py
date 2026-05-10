"""Debug HTTP router.

Phase 7 step 75. Exposes the turn_timeline view (Phase 6 step 61) for
replay/debugging. Caller passes session_id + turn_id; returns the
chronological raw_events for that turn joined with tool_calls.

Synthesis §3 + §4 Phase 7 step 75. B3 step 9.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request


router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/turns/{session_id}/{turn_id}")
async def get_turn_timeline(
    session_id: str,
    turn_id: str,
    request: Request,
) -> Dict[str, Any]:
    """Return chronological events for a turn (raw_events JOINed with tool_calls).

    404 if no events exist for the given (session_id, turn_id) pair.
    """
    engine = request.app.state.gen_svc

    store = engine.store
    if not hasattr(store, "_conn"):
        raise HTTPException(
            status_code=501,
            detail="Debug turn endpoint only supported with SqliteSessionStore",
        )

    conn = store._conn
    if conn is None:
        if hasattr(store, "connect"):
            await store.connect()
            conn = store._conn
        if conn is None:
            raise HTTPException(
                status_code=503,
                detail="SessionStore connection not available",
            )

    async with conn.execute(
        "SELECT seq, type, ts, protocol_version, tool_call_id, tool_name, tool_status, payload_json "
        "FROM turn_timeline WHERE session_id = ? AND turn_id = ? "
        "ORDER BY seq ASC",
        (session_id, turn_id),
    ) as cur:
        rows = await cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No events found for session_id={session_id!r}, turn_id={turn_id!r}",
        )

    events: List[Dict[str, Any]] = [
        {
            "seq": r["seq"],
            "type": r["type"],
            "ts": r["ts"],
            "protocol_version": r["protocol_version"],
            "tool_call_id": r["tool_call_id"],
            "tool_name": r["tool_name"],
            "tool_status": r["tool_status"],
            "payload_json": r["payload_json"],
        }
        for r in rows
    ]

    return {
        "session_id": session_id,
        "turn_id": turn_id,
        "events": events,
    }


__all__ = ["router"]
