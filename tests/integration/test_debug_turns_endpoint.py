"""GET /api/v1/debug/turns/{session_id}/{turn_id} integration tests.

Phase 7 step 75: exposes turn_timeline view to debugging tools.

Test strategy: build a minimal FastAPI app backed by an isolated tmp-path
SQLite DB. Insert test data via stdlib sqlite3 (WAL mode allows concurrent
access with the aiosqlite connection opened by the TestClient lifespan).
This avoids asyncio event-loop cross-contamination when writing from sync
test code.
"""
from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from tether.app.http.api import lifespan
from tether.app.http.routers.debug import router as debug_router
from tether.app.http.routers.health import router as health_router
from tether.config.settings import Settings
from tether.engine import Engine
from tether.runtime.watchdog_mode import WatchdogMode

# yoyo 8.x imports pkg_resources which emits UserWarning attributed to
# yoyo.migrations (stacklevel=2). Suppress so -W error sweeps stay clean.
# Same pattern as test_engine_applies_migrations_on_startup.py.
pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning:yoyo",
    "ignore::UserWarning:yoyo",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(db_path: str) -> Settings:
    """Minimal Settings with DummyProvider + SqliteSessionStore at *db_path*."""
    dsn = f"sqlite:///{db_path}"
    return Settings.model_validate(
        {
            "system": {"prompt": "debug-turns-test"},
            "providers": {
                "model": {
                    "impl": "tether.providers.dummy.provider.DummyProvider",
                    "args": {},
                },
                "parser": {
                    "impl": "tether.protocol.parsers.sliding.SlidingParser",
                    "args": {},
                },
                "session_store": {
                    "impl": "tether.context.sqlite_store.SqliteSessionStore",
                    "args": {"dsn": dsn},
                },
            },
            "storage": {"sqlite": {"dsn": dsn}},
            "tools": {
                "registry": [],
                "enabled": [],
                "disabled": ["time", "weather", "forecast", "web_search"],
            },
        }
    )


def _build_test_app(db_path: str) -> FastAPI:
    """Build a minimal FastAPI app with the debug router, backed by *db_path*.

    Uses the real lifespan so engine startup/shutdown run, and the
    SqliteSessionStore opens its aiosqlite connection before tests run.
    """
    engine = Engine.from_settings(
        _settings(db_path), watchdog_mode=WatchdogMode.SERVER
    )
    app = FastAPI(lifespan=lifespan)
    app.state.gen_svc = engine

    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(debug_router)
    v1.include_router(health_router)
    app.include_router(v1)
    return app


def _insert_raw_events(db_path: str, events: list[dict]) -> None:
    """Write *events* directly via stdlib sqlite3 (WAL mode is concurrent-safe).

    Each event dict must contain: session_id, turn_id, seq, type, payload.
    Optional: tool_call_id.

    ``apply_pending_migrations`` has already run via Engine.from_settings,
    so all tables and the turn_timeline view exist before we write here.
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for evt in events:
            session_id = evt["session_id"]
            # Ensure sessions + turns rows exist (raw_events has FK to turns).
            conn.execute(
                "INSERT OR IGNORE INTO sessions(id, created_at, metadata) VALUES (?, datetime('now'), '{}')",
                (session_id,),
            )
            conn.execute(
                "INSERT OR IGNORE INTO turns(turn_id, session_id, model_name, status)"
                " VALUES (?, ?, 'dummy', 'running')",
                (evt["turn_id"], session_id),
            )
            conn.execute(
                "INSERT INTO raw_events(session_id, turn_id, seq, type, tool_call_id, payload_json)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    evt["turn_id"],
                    evt["seq"],
                    evt["type"],
                    evt.get("tool_call_id"),
                    json.dumps(evt.get("payload", {})),
                ),
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def debug_client(tmp_path):
    """TestClient backed by an isolated tmp SQLite DB with the debug router."""
    db_path = (tmp_path / "debug_test.db").as_posix()
    app = _build_test_app(db_path)
    with TestClient(app) as c:
        # Expose db_path so tests can call _insert_raw_events.
        c._test_db_path = db_path  # type: ignore[attr-defined]
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_unknown_session_returns_404(debug_client):
    """No events for an unknown session_id/turn_id -> 404."""
    resp = debug_client.get("/api/v1/debug/turns/ghost-session/ghost-turn")
    assert resp.status_code == 404


def test_directly_inserted_events_visible(debug_client):
    """Events written directly to raw_events appear via the endpoint."""
    db_path = debug_client._test_db_path
    _insert_raw_events(
        db_path,
        [
            {
                "session_id": "direct-sess",
                "turn_id": "t-direct",
                "seq": 0,
                "type": "text_delta",
                "payload": {"text": "hello"},
            },
            {
                "session_id": "direct-sess",
                "turn_id": "t-direct",
                "seq": 1,
                "type": "message_stop",
                "payload": {"stop_reason": "complete"},
            },
        ],
    )

    resp = debug_client.get("/api/v1/debug/turns/direct-sess/t-direct")
    assert resp.status_code == 200

    body = resp.json()
    assert body["session_id"] == "direct-sess"
    assert body["turn_id"] == "t-direct"
    assert len(body["events"]) == 2
    assert body["events"][0]["seq"] == 0
    assert body["events"][0]["type"] == "text_delta"
    assert body["events"][1]["seq"] == 1
    assert body["events"][1]["type"] == "message_stop"


def test_events_ordered_by_seq(debug_client):
    """Events are returned in ascending seq order regardless of insertion order."""
    db_path = debug_client._test_db_path
    # Insert in reverse order to confirm ORDER BY seq.
    _insert_raw_events(
        db_path,
        [
            {
                "session_id": "order-sess",
                "turn_id": "t-order",
                "seq": 2,
                "type": "message_stop",
                "payload": {},
            },
            {
                "session_id": "order-sess",
                "turn_id": "t-order",
                "seq": 0,
                "type": "message_start",
                "payload": {},
            },
            {
                "session_id": "order-sess",
                "turn_id": "t-order",
                "seq": 1,
                "type": "text_delta",
                "payload": {"text": "hi"},
            },
        ],
    )

    resp = debug_client.get("/api/v1/debug/turns/order-sess/t-order")
    assert resp.status_code == 200

    seqs = [e["seq"] for e in resp.json()["events"]]
    assert seqs == [0, 1, 2]


def test_session_turn_mismatch_returns_404(debug_client):
    """A valid turn_id with a wrong session_id returns 404."""
    db_path = debug_client._test_db_path
    _insert_raw_events(
        db_path,
        [
            {
                "session_id": "real-sess",
                "turn_id": "t-real",
                "seq": 0,
                "type": "text_delta",
                "payload": {"text": "hi"},
            },
        ],
    )

    # Correct session_id works.
    ok = debug_client.get("/api/v1/debug/turns/real-sess/t-real")
    assert ok.status_code == 200

    # Wrong session_id → 404 even though turn_id exists.
    resp = debug_client.get("/api/v1/debug/turns/wrong-sess/t-real")
    assert resp.status_code == 404


def test_endpoint_path_is_under_api_v1_debug(debug_client):
    """Endpoint lives at /api/v1/debug/turns/..., not /debug/turns/..."""
    resp = debug_client.get("/debug/turns/any/any")
    # FastAPI returns 404 for unmounted paths.
    assert resp.status_code == 404


def test_event_envelope_has_required_fields(debug_client):
    """Each event must contain all required envelope fields."""
    db_path = debug_client._test_db_path
    _insert_raw_events(
        db_path,
        [
            {
                "session_id": "env-sess",
                "turn_id": "t-env",
                "seq": 0,
                "type": "text_delta",
                "payload": {"text": "hi"},
            },
        ],
    )

    resp = debug_client.get("/api/v1/debug/turns/env-sess/t-env")
    assert resp.status_code == 200

    event = resp.json()["events"][0]
    required_fields = (
        "seq",
        "type",
        "ts",
        "protocol_version",
        "tool_call_id",
        "tool_name",
        "tool_status",
        "payload_json",
    )
    for field in required_fields:
        assert field in event, f"event missing required field: {field!r}"
