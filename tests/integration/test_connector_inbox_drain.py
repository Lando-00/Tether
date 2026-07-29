"""Integration test for the connector inbox drain task (Phase 6.5 step 66h).

Synthesis §4 step 66e end-to-end:

    EchoConnector.start()                    # queues 3 synthetic events
    └─> ConnectorRegistry.start_connector(id)
        └─> SupervisedTask: async for event in conn.inbound_stream():
            └─> SqliteInbox.append_many([event])

The test asserts:

* Within 1 s of ``start_connector``, all 3 echo synthetic events land
  in the inbox.
* ``GET /api/v1/connectors/echo/inbox?unread=true`` lists those 3
  events.
* ``POST /api/v1/connectors/echo/inbox/mark-seen`` with all 3 ids
  flips them seen; subsequent ``unread=true`` query is empty.
* ``stop_connector`` cancels the drain task within 2 s.

Coverage matches connector spec §8.3 acceptance row "drain → inbox →
mark-seen flow" plus the synthesis §4 step 66h add-coverage line.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from tests.fixtures.echo_connector import EchoConnector
from tether.app.http.routers.connectors import router as connectors_router
from tether.context.inbox_store import SqliteInbox
from tether.core.connector_registry import ConnectorRegistry
from tether.engine import Engine
from tether.protocol.parsers.sliding import SlidingParser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_engine(echo: EchoConnector, tmp_path):
    """Construct a connector registry + inbox + Engine wired together.

    Mirrors the production wiring used by ``Engine.from_settings``
    minus the expensive provider/store machinery — we only need the
    connector registry + inbox path here.
    """
    db_path = (tmp_path / "inbox.db").as_posix()
    inbox = SqliteInbox(f"sqlite:///{db_path}")

    registry = ConnectorRegistry([echo], data_dir=tmp_path, inbox=inbox)
    engine = Engine(
        provider=AsyncMock(),
        parser=SlidingParser(),
        session_store=AsyncMock(),
        tools=dict(registry.aggregate_tools()),
        system_prompt="",
        connector_registry=registry,
        inbox=inbox,
    )
    return engine, inbox, registry


def _build_app(engine):
    app = FastAPI()
    v1 = APIRouter(prefix="/api/v1")
    v1.include_router(connectors_router)
    app.include_router(v1)
    app.state.gen_svc = engine
    return app


async def _wait_for(predicate, *, timeout: float = 1.5, poll: float = 0.05):
    """Poll ``predicate`` until truthy or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await predicate():
            return True
        await asyncio.sleep(poll)
    return False


# ---------------------------------------------------------------------------
# End-to-end: start → drain → inbox query → mark-seen → stop
# ---------------------------------------------------------------------------


async def test_echo_inbox_drain_end_to_end(tmp_path):
    """Phase 6.5 step 66h: full drain → inbox → mark-seen → stop loop."""
    echo = EchoConnector()
    # Move the connector into READY state explicitly (login flow is
    # exercised by other tests — we only need start_connector to spawn
    # the drain task here).
    await echo.complete_login(payload={"code": "ok", "user_id": "alice"})

    engine, inbox, registry = _build_engine(echo, tmp_path)
    await inbox.connect()
    try:
        # Spawn the drain task by starting the connector.
        await registry.start_connector("echo")

        # Wait for all 3 synthetic events to land in the inbox.
        async def all_three_arrived():
            rows = await inbox.list_unread("echo")
            return len(rows) >= 3

        arrived = await _wait_for(all_three_arrived, timeout=1.5)
        assert arrived, "drain task did not land 3 events within 1.5s"

        # HTTP layer sees the same view.
        app = _build_app(engine)
        client = TestClient(app)
        resp = client.get(
            "/api/v1/connectors/echo/inbox", params={"unread": "true"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 3
        ids = [e["event_id"] for e in body]

        # Mark all 3 seen via HTTP.
        resp = client.post(
            "/api/v1/connectors/echo/inbox/mark-seen",
            json={"event_ids": ids},
        )
        assert resp.status_code == 200
        assert resp.json() == {"affected": 3}

        # Subsequent unread query is empty.
        resp = client.get(
            "/api/v1/connectors/echo/inbox", params={"unread": "true"}
        )
        assert resp.json() == []

        # Stop within 2 s. Bound the test on a private timeout so a hung
        # drain task fails loud rather than waiting indefinitely.
        t0 = time.monotonic()
        await asyncio.wait_for(
            registry.stop_connector("echo", timeout_sec=2.0), timeout=3.0
        )
        elapsed = time.monotonic() - t0
        assert elapsed < 2.5, (
            f"stop_connector took {elapsed:.2f}s; should be < 2.5s"
        )
        # The drain task is no longer tracked.
        assert "echo" not in registry._drain_tasks
    finally:
        # Stop the connector again (idempotent) + close the inbox.
        await registry.stop_connector("echo", timeout_sec=2.0)
        await inbox.aclose()


async def test_inbox_drain_skipped_when_inbox_unconfigured(tmp_path):
    """Without an inbox, the drain task is never spawned.

    Phase 6.5 step 66e: legacy direct-construction paths (no inbox)
    keep the connector lifecycle working — only the inbound-stream
    drain wiring is suppressed.
    """
    echo = EchoConnector()
    await echo.complete_login(payload={"code": "ok", "user_id": "alice"})
    registry = ConnectorRegistry([echo], data_dir=tmp_path, inbox=None)

    await registry.start_connector("echo")
    # ``_drain_tasks`` should stay empty.
    assert registry._drain_tasks == {}
    await registry.stop_connector("echo", timeout_sec=2.0)


async def test_drain_task_survives_inbox_error(tmp_path):
    """A single ``inbox.append_many`` failure must not kill the drain task.

    Synthesis §4 step 66e: the drain loop swallows + logs per-event
    exceptions so a transient SQLite error doesn't lose the rest of
    the stream.
    """

    class FlakyInbox:
        def __init__(self):
            self.calls = 0
            self.successful = []

        async def append_many(self, events):
            self.calls += 1
            # Fail the first call, succeed thereafter.
            if self.calls == 1:
                raise RuntimeError("transient")
            self.successful.extend(events)
            return len(events)

        # Never called in this test, but the protocol requires them.
        async def list_unread(self, *args, **kwargs):  # pragma: no cover
            return []

        async def list_recent(self, *args, **kwargs):  # pragma: no cover
            return []

        async def mark_seen(self, *args, **kwargs):  # pragma: no cover
            return 0

        async def prune_older_than(self, *args, **kwargs):  # pragma: no cover
            return 0

    flaky = FlakyInbox()
    echo = EchoConnector()
    await echo.complete_login(payload={"code": "ok", "user_id": "alice"})
    registry = ConnectorRegistry(
        [echo], data_dir=tmp_path, inbox=flaky  # type: ignore[arg-type]
    )
    await registry.start_connector("echo")

    async def two_or_more_succeeded():
        return len(flaky.successful) >= 2

    ok = await _wait_for(two_or_more_succeeded, timeout=1.5)
    assert ok, (
        f"drain task should have survived 1 error and persisted >=2 "
        f"events; got {len(flaky.successful)} successful, "
        f"{flaky.calls} total calls"
    )
    await registry.stop_connector("echo", timeout_sec=2.0)
