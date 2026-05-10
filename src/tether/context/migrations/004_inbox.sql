-- 004_inbox.sql
-- Phase 6.5 step 66d (synthesis §4): inbound events table for the
-- :class:`tether.context.inbox_store.SqliteInbox` shared-DB inbox.
-- Schema owned by Phase 6.5; ratified per ADR-0009 (single
-- ``data/tether.db`` file, deviates from connector spec §3.6).
--
-- Per-row size caps are enforced at append time by SqliteInbox itself
-- (`inbox.max_payload_bytes` / `inbox.max_summary_chars`); SQLite has
-- no fixed-width TEXT, so the schema is permissive and the inbox
-- layer is the gate.
--
-- ``inbox_seen`` is the orchestrator-side flag — flipped when the
-- ``/api/v1/connectors/{id}/inbox/mark-seen`` route acknowledges an
-- event. Connector-platform read state (``platform_read``) is
-- intentionally NOT in v1; spec-§3.4 separates the two concepts and
-- a future migration will add a ``platform_read`` column once the
-- platforms that support it (Gmail, WhatsApp Web) need it.
--
-- Synthesis §4 Phase 6.5 + §13.4 M2; connector spec §3.4 + §3.7.

CREATE TABLE inbound_events (
    connector_id    TEXT NOT NULL,
    event_id        TEXT NOT NULL,
    kind            TEXT NOT NULL,
    received_at     TEXT NOT NULL,                  -- ISO-8601 UTC
    payload         TEXT NOT NULL,                  -- JSON, size-capped
    summary         TEXT,                           -- short preview, size-capped
    inbox_seen      INTEGER NOT NULL DEFAULT 0,     -- 0/1; orchestrator-side
    PRIMARY KEY (connector_id, event_id)
);

-- Pruning + recent-list scans hit ``received_at`` directly.
CREATE INDEX idx_inbound_events_received_at
    ON inbound_events (received_at);

-- Hot path: list-unread for one connector. Composite covers
-- (connector_id, inbox_seen) filter + received_at ordering.
CREATE INDEX idx_inbound_events_unread
    ON inbound_events (connector_id, inbox_seen, received_at);
