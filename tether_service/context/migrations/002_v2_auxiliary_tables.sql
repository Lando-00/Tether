-- 002_v2_auxiliary_tables.sql
-- Phase 6 step 61: ADDITIVE — adds turns, tool_calls, raw_events
-- alongside the v1 sessions/messages tables. Step 63 (aiosqlite store
-- rewrite) reshapes sessions/messages to v2 (or via a 004 migration);
-- this PR only adds the new auxiliary tables.
--
-- Synthesis §3.6 + b1-persistence.md schema design.

CREATE TABLE turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    model_name TEXT,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running','completed','failed','cancelled')),
    stop_reason TEXT,
    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    completed_at TEXT,
    error_json TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
    -- NOTE: FK to sessions(id) deferred. v1 sessions schema is
    -- (id TEXT PRIMARY KEY, created_at TEXT, metadata TEXT) — compatible
    -- with the FK column type but adding the FK now requires PRAGMA
    -- foreign_keys=ON at all times. _init_pragmas already enables it,
    -- but a missing parent row in v1 sessions would block turn insertion.
    -- The store currently auto-creates sessions on first INSERT, so the
    -- FK would be safe — but v2 sessions reshape (step 63) is right
    -- around the corner. Defer FK declaration to that migration.
);

CREATE TABLE tool_calls (
    tool_call_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL REFERENCES turns(turn_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    arguments_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','running','ok','error','cancelled')),
    result_json TEXT,
    error_json TEXT,
    call_seq INTEGER,
    result_seq INTEGER,
    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    completed_at TEXT,
    duration_ms INTEGER
);

CREATE TABLE raw_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL REFERENCES turns(turn_id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    protocol_version TEXT NOT NULL DEFAULT '1.0',
    type TEXT NOT NULL,
    ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    tool_call_id TEXT REFERENCES tool_calls(tool_call_id) ON DELETE SET NULL,
    -- message_id REFERENCES messages(message_id) deferred — v1 messages
    -- has 'id', not 'message_id'. Step 63's reshape adds this FK.
    message_id INTEGER,
    payload_json TEXT NOT NULL,
    UNIQUE (turn_id, seq)
);

-- Indexes for typical query patterns:
CREATE INDEX idx_turns_session_started ON turns(session_id, started_at, turn_id);
CREATE INDEX idx_tool_calls_turn ON tool_calls(turn_id, call_seq);
CREATE INDEX idx_tool_calls_session_name ON tool_calls(session_id, name, started_at);
CREATE INDEX idx_raw_events_turn_seq ON raw_events(turn_id, seq);
CREATE INDEX idx_raw_events_session_turn ON raw_events(session_id, turn_id, seq);
CREATE INDEX idx_raw_events_tool_call ON raw_events(tool_call_id);

-- turn_timeline view (reduced; messages join added in step 63 reshape)
CREATE VIEW turn_timeline AS
SELECT
    e.session_id,
    e.turn_id,
    e.seq,
    e.ts,
    e.protocol_version,
    e.type,
    e.tool_call_id,
    tc.name AS tool_name,
    tc.status AS tool_status,
    e.payload_json
FROM raw_events e
LEFT JOIN tool_calls tc ON tc.tool_call_id = e.tool_call_id;
