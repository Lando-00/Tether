-- 001_current_schema.sql
-- Phase 6 step 59: baseline migration. Captures the schema previously
-- created by SqliteSessionStore._init_schema() exactly.
-- Phase 6 step 61 (002_v2_schema.sql) introduces turns, tool_calls,
-- raw_events. This file is the permanent foundation.
--
-- Synthesis §3.6, B1 step 2.

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    thinking_text TEXT,
    tool_name TEXT,
    args TEXT,
    result TEXT,
    ts TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_session_ts ON messages(session_id, ts);
