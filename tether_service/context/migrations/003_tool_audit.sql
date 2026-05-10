-- 003_tool_audit.sql
-- Phase 7 step 73: append-only audit log for tool invocations.
-- B5 step 7. Columns owned by B5; coordination with B1.
--
-- Purpose: every tool call gets ONE row written by the orchestrator
-- (Phase 7 step 74 / p7-tool-audit-writes). Args stored as SHA-256
-- hash by default (privacy-preserving). Raw args optional via
-- Settings.security.audit_log.store_args=true (debug-only).
--
-- Synthesis §3.6 + B5 step 7.

CREATE TABLE tool_audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    correlation_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    tool_call_id TEXT REFERENCES tool_calls(tool_call_id) ON DELETE SET NULL,
    tool_name TEXT NOT NULL,
    args_sha256 TEXT NOT NULL,
    args_json TEXT,
    capabilities TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL CHECK (status IN ('ok','error','cancelled')),
    error_kind TEXT,
    duration_ms INTEGER,
    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    completed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX idx_tool_audit_session_started ON tool_audit(session_id, started_at);
CREATE INDEX idx_tool_audit_turn ON tool_audit(turn_id);
CREATE INDEX idx_tool_audit_tool_name_started ON tool_audit(tool_name, started_at);
CREATE INDEX idx_tool_audit_correlation ON tool_audit(correlation_id);
