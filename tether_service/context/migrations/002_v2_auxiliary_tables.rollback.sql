-- 002_v2_auxiliary_tables.rollback.sql
-- Rollback for 002_v2_auxiliary_tables.sql
-- Drops the v2 auxiliary tables, indexes, and view added in the up migration.
-- v1 sessions and messages tables are NOT touched.

DROP VIEW IF EXISTS turn_timeline;
DROP INDEX IF EXISTS idx_raw_events_tool_call;
DROP INDEX IF EXISTS idx_raw_events_session_turn;
DROP INDEX IF EXISTS idx_raw_events_turn_seq;
DROP INDEX IF EXISTS idx_tool_calls_session_name;
DROP INDEX IF EXISTS idx_tool_calls_turn;
DROP INDEX IF EXISTS idx_turns_session_started;
DROP TABLE IF EXISTS raw_events;
DROP TABLE IF EXISTS tool_calls;
DROP TABLE IF EXISTS turns;
