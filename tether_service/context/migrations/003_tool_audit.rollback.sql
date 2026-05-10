-- 003_tool_audit.rollback.sql
-- Rollback for 003_tool_audit.sql
-- Drops the tool_audit table and its indexes.

DROP INDEX IF EXISTS idx_tool_audit_correlation;
DROP INDEX IF EXISTS idx_tool_audit_tool_name_started;
DROP INDEX IF EXISTS idx_tool_audit_turn;
DROP INDEX IF EXISTS idx_tool_audit_session_started;
DROP TABLE IF EXISTS tool_audit;
