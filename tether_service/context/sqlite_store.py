"""SQLite-backed session store implementing SessionStore with WAL + safe PRAGMAs"""
from __future__ import annotations

import datetime
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from tether_service.core.interfaces import SessionStore

logger = logging.getLogger(__name__)


class SqliteSessionStore(SessionStore):
    def __init__(self, dsn: str = "sqlite:///./data/tether.db"):
        # Apply all pending migrations BEFORE opening the connection.
        # yoyo is idempotent — Engine.from_settings's earlier call (if any)
        # is a no-op via the in-process DSN cache. Direct construction paths
        # (contract tests, CLI one-shots, future steps) also get the latest
        # schema automatically, including 002, 003, … as they ship.
        from tether_service.context.migration_runner import apply_pending_migrations
        apply_pending_migrations(dsn)

        # Parse DSN
        if dsn.startswith("sqlite:///"):
            path = dsn[len("sqlite:///"):]
        else:
            path = dsn

        # Ensure parent directory exists
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        # Connect with WAL and pragmas
        self.conn = sqlite3.connect(str(p), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_pragmas()

    def _init_pragmas(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA foreign_keys=ON;")
        self.conn.commit()

    async def create_session(self, session_id: str, created_at: int) -> None:
        """Create a new session."""
        self.conn.execute(
            "INSERT OR IGNORE INTO sessions(id, created_at, metadata) VALUES (?, ?, ?)",
            (session_id, datetime.datetime.fromtimestamp(created_at).isoformat(), "{}"),
        )
        self.conn.commit()

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions."""
        rows = self.conn.execute(
            "SELECT id, created_at FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        return [
            {
                "session_id": r["id"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID. Returns True if deleted, False if not found. Deletes all child messages first."""
        cur = self.conn.cursor()
        # Delete all messages for this session first
        cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        # Now delete the session
        cur.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        deleted_count = cur.rowcount
        self.conn.commit()
        return deleted_count > 0

    async def delete_all_sessions(self) -> int:
        """Delete all sessions and all child messages. Returns the count of deleted sessions."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sessions")
        count = cur.fetchone()[0]
        # Delete all messages first
        cur.execute("DELETE FROM messages")
        # Now delete all sessions
        cur.execute("DELETE FROM sessions")
        self.conn.commit()
        return count

    async def _ensure_session(self, session_id: str) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.conn.execute(
            "INSERT OR IGNORE INTO sessions(id, created_at, metadata) VALUES (?, ?, ?)",
            (session_id, now, "{}"),
        )
        self.conn.commit()

    async def add_user(
        self,
        session_id: str,
        text: str,
        *,
        turn_id: Optional[str] = None,
        seq_start: Optional[int] = None,
    ) -> None:
        await self._ensure_session(session_id)
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO messages(session_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (session_id, "user", text, ts),
        )
        self.conn.commit()
        # turn_id/seq_start accepted for future step-63 reshape; not yet written
        # to v2 tables (messages.turn_id column not yet added). Synthesis §3.6.
        if turn_id is not None:
            logger.debug(
                "add_user: turn_id=%s seq_start=%s (v2 write deferred to step 63)",
                turn_id, seq_start,
            )

    async def add_assistant_text(
        self,
        session_id: str,
        text: str,
        thinking_text: Optional[str] = None,
        save_thinking: bool = True,
        *,
        turn_id: Optional[str] = None,
        seq_start: Optional[int] = None,
    ) -> None:
        await self._ensure_session(session_id)
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        thinking_value = thinking_text if (save_thinking and thinking_text) else None
        self.conn.execute(
            "INSERT INTO messages(session_id, role, content, thinking_text, ts) VALUES (?, ?, ?, ?, ?)",
            (session_id, "assistant", text, thinking_value, ts),
        )
        self.conn.commit()
        # turn_id/seq_start accepted for future step-63 reshape. Synthesis §3.6.
        if turn_id is not None:
            logger.debug(
                "add_assistant_text: turn_id=%s seq_start=%s (v2 write deferred to step 63)",
                turn_id, seq_start,
            )

    async def add_assistant_toolcall(
        self,
        session_id: str,
        tool_name: str,
        args: Dict[str, Any],
        *,
        turn_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        seq_start: Optional[int] = None,
    ) -> None:
        await self._ensure_session(session_id)
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # v1 messages row (unchanged shape — synthesis §3.6 anti-scope)
        self.conn.execute(
            "INSERT INTO messages(session_id, role, tool_name, args, ts) VALUES (?, ?, ?, ?, ?)",
            (session_id, "tool", tool_name, json.dumps(args or {}), ts),
        )
        # v2 tool_calls row when turn_id + tool_call_id are provided
        if turn_id is not None and tool_call_id is not None:
            self.conn.execute(
                "INSERT OR IGNORE INTO tool_calls"
                "(tool_call_id, session_id, turn_id, name, arguments_json, status, call_seq)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    tool_call_id, session_id, turn_id, tool_name,
                    json.dumps(args or {}), "running", seq_start,
                ),
            )
        self.conn.commit()

    async def add_tool_result(
        self,
        session_id: str,
        tool_name: str,
        result: Any,
        *,
        turn_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        seq_start: Optional[int] = None,
        status: str = "ok",
        error: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        await self._ensure_session(session_id)
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # v1 messages row (unchanged shape — synthesis §3.6 anti-scope)
        self.conn.execute(
            "INSERT INTO messages(session_id, role, tool_name, result, ts) VALUES (?, ?, ?, ?, ?)",
            (session_id, "tool_result", tool_name, json.dumps(result), ts),
        )
        # v2 tool_calls UPDATE when turn_id + tool_call_id are provided
        if turn_id is not None and tool_call_id is not None:
            completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self.conn.execute(
                "UPDATE tool_calls SET status=?, result_json=?, error_json=?,"
                " result_seq=?, completed_at=?, duration_ms=?"
                " WHERE tool_call_id=?",
                (
                    status, json.dumps(result), error,
                    seq_start, completed_at, duration_ms,
                    tool_call_id,
                ),
            )
        self.conn.commit()

    async def get_history(
        self, session_id: str, include_thinking: bool = False
    ) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT role, content, thinking_text, tool_name, args, result FROM messages WHERE session_id = ? ORDER BY ts ASC",
            (session_id,),
        ).fetchall()
        history: List[Dict[str, Any]] = []
        for r in rows:
            role = r["role"]
            if role in ("user", "assistant", "system"):
                content = r["content"] or ""
                if role == "assistant" and include_thinking:
                    thinking = r["thinking_text"] or ""
                    if thinking:
                        content = f"{thinking}{content}"
                history.append({"role": role, "content": content})
            elif role == "tool":
                # Assistant made a tool call - format as assistant message with function_call syntax
                tool_name = r["tool_name"]
                args = json.loads(r["args"] or "{}")
                tool_call_json = json.dumps({"name": tool_name, "arguments": args})
                content = f"<<function_call>> {tool_call_json}"
                history.append({"role": "assistant", "content": content})
            elif role == "tool_result":
                # Tool execution result - format as user message so model can see the result
                tool_name = r["tool_name"]
                result = json.loads(r["result"] or "{}")
                result_text = json.dumps(result, indent=2)
                content = f"Tool '{tool_name}' returned:\n{result_text}"
                history.append({"role": "user", "content": content})
        return history

    async def ensure_system_prompt(self, session_id: str, prompt: str) -> None:
        # If no messages, seed with system prompt (empty ok)
        count = self.conn.execute(
            "SELECT COUNT(1) AS c FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()["c"]
        if count == 0:
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self.conn.execute(
                "INSERT INTO messages(session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (session_id, "system", prompt, ts),
            )
            self.conn.commit()

    # ------------------------------------------------------------------
    # v2 turn lifecycle (synthesis §3.6 + b1-persistence.md)
    # ------------------------------------------------------------------

    async def start_turn(
        self,
        session_id: str,
        turn_id: str,
        *,
        model_name: Optional[str] = None,
    ) -> None:
        """Insert a turns row in 'running' state.

        _ensure_session guarantees a parent sessions row even when the
        orchestrator calls this before the first message write.
        Synthesis §3.6.
        """
        await self._ensure_session(session_id)
        self.conn.execute(
            "INSERT OR IGNORE INTO turns(turn_id, session_id, model_name, status)"
            " VALUES (?, ?, ?, ?)",
            (turn_id, session_id, model_name, "running"),
        )
        self.conn.commit()

    async def complete_turn(
        self,
        turn_id: str,
        *,
        status: str = "completed",
        stop_reason: Optional[str] = None,
        error_json: Optional[str] = None,
    ) -> None:
        """Stamp the turns row with completed_at + final status.

        status must satisfy the CHECK constraint:
          running | completed | failed | cancelled
        Synthesis §3.6.
        """
        completed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE turns SET status=?, stop_reason=?, completed_at=?, error_json=?"
            " WHERE turn_id=?",
            (status, stop_reason, completed_at, error_json, turn_id),
        )
        self.conn.commit()

    async def record_raw_event(
        self,
        session_id: str,
        turn_id: str,
        seq: int,
        event_type: str,
        payload: Dict[str, Any],
        *,
        tool_call_id: Optional[str] = None,
    ) -> None:
        """Insert a raw_events row.

        UNIQUE(turn_id, seq) violations are logged at WARNING and
        swallowed; the replay log can tolerate sparse gaps.
        Synthesis §3.6.
        """
        payload_json = json.dumps(payload, default=str)
        try:
            self.conn.execute(
                "INSERT INTO raw_events"
                "(session_id, turn_id, seq, type, tool_call_id, payload_json)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, turn_id, seq, event_type, tool_call_id, payload_json),
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            logger.warning(
                "Duplicate raw_event skipped: turn_id=%s seq=%s", turn_id, seq
            )
