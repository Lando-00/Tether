"""SQLite-backed session store implementing SessionStore with WAL + safe PRAGMAs"""
from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from tether_service.core.interfaces import SessionStore


class SqliteSessionStore(SessionStore):
    def __init__(self, dsn: str = "sqlite:///./data/tether.db"):
        # Parse DSN
        if dsn.startswith("sqlite:///"):
            path = dsn[len("sqlite:///") :]
        else:
            path = dsn

        # Ensure parent directory exists
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        # Connect with WAL and pragmas
        self.conn = sqlite3.connect(str(p), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_pragmas()
        self._init_schema()

    def _init_pragmas(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute("PRAGMA foreign_keys=ON;")
        self.conn.commit()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_name TEXT,
                args TEXT,
                result TEXT,
                ts TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_ts ON messages(session_id, ts)"
        )
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
        now = datetime.datetime.utcnow().isoformat()
        self.conn.execute(
            "INSERT OR IGNORE INTO sessions(id, created_at, metadata) VALUES (?, ?, ?)",
            (session_id, now, "{}"),
        )
        self.conn.commit()

    async def add_user(self, session_id: str, text: str) -> None:
        await self._ensure_session(session_id)
        ts = datetime.datetime.utcnow().isoformat()
        self.conn.execute(
            "INSERT INTO messages(session_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (session_id, "user", text, ts),
        )
        self.conn.commit()

    async def add_assistant_text(self, session_id: str, text: str) -> None:
        await self._ensure_session(session_id)
        ts = datetime.datetime.utcnow().isoformat()
        self.conn.execute(
            "INSERT INTO messages(session_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (session_id, "assistant", text, ts),
        )
        self.conn.commit()

    async def add_assistant_toolcall(
        self, session_id: str, tool_name: str, args: Dict[str, Any]
    ) -> None:
        await self._ensure_session(session_id)
        ts = datetime.datetime.utcnow().isoformat()
        self.conn.execute(
            "INSERT INTO messages(session_id, role, tool_name, args, ts) VALUES (?, ?, ?, ?, ?)",
            (session_id, "tool", tool_name, json.dumps(args or {}), ts),
        )
        self.conn.commit()

    async def add_tool_result(
        self, session_id: str, tool_name: str, result: Any
    ) -> None:
        await self._ensure_session(session_id)
        ts = datetime.datetime.utcnow().isoformat()
        self.conn.execute(
            "INSERT INTO messages(session_id, role, tool_name, result, ts) VALUES (?, ?, ?, ?, ?)",
            (session_id, "tool_result", tool_name, json.dumps(result), ts),
        )
        self.conn.commit()

    async def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT role, content, tool_name, args, result FROM messages WHERE session_id = ? ORDER BY ts ASC",
            (session_id,),
        ).fetchall()
        history: List[Dict[str, Any]] = []
        for r in rows:
            role = r["role"]
            if role in ("user", "assistant", "system"):
                history.append({"role": role, "content": r["content"] or ""})
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
            ts = datetime.datetime.utcnow().isoformat()
            self.conn.execute(
                "INSERT INTO messages(session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                (session_id, "system", prompt, ts),
            )
            self.conn.commit()
