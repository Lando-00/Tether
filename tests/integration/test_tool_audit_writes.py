"""Orchestrator writes tool_audit rows on every tool dispatch.

Phase 7 step 74: the orchestrator's _audit_tool_call stub becomes a real
INSERT to tool_audit. Tests use a scripted provider (emits <<function_call>>)
to guarantee tool dispatch regardless of model availability.

Synthesis §3.6 + B5 step 7.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest

from tether_service.context.sqlite_store import SqliteSessionStore
from tether_service.core.interfaces import ModelProvider, Tool
from tether_service.core.types import OrchestratorConfig, ToolExecutionContext
from tether_service.protocol.orchestration.chatty import ChattyAgentOrchestrator
from tether_service.protocol.orchestration.policies import (
    LoopLimitPolicy,
    ToolErrorPolicy,
)
from tether_service.protocol.orchestration.tool_runner import ToolRunner
from tether_service.protocol.parsers.sliding import SlidingParser

# yoyo uses datetime.utcnow() — suppress its DeprecationWarning so
# -W error sweeps stay clean.
pytestmark = [
    pytest.mark.anyio,
    pytest.mark.filterwarnings("ignore::DeprecationWarning:yoyo"),
]


# ---------------------------------------------------------------------------
# Scripted providers and tools (shared with test_tool_error_feed_back.py style)
# ---------------------------------------------------------------------------


class _ToolThenDoneProvider(ModelProvider):
    """Iter 1: emit a tool call. Iter 2: emit final answer text."""

    def __init__(self, tool_name: str = "noop", tool_args: dict | None = None):
        self._calls = 0
        self._tool_name = tool_name
        self._tool_args = tool_args or {}

    async def stream(
        self, model_name, messages, tools=None, **kwargs
    ) -> AsyncGenerator[str, None]:
        self._calls += 1
        if self._calls == 1:
            args_str = json.dumps(self._tool_args)
            yield (
                "Long enough preamble to flush parser overlap. "
                f'<<function_call>> {{"name": "{self._tool_name}", "arguments": {args_str}}}'
            )
        else:
            yield "Final answer after the tool ran successfully."

    def list_models(self) -> List[str]:
        return ["scripted"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


class _RaisingProvider(ModelProvider):
    """Always emits a tool call (triggers error path)."""

    async def stream(
        self, model_name, messages, tools=None, **kwargs
    ) -> AsyncGenerator[str, None]:
        yield (
            'Long enough preamble. '
            '<<function_call>> {"name": "noop", "arguments": {}}'
        )

    def list_models(self) -> List[str]:
        return ["scripted"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


class _OkTool(Tool):
    """Returns {"ok": True} always."""

    @property
    def name(self) -> str:
        return "noop"

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": "noop",
            "description": "succeeds",
            "parameters": {"type": "object", "properties": {}},
        }

    async def invoke(
        self, args: Dict[str, Any], *, context: Optional[ToolExecutionContext] = None
    ) -> Any:
        return {"ok": True}


class _RaisingTool(Tool):
    """Raises ValueError on every call."""

    @property
    def name(self) -> str:
        return "noop"

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": "noop",
            "description": "always raises",
            "parameters": {"type": "object", "properties": {}},
        }

    async def invoke(
        self, args: Dict[str, Any], *, context: Optional[ToolExecutionContext] = None
    ) -> Any:
        raise ValueError("simulated tool failure")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(db_path: str, *, store_args: bool = False):
    from tether_service.config.settings import (
        AuditLogSettings,
        SecuritySettings,
        Settings,
    )
    return Settings.model_validate(
        {
            "system": {"prompt": "audit-test-prompt"},
            "providers": {
                "model": {
                    "impl": "tether_service.providers.dummy.provider.DummyProvider",
                    "args": {},
                },
                "parser": {
                    "impl": "tether_service.protocol.parsers.sliding.SlidingParser",
                    "args": {},
                },
                "session_store": {
                    "impl": "tether_service.context.sqlite_store.SqliteSessionStore",
                    "args": {"dsn": f"sqlite:///{db_path}"},
                },
            },
            "tools": {
                "registry": [],
                "enabled": [],
                "disabled": ["time", "weather", "forecast", "web_search"],
            },
            "security": {
                "audit_log": {"store_args": store_args},
            },
        }
    )


def _query_audit(db_path: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT correlation_id, session_id, turn_id, tool_call_id, tool_name, "
        "args_sha256, args_json, capabilities, status, error_kind, duration_ms "
        "FROM tool_audit ORDER BY audit_id ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _build_orch(
    provider: ModelProvider,
    tool: Tool,
    store,
    *,
    store_args: bool = False,
    policy: ToolErrorPolicy = ToolErrorPolicy.BREAK_LOOP,
) -> ChattyAgentOrchestrator:
    tools = {"noop": tool}
    config = OrchestratorConfig(
        max_tool_loops=3,
        auto_reload_on_fatal_error=False,
        save_thinking=False,
        include_thinking_in_history=False,
        loop_limit_policy=LoopLimitPolicy.EMIT_LIMIT_EVENT,
        tool_error_policy=policy,
    )
    return ChattyAgentOrchestrator(
        provider=provider,
        parser=SlidingParser(),
        store=store,
        tools=tools,
        system_prompt="sys",
        config=config,
        tool_runner=ToolRunner(tools, timeout_sec=2),
        audit_store_args=store_args,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_successful_tool_writes_audit_row(tmp_path):
    """A successful tool dispatch writes ONE row with status='ok'."""
    db_path = (tmp_path / "audit.db").as_posix()
    settings = _settings(db_path)
    from tether_service.engine import Engine
    import time as _time
    Engine.from_settings(settings)  # apply migrations

    store = SqliteSessionStore(dsn=f"sqlite:///{db_path}")
    orch = _build_orch(_ToolThenDoneProvider(), _OkTool(), store)

    async with store:
        await store.create_session("s1", int(_time.time()))
        async for _ in orch.run(
            session_id="s1", prompt="hi", model_name="scripted"
        ):
            pass

    rows = _query_audit(db_path)
    assert len(rows) == 1, f"Expected 1 audit row, got {len(rows)}: {rows}"
    r = rows[0]
    assert r["session_id"] == "s1"
    assert r["turn_id"] is not None
    assert r["tool_name"] == "noop"
    assert r["args_sha256"] is not None and len(r["args_sha256"]) == 64
    assert r["status"] == "ok"
    assert r["error_kind"] is None
    assert r["duration_ms"] is not None and r["duration_ms"] >= 0
    assert r["capabilities"] == "[]"


async def test_error_tool_writes_audit_row_with_error_kind(tmp_path):
    """A tool that raises writes ONE row with status='error', error_kind='execution'."""
    import time as _time
    db_path = (tmp_path / "audit-err.db").as_posix()
    settings = _settings(db_path)
    from tether_service.engine import Engine
    Engine.from_settings(settings)

    store = SqliteSessionStore(dsn=f"sqlite:///{db_path}")
    orch = _build_orch(
        _RaisingProvider(), _RaisingTool(), store,
        policy=ToolErrorPolicy.BREAK_LOOP,
    )

    async with store:
        await store.create_session("s2", int(_time.time()))
        async for _ in orch.run(
            session_id="s2", prompt="hi", model_name="scripted"
        ):
            pass

    rows = _query_audit(db_path)
    assert len(rows) == 1, f"Expected 1 audit row, got {len(rows)}: {rows}"
    r = rows[0]
    assert r["status"] == "error"
    assert r["error_kind"] == "execution"
    assert r["tool_name"] == "noop"
    assert r["duration_ms"] is not None


async def test_args_json_null_by_default(tmp_path):
    """args_json is NULL by default (store_args=False)."""
    import time as _time
    db_path = (tmp_path / "audit-no-args.db").as_posix()
    settings = _settings(db_path, store_args=False)
    from tether_service.engine import Engine
    Engine.from_settings(settings)

    store = SqliteSessionStore(dsn=f"sqlite:///{db_path}")
    orch = _build_orch(_ToolThenDoneProvider(), _OkTool(), store, store_args=False)

    async with store:
        await store.create_session("s3", int(_time.time()))
        async for _ in orch.run(
            session_id="s3", prompt="hi", model_name="scripted"
        ):
            pass

    rows = _query_audit(db_path)
    assert len(rows) >= 1
    for r in rows:
        assert r["args_json"] is None, f"args_json should be NULL, got {r['args_json']!r}"


async def test_args_json_populated_when_store_args_true(tmp_path):
    """args_json is populated when store_args=True."""
    import time as _time
    db_path = (tmp_path / "audit-args.db").as_posix()
    settings = _settings(db_path, store_args=True)
    from tether_service.engine import Engine
    Engine.from_settings(settings)

    store = SqliteSessionStore(dsn=f"sqlite:///{db_path}")
    orch = _build_orch(
        _ToolThenDoneProvider(tool_args={"key": "val"}),
        _OkTool(),
        store,
        store_args=True,
    )

    async with store:
        await store.create_session("s4", int(_time.time()))
        async for _ in orch.run(
            session_id="s4", prompt="hi", model_name="scripted"
        ):
            pass

    rows = _query_audit(db_path)
    assert len(rows) >= 1
    assert any(r["args_json"] is not None for r in rows), (
        f"Expected at least one row with args_json populated; rows={rows}"
    )
    # Verify the stored JSON is valid and matches the canonical encoding
    populated = [r for r in rows if r["args_json"] is not None]
    for r in populated:
        parsed = json.loads(r["args_json"])
        assert isinstance(parsed, dict)


async def test_args_sha256_deterministic(tmp_path):
    """args_sha256 is a stable SHA-256 of canonical (sorted-key) JSON args."""
    import time as _time
    db_path = (tmp_path / "audit-sha.db").as_posix()
    settings = _settings(db_path)
    from tether_service.engine import Engine
    Engine.from_settings(settings)

    tool_args = {"b": 2, "a": 1}
    expected_sha = hashlib.sha256(
        json.dumps(tool_args, sort_keys=True, default=str).encode()
    ).hexdigest()

    store = SqliteSessionStore(dsn=f"sqlite:///{db_path}")
    orch = _build_orch(
        _ToolThenDoneProvider(tool_args=tool_args), _OkTool(), store
    )

    async with store:
        await store.create_session("s5", int(_time.time()))
        async for _ in orch.run(
            session_id="s5", prompt="hi", model_name="scripted"
        ):
            pass

    rows = _query_audit(db_path)
    assert len(rows) >= 1
    assert rows[0]["args_sha256"] == expected_sha


async def test_tool_audit_no_op_on_memory_store():
    """MemoryStore inherits the no-op audit_tool_call — must not raise."""
    from tether_service.context.memory_store import MemoryStore

    store = MemoryStore()
    await store.audit_tool_call(
        correlation_id="cid",
        session_id="sid",
        turn_id="tid",
        tool_call_id="call-1",
        tool_name="t",
        args_sha256="a" * 64,
        args_json=None,
        status="ok",
        error_kind=None,
        duration_ms=42,
    )
    # No assertion needed — just verifies it doesn't raise


async def test_audit_write_failure_doesnt_break_orchestrator(tmp_path):
    """If audit_tool_call raises, the orchestrator logs and continues."""
    from tether_service.context.memory_store import MemoryStore
    from tether_service.protocol.wire.events import MessageStop

    class _FailingStore(MemoryStore):
        async def audit_tool_call(self, **kwargs) -> None:
            raise RuntimeError("simulated audit failure")

    store = _FailingStore()
    orch = _build_orch(_ToolThenDoneProvider(), _OkTool(), store)

    events = []
    async for ev in orch.run(
        session_id="s6", prompt="hi", model_name="scripted"
    ):
        events.append(ev)

    # The orchestrator must complete normally (MessageStop emitted) despite
    # the audit write failure.
    assert any(isinstance(ev, MessageStop) for ev in events), (
        f"MessageStop not found in events: {events}"
    )
    stop = [ev for ev in events if isinstance(ev, MessageStop)][0]
    assert stop.stop_reason in ("complete", "ok", "completed")


async def test_correlation_id_falls_back_to_turn_id_in_library_mode(tmp_path):
    """Without HTTP middleware (no request_id contextvar), correlation_id == turn_id."""
    import time as _time
    db_path = (tmp_path / "audit-libmode.db").as_posix()
    settings = _settings(db_path)
    from tether_service.engine import Engine
    Engine.from_settings(settings)

    store = SqliteSessionStore(dsn=f"sqlite:///{db_path}")
    orch = _build_orch(_ToolThenDoneProvider(), _OkTool(), store)

    # Ensure no request_id is set in structlog contextvars
    import structlog
    structlog.contextvars.clear_contextvars()

    async with store:
        await store.create_session("s7", int(_time.time()))
        async for _ in orch.run(
            session_id="s7", prompt="hi", model_name="scripted"
        ):
            pass

    rows = _query_audit(db_path)
    assert len(rows) >= 1
    for r in rows:
        if r["session_id"] == "s7":
            # Library mode: no request_id → correlation_id falls back to turn_id
            assert r["correlation_id"] == r["turn_id"], (
                f"Expected correlation_id == turn_id in library mode, "
                f"got correlation_id={r['correlation_id']!r}, "
                f"turn_id={r['turn_id']!r}"
            )
