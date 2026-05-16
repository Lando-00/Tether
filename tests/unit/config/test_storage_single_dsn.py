"""P0-A regression: ``storage.sqlite.dsn`` is the single source of truth.

Tribunal §3 P0-01 (findings A3-F1, A12-F1).

Two invariants are pinned here:

1. The packaged ``default.yml`` does NOT hard-code a session-store DSN
   (legacy ``providers.session_store.args.dsn``). The authoritative DSN
   lives at ``storage.sqlite.dsn`` per ADR-0009.
2. ``Engine.from_settings`` rejects a configuration where a legacy
   session-store DSN is *also* set and disagrees with the resolved
   storage DSN — a previously-silent split that routed sessions and
   the inbox onto two different SQLite files.
"""
from __future__ import annotations

import asyncio

import pytest

from tether.config.settings import Settings, load_settings
from tether.core.errors import ConfigError
from tether.engine import Engine


def test_default_config_no_session_store_dsn():
    """Default packaged YAML must NOT hard-code a session-store DSN."""
    settings = load_settings()
    assert "dsn" not in settings.providers.session_store.args, (
        "providers.session_store.args still contains a hard-coded dsn: "
        f"{settings.providers.session_store.args!r}. Phase-9 P0-A requires "
        "removing it and relying on storage.sqlite.dsn instead "
        "(ADR-0009 single source of truth)."
    )


def _engine_settings_dict(*, storage_dsn: str, legacy_dsn: str | None) -> dict:
    """Minimal Settings dict driving Engine.from_settings."""
    session_args: dict = {}
    if legacy_dsn is not None:
        session_args["dsn"] = legacy_dsn
    return {
        "system": {"prompt": "test-prompt"},
        "providers": {
            "model": {
                "impl": "tether.providers.dummy.provider.DummyProvider",
                "args": {},
            },
            "parser": {
                "impl": "tether.protocol.parsers.sliding.SlidingParser",
                "args": {},
            },
            "session_store": {
                "impl": "tether.context.sqlite_store.SqliteSessionStore",
                "args": session_args,
            },
        },
        "storage": {"sqlite": {"dsn": storage_dsn}},
        "tools": {
            "registry": [],
            "enabled": [],
            "disabled": ["time", "weather", "forecast", "web_search"],
        },
        "orchestrator": {
            "registry": {
                "chat": "tether.protocol.orchestration.chatty.ChattyAgentOrchestrator",
            },
        },
    }


def test_engine_rejects_disagreeing_dsns(tmp_path):
    """If both DSNs are set and disagree, Engine.from_settings must raise."""
    storage_dsn = f"sqlite:///{(tmp_path / 'right.db').as_posix()}"
    legacy_dsn = f"sqlite:///{(tmp_path / 'wrong.db').as_posix()}"
    bad = Settings.model_validate(
        _engine_settings_dict(storage_dsn=storage_dsn, legacy_dsn=legacy_dsn)
    )

    with pytest.raises(ConfigError, match="single source of truth"):
        Engine.from_settings(bad)


def test_engine_accepts_matching_legacy_dsn(tmp_path):
    """Strict-equality match: legacy DSN equal to storage DSN is allowed."""
    same_dsn = f"sqlite:///{(tmp_path / 'same.db').as_posix()}"
    ok = Settings.model_validate(
        _engine_settings_dict(storage_dsn=same_dsn, legacy_dsn=same_dsn)
    )
    engine = Engine.from_settings(ok)
    try:
        assert engine.store is not None
    finally:
        asyncio.run(engine.aclose())


def test_engine_accepts_storage_only_dsn(tmp_path):
    """The canonical post-fix path: only storage.sqlite.dsn is set."""
    storage_dsn = f"sqlite:///{(tmp_path / 'storage_only.db').as_posix()}"
    ok = Settings.model_validate(
        _engine_settings_dict(storage_dsn=storage_dsn, legacy_dsn=None)
    )
    engine = Engine.from_settings(ok)
    try:
        assert engine.store is not None
    finally:
        asyncio.run(engine.aclose())
