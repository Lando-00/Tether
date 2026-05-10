"""Tests for :class:`LoopLimitPolicy` + :class:`ToolErrorPolicy` enums and
their plumbing through :class:`OrchestratorConfig` /
:class:`LimitsSettings` (synthesis §3.5)."""
from __future__ import annotations

import pytest

from tether.config.settings import LimitsSettings, Settings
from tether.core.types import OrchestratorConfig
from tether.protocol.orchestration.policies import (
    LoopLimitPolicy,
    ToolErrorPolicy,
)


def _settings_dict(**limits_overrides) -> dict:
    base = {
        "system": {"prompt": ""},
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
                "args": {"dsn": "sqlite:///:memory:"},
            },
        },
        "tools": {"registry": [], "enabled": [], "disabled": ["time", "weather", "forecast", "web_search"]},
    }
    if limits_overrides:
        base["limits"] = limits_overrides
    return base


def test_loop_limit_policy_values():
    assert LoopLimitPolicy.RAISE == "raise"
    assert LoopLimitPolicy.EMIT_LIMIT_EVENT == "emit_limit_event"
    assert LoopLimitPolicy("raise") is LoopLimitPolicy.RAISE
    assert LoopLimitPolicy("emit_limit_event") is LoopLimitPolicy.EMIT_LIMIT_EVENT


def test_tool_error_policy_values():
    assert ToolErrorPolicy.BREAK_LOOP == "break_loop"
    assert ToolErrorPolicy.FEED_BACK_TO_MODEL == "feed_back_to_model"
    assert ToolErrorPolicy("break_loop") is ToolErrorPolicy.BREAK_LOOP
    assert ToolErrorPolicy("feed_back_to_model") is ToolErrorPolicy.FEED_BACK_TO_MODEL


def test_limits_settings_defaults():
    """Defaults match the user-ratified plan: emit_limit_event +
    feed_back_to_model."""
    s = LimitsSettings()
    assert s.loop_limit_policy == "emit_limit_event"
    assert s.tool_error_policy == "feed_back_to_model"


def test_orchestrator_config_default_loop_limit():
    """``from_settings`` produces ``LoopLimitPolicy.EMIT_LIMIT_EVENT`` by
    default (synthesis §3.5; user-ratified)."""
    settings = Settings.model_validate(_settings_dict())
    cfg = OrchestratorConfig.from_settings(settings)
    assert cfg.loop_limit_policy is LoopLimitPolicy.EMIT_LIMIT_EVENT


def test_orchestrator_config_default_tool_error():
    """``from_settings`` produces ``ToolErrorPolicy.FEED_BACK_TO_MODEL`` by
    default (synthesis §3.5; A5 P2)."""
    settings = Settings.model_validate(_settings_dict())
    cfg = OrchestratorConfig.from_settings(settings)
    assert cfg.tool_error_policy is ToolErrorPolicy.FEED_BACK_TO_MODEL


def test_orchestrator_config_overrides():
    """Setting ``raise`` / ``break_loop`` in YAML produces the right enum
    values."""
    settings = Settings.model_validate(
        _settings_dict(loop_limit_policy="raise", tool_error_policy="break_loop")
    )
    cfg = OrchestratorConfig.from_settings(settings)
    assert cfg.loop_limit_policy is LoopLimitPolicy.RAISE
    assert cfg.tool_error_policy is ToolErrorPolicy.BREAK_LOOP


def test_orchestrator_config_dataclass_default():
    """Direct constructor (no settings): policy fields default to the
    same user-ratified values."""
    cfg = OrchestratorConfig(
        max_tool_loops=5,
        auto_reload_on_fatal_error=False,
        save_thinking=True,
        include_thinking_in_history=False,
    )
    assert cfg.loop_limit_policy is LoopLimitPolicy.EMIT_LIMIT_EVENT
    assert cfg.tool_error_policy is ToolErrorPolicy.FEED_BACK_TO_MODEL


def test_limits_settings_rejects_unknown_policy():
    """Pydantic ``Literal`` rejects garbage values at validation time."""
    with pytest.raises(Exception):
        LimitsSettings(loop_limit_policy="invalid_policy")
    with pytest.raises(Exception):
        LimitsSettings(tool_error_policy="silent_swallow")
