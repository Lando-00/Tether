"""``Engine.from_settings`` must validate the orchestrator registry at boot
rather than letting a typo surface as a 500 mid-handler.

Phase 5 followups F6 (rubber-duck review): a typo in ``default.yml``'s
orchestrator registry path previously surfaced as ``ImportError`` from
inside the request handler at first ``Engine.chat`` call. Now
``Engine.from_settings`` calls ``resolve_orchestrator_class`` for every
registered mode at construction time. ``importlib.import_module`` is
cached, so this is essentially free.

Synthesis §3.5; briefing §2 Seam B.
"""
from __future__ import annotations

import pytest

from tether.config.settings import Settings


def _settings_dict(tmp_db: str, *, orchestrator: dict | None = None) -> dict:
    """Minimal Settings dict using DummyProvider + sqlite session store."""
    cfg: dict = {
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
                "args": {},
            },
        },
        "storage": {"sqlite": {"dsn": f"sqlite:///{tmp_db}"}},
        "tools": {
            "registry": [],
            "enabled": [],
            "disabled": ["time", "weather", "forecast", "web_search"],
        },
    }
    if orchestrator is not None:
        cfg["orchestrator"] = orchestrator
    return cfg


def test_invalid_dotted_path_raises_at_boot(tmp_path):
    """A registry entry pointing at a non-existent module raises
    ``ValueError`` during ``from_settings`` — not at first request.
    """
    db = tmp_path / "f6.db"
    settings = Settings.model_validate(
        _settings_dict(
            str(db),
            orchestrator={
                "default": "chat",
                "registry": {
                    "chat": (
                        "tether.protocol.orchestration.chatty"
                        ".ChattyAgentOrchestrator"
                    ),
                    "research": (
                        "tether.protocol.orchestration.notebook"
                        ".NotebookOrchestrator"
                    ),
                    "broken": (
                        "tether.does_not_exist.NotAClass"
                    ),
                },
            },
        )
    )

    from tether.engine import Engine

    with pytest.raises(ValueError, match="invalid"):
        Engine.from_settings(settings)


def test_non_orchestrator_class_raises_at_boot(tmp_path):
    """A registry entry pointing at a real class that is NOT an Orchestrator
    raises ``ValueError`` during ``from_settings``.
    """
    db = tmp_path / "f6.db"
    settings = Settings.model_validate(
        _settings_dict(
            str(db),
            orchestrator={
                "default": "chat",
                "registry": {
                    "chat": (
                        "tether.protocol.orchestration.chatty"
                        ".ChattyAgentOrchestrator"
                    ),
                    "wrong_type": "json.JSONDecoder",
                },
            },
        )
    )

    from tether.engine import Engine

    with pytest.raises(ValueError, match="invalid"):
        Engine.from_settings(settings)


def test_unresolvable_attribute_raises_at_boot(tmp_path):
    """A registry entry pointing at a real module but a non-existent
    attribute raises ``ValueError`` during ``from_settings``.
    """
    db = tmp_path / "f6.db"
    settings = Settings.model_validate(
        _settings_dict(
            str(db),
            orchestrator={
                "default": "chat",
                "registry": {
                    "chat": (
                        "tether.protocol.orchestration.chatty"
                        ".ChattyAgentOrchestrator"
                    ),
                    "missing_attr": (
                        "tether.protocol.orchestration.chatty"
                        ".NotARealClass"
                    ),
                },
            },
        )
    )

    from tether.engine import Engine

    with pytest.raises(ValueError, match="invalid"):
        Engine.from_settings(settings)


def test_valid_registry_loads_normally(tmp_path):
    """The default registry validates cleanly at boot."""
    db = tmp_path / "f6.db"
    settings = Settings.model_validate(_settings_dict(str(db)))

    from tether.engine import Engine

    engine = Engine.from_settings(settings)
    assert engine is not None
    # Both default modes resolved successfully — no ImportError surfaced.
    assert "chat" in engine._orchestrator_registry
    assert "research" in engine._orchestrator_registry
