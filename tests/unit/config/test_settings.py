"""Tests for tether.config.settings — typed Settings + load_settings."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from tether.config.settings import (
    ConnectorSpec,
    ConnectorsSettings,
    HttpSettings,
    InboxSettings,
    ProvidersSettings,
    ProviderSpec,
    Settings,
    load_settings,
)


_MIN_PROVIDERS = {
    "model": {"impl": "x.M", "args": {}},
    "parser": {"impl": "x.P", "args": {}},
    "session_store": {"impl": "x.S", "args": {}},
}


def _min_settings_dict() -> dict:
    return {"providers": _MIN_PROVIDERS}


def test_settings_construct_from_dict():
    s = Settings.model_validate(_min_settings_dict())
    assert isinstance(s, Settings)
    assert s.http.host == "127.0.0.1"
    assert s.http.port == 8080
    assert s.providers.model.impl == "x.M"


def test_settings_extra_forbid():
    bad = {**_min_settings_dict(), "totally_unknown_section": {}}
    with pytest.raises(ValidationError):
        Settings.model_validate(bad)


def test_settings_frozen():
    s = Settings.model_validate(_min_settings_dict())
    with pytest.raises(ValidationError):
        s.http = HttpSettings(host="other", port=1)  # type: ignore[misc]


def test_settings_yaml_round_trip():
    s = load_settings(env={})
    # Default YAML: host/port nested under http; MLCProvider as model impl.
    assert s.http.host == "127.0.0.1"
    assert s.http.port == 8080
    assert s.providers.model.impl == "tether.providers.mlc.provider.MLCProvider"
    # Tools registry from default.yml.
    names = [t.name for t in s.tools.registry]
    assert "time" in names
    assert "web_search" in names
    assert s.context.save_thinking is True


def test_settings_env_override():
    s = load_settings(env={"TETHER__http__port": "9999"})
    assert s.http.port == 9999


def test_settings_overlay_yaml(tmp_path: Path):
    overlay = tmp_path / "dev.yml"
    overlay.write_text(
        textwrap.dedent(
            """
            http:
              port: 7777
            limits:
              max_tool_loops: 9
            """
        ).strip(),
        encoding="utf-8",
    )
    s = load_settings(overlay_yaml=overlay, env={})
    assert s.http.port == 7777
    assert s.http.host == "127.0.0.1"  # base preserved
    assert s.limits.max_tool_loops == 9


def test_settings_replaces_default(tmp_path: Path):
    overlay = tmp_path / "dev.yml"
    overlay.write_text(
        textwrap.dedent(
            """
            _replaces_default: true
            providers:
              model:
                impl: "fake.M"
                args: {}
              parser:
                impl: "fake.P"
                args: {}
              session_store:
                impl: "fake.S"
                args: {}
            http:
              host: "0.0.0.0"
              port: 1234
            """
        ).strip(),
        encoding="utf-8",
    )
    s = load_settings(overlay_yaml=overlay, env={})
    assert s.providers.model.impl == "fake.M"
    assert s.http.host == "0.0.0.0"
    assert s.http.port == 1234
    # default.yml content (e.g. tools registry) should NOT have leaked through.
    assert s.tools.registry == []


def test_load_settings_legacy_still_dict():
    """The legacy loader is preserved (one-cycle coexistence)."""
    from tether.core.config import load_settings_legacy

    cfg = load_settings_legacy()
    assert isinstance(cfg, dict)
    assert "http" in cfg or "host" in cfg


def test_providers_settings_requires_all_three():
    with pytest.raises(ValidationError):
        ProvidersSettings.model_validate(
            {"model": {"impl": "x", "args": {}}}  # missing parser, session_store
        )


def test_provider_spec_default_args():
    spec = ProviderSpec.model_validate({"impl": "x.Y"})
    assert spec.args == {}


# ---------------------------------------------------------------------------
# Phase 4.5: ConnectorsSettings + InboxSettings (connector spec §3.7)
# ---------------------------------------------------------------------------


def test_connectors_settings_default_construct():
    """Empty registry is valid (no concrete connectors land in this PR)."""
    c = ConnectorsSettings()
    assert c.registry == {}


def test_connectors_settings_default_factory_independent():
    """``registry`` uses ``Field(default_factory=dict)`` so two instances
    don't share a mutable map (defensive against the classic
    ``def f(x={})`` pitfall — though pydantic v2 protects against it,
    documenting that we rely on it)."""
    a = ConnectorsSettings()
    b = ConnectorsSettings()
    # a.registry is frozen so we can't mutate it; equality + identity check.
    assert a.registry == b.registry == {}
    assert a.registry is not b.registry


def test_connector_spec_required_fields():
    """``impl`` is required; ``args`` defaults to {} and ``enabled`` to True."""
    spec = ConnectorSpec.model_validate({"impl": "x.Echo"})
    assert spec.impl == "x.Echo"
    assert spec.args == {}
    assert spec.enabled is True


def test_connector_spec_extra_forbid():
    with pytest.raises(ValidationError):
        ConnectorSpec.model_validate(
            {"impl": "x.Y", "completely_unknown_field": True}
        )


def test_connector_spec_disabled_round_trip():
    spec = ConnectorSpec.model_validate(
        {"impl": "x.Y", "args": {"k": 1}, "enabled": False}
    )
    assert spec.enabled is False
    assert spec.args == {"k": 1}


def test_connectors_registry_validates_specs():
    """The registry's value type is ConnectorSpec, so dicts are coerced
    and bad dicts fail validation."""
    c = ConnectorsSettings.model_validate(
        {
            "registry": {
                "echo": {"impl": "tether.connectors.echo.EchoConnector"},
            }
        }
    )
    assert "echo" in c.registry
    spec = c.registry["echo"]
    assert isinstance(spec, ConnectorSpec)
    assert spec.enabled is True

    with pytest.raises(ValidationError):
        # Missing required ``impl``.
        ConnectorsSettings.model_validate({"registry": {"bad": {}}})


def test_inbox_settings_defaults():
    i = InboxSettings()
    assert i.enabled is True
    assert i.retention_days == 30
    assert i.max_payload_bytes == 64_000
    assert i.max_summary_chars == 512


def test_inbox_settings_validates_bounds():
    """Pydantic-level Field constraints reject out-of-range values."""
    with pytest.raises(ValidationError):
        InboxSettings.model_validate({"retention_days": 0})
    with pytest.raises(ValidationError):
        InboxSettings.model_validate({"retention_days": 5000})
    with pytest.raises(ValidationError):
        InboxSettings.model_validate({"max_payload_bytes": 100})
    with pytest.raises(ValidationError):
        InboxSettings.model_validate({"max_summary_chars": 0})


def test_settings_loads_default_yaml_connectors_inbox():
    """The packaged default.yml ships a populated ``inbox:`` section
    (Phase 4.5 step 47a) and an empty ``connectors.registry``. Round-trip
    via load_settings() to verify the YAML matches the schema."""
    s = load_settings(env={})
    assert s.connectors.registry == {}
    assert s.inbox.enabled is True
    assert s.inbox.retention_days == 30
    assert s.inbox.max_payload_bytes == 64_000
    assert s.inbox.max_summary_chars == 512


def test_settings_overlay_adds_connector_entry(tmp_path: Path):
    overlay = tmp_path / "dev.yml"
    overlay.write_text(
        textwrap.dedent(
            """
            connectors:
              registry:
                echo:
                  impl: "tether.connectors.echo.EchoConnector"
                  args:
                    greeting: "hi"
            """
        ).strip(),
        encoding="utf-8",
    )
    s = load_settings(overlay_yaml=overlay, env={})
    assert "echo" in s.connectors.registry
    assert s.connectors.registry["echo"].impl == (
        "tether.connectors.echo.EchoConnector"
    )
    assert s.connectors.registry["echo"].args == {"greeting": "hi"}
    # Inbox still picks up the default.yml values:
    assert s.inbox.enabled is True
