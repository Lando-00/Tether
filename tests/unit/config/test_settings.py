"""Tests for tether_service.config.settings — typed Settings + load_settings."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from tether_service.config.settings import (
    HttpSettings,
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
    assert s.providers.model.impl == "tether_service.providers.mlc.provider.MLCProvider"
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
    from tether_service.core.config import load_settings_legacy

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
