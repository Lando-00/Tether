"""Tests for ProvidersSettings multi-provider registry (ADR-0021 P1.2)."""
from __future__ import annotations

import warnings

import pytest

from tether.config.settings import ProviderSpec, ProvidersSettings
from tether.core.errors import ConfigError

_PARSER = {"impl": "x.P", "args": {}}
_STORE = {"impl": "x.S", "args": {}}
_SPEC_A = {"impl": "x.A", "args": {}}
_SPEC_B = {"impl": "x.B", "args": {}}


def test_singular_model_synthesises_registry():
    with pytest.warns(DeprecationWarning):
        p = ProvidersSettings.model_validate(
            {"model": _SPEC_A, "parser": _PARSER, "session_store": _STORE}
        )
    assert "default" in p.model_registry
    assert p.model_registry["default"].impl == "x.A"
    assert p.default_model_provider == "default"
    # Legacy field is preserved (not removed) for one cycle.
    assert isinstance(p.model, ProviderSpec)


def test_explicit_registry_works():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        p = ProvidersSettings.model_validate(
            {
                "model_registry": {"a": _SPEC_A, "b": _SPEC_B},
                "default_model_provider": "a",
                "parser": _PARSER,
                "session_store": _STORE,
            }
        )
    assert set(p.model_registry) == {"a", "b"}
    assert p.default_model_provider == "a"
    assert p.model is None


def test_both_singular_and_registry_raises_config_error():
    with pytest.raises(ConfigError, match="mutually exclusive"):
        ProvidersSettings.model_validate(
            {
                "model": _SPEC_A,
                "model_registry": {"a": _SPEC_A},
                "default_model_provider": "a",
                "parser": _PARSER,
                "session_store": _STORE,
            }
        )


def test_empty_registry_raises_config_error():
    with pytest.raises(ConfigError, match="empty"):
        ProvidersSettings.model_validate(
            {"parser": _PARSER, "session_store": _STORE}
        )


def test_default_missing_raises():
    with pytest.raises(ConfigError, match="default_model_provider is required"):
        ProvidersSettings.model_validate(
            {
                "model_registry": {"a": _SPEC_A},
                "parser": _PARSER,
                "session_store": _STORE,
            }
        )


def test_default_not_in_registry_raises():
    with pytest.raises(ConfigError, match="not a key"):
        ProvidersSettings.model_validate(
            {
                "model_registry": {"a": _SPEC_A},
                "default_model_provider": "missing",
                "parser": _PARSER,
                "session_store": _STORE,
            }
        )


def test_legacy_load_warning_message_mentions_adr():
    with pytest.warns(DeprecationWarning, match="ADR-0021"):
        ProvidersSettings.model_validate(
            {"model": _SPEC_A, "parser": _PARSER, "session_store": _STORE}
        )
