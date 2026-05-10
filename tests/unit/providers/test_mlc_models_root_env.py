"""Verify ``TETHER_MODELS_DIR`` env var overrides ``models_root`` for ``MLCProvider``.

Phase 8 step 85 (``p8-models-rename``): the model storage directory was
renamed from ``dist/`` to ``models/`` and ``MLCProvider`` grew a typed
``models_root`` constructor argument plus a ``TETHER_MODELS_DIR``
environment-variable override. The env var lets operators relocate large
model artifacts off the project tree without editing config — it must win
over both the constructor argument (which carries the YAML default) and
the hard-coded ``"models"`` default.

These tests pin the precedence ordering and the default literal so a future
refactor can't silently break either.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tether.providers.mlc.provider import MLCProvider


def test_models_root_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env override, ``MLCProvider`` defaults to ``models``."""
    monkeypatch.delenv("TETHER_MODELS_DIR", raising=False)
    p = MLCProvider()
    assert p.models_root == Path("models")
    assert p.libs_dir == Path("models") / "libs"


def test_models_root_explicit_arg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An explicit ``models_root=...`` constructor arg is honoured when no env override is set."""
    monkeypatch.delenv("TETHER_MODELS_DIR", raising=False)
    custom = tmp_path / "custom_models"
    p = MLCProvider(models_root=str(custom))
    assert p.models_root == custom
    assert p.libs_dir == custom / "libs"


def test_tether_models_dir_env_overrides_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``TETHER_MODELS_DIR`` overrides the hard-coded ``"models"`` default."""
    override = tmp_path / "env_models"
    monkeypatch.setenv("TETHER_MODELS_DIR", str(override))
    p = MLCProvider()
    assert p.models_root == override
    assert p.libs_dir == override / "libs"


def test_tether_models_dir_env_overrides_explicit_arg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``TETHER_MODELS_DIR`` overrides even an explicit ``models_root=...`` arg.

    The explicit constructor arg comes from YAML (``providers.model.args``);
    env vars sit above YAML in the standard precedence ladder. This test
    pins that ordering so a future refactor can't silently invert it.
    """
    yaml_value = tmp_path / "from_yaml"
    env_value = tmp_path / "from_env"
    monkeypatch.setenv("TETHER_MODELS_DIR", str(env_value))
    p = MLCProvider(models_root=str(yaml_value))
    assert p.models_root == env_value
    assert p.libs_dir == env_value / "libs"


def test_tether_models_dir_env_empty_string_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty ``TETHER_MODELS_DIR`` must not override (treat as unset)."""
    monkeypatch.setenv("TETHER_MODELS_DIR", "")
    p = MLCProvider()
    assert p.models_root == Path("models")
