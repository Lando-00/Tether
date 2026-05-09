"""Unit tests for ``tether_service.core.secrets``.

Per connector spec §3.5 + §3.6; synthesis §4 Phase 4.5 step 47a.
"""
from __future__ import annotations

import os
import platform
import stat
from pathlib import Path
from typing import Iterator
from unittest import mock

import pytest

from tether_service.core.secrets import (
    EnvFileSecretsProvider,
    SecretsProvider,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_data_dir(tmp_path: Path) -> Iterator[Path]:
    """Per-test data directory so file writes don't leak across tests."""
    yield tmp_path


@pytest.fixture
def provider(isolated_data_dir: Path) -> EnvFileSecretsProvider:
    return EnvFileSecretsProvider(data_dir=isolated_data_dir)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


def test_envfile_is_secrets_provider():
    """The default impl satisfies the abstract base class — type checkers
    and ``isinstance(..., SecretsProvider)`` checks both work."""
    p = EnvFileSecretsProvider(data_dir=Path("data"))
    assert isinstance(p, SecretsProvider)


def test_secrets_provider_cannot_instantiate_directly():
    with pytest.raises(TypeError):
        SecretsProvider()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# get() — env priority + file fallback
# ---------------------------------------------------------------------------


def test_envfile_get_env_priority(
    provider: EnvFileSecretsProvider, monkeypatch: pytest.MonkeyPatch
):
    """Even when a file exists, the env var wins. This matches the locked
    'env-first, file fallback' semantic in connector spec §3.5: developers
    can override file-stored secrets without touching disk.
    """
    monkeypatch.setenv("TETHER_TEST_SECRET", "from-env")
    provider.set("TETHER_TEST_SECRET", "from-file")
    assert provider.get("TETHER_TEST_SECRET") == "from-env"


def test_envfile_get_file_fallback(
    provider: EnvFileSecretsProvider, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("TETHER_FILE_ONLY", raising=False)
    provider.set("TETHER_FILE_ONLY", "secret-on-disk")
    assert provider.get("TETHER_FILE_ONLY") == "secret-on-disk"


def test_envfile_get_strips_trailing_newline(
    provider: EnvFileSecretsProvider,
    monkeypatch: pytest.MonkeyPatch,
    isolated_data_dir: Path,
):
    """``set()`` writes a trailing newline; ``get()`` must strip exactly
    one (so ``set('K', 'v') -> get('K')`` round-trips)."""
    monkeypatch.delenv("ROUND_TRIP", raising=False)
    provider.set("ROUND_TRIP", "value")
    # Verify the file actually has the trailing newline:
    raw = (isolated_data_dir / "secrets" / "ROUND_TRIP").read_text(
        encoding="utf-8"
    )
    assert raw == "value\n"
    # And get() strips it back:
    assert provider.get("ROUND_TRIP") == "value"


def test_envfile_get_returns_none_when_neither(
    provider: EnvFileSecretsProvider, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("DEFINITELY_NOT_SET_4831", raising=False)
    assert provider.get("DEFINITELY_NOT_SET_4831") is None


# ---------------------------------------------------------------------------
# set() — file mode + idempotency
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX file permission bits are not enforced on NTFS; ACLs handle this on Windows.",
)
def test_envfile_set_writes_file_with_mode_600(
    provider: EnvFileSecretsProvider, isolated_data_dir: Path
):
    """POSIX: set() chmods the file to 0o600 (owner-read+write only).
    Skipped on Windows where chmod is effectively a no-op."""
    provider.set("MODE_TEST", "value")
    path = isolated_data_dir / "secrets" / "MODE_TEST"
    assert path.exists()
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"expected 0o600, got 0o{mode:o}"


def test_envfile_set_creates_secrets_dir(
    provider: EnvFileSecretsProvider, isolated_data_dir: Path
):
    """``data/secrets/`` is created lazily on first ``set()``."""
    secrets_dir = isolated_data_dir / "secrets"
    assert not secrets_dir.exists()
    provider.set("LAZY", "value")
    assert secrets_dir.is_dir()


def test_envfile_set_overwrites_existing(provider: EnvFileSecretsProvider):
    """``set()`` is idempotent on key — overwrites prior value."""
    provider.set("OVERWRITE", "v1")
    provider.set("OVERWRITE", "v2")
    assert provider.get("OVERWRITE") == "v2"


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


def test_envfile_delete_removes_file(
    provider: EnvFileSecretsProvider, isolated_data_dir: Path
):
    provider.set("DELETE_ME", "value")
    path = isolated_data_dir / "secrets" / "DELETE_ME"
    assert path.exists()
    assert provider.delete("DELETE_ME") is True
    assert not path.exists()


def test_envfile_delete_returns_false_when_missing(
    provider: EnvFileSecretsProvider,
):
    assert provider.delete("NEVER_EXISTED") is False


# ---------------------------------------------------------------------------
# Key validation (defense against traversal / weird inputs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_key",
    [
        "",
        "../traversal",
        "a/b",
        "a\\b",
        "with space",
        "with.dot",
        "weird:colon",
        "newline\nkey",
    ],
)
def test_envfile_invalid_key_raises(
    provider: EnvFileSecretsProvider, bad_key: str
):
    """Path-traversal-ish or otherwise malformed keys are rejected on
    every operation — defense in depth, not a substitute for a sandbox.
    """
    with pytest.raises(ValueError):
        provider.get(bad_key)
    with pytest.raises(ValueError):
        provider.set(bad_key, "value")
    with pytest.raises(ValueError):
        provider.delete(bad_key)


# ---------------------------------------------------------------------------
# data_dir resolution
# ---------------------------------------------------------------------------


def test_envfile_explicit_data_dir(tmp_path: Path):
    """Explicit ``data_dir`` short-circuits any platformdirs lookup."""
    p = EnvFileSecretsProvider(data_dir=tmp_path)
    assert p.data_dir == tmp_path
    assert p.secrets_dir == tmp_path / "secrets"


def test_envfile_uses_platformdirs_when_available(tmp_path: Path):
    """When ``data_dir`` is omitted and platformdirs is importable, the
    provider routes secrets to ``platformdirs.user_data_dir('Tether')``.

    Patches the import so the test works regardless of whether
    platformdirs is actually installed.
    """
    fake_pd = mock.MagicMock()
    fake_pd.user_data_dir.return_value = str(tmp_path / "platform-Tether")
    with mock.patch.dict("sys.modules", {"platformdirs": fake_pd}):
        p = EnvFileSecretsProvider()
    assert p.data_dir == tmp_path / "platform-Tether"
    assert p.secrets_dir == tmp_path / "platform-Tether" / "secrets"
    fake_pd.user_data_dir.assert_called_once_with("Tether")


def test_envfile_falls_back_to_local_data_when_no_platformdirs():
    """When platformdirs is missing, default ``data_dir`` is ``./data``
    (matches the existing session_store dsn convention).
    """
    # Force ImportError by injecting a sentinel into sys.modules that
    # raises on attribute access — simpler: clear and have import fail.
    import builtins

    real_import = builtins.__import__

    def _fail_platformdirs(name, *args, **kwargs):
        if name == "platformdirs":
            raise ImportError("simulated missing platformdirs")
        return real_import(name, *args, **kwargs)

    with mock.patch.object(builtins, "__import__", side_effect=_fail_platformdirs):
        p = EnvFileSecretsProvider()
    assert p.data_dir == Path("data")


# ---------------------------------------------------------------------------
# Env-only path (no file ever created)
# ---------------------------------------------------------------------------


def test_envfile_env_only_get_does_not_touch_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """If the env var is set, ``get()`` does NOT lazy-create the secrets dir."""
    monkeypatch.setenv("ENV_ONLY_KEY", "from-env")
    p = EnvFileSecretsProvider(data_dir=tmp_path)
    assert p.get("ENV_ONLY_KEY") == "from-env"
    assert not (tmp_path / "secrets").exists()
