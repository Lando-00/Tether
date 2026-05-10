"""Tests for the typed StorageSettings + SqliteSettings sub-models.

Phase 6 step 60: dsn=None resolves via platformdirs.user_data_dir,
not the CWD-relative legacy default. Synthesis §3.6, §4.
"""
import pytest
from pathlib import Path

from tether.config.settings import (
    Settings,
    StorageSettings,
    SqliteSettings,
)


_MIN_PROVIDERS = {
    "model": {"impl": "x.M", "args": {}},
    "parser": {"impl": "x.P", "args": {}},
    "session_store": {"impl": "x.S", "args": {}},
}


def _min_settings() -> Settings:
    return Settings.model_validate({"providers": _MIN_PROVIDERS})


def test_default_settings_has_storage():
    """Settings has a default StorageSettings."""
    s = _min_settings()
    assert isinstance(s.storage, StorageSettings)
    assert isinstance(s.storage.sqlite, SqliteSettings)
    assert s.storage.sqlite.dsn is None


def test_resolved_dsn_with_none_uses_platformdirs():
    """When sqlite.dsn is None, resolved_dsn() returns a platformdirs-rooted path."""
    s = _min_settings()
    resolved = s.storage.resolved_dsn()

    assert resolved.startswith("sqlite:///")
    assert resolved.endswith("tether.db")
    # Must contain the 'Tether' app name from platformdirs
    assert "Tether" in resolved

    # Dir must exist (auto-created by resolved_dsn)
    db_path = resolved.replace("sqlite:///", "")
    db_dir = Path(db_path).parent
    assert db_dir.exists()


def test_resolved_dsn_with_explicit_value():
    """An explicit dsn is returned verbatim."""
    s = Settings.model_validate(
        {
            "providers": _MIN_PROVIDERS,
            "storage": {"sqlite": {"dsn": "sqlite:///explicit/path.db"}},
        }
    )
    assert s.storage.resolved_dsn() == "sqlite:///explicit/path.db"


def test_storage_settings_is_strict():
    """Extra fields rejected (StrictModel inherited)."""
    with pytest.raises(Exception):  # ValidationError
        StorageSettings(unknown_field="x")


def test_sqlite_settings_is_strict():
    """Extra fields rejected on SqliteSettings."""
    with pytest.raises(Exception):  # ValidationError
        SqliteSettings(unknown_field="x")


def test_resolved_dsn_no_doubled_tether_segment():
    """resolved_dsn must NOT include 'Tether/Tether' or 'Tether\\Tether'."""
    s = _min_settings()
    resolved = s.storage.resolved_dsn()
    assert "Tether/Tether" not in resolved, (
        f"resolved_dsn has doubled Tether segment (platformdirs appauthor): {resolved}"
    )
    assert "Tether\\Tether" not in resolved, (
        f"resolved_dsn has doubled Tether segment (Windows backslash): {resolved}"
    )


def test_resolved_dsn_idempotent():
    """Calling resolved_dsn twice returns the same string."""
    s = _min_settings()
    a = s.storage.resolved_dsn()
    b = s.storage.resolved_dsn()
    assert a == b


