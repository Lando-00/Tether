"""P0-B3 regression: CSRF token persists to a 0600 file atomically.

Tribunal §3 P1-10 (A4-F1, ADR-0012): the generated CSRF token used to
be ``print()``-ed to stderr only, so CLI clients couldn't bootstrap.
Now ``CSRFTokenMiddleware`` writes it to
``platformdirs.user_config_dir('Tether', appauthor=False)/csrf_token``
atomically at mode 0600.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def csrf_config_dir(monkeypatch, tmp_path):
    """Force ``platformdirs.user_config_dir`` to a tmp dir for isolation.

    The middleware lazy-imports ``platformdirs`` inside
    ``CSRFTokenSettings.resolved_token_file``; patching the attribute on
    the ``platformdirs`` module is sufficient to redirect the lookup.
    """
    import platformdirs

    monkeypatch.setattr(
        platformdirs,
        "user_config_dir",
        lambda *a, **kw: str(tmp_path),
    )
    return tmp_path


def _build_middleware():
    from tether.app.http.csrf_middleware import CSRFTokenMiddleware
    from tether.config.settings import CSRFTokenSettings

    cfg = CSRFTokenSettings(enabled=True)
    return CSRFTokenMiddleware(app=MagicMock(), settings=cfg), cfg


def test_csrf_token_written_to_file(csrf_config_dir):
    mw, _ = _build_middleware()

    token_file = csrf_config_dir / "csrf_token"
    assert token_file.exists(), f"Token file missing at {token_file}"

    on_disk = token_file.read_text(encoding="utf-8").strip()
    assert on_disk, "Token file is empty"
    assert len(on_disk) >= 16, f"Token suspiciously short: {on_disk!r}"
    # The on-disk token must equal the in-memory token the middleware
    # will compare against — otherwise the CLI bootstrap is broken.
    assert on_disk == mw._token


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX 0o600 check")
def test_csrf_token_file_mode_0600(csrf_config_dir):
    _build_middleware()
    token_file = csrf_config_dir / "csrf_token"
    mode = os.stat(token_file).st_mode & 0o777
    assert mode == 0o600, f"Token file mode is {oct(mode)}; want 0o600"


def test_atomic_write_no_tmp_residue(csrf_config_dir):
    _build_middleware()
    tmps = list(csrf_config_dir.glob(".csrf_token.*"))
    assert tmps == [], f"Residual atomic-write tmpfiles: {tmps}"


def test_resolved_token_file_default_uses_platformdirs(csrf_config_dir):
    """``CSRFTokenSettings.resolved_token_file()`` defaults to platformdirs."""
    from tether.config.settings import CSRFTokenSettings

    cfg = CSRFTokenSettings(enabled=True)
    resolved = cfg.resolved_token_file()
    assert resolved == csrf_config_dir / "csrf_token"


def test_resolved_token_file_honors_override(tmp_path):
    """Explicit ``token_file`` overrides the platformdirs default."""
    from tether.config.settings import CSRFTokenSettings

    custom = tmp_path / "custom" / "tok"
    cfg = CSRFTokenSettings(enabled=True, token_file=custom)
    assert cfg.resolved_token_file() == custom
