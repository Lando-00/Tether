"""Secrets retrieval interface + env→file fallback default impl.

Per connector spec §3.5 + §3.6. Env vars take precedence over the file
fallback at ``<data_dir>/secrets/<key>`` (newline-stripped, mode 0600 on
POSIX). Phase 6 may layer in OS keyring support; this PR ships env→file
only — no third-party deps.

``BRAVE_API_KEY``, future ``GMAIL_CLIENT_SECRET``, ``WHATSAPP_*`` etc. all
flow through this single interface. Connectors declare their required
secret keys and call :meth:`SecretsProvider.get` from ``start()``; if a
required secret is missing the connector reports
``ConnectorState.UNCONFIGURED`` and its tools raise
:class:`tether.core.errors.ConnectorNotConfiguredError`.

Citations:
    - Connector spec §3.5 (SecretsProvider get/set/delete contract).
    - Connector spec §3.6 (data/secrets/ directory layout).
    - Synthesis §4 Phase 4.5 step 47a.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

_VALID_KEY_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
)


def _validate_key(key: str) -> None:
    """Reject keys that aren't pure ``[A-Za-z0-9_-]+``.

    Defense in depth against path traversal: even if a caller passes
    ``"../../etc/passwd"`` we refuse rather than rely on Path normalization.
    """
    if not key or not all(c in _VALID_KEY_CHARS for c in key):
        raise ValueError(
            f"invalid secret key: {key!r} (must be non-empty and match [A-Za-z0-9_-]+)"
        )


class SecretsProvider(ABC):
    """Abstract secret retrieval.

    Three operations: ``get`` (read), ``set`` (write), ``delete`` (remove).
    Implementations decide where secrets live; the default impl
    (:class:`EnvFileSecretsProvider`) prefers env vars and falls back to
    per-key files under ``<data_dir>/secrets/``.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Return the secret value for ``key``, or ``None`` if not set."""

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Persist ``key=value``. Idempotent — overwrites any prior value."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove ``key``. Return ``True`` if it existed, ``False`` if not."""


class EnvFileSecretsProvider(SecretsProvider):
    """Env-first, file-fallback :class:`SecretsProvider`.

    - :meth:`get`: returns ``os.environ[key]`` if set; else reads
      ``<data_dir>/secrets/<key>`` (newline-stripped). Returns ``None`` if
      neither exists.
    - :meth:`set`: writes ``<data_dir>/secrets/<key>`` with mode ``0o600``
      on POSIX. The env var (if set) is NOT modified — env wins on the
      next ``get``, which is intentional: env-set secrets are
      developer/operator-controlled, not Tether-managed.
    - :meth:`delete`: removes the file (no-op on env-only keys; returns
      ``False`` if neither file nor env var exists / file is missing).

    Per connector spec §3.5 + §3.6.
    """

    def __init__(self, *, data_dir: Optional[Path] = None) -> None:
        if data_dir is None:
            # Optional: respect the user's platformdirs convention if it's
            # installed. Otherwise fall back to ./data (relative to cwd —
            # matches the existing ``providers.session_store.args.dsn``
            # default of ``sqlite:///./data/tether.db``).
            try:
                from platformdirs import user_data_dir  # type: ignore[import-not-found]

                data_dir = Path(user_data_dir("Tether"))
            except ImportError:
                data_dir = Path("data")
        self._data_dir = Path(data_dir)
        self._secrets_dir = self._data_dir / "secrets"

    @property
    def data_dir(self) -> Path:
        """Root data directory (parent of ``secrets/``). Exposed for tests
        and for the future ``ConnectorRegistry`` (``data/connectors/<id>/``
        layout)."""
        return self._data_dir

    @property
    def secrets_dir(self) -> Path:
        """``<data_dir>/secrets/`` directory."""
        return self._secrets_dir

    def _path(self, key: str) -> Path:
        _validate_key(key)
        return self._secrets_dir / key

    def get(self, key: str) -> Optional[str]:
        _validate_key(key)
        env_val = os.environ.get(key)
        if env_val is not None:
            return env_val
        path = self._path(key)
        if path.exists():
            return path.read_text(encoding="utf-8").rstrip("\n")
        return None

    def set(self, key: str, value: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n", encoding="utf-8")
        # POSIX: restrict to owner read/write. On Windows ``chmod`` is a
        # near-no-op (NTFS uses ACLs); the per-user data dir is already
        # private under typical configurations. We swallow OSError so a
        # cross-platform fallback path (e.g. Path("data") under a CI
        # working tree) doesn't fail.
        try:
            path.chmod(0o600)
        except OSError:  # pragma: no cover - platform-dependent
            pass

    def delete(self, key: str) -> bool:
        path = self._path(key)
        if path.exists():
            path.unlink()
            return True
        return False


__all__ = ["SecretsProvider", "EnvFileSecretsProvider"]
