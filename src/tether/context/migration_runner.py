"""Schema migration runner backed by yoyo-migrations.

Phase 6 step 59: introduces yoyo as the schema-versioning tool. The
migrations directory at tether_service/context/migrations/ contains
the baseline (001_current_schema.sql) and any subsequent versions.
Engine.from_settings calls apply_pending_migrations before constructing
the store so the DB schema is always current. SqliteSessionStore.__init__
also calls it directly so that store instances created outside
Engine.from_settings (e.g., contract tests, CLI one-shots) still get
a valid schema.

Synthesis §3.6, B1 step 2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from tether.core.logging import logger

_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

# In-process cache of normalized DSNs already migrated in this process.
# Avoids redundant yoyo I/O when Engine.from_settings and SqliteSessionStore.__init__
# both call apply_pending_migrations for the same DB in the same process.
# Safe: schema only changes between process restarts; fresh processes start
# with an empty cache and always run yoyo on the first call.
_MIGRATED_DSNS: set[str] = set()


def apply_pending_migrations(dsn: str, *, migrations_dir: Optional[Path] = None) -> int:
    """Apply any pending yoyo migrations to the SQLite database at ``dsn``.

    ``dsn`` is a SQLite DSN, e.g. ``sqlite:///./data/tether.db`` (relative)
    or ``sqlite:///C:/abs/path/tether.db`` (absolute). Relative paths are
    resolved against the process cwd before being handed to yoyo, which
    requires an absolute path on Windows.

    Returns the number of migrations applied (0 when already up to date —
    idempotent). Safe to call on every Engine startup.

    Raises whatever yoyo raises on a corrupt migration or DB issue; the
    Engine treats a migration failure at startup as fatal.
    """
    # Lazy-import yoyo so that ``import tether`` does not pull it
    # into sys.modules. Library-first invariant (R8 lazy-import rule).
    from yoyo import get_backend, read_migrations  # noqa: PLC0415

    target = migrations_dir or _MIGRATIONS_DIR

    # Normalise the DSN to an absolute path for cross-platform compatibility.
    # yoyo on Windows requires an absolute path in the DSN.
    normalized_dsn = _normalize_dsn(dsn)

    # Fast path: already migrated in this process (idempotent no-op).
    if normalized_dsn in _MIGRATED_DSNS:
        logger.debug("DB schema migration skipped (already applied in this process)")
        return 0

    backend = get_backend(normalized_dsn)
    migrations = read_migrations(str(target))

    try:
        with backend.lock():
            pending = backend.to_apply(migrations)
            applied_count = len(pending)
            if pending:
                logger.info(
                    "Applying %d pending schema migration(s) from %s",
                    applied_count,
                    target,
                )
                backend.apply_migrations(pending)
            else:
                logger.debug("DB schema is up to date; no migrations to apply")
    finally:
        # Close the backend's SQLite connection immediately so it is not left
        # open until GC runs. Without explicit close, processes that call
        # apply_pending_migrations for many unique DSNs (e.g. test suites with
        # per-test tmp_path) accumulate open file handles that are all closed
        # during interpreter teardown, causing unpredictable cleanup delays.
        try:
            backend.connection.close()
        except Exception:
            pass  # best-effort; never mask a migration error

    _MIGRATED_DSNS.add(normalized_dsn)
    return applied_count


def _normalize_dsn(dsn: str) -> str:
    """Convert a sqlite:/// DSN (possibly relative) to an absolute-path DSN."""
    if not dsn.startswith("sqlite:///"):
        return dsn
    raw_path = dsn[len("sqlite:///"):]
    abs_path = Path(raw_path).expanduser().resolve()
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    # Use POSIX separators in the DSN (yoyo/SQLite accept forward slashes on
    # Windows; backslashes in a URI cause issues).
    return "sqlite:///" + abs_path.as_posix()


__all__ = ["apply_pending_migrations"]
