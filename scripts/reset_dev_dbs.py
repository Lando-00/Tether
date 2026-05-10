#!/usr/bin/env python
"""Reset Tether dev DBs by backing up to data/.archive/ then deleting.

Phase 6 introduces schema changes that may render old dev data
inconsistently. Per user ratification (drop-and-rebuild OK), this
script archives any existing dev DBs and removes them so the next
Engine.from_settings run starts with a clean schema applied via yoyo.

Usage:
    python scripts/reset_dev_dbs.py

Safe to re-run: idempotent. Backups land in data/.archive/<name>.<UTC ISO>.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


# Known dev DB locations, relative to repo root.
KNOWN_DEV_DBS = [
    Path("data/tether.db"),
    Path("data/tether_dev.db"),
    Path("mlc_sessions.db"),
]


def reset_dev_dbs(repo_root: Path) -> int:
    archive_dir = repo_root / "data" / ".archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    failed = 0

    for rel_path in KNOWN_DEV_DBS:
        src = repo_root / rel_path
        if not src.exists():
            print(f"[skip] {rel_path}: not present")
            continue

        # Build a non-clobbering archive filename
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        archive_name = f"{src.name}.{ts}"
        archive_path = archive_dir / archive_name

        try:
            shutil.copy2(src, archive_path)
            src.unlink()
            print(f"[ok]   {rel_path} -> {archive_path.relative_to(repo_root)}")
            moved += 1
        except OSError as e:
            print(f"[fail] {rel_path}: {e}", file=sys.stderr)
            failed += 1

    print(f"\n{moved} archived + deleted, {failed} failed")
    return 0 if failed == 0 else 1


def main() -> int:
    # __file__ is scripts/reset_dev_dbs.py — repo root is its parent's parent
    repo_root = Path(__file__).resolve().parent.parent
    return reset_dev_dbs(repo_root)


if __name__ == "__main__":
    sys.exit(main())
