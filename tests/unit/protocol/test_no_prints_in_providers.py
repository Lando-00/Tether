"""Regression test: tether/providers/ must contain no bare print() calls.

Phase 7 step 70 replaced all 27 print() sites in providers/ with structured
structlog events. This test prevents regressions (new prints being added).

Citations: _synthesis.md §4 Phase 7 step 70 (B3 step 5-6, B5 step 4).
"""
from __future__ import annotations

import pathlib
import re


def _collect_print_sites(root: pathlib.Path) -> list[str]:
    """Return a list of 'path:lineno: code' strings for bare print() calls."""
    hits: list[str] = []
    for py_file in root.rglob("*.py"):
        try:
            lines = py_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for lineno, raw_line in enumerate(lines, start=1):
            # Match lines where print( is the first non-whitespace token.
            # This catches production prints but not string literals that
            # happen to contain the word "print(", nor test fixture helpers
            # that explicitly call print() as a deliberate side effect.
            if re.match(r"\s*print\(", raw_line):
                hits.append(f"{py_file}:{lineno}: {raw_line.strip()}")
    return hits


def test_no_bare_prints_in_providers() -> None:
    """All print() calls in tether/providers/ must have been replaced
    with structured structlog logger calls (Phase 7 step 70)."""
    providers_root = pathlib.Path("src") / "tether" / "providers"
    assert providers_root.is_dir(), (
        f"providers root not found at {providers_root.resolve()} — "
        "run pytest from the repo root"
    )

    hits = _collect_print_sites(providers_root)

    assert not hits, (
        f"Found {len(hits)} bare print() call(s) in tether/providers/ "
        f"— replace with structlog _log.* calls (Phase 7 step 70):\n"
        + "\n".join(f"  {h}" for h in hits)
    )
