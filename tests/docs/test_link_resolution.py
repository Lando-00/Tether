"""P0-J drift gate: every relative .md link must resolve.

Tribunal §3 P0-20 (A10-F1, A10-F2, B4-F1, B5-F2).

Scope: canonical user-facing docs (README, AGENTS, docs/). The
``.github/agents/`` orchestrator-template directory is intentionally
excluded — those files are operational templates that legitimately
reference per-session state and are consumed by tooling, not by humans
landing on the repo.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN: list[Path] = [REPO_ROOT / "README.md", REPO_ROOT / "AGENTS.md"]
SCAN += list((REPO_ROOT / "docs").rglob("*.md"))

# Matches Markdown links: [text](url) — capture url group.
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


@pytest.mark.docs
def test_all_relative_md_links_resolve():
    broken: list[tuple[str, str]] = []
    for md in SCAN:
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8")
        for m in LINK_RE.finditer(text):
            url = m.group(1).split("#", 1)[0]  # strip anchor
            if not url:
                continue  # pure anchor
            if url.startswith(("http://", "https://", "mailto:")):
                continue  # don't check external
            # Resolve relative to the source file's directory.
            target = (md.parent / url).resolve()
            if not target.exists():
                broken.append((str(md.relative_to(REPO_ROOT)), url))
    assert not broken, (
        "Broken relative .md links (Tribunal P0-20). "
        "Fix or remove each link:\n  - "
        + "\n  - ".join(f"{src}  ->  {tgt}" for src, tgt in broken)
    )


@pytest.mark.docs
def test_no_session_state_citations():
    """Repo .md files must not cite session-state paths as canonical."""
    offenders: list[str] = []
    for md in SCAN:
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8")
        for pat in ("files/investigations/", "session-state/", "~/.copilot/"):
            if pat in text:
                offenders.append(f"{md.relative_to(REPO_ROOT)} contains {pat!r}")
    # ALLOW mentions within the new digest (where it's intentional).
    offenders = [o for o in offenders if "synthesis-2026-05.md" not in o]
    assert not offenders, (
        "Session-state paths cited as canonical in repo docs (Tribunal P0-20). "
        "Move the content into docs/refactor/ instead:\n  - "
        + "\n  - ".join(offenders)
    )
