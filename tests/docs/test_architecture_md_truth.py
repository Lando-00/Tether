"""P0-H regression: architecture.md must not claim non-existent modules.

Tribunal §3 P0-18 (A9-F1).
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCH_MD = REPO_ROOT / "docs" / "architecture.md"


@pytest.mark.docs
def test_architecture_md_does_not_claim_runtime_spans_py():
    """architecture.md mentioning runtime/spans.py must be a deferred-status row, not a live claim."""
    text = ARCH_MD.read_text(encoding="utf-8")
    if "runtime/spans" in text or "async_span" in text:
        # If mentioned, it must be marked DEFERRED.
        for marker in ("DEFERRED", "deferred", "NOT IMPLEMENTED"):
            if marker in text:
                return
        raise AssertionError(
            "architecture.md mentions runtime/spans.py / async_span but does not mark it DEFERRED — "
            "the file does not exist on disk (Tribunal P0-18)."
        )


@pytest.mark.docs
def test_runtime_spans_py_absence_consistent_with_doc():
    """If architecture.md claims runtime/spans.py exists, the file must actually exist."""
    text = ARCH_MD.read_text(encoding="utf-8")
    spans_py = REPO_ROOT / "src" / "tether" / "runtime" / "spans.py"
    if spans_py.exists():
        # Existence is fine; no further check.
        return
    # Spans.py absent — doc must not claim it's implemented.
    text_l = text.lower()
    if "runtime/spans" in text_l or "async_span" in text_l:
        # Must be marked as deferred/not implemented.
        assert ("deferred" in text_l) or ("not implemented" in text_l), (
            "architecture.md mentions runtime/spans without marking it deferred/not implemented "
            "and the file does not exist on disk (Tribunal P0-18)."
        )
