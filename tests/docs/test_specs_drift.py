"""Spec drift checks: generated artefacts must match shipped files.

Run with:
    pytest -m docs

CI gate (synthesis §4 Phase 8 step 92): if generators are updated but the
shipped specs are NOT regenerated, these tests fail loud. The fix is always:

    python -m scripts.docs.generate
    git add docs/specs/ && git commit -m 'regen specs'
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.docs.generate import (
    regenerate_events_schema,
    regenerate_openapi,
    regenerate_settings_schema,
    regenerate_tools_md,
)

SPECS = Path(__file__).resolve().parent.parent.parent / "docs" / "specs"


pytestmark = pytest.mark.docs


def _read(p: Path) -> str:
    """Read a spec file, normalising CRLF to LF.

    Git ``autocrlf=true`` (the Windows default) checks files out with
    CRLF in the working tree even though they're stored as LF in the
    index. The generator always emits LF (``newline="\\n"``), so we
    normalise on read to keep the comparison line-ending-agnostic.
    """
    return p.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_events_schema_no_drift() -> None:
    assert regenerate_events_schema() == _read(SPECS / "events.schema.json")


def test_settings_schema_no_drift() -> None:
    assert regenerate_settings_schema() == _read(SPECS / "settings.schema.json")


def test_openapi_no_drift() -> None:
    assert regenerate_openapi() == _read(SPECS / "openapi.json")


def test_tools_md_no_drift() -> None:
    assert regenerate_tools_md() == _read(SPECS / "tools.md")
