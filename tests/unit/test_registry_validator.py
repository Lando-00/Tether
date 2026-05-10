"""Tests for ``tether.core.registry_validator.validate_unique_names``.

Synthesis §13.4 M5. The validator is a pure function shared by
:class:`tether.core.tool_registry.ToolRegistry` (Phase 4) and the
future connector registry (Phase 4.5).
"""
from __future__ import annotations

import pytest

from tether.core.registry_validator import validate_unique_names


def test_validate_unique_names_clean_passes():
    """Valid mapping with no constraints triggered → no exception."""
    validate_unique_names({"time": object(), "weather": object()})


def test_validate_unique_names_forbidden_raises():
    """A forbidden name in items → ValueError naming the offender."""
    with pytest.raises(ValueError) as exc_info:
        validate_unique_names(
            {"time": object(), "evil": object()},
            forbidden=["evil", "bad"],
        )
    msg = str(exc_info.value)
    assert "'evil'" in msg
    assert "forbidden" in msg.lower()


def test_validate_unique_names_require_prefix_passes():
    """All items have the required prefix → no error."""
    validate_unique_names(
        {"gmail_send": object(), "gmail_list": object()},
        require_prefix="gmail_",
    )


def test_validate_unique_names_require_prefix_fails():
    """One item missing prefix → ValueError naming offender + the prefix."""
    with pytest.raises(ValueError) as exc_info:
        validate_unique_names(
            {"gmail_send": object(), "send": object()},
            require_prefix="gmail_",
        )
    msg = str(exc_info.value)
    assert "'send'" in msg
    assert "'gmail_'" in msg


def test_validate_unique_names_combined_constraints():
    """Prefix + forbidden together; both kinds of violation surface
    (whichever comes first in iteration)."""
    # Missing prefix violates first.
    with pytest.raises(ValueError) as exc_info:
        validate_unique_names(
            {"foo": object()},
            require_prefix="x_",
            forbidden=["foo"],
        )
    assert "'foo'" in str(exc_info.value)

    # If prefix is satisfied but name is forbidden, forbidden fires.
    with pytest.raises(ValueError) as exc_info:
        validate_unique_names(
            {"x_evil": object(), "x_ok": object()},
            require_prefix="x_",
            forbidden=["x_evil"],
        )
    msg = str(exc_info.value)
    assert "'x_evil'" in msg
    assert "forbidden" in msg.lower()


def test_validate_unique_names_empty_mapping():
    """Empty mapping → no error regardless of constraints."""
    validate_unique_names({})
    validate_unique_names({}, require_prefix="x_", forbidden=["a"])
