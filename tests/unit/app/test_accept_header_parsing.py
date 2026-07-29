"""Unit tests for _has_version_0 / _has_version_1_0 Accept-header helpers.

The helpers detect whether the 'application/x-ndjson' media type carries
a 'version=0' or 'version=1.0' parameter for content negotiation.

Both helpers are boundary-aware (regex with negative lookahead) so that
'version=00', 'version=1.00', etc. do NOT false-positive match.

Synthesis §3.4 (HTTP transport routing); §11.3 R18 (split big-bang cutover).
p5-cutover-a-dual-emit (v1.0); boundary fix in p5-cutover-c-flip-default.
"""
from __future__ import annotations

from tether.app.http.routers.chat import _has_version_0, _has_version_1_0

# ---------------------------------------------------------------------------
# _has_version_1_0 — existing tests (must still pass after regex change)
# ---------------------------------------------------------------------------


def test_no_accept_returns_false():
    """Empty accept string -> False (default v2 path)."""
    assert _has_version_1_0("") is False


def test_empty_string_returns_false():
    """Sanity: empty string -> False."""
    assert _has_version_1_0("") is False


def test_v0_explicit_returns_false():
    """Bare application/x-ndjson without version param -> False."""
    assert _has_version_1_0("application/x-ndjson") is False


def test_v2_with_version_param():
    """application/x-ndjson; version=1.0 (spaced) -> True."""
    assert _has_version_1_0("application/x-ndjson; version=1.0") is True


def test_v2_with_quoted_version():
    """application/x-ndjson; version=\"1.0\" -> True (quoted form)."""
    assert _has_version_1_0('application/x-ndjson; version="1.0"') is True


def test_v2_no_space_before_param():
    """application/x-ndjson;version=1.0 (no space) -> True."""
    assert _has_version_1_0("application/x-ndjson;version=1.0") is True


def test_v2_with_other_params():
    """Other parameters alongside version=1.0 -> True."""
    assert _has_version_1_0("application/x-ndjson; q=0.5; version=1.0") is True


def test_v2_wrong_version():
    """version=2.0 does NOT opt in -> False."""
    assert _has_version_1_0("application/x-ndjson; version=2.0") is False


def test_v2_partial_version():
    """version=1 (no .0) does NOT match -> False."""
    assert _has_version_1_0("application/x-ndjson; version=1") is False


def test_uppercase_input_handled_by_caller():
    """_has_version_1_0 expects already-lowercased input (caller lowers).

    If the caller passes un-lowercased input, the match won't fire because
    we search for the literal lowercase token. This test documents that
    contract: callers MUST lowercase before passing in.
    """
    # Un-lowercased 'VERSION=1.0' should NOT match (case contract)
    assert _has_version_1_0("application/x-ndjson; VERSION=1.0") is False
    # Properly lowercased should match
    assert _has_version_1_0("application/x-ndjson; version=1.0") is True


# ---------------------------------------------------------------------------
# _has_version_1_0 — boundary tests (regression for P8 bug fix)
# ---------------------------------------------------------------------------


def test_v1_no_match_for_one_zero_zero():
    """version=1.00 must NOT match (trailing '0' after '1.0' triggers lookahead)."""
    assert _has_version_1_0("application/x-ndjson; version=1.00") is False


def test_v1_no_match_for_one_zero_alpha():
    """version=1.0a must NOT match (trailing 'a' is a word char)."""
    assert _has_version_1_0("application/x-ndjson; version=1.0a") is False


def test_v1_no_match_for_one_zero_dot():
    """version=1.0.1 must NOT match (trailing '.' triggers lookahead)."""
    assert _has_version_1_0("application/x-ndjson; version=1.0.1") is False


def test_v1_match_when_followed_by_semicolon():
    """version=1.0 followed by ';' (not a word char or dot) -> True."""
    assert _has_version_1_0("application/x-ndjson; version=1.0; q=0.9") is True


def test_v1_match_when_followed_by_space():
    """version=1.0 followed by ' ' -> True."""
    assert _has_version_1_0("application/x-ndjson; version=1.0 ") is True


# ---------------------------------------------------------------------------
# _has_version_0 — baseline true/false cases
# ---------------------------------------------------------------------------


def test_v0_bare_ndjson_no_match():
    """Bare application/x-ndjson (no version param) -> False."""
    assert _has_version_0("application/x-ndjson") is False


def test_v0_empty_string_no_match():
    """Empty string -> False."""
    assert _has_version_0("") is False


def test_v0_version_zero_match():
    """application/x-ndjson; version=0 -> True."""
    assert _has_version_0("application/x-ndjson; version=0") is True


def test_v0_quoted_version_zero_match():
    """application/x-ndjson; version=\"0\" (quoted) -> True."""
    assert _has_version_0('application/x-ndjson; version="0"') is True


def test_v0_no_space_before_param():
    """application/x-ndjson;version=0 (no space) -> True."""
    assert _has_version_0("application/x-ndjson;version=0") is True


def test_v0_with_other_params():
    """Other parameters alongside version=0 -> True."""
    assert _has_version_0("application/x-ndjson; q=0.5; version=0") is True


def test_v0_version_1_0_no_match():
    """version=1.0 does NOT match _has_version_0 -> False."""
    assert _has_version_0("application/x-ndjson; version=1.0") is False


# ---------------------------------------------------------------------------
# _has_version_0 — boundary tests (P8 bug fix)
# ---------------------------------------------------------------------------


def test_v0_no_match_for_double_zero():
    """version=00 must NOT match (trailing '0' is a word char)."""
    assert _has_version_0("application/x-ndjson; version=00") is False


def test_v0_no_match_for_zero_alpha():
    """version=0a must NOT match (trailing 'a' is a word char)."""
    assert _has_version_0("application/x-ndjson; version=0a") is False


def test_v0_no_match_for_zero_dot():
    """version=0.5 must NOT match (trailing '.' triggers lookahead)."""
    assert _has_version_0("application/x-ndjson; version=0.5") is False


def test_v0_match_when_followed_by_semicolon():
    """version=0 followed by ';' -> True."""
    assert _has_version_0("application/x-ndjson; version=0; q=0.9") is True


def test_v0_match_when_followed_by_space():
    """version=0 followed by ' ' -> True."""
    assert _has_version_0("application/x-ndjson; version=0 ") is True
