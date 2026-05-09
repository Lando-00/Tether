"""Unit tests for _has_version_1_0 Accept-header helper in chat router.

The helper detects whether the 'application/x-ndjson' media type carries
a 'version=1.0' parameter, opting into v2 NDJSON vocab.

Synthesis §3.4 (HTTP transport routing); §11.3 R18 (split big-bang cutover).
p5-cutover-a-dual-emit.
"""
from __future__ import annotations

import pytest

from tether_service.app.http.routers.chat import _has_version_1_0


def test_no_accept_returns_false():
    """Empty accept string -> False (default v0 path)."""
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
