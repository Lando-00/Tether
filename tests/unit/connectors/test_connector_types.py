"""Unit tests for ``tether_service.connectors.types``.

Covers ConnectorState enum, frozen dataclass invariants, and LoginPrompt
literal validation. Per connector spec §3.2 + §3.5; synthesis §4 Phase 4.5
step 47a.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from tether_service.connectors.types import (
    AuthStatus,
    ConnectorState,
    HealthStatus,
    InboundEvent,
    LoginContinueResult,
    LoginPrompt,
)


def test_connector_state_enum_values():
    """All six lifecycle states are present, with stable string values
    that the future HTTP routes (``/api/v1/connectors/<id>/health``) and
    the Phase 6.5 SqliteInbox can serialize directly.
    """
    expected = {
        "UNCONFIGURED": "unconfigured",
        "AUTHENTICATING": "authenticating",
        "READY": "ready",
        "DEGRADED": "degraded",
        "ERROR": "error",
        "LOGGED_OUT": "logged_out",
    }
    actual = {member.name: member.value for member in ConnectorState}
    assert actual == expected
    assert len(ConnectorState) == 6


def test_health_status_dataclass_frozen():
    h = HealthStatus(state=ConnectorState.READY, detail="ok")
    assert h.state is ConnectorState.READY
    assert h.detail == "ok"
    assert h.last_success is None
    with pytest.raises(FrozenInstanceError):
        h.detail = "mutated"  # type: ignore[misc]


def test_auth_status_dataclass_frozen():
    a = AuthStatus(state=ConnectorState.READY, user_id="alice@example.com")
    assert a.state is ConnectorState.READY
    assert a.user_id == "alice@example.com"
    assert a.expires_at is None
    with pytest.raises(FrozenInstanceError):
        a.user_id = "eve@example.com"  # type: ignore[misc]


def test_login_prompt_dataclass_frozen():
    p = LoginPrompt(kind="qr_code", payload="data:image/png;base64,...")
    assert p.kind == "qr_code"
    assert p.payload.startswith("data:image/png")
    assert p.extra == {}
    with pytest.raises(FrozenInstanceError):
        p.payload = "x"  # type: ignore[misc]


@pytest.mark.parametrize("kind", ["qr_code", "url", "password", "code"])
def test_login_prompt_kind_accepts_documented_values(kind):
    """Every value listed in the Literal[...] type hint constructs cleanly."""
    p = LoginPrompt(kind=kind, payload="x")  # type: ignore[arg-type]
    assert p.kind == kind


def test_login_prompt_kind_rejects_other_values():
    """Runtime validation enforces the Literal[...] type hint — type hints
    alone are not checked at construction time, so __post_init__ does it.
    """
    with pytest.raises(ValueError):
        LoginPrompt(kind="bogus", payload="x")  # type: ignore[arg-type]


def test_login_prompt_extra_default_is_independent_dict():
    """``extra`` uses ``field(default_factory=dict)`` so two prompts don't
    share the same empty dict (would be a bug if we'd written ``= {}``).
    """
    a = LoginPrompt(kind="url", payload="https://example.com/auth")
    b = LoginPrompt(kind="url", payload="https://example.com/auth")
    # Both default-constructed extras are equal but distinct objects.
    assert a.extra == b.extra == {}
    assert a.extra is not b.extra


def test_login_continue_result_dataclass_frozen():
    nested = LoginPrompt(kind="code", payload="enter the 6-digit code")
    r = LoginContinueResult(
        state=ConnectorState.AUTHENTICATING,
        detail="MFA required",
        next_prompt=nested,
    )
    assert r.state is ConnectorState.AUTHENTICATING
    assert r.next_prompt is nested
    with pytest.raises(FrozenInstanceError):
        r.detail = "x"  # type: ignore[misc]


def test_login_continue_result_default_next_prompt_is_none():
    r = LoginContinueResult(state=ConnectorState.READY)
    assert r.next_prompt is None
    assert r.detail is None


def test_inbound_event_dataclass_frozen():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    e = InboundEvent(
        event_id="evt-1",
        connector_id="echo",
        kind="message",
        received_at=now,
        payload={"text": "hi"},
        summary="hi",
    )
    assert e.event_id == "evt-1"
    assert e.payload == {"text": "hi"}
    assert e.summary == "hi"
    with pytest.raises(FrozenInstanceError):
        e.summary = "tampered"  # type: ignore[misc]


def test_inbound_event_payload_default_is_independent_dict():
    """``payload`` uses ``field(default_factory=dict)`` so default-constructed
    events do NOT share a mutable dict (defensive against the classic
    ``def f(x={}):`` pitfall — frozen=True does not protect against that).
    """
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    a = InboundEvent(
        event_id="a", connector_id="echo", kind="message", received_at=now
    )
    b = InboundEvent(
        event_id="b", connector_id="echo", kind="message", received_at=now
    )
    assert a.payload == b.payload == {}
    assert a.payload is not b.payload


def test_inbound_event_summary_optional():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    e = InboundEvent(
        event_id="evt", connector_id="echo", kind="message", received_at=now
    )
    assert e.summary is None
