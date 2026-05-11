"""Unit tests for WhatsApp target resolution."""
from __future__ import annotations

import pytest

from tether.connectors.whatsapp.adapter import Contact
from tether.connectors.whatsapp.target_resolver import (
    is_whatsapp_group_jid,
    is_whatsapp_newsletter_jid,
    is_whatsapp_user_jid,
    normalize_whatsapp_target,
    resolve_target,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("41796666864@s.whatsapp.net", "41796666864@s.whatsapp.net"),
        ("41796666864:0@s.whatsapp.net", "41796666864@s.whatsapp.net"),
        ("+41 79 666 6864", "41796666864@s.whatsapp.net"),
        ("41796666864", "41796666864@s.whatsapp.net"),
        ("41796666864@c.us", "41796666864@s.whatsapp.net"),
        ("41796666864@lid", "41796666864@s.whatsapp.net"),
        ("123-456@g.us", "123-456@g.us"),
        ("123@g.us", "123@g.us"),
        ("999@newsletter", "999@newsletter"),
        ("whatsapp:+1555", None),
        ("whatsapp:whatsapp:+15551234567", "15551234567@s.whatsapp.net"),
        ("telegram:+1555", None),
        ("sms:+15551234567", None),
        ("abc@s.whatsapp.net", None),
        ("@g.us", None),
        ("", None),
    ],
)
def test_normalize_whatsapp_target_corner_cases(value: str, expected: str | None):
    assert normalize_whatsapp_target(value) == expected


def test_resolve_target_single_name_match():
    contacts = [
        Contact(jid="111@s.whatsapp.net", name="John Doe", e164="+111", is_business=False),
        Contact(jid="222@s.whatsapp.net", name="Jane", e164="+222", is_business=False),
    ]

    assert resolve_target("john", contacts) == "111@s.whatsapp.net"


def test_resolve_target_ambiguous_name_match_returns_none():
    contacts = [
        Contact(jid="111@s.whatsapp.net", name="John Doe", e164="+111", is_business=False),
        Contact(jid="222@s.whatsapp.net", name="Johnny", e164="+222", is_business=False),
    ]

    assert resolve_target("john", contacts) is None


def test_resolve_target_no_match_returns_none():
    assert resolve_target("john", []) is None


def test_resolve_target_phone_bypasses_lookup():
    contacts = [Contact(jid="111@s.whatsapp.net", name="+15551234567", e164="+111", is_business=False)]

    assert resolve_target("+15551234567", contacts) == "15551234567@s.whatsapp.net"


def test_resolve_target_jid_bypasses_lookup():
    contacts = [Contact(jid="111@s.whatsapp.net", name="999@s.whatsapp.net", e164="+111", is_business=False)]

    assert resolve_target("999@s.whatsapp.net", contacts) == "999@s.whatsapp.net"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("123@s.whatsapp.net", True),
        ("123:0@s.whatsapp.net", True),
        ("123@c.us", True),
        ("123@lid", True),
        ("123@g.us", False),
        ("abc@s.whatsapp.net", False),
    ],
)
def test_is_whatsapp_user_jid(value: str, expected: bool):
    assert is_whatsapp_user_jid(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("123@g.us", True),
        ("123-456@g.us", True),
        ("123@s.whatsapp.net", False),
        ("@g.us", False),
    ],
)
def test_is_whatsapp_group_jid(value: str, expected: bool):
    assert is_whatsapp_group_jid(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("999@newsletter", True),
        ("123@newsletter", True),
        ("abc@newsletter", False),
        ("999@g.us", False),
    ],
)
def test_is_whatsapp_newsletter_jid(value: str, expected: bool):
    assert is_whatsapp_newsletter_jid(value) is expected
