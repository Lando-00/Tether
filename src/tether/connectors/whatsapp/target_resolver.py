"""WhatsApp target resolution: display-name / E.164 / JID → JID.

Ports OpenClaw's ``normalizeWhatsAppTarget`` (extensions/whatsapp/src/
normalize-target.ts:676) to Python. Handles the five JID variants
(user @s.whatsapp.net, legacy @c.us, LID @lid, group @g.us, newsletter
@newsletter) plus bare E.164 phones. Display-name lookup matches
case-insensitive substring against the contacts list the connector
gives us (synthesis §D5).

Citations: wa-D-openclaw.md §4 + wa-D-synthesis.md §D5.
"""
from __future__ import annotations

import re
from typing import Final, Optional, Sequence

from tether.connectors.whatsapp.adapter import Contact

_WHATSAPP_USER_JID_RE: Final = re.compile(r"^(\d+)(?::\d+)?@s\.whatsapp\.net$", re.IGNORECASE)
_WHATSAPP_LEGACY_USER_JID_RE: Final = re.compile(r"^(\d+)@c\.us$", re.IGNORECASE)
_WHATSAPP_LID_RE: Final = re.compile(r"^(\d+)@lid$", re.IGNORECASE)
_WHATSAPP_NEWSLETTER_JID_RE: Final = re.compile(r"^([0-9]+)@newsletter$", re.IGNORECASE)
_WHATSAPP_GROUP_JID_RE: Final = re.compile(r"^(\d+(?:-\d+)*)@g\.us$", re.IGNORECASE)
_WHATSAPP_PREFIX_RE: Final = re.compile(r"^whatsapp:", re.IGNORECASE)
_NON_WA_PROVIDER_RE: Final = re.compile(r"^(?:telegram|sms|signal):", re.IGNORECASE)
_PHONE_PUNCTUATION_RE: Final = re.compile(r"[\s().-]+")


def is_whatsapp_user_jid(value: str) -> bool:
    return bool(
        _WHATSAPP_USER_JID_RE.match(value)
        or _WHATSAPP_LEGACY_USER_JID_RE.match(value)
        or _WHATSAPP_LID_RE.match(value)
    )


def is_whatsapp_group_jid(value: str) -> bool:
    return bool(_WHATSAPP_GROUP_JID_RE.match(value))


def is_whatsapp_newsletter_jid(value: str) -> bool:
    return bool(_WHATSAPP_NEWSLETTER_JID_RE.match(value))


def _strict_e164_candidate(phone: str) -> Optional[str]:
    s = _PHONE_PUNCTUATION_RE.sub("", phone.strip())
    digits = s[1:] if s.startswith("+") else s
    if not digits.isdigit() or not (7 <= len(digits) <= 15):
        return None
    return f"+{digits}"


def _normalize_e164(phone: str) -> Optional[str]:
    """Normalize a bare phone to E.164 or return None if invalid.

    Uses ``phonenumbers`` when available; falls back to a strict regex
    (digits + optional leading +; 7-15 digits per ITU-T E.164) when not.
    WhatsApp accepts possible E.164 numbers that public numbering metadata
    may mark unassigned, so ``is_possible_number`` is intentionally enough.
    """
    try:
        import phonenumbers
    except ImportError:
        return _strict_e164_candidate(phone)

    try:
        parsed = phonenumbers.parse(phone, None)
    except phonenumbers.NumberParseException:
        return _strict_e164_candidate(phone)

    if not phonenumbers.is_possible_number(parsed):
        return _strict_e164_candidate(phone)

    e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    return _strict_e164_candidate(e164)


def normalize_whatsapp_target(value: str) -> Optional[str]:
    """Convert any user input to a canonical WhatsApp JID (or None if invalid).

    Returns:
        - For groups: ``"<local>@g.us"``
        - For newsletters: ``"<digits>@newsletter"``
        - For user JIDs / legacy / LID / bare E.164: ``"<digits>@s.whatsapp.net"`` (device suffix stripped)
        - None for invalid inputs

    Mirrors OpenClaw's ``normalizeWhatsAppTarget`` semantics
    (extensions/whatsapp/src/normalize-target.ts:676).
    """
    if not value:
        return None

    s = value.strip()
    while _WHATSAPP_PREFIX_RE.match(s):
        s = _WHATSAPP_PREFIX_RE.sub("", s, count=1)

    if _NON_WA_PROVIDER_RE.match(s):
        return None

    if m := _WHATSAPP_GROUP_JID_RE.match(s):
        return f"{m.group(1)}@g.us"

    if m := _WHATSAPP_NEWSLETTER_JID_RE.match(s):
        return f"{m.group(1)}@newsletter"

    for pattern in (_WHATSAPP_USER_JID_RE, _WHATSAPP_LEGACY_USER_JID_RE, _WHATSAPP_LID_RE):
        if m := pattern.match(s):
            digits = m.group(1)
            return f"{digits}@s.whatsapp.net"

    if "@" in s:
        return None

    e164 = _normalize_e164(s)
    if e164 is None:
        return None
    digits = e164.lstrip("+")
    return f"{digits}@s.whatsapp.net"


def _looks_phone_like(value: str) -> bool:
    return _PHONE_PUNCTUATION_RE.sub("", value).replace("+", "", 1).isdigit()


def resolve_target(value: str, contacts: Sequence[Contact]) -> Optional[str]:
    """Resolve a user-supplied target string to a WhatsApp JID.

    Resolution order (synthesis §D5):
    1. Display-name lookup: case-insensitive substring match against
       contact names. If exactly one match, return that contact's JID.
       If multiple matches, return None (ambiguous — caller should
       prompt user to disambiguate).
    2. Fall through to ``normalize_whatsapp_target(value)``.
    """
    if not value:
        return None

    q = value.strip().lower()
    if not q:
        return None

    if "@" in q or _looks_phone_like(q):
        return normalize_whatsapp_target(value)

    matches = [c for c in contacts if c.name and q in c.name.lower()]
    if len(matches) == 1:
        return matches[0].jid
    if len(matches) > 1:
        return None
    return normalize_whatsapp_target(value)


__all__ = [
    "is_whatsapp_user_jid",
    "is_whatsapp_group_jid",
    "is_whatsapp_newsletter_jid",
    "normalize_whatsapp_target",
    "resolve_target",
]
