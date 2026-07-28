"""Shared validation rules for public provider identifiers."""
from __future__ import annotations

import re

PROVIDER_ID_PATTERN = r"^[A-Za-z0-9._-]{1,64}$"
_PROVIDER_ID_RE = re.compile(PROVIDER_ID_PATTERN)

# ModelDetails uses this only for bare providers that bypass Engine wrapping.
# It must never be a configured provider ID because the CLI treats it as
# absent for compatibility with pre-registry servers.
PROVIDER_ID_SENTINEL = "_unwrapped_"


def is_valid_provider_id(value: str) -> bool:
    """Return whether *value* is a requestable, non-reserved provider ID."""
    return value != PROVIDER_ID_SENTINEL and bool(_PROVIDER_ID_RE.fullmatch(value))


__all__ = [
    "PROVIDER_ID_PATTERN",
    "PROVIDER_ID_SENTINEL",
    "is_valid_provider_id",
]
