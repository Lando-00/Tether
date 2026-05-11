"""In-memory draft store for the WhatsApp prepare/confirm send flow.

A draft is a (draft_id, target_jid, body, created_at) tuple created by
``whatsapp_prepare_send`` and consumed by ``whatsapp_confirm_send`` only
after the user has explicitly affirmed (gated by
:class:`tether.protocol.intent.ConfirmIntentClassifier`).

Drafts expire after ``ttl_sec`` (default 300 = 5 min) — synthesis §D9 +
connector spec §4 footer. Expired drafts are deleted on access.
Stdlib-only; no third-party deps (R6 anti-over-engineering).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Final, Optional


@dataclass(frozen=True)
class Draft:
    draft_id: str
    target_jid: str
    body: str
    media_path: Optional[str] = None
    media_mime: Optional[str] = None
    media_caption: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)


class DraftStore:
    """In-memory TTL store. NOT thread-safe; caller serialises access.

    The connector is the sole owner; the orchestrator runs tools
    sequentially per turn so single-threaded access is the contract.
    """

    _DEFAULT_TTL_SEC: Final[float] = 300.0

    def __init__(self, *, ttl_sec: float = _DEFAULT_TTL_SEC) -> None:
        if ttl_sec <= 0:
            raise ValueError(f"ttl_sec must be > 0, got {ttl_sec}")
        self._ttl_sec = ttl_sec
        self._store: dict[str, Draft] = {}

    def prepare(
        self,
        *,
        target_jid: str,
        body: str,
        media_path: Optional[str] = None,
        media_mime: Optional[str] = None,
        media_caption: Optional[str] = None,
    ) -> Draft:
        """Create a new draft. Returns the Draft (with generated draft_id)."""
        self._sweep_expired()
        draft = Draft(
            draft_id=uuid.uuid4().hex,
            target_jid=target_jid,
            body=body,
            media_path=media_path,
            media_mime=media_mime,
            media_caption=media_caption,
        )
        self._store[draft.draft_id] = draft
        return draft

    def consume(self, draft_id: str) -> Optional[Draft]:
        """Atomically remove and return a draft. Returns None if not found or expired."""
        self._sweep_expired()
        return self._store.pop(draft_id, None)

    def peek(self, draft_id: str) -> Optional[Draft]:
        """Return without removing. None if expired/missing."""
        self._sweep_expired()
        return self._store.get(draft_id)

    def size(self) -> int:
        self._sweep_expired()
        return len(self._store)

    def _sweep_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, v in self._store.items() if now - v.created_at > self._ttl_sec]
        for k in expired:
            del self._store[k]


__all__ = ["Draft", "DraftStore"]
