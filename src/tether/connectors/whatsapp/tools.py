"""Nine WhatsApp tools for Tether (Phase 2b · per ADR-0018 § 9-tool spec).

All tools inherit from BaseTool. The connector wires them in tools()
(src/tether/connectors/whatsapp/connector.py). Mocked-adapter tests
live alongside this module in tests/unit/connectors/whatsapp/.

Doctrine compliance (ADR-0015):
* `whatsapp_prepare_send` + `whatsapp_send_media` build DRAFTS only
  (the latter despite its name — see its docstring).
* `whatsapp_confirm_send` is the only tool that actually dispatches
  outbound traffic. It gates on ToolExecutionContext.user_confirmed_send
  (ADR-0019).
* `whatsapp_mark_platform_read` is a social-visible side effect; the
  rename from `whatsapp_send_read_receipt` (Reconcile R10) makes that
  explicit in the schema.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional

import structlog

from tether.connectors.types import ConnectorState, InboundEvent
from tether.connectors.whatsapp.adapter import Contact, WhatsAppClientAdapter
from tether.connectors.whatsapp.draft_store import DraftStore
from tether.context.inbox_store import InboundInbox
from tether.core.errors import ConnectorNotConfiguredError
from tether.core.types import ToolExecutionContext
from tether.tools.base import BaseTool

logger = structlog.get_logger("tether.whatsapp.tools")

StateProvider = Callable[[], ConnectorState]
TargetResolver = Callable[[str], Optional[str]]


def _require_ready(state_provider: StateProvider, label: str) -> None:
    """Raise ConnectorNotConfiguredError if connector isn't READY."""
    state = state_provider()
    if state is not ConnectorState.READY:
        raise ConnectorNotConfiguredError(
            f"WhatsApp tool {label!r} requires READY state (current: {state.value})"
        )


def _serialize_event(e: InboundEvent) -> dict:
    return {
        "event_id": e.event_id,
        "kind": e.kind,
        "received_at_iso": e.received_at.isoformat(),
        "summary": e.summary,
        "payload": e.payload,
    }


class WhatsAppPrepareSendTool(BaseTool):
    """Build a draft text message. Does NOT send."""

    _tether_tool_registered_name: ClassVar[str] = "whatsapp_prepare_send"

    def __init__(
        self,
        *,
        adapter: WhatsAppClientAdapter,
        draft_store: DraftStore,
        target_resolver_fn: TargetResolver,
        state_provider: StateProvider,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._draft_store = draft_store
        self._target_resolver_fn = target_resolver_fn
        self._state_provider = state_provider

    @property
    def schema(self) -> dict[str, Any]:
        return self.auto_schema

    async def run(self, to: str, text: str) -> dict:
        """Build a draft text message. Does NOT send. Use whatsapp_confirm_send(draft_id) to dispatch.

        Args:
            to: Display name, E.164 phone, or JID. Resolves via the connector's contact cache.
            text: Message body. UTF-8, no length limit (WhatsApp accepts ~65k characters).
        """
        _require_ready(self._state_provider, "whatsapp_prepare_send")
        if not text or not text.strip():
            return {"error": "text must be non-empty"}
        jid = self._target_resolver_fn(to)
        if jid is None:
            return {"error": f"could not resolve target {to!r} to a WhatsApp JID"}
        draft = self._draft_store.prepare(target_jid=jid, body=text)
        preview = text[:100] + ("…" if len(text) > 100 else "")
        return {
            "draft_id": draft.draft_id,
            "preview": preview,
            "resolved_target": jid,
            "expires_in_sec": int(self._draft_store._ttl_sec),
        }


class WhatsAppConfirmSendTool(BaseTool):
    """Send a previously-prepared WhatsApp draft after user confirmation."""

    _tether_tool_registered_name: ClassVar[str] = "whatsapp_confirm_send"

    def __init__(
        self,
        *,
        adapter: WhatsAppClientAdapter,
        draft_store: DraftStore,
        state_provider: StateProvider,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._draft_store = draft_store
        self._state_provider = state_provider

    @property
    def schema(self) -> dict[str, Any]:
        return self.auto_schema

    async def run(
        self,
        draft_id: str,
        *,
        context: Optional[ToolExecutionContext] = None,
    ) -> dict:
        """Send a previously-prepared draft. REQUIRES user confirmation.

        Refuses unless ToolExecutionContext.user_confirmed_send is True
        (set by RegexConfirmIntentClassifier per ADR-0019). The classifier
        runs on the user's last message; a "yes/send/go ahead" affirmation
        flips the flag. If the flag is False, the draft is preserved (not
        consumed) so the user can re-confirm.

        Args:
            draft_id: Identifier returned by whatsapp_prepare_send or whatsapp_send_media.
        """
        _require_ready(self._state_provider, "whatsapp_confirm_send")
        if context is None or not context.user_confirmed_send:
            return {
                "sent": False,
                "reason": (
                    "user_confirmed_send is False; draft requires explicit "
                    "user affirmation"
                ),
            }
        draft = self._draft_store.peek(draft_id)
        if draft is None:
            return {"sent": False, "reason": f"draft {draft_id!r} not found or expired"}
        try:
            if draft.media_path:
                p = Path(draft.media_path)
                if not p.exists():
                    return {
                        "sent": False,
                        "reason": f"media file not found: {draft.media_path}",
                    }
                body = p.read_bytes()
                result = await self._adapter.send_media(
                    to_jid=draft.target_jid,
                    body=body,
                    mime=draft.media_mime or "application/octet-stream",
                    filename=p.name,
                    caption=draft.media_caption,
                )
            else:
                result = await self._adapter.send_text(
                    to_jid=draft.target_jid,
                    text=draft.body,
                )
        except Exception as exc:
            return {
                "sent": False,
                "reason": (
                    f"adapter error: {exc}; draft preserved, "
                    "retry whatsapp_confirm_send"
                ),
            }
        self._draft_store.consume(draft_id)
        return {
            "sent": True,
            "message_id": result.message_id,
            "to_jid": result.to_jid,
        }


class WhatsAppListUnreadTool(BaseTool):
    """List unread WhatsApp inbound events from Tether's local inbox."""

    _tether_tool_registered_name: ClassVar[str] = "whatsapp_list_unread"

    def __init__(
        self,
        *,
        inbox: InboundInbox,
        state_provider: StateProvider,
    ) -> None:
        super().__init__()
        self._inbox = inbox
        self._state_provider = state_provider

    @property
    def schema(self) -> dict[str, Any]:
        return self.auto_schema

    async def run(self, limit: int = 20) -> dict:
        """List unread WhatsApp inbound events from Tether's local inbox.

        Args:
            limit: Maximum unread events to return. Must be in [1, 500].
        """
        _require_ready(self._state_provider, "whatsapp_list_unread")
        if limit < 1 or limit > 500:
            return {"error": f"limit must be in [1, 500], got {limit}"}
        events = await self._inbox.list_unread(connector_id="whatsapp", limit=limit)
        return {
            "events": [_serialize_event(e) for e in events],
            "count": len(events),
        }


class WhatsAppGetThreadTool(BaseTool):
    """Return recent INBOUND messages with the given peer.

    LIMITATIONS:
    - Returns only messages Tether has seen since first connect (no arbitrary
      history fetch — WhatsApp Web limitation).
    - Outgoing messages (sent by you) are NOT included; the inbox stores only
      inbound events.
    - Filters in Python from the newest N messages across all peers, then
      truncates to `limit`. If the peer is quiet relative to other chats,
      results may be sparse even when older messages exist.
    """

    _tether_tool_registered_name: ClassVar[str] = "whatsapp_get_thread"

    def __init__(
        self,
        *,
        inbox: InboundInbox,
        target_resolver_fn: TargetResolver,
        state_provider: StateProvider,
    ) -> None:
        super().__init__()
        self._inbox = inbox
        self._target_resolver_fn = target_resolver_fn
        self._state_provider = state_provider

    @property
    def schema(self) -> dict[str, Any]:
        return self.auto_schema

    async def run(self, peer: str, limit: int = 20) -> dict:
        """Return recent INBOUND messages with the given peer.

        Returns:
            {
              "peer_jid": "...",
              "events": [...],
              "count": N,
              "scanned": M,
              "scanned_window": N,
              "inbound_only": true,
              "note": "Returns inbound messages only; outgoing not stored. ...",
            }

        Args:
            peer: Display name, E.164 phone, or JID to resolve to a peer JID.
            limit: Maximum recent events for this peer. Must be in [1, 500].
        """
        _require_ready(self._state_provider, "whatsapp_get_thread")
        if limit < 1 or limit > 500:
            return {"error": f"limit must be in [1, 500], got {limit}"}
        peer_jid = self._target_resolver_fn(peer)
        if peer_jid is None:
            return {"error": f"could not resolve peer {peer!r} to a JID"}
        fetch_limit = min(limit * 10, 500)
        all_recent = await self._inbox.list_recent(
            connector_id="whatsapp",
            limit=fetch_limit,
        )
        filtered = [e for e in all_recent if e.payload.get("from") == peer_jid][:limit]
        note = (
            "Returns inbound messages only; outgoing messages are not stored. "
            f"Filtered from the newest {fetch_limit} messages across all peers."
        )
        return {
            "peer_jid": peer_jid,
            "events": [_serialize_event(e) for e in filtered],
            "count": len(filtered),
            "scanned": len(all_recent),
            "scanned_window": fetch_limit,
            "inbound_only": True,
            "note": note,
        }


class WhatsAppInboxMarkSeenTool(BaseTool):
    """Mark Tether-local WhatsApp inbox events as seen."""

    _tether_tool_registered_name: ClassVar[str] = "whatsapp_inbox_mark_seen"

    def __init__(
        self,
        *,
        inbox: InboundInbox,
        state_provider: StateProvider,
    ) -> None:
        super().__init__()
        self._inbox = inbox
        self._state_provider = state_provider

    @property
    def schema(self) -> dict[str, Any]:
        return self.auto_schema

    async def run(self, event_ids: list[str]) -> dict:
        """Mark Tether-local inbox events as seen. Does NOT change WhatsApp UI state.

        Max 900 event_ids per call (SqliteInbox limit). For larger sets, call repeatedly.

        Args:
            event_ids: Tether-local inbound event ids to mark seen.
        """
        _require_ready(self._state_provider, "whatsapp_inbox_mark_seen")
        if len(event_ids) > 900:
            return {"error": f"max 900 event_ids per call, got {len(event_ids)}"}
        affected = await self._inbox.mark_seen("whatsapp", event_ids)
        return {"affected": affected, "requested": len(event_ids)}


class WhatsAppMarkPlatformReadTool(BaseTool):
    """Mark messages read on WhatsApp itself; sends visible read receipts."""

    _tether_tool_registered_name: ClassVar[str] = "whatsapp_mark_platform_read"

    def __init__(
        self,
        *,
        adapter: WhatsAppClientAdapter,
        state_provider: StateProvider,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._state_provider = state_provider

    @property
    def schema(self) -> dict[str, Any]:
        return self.auto_schema

    async def run(self, jid: str, message_ids: list[str]) -> dict:
        """Mark messages read on the WhatsApp platform (sends read receipts).

        SIDE EFFECT: the other party sees blue checkmarks. Use only when the
        user explicitly wants to signal "I read this" — not as a side effect
        of inbox enumeration. Distinct from whatsapp_inbox_mark_seen which
        is Tether-local only.

        Args:
            jid: WhatsApp JID or E.164 phone number to normalize.
            message_ids: WhatsApp message ids to mark read.
        """
        from tether.connectors.whatsapp.target_resolver import normalize_whatsapp_target

        _require_ready(self._state_provider, "whatsapp_mark_platform_read")
        resolved_jid = normalize_whatsapp_target(jid) if jid else None
        if resolved_jid is None:
            return {"error": f"could not resolve jid {jid!r}"}
        if not message_ids:
            return {"error": "message_ids must be non-empty"}
        try:
            await self._adapter.send_read_receipt(
                jid=resolved_jid,
                message_ids=message_ids,
            )
        except Exception as exc:
            return {"error": f"adapter error: {exc}"}
        return {"marked": len(message_ids), "jid": resolved_jid}


class WhatsAppSendMediaTool(BaseTool):
    """Build a media DRAFT — does NOT send. Use whatsapp_confirm_send(draft_id) to dispatch."""

    _tether_tool_registered_name: ClassVar[str] = "whatsapp_send_media"

    def __init__(
        self,
        *,
        adapter: WhatsAppClientAdapter,
        draft_store: DraftStore,
        target_resolver_fn: TargetResolver,
        state_provider: StateProvider,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._draft_store = draft_store
        self._target_resolver_fn = target_resolver_fn
        self._state_provider = state_provider

    @property
    def schema(self) -> dict[str, Any]:
        return self.auto_schema

    async def run(
        self,
        to: str,
        file_path: str,
        mime: Optional[str] = None,
        caption: Optional[str] = None,
    ) -> dict:
        """Build a media DRAFT — does NOT send.

        Use whatsapp_confirm_send(draft_id) to dispatch the draft after the
        user has affirmed. The model MUST relay the draft to the user
        ("I drafted a video send to X. Confirm?") before calling
        whatsapp_confirm_send.

        Args:
            to: Display name / E.164 / JID — same resolver as whatsapp_prepare_send.
            file_path: Path to a local file. The file is read at confirm-time
                (not at prepare-time) — the draft only stores the path.
            mime: MIME type (e.g. "image/jpeg"). If None, guessed from the
                file extension via mimetypes.guess_type.
            caption: Optional caption text (for image/video/document; ignored for audio).
        """
        _require_ready(self._state_provider, "whatsapp_send_media")
        p = Path(file_path)
        if not p.is_absolute():
            return {"error": f"file_path must be absolute, got {file_path!r}"}
        if ".." in p.parts:
            return {"error": f"parent traversal not allowed: {file_path!r}"}
        if str(p).startswith("\\\\") or str(p).startswith("//"):
            return {"error": "UNC paths not allowed"}
        try:
            if p.is_symlink():
                return {"error": f"symlinks not allowed: {file_path!r}"}
        except OSError as exc:
            return {"error": f"cannot inspect file_path: {exc}"}
        try:
            p = p.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            return {"error": f"cannot resolve file_path: {exc}"}
        if p.is_symlink():
            return {"error": f"symlinks not allowed: {file_path!r}"}
        if str(p).startswith("\\\\") or str(p).startswith("//"):
            return {"error": "UNC paths not allowed"}
        if not p.is_file():
            return {"error": f"not a file: {file_path}"}
        size = p.stat().st_size
        if size > 64 * 1024 * 1024:
            return {"error": f"file too large: {size} bytes (max 64 MiB)"}
        if mime is None:
            guessed, _ = mimetypes.guess_type(p.name)
            mime = guessed or "application/octet-stream"
        jid = self._target_resolver_fn(to)
        if jid is None:
            return {"error": f"could not resolve target {to!r}"}
        draft = self._draft_store.prepare(
            target_jid=jid,
            body=caption or "",
            media_path=str(p),
            media_mime=mime,
            media_caption=caption,
        )
        preview = f"<{mime} from {p.name}, {size} bytes>"
        if caption:
            preview += f" caption={caption!r}"
        return {
            "draft_id": draft.draft_id,
            "preview": preview,
            "resolved_target": jid,
            "expires_in_sec": int(self._draft_store._ttl_sec),
        }


class WhatsAppGetContactsTool(BaseTool):
    """Search the connector's contacts cache by name or E.164 substring."""

    _tether_tool_registered_name: ClassVar[str] = "whatsapp_get_contacts"

    def __init__(
        self,
        *,
        adapter: WhatsAppClientAdapter,
        contacts_cache: list[Contact],
        state_provider: StateProvider,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._contacts_cache = contacts_cache
        self._state_provider = state_provider

    @property
    def schema(self) -> dict[str, Any]:
        return self.auto_schema

    async def run(self, query: str, limit: int = 20) -> dict:
        """Search the connector's contacts cache by name or E.164 substring.

        Args:
            query: Non-empty case-insensitive substring to match against name or E.164.
            limit: Max results (1-100).
        """
        _require_ready(self._state_provider, "whatsapp_get_contacts")
        if not query or not query.strip():
            return {"error": "query must be non-empty"}
        if limit < 1 or limit > 100:
            return {"error": f"limit must be in [1, 100], got {limit}"}
        q = query.strip().lower()
        matches = []
        for c in self._contacts_cache:
            name_match = c.name and q in c.name.lower()
            e164_match = c.e164 and q in c.e164.lower()
            if name_match or e164_match:
                matches.append(
                    {
                        "jid": c.jid,
                        "name": c.name,
                        "e164": c.e164,
                        "is_business": c.is_business,
                    }
                )
                if len(matches) >= limit:
                    break
        return {"contacts": matches, "count": len(matches), "query": query}


class WhatsAppResolveContactTool(BaseTool):
    """Resolve a display name, E.164, or JID to a canonical WhatsApp JID."""

    _tether_tool_registered_name: ClassVar[str] = "whatsapp_resolve_contact"

    def __init__(
        self,
        *,
        adapter: WhatsAppClientAdapter,
        contacts_cache: list[Contact],
        state_provider: StateProvider,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._contacts_cache = contacts_cache
        self._state_provider = state_provider

    @property
    def schema(self) -> dict[str, Any]:
        return self.auto_schema

    async def run(self, value: str) -> dict:
        """Resolve a display name, E.164, or JID to a canonical WhatsApp JID.

        Returns the canonical JID, or an error if ambiguous (multiple
        contacts match the name) or unresolvable.

        Args:
            value: Display name, E.164 phone number, or WhatsApp JID.
        """
        from tether.connectors.whatsapp.target_resolver import resolve_target

        _require_ready(self._state_provider, "whatsapp_resolve_contact")
        if not value or not value.strip():
            return {"error": "value must be non-empty"}
        jid = resolve_target(value, self._contacts_cache)
        if jid is None:
            q = value.strip().lower()
            if "@" not in q and not q.replace("+", "").replace(" ", "").isdigit():
                ambiguous = [
                    c for c in self._contacts_cache if c.name and q in c.name.lower()
                ]
                if len(ambiguous) > 1:
                    return {
                        "error": "ambiguous",
                        "candidates": [
                            {"jid": c.jid, "name": c.name, "e164": c.e164}
                            for c in ambiguous[:10]
                        ],
                    }
            return {"error": f"could not resolve {value!r}"}
        return {"jid": jid}


__all__ = [
    "WhatsAppPrepareSendTool",
    "WhatsAppConfirmSendTool",
    "WhatsAppListUnreadTool",
    "WhatsAppGetThreadTool",
    "WhatsAppInboxMarkSeenTool",
    "WhatsAppMarkPlatformReadTool",
    "WhatsAppSendMediaTool",
    "WhatsAppGetContactsTool",
    "WhatsAppResolveContactTool",
]
