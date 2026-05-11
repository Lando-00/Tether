"""Unit tests for WhatsApp draft TTL storage."""
from __future__ import annotations

import time

import pytest

from tether.connectors.whatsapp.draft_store import DraftStore


def test_prepare_returns_draft_with_uuid_id():
    store = DraftStore()

    first = store.prepare(target_jid="111@s.whatsapp.net", body="hello")
    second = store.prepare(target_jid="111@s.whatsapp.net", body="hello")

    assert first.draft_id
    assert second.draft_id
    assert first.draft_id != second.draft_id
    assert first.target_jid == "111@s.whatsapp.net"
    assert first.body == "hello"


def test_consume_returns_draft_and_removes():
    store = DraftStore()
    draft = store.prepare(target_jid="111@s.whatsapp.net", body="hello")

    consumed = store.consume(draft.draft_id)

    assert consumed == draft
    assert store.consume(draft.draft_id) is None


def test_peek_returns_draft_without_removing():
    store = DraftStore()
    draft = store.prepare(target_jid="111@s.whatsapp.net", body="hello")

    assert store.peek(draft.draft_id) == draft
    assert store.peek(draft.draft_id) == draft
    assert store.consume(draft.draft_id) == draft


def test_expired_draft_returns_none():
    store = DraftStore(ttl_sec=0.01)
    draft = store.prepare(target_jid="111@s.whatsapp.net", body="hello")

    time.sleep(0.05)

    assert store.consume(draft.draft_id) is None


def test_size_excludes_expired():
    store = DraftStore(ttl_sec=0.01)
    store.prepare(target_jid="111@s.whatsapp.net", body="hello")
    store.prepare(target_jid="222@s.whatsapp.net", body="hi")

    time.sleep(0.05)

    assert store.size() == 0


def test_invalid_ttl_raises():
    with pytest.raises(ValueError, match="ttl_sec must be > 0"):
        DraftStore(ttl_sec=0)
    with pytest.raises(ValueError, match="ttl_sec must be > 0"):
        DraftStore(ttl_sec=-1)


def test_media_fields_round_trip():
    store = DraftStore()
    draft = store.prepare(
        target_jid="111@s.whatsapp.net",
        body="hello",
        media_path="x.jpg",
        media_mime="image/jpeg",
        media_caption="hi",
    )

    consumed = store.consume(draft.draft_id)

    assert consumed is not None
    assert consumed.media_path == "x.jpg"
    assert consumed.media_mime == "image/jpeg"
    assert consumed.media_caption == "hi"
