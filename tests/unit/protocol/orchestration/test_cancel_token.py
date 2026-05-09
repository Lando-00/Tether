"""Tests for :class:`CancelToken` Protocol + :class:`AsyncEventCancelToken`
adapter (synthesis §11.3 R7).
"""
from __future__ import annotations

import asyncio

import pytest

from tether_service.protocol.orchestration.cancel import (
    AsyncEventCancelToken,
    CancelToken,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_protocol_runtime_checkable():
    """``AsyncEventCancelToken`` satisfies the ``CancelToken`` Protocol via
    ``isinstance``-style runtime check (the Protocol is decorated with
    ``@runtime_checkable``)."""
    token = AsyncEventCancelToken()
    assert isinstance(token, CancelToken)


def test_async_event_cancel_token_default():
    """Fresh adapter: not cancelled until ``set()`` is called."""
    token = AsyncEventCancelToken()
    assert token.cancelled() is False


def test_async_event_cancel_token_after_set():
    """After ``set()``, ``cancelled()`` returns True."""
    token = AsyncEventCancelToken()
    token.set()
    assert token.cancelled() is True


def test_async_event_cancel_token_wraps_external_event():
    """Caller-supplied ``asyncio.Event``: setting the event externally
    flips ``cancelled()`` to True (back-compat with ``Engine.stream``'s
    ``cancel_event`` parameter)."""
    event = asyncio.Event()
    token = AsyncEventCancelToken(event)
    assert token.cancelled() is False

    event.set()
    assert token.cancelled() is True


def test_async_event_cancel_token_event_property():
    """``.event`` exposes the underlying ``asyncio.Event`` for callers that
    need direct access (legacy ``cancel_event`` plumbing in
    ``chat.py``)."""
    event = asyncio.Event()
    token = AsyncEventCancelToken(event)
    assert token.event is event


def test_protocol_accepts_duck_typed_cancel_token():
    """Any object with ``cancelled() -> bool`` is a ``CancelToken``."""

    class CustomToken:
        def __init__(self):
            self._flag = False

        def cancelled(self) -> bool:
            return self._flag

    custom = CustomToken()
    assert isinstance(custom, CancelToken)
    custom._flag = True
    assert isinstance(custom, CancelToken)
    assert custom.cancelled() is True
