"""Lifecycle tests for :class:`BraveSearchClient`.

p4-brave-client-lifecycle (synthesis §4 Phase 4 step 44; §6 row 17): the
client now exposes :meth:`aopen` / :meth:`aclose` so a single
``httpx.AsyncClient`` is reused across :meth:`search` calls instead of
spinning up a fresh TLS handshake per query (~150-300 ms cold-start).

Covers:

* :meth:`aopen` constructs the underlying ``httpx.AsyncClient`` exactly
  once and toggles ``_opened``.
* :meth:`aclose` is idempotent and resets ``_client`` / ``_opened``.
* :meth:`search` raises :class:`RuntimeError` when called without
  :meth:`aopen` — the contract that motivates the explicit lifecycle.
* Multiple :meth:`search` calls reuse the same ``_client`` instance
  (the actual perf claim — same ``id()`` across calls).
* The ``async with`` shortcut opens + closes correctly.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tether.tools.brave_client import BraveSearchClient


@pytest.mark.asyncio
async def test_aopen_creates_client():
    """``aopen`` constructs a real ``httpx.AsyncClient`` and flips
    ``_opened`` to True."""
    client = BraveSearchClient(api_key="k")
    assert client._client is None
    assert client._opened is False

    await client.aopen()
    try:
        assert client._opened is True
        assert isinstance(client._client, httpx.AsyncClient)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_aopen_idempotent():
    """Calling ``aopen`` twice does not allocate a second client (no
    leaked transport)."""
    client = BraveSearchClient(api_key="k")
    await client.aopen()
    try:
        first = client._client
        assert first is not None
        await client.aopen()
        assert client._client is first  # same instance, not a fresh one
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_aclose_closes_and_unsets():
    """``aclose`` calls the underlying ``aclose`` and resets state."""
    client = BraveSearchClient(api_key="k")
    await client.aopen()
    underlying = client._client
    assert underlying is not None

    await client.aclose()

    assert client._client is None
    assert client._opened is False
    # The underlying httpx.AsyncClient was actually closed.
    assert underlying.is_closed is True


@pytest.mark.asyncio
async def test_aclose_idempotent():
    """``aclose`` without a prior ``aopen`` (or after one) is a no-op."""
    client = BraveSearchClient(api_key="k")
    await client.aclose()  # never opened — must not raise

    await client.aopen()
    await client.aclose()
    await client.aclose()  # second close — must not raise


@pytest.mark.asyncio
async def test_search_before_aopen_raises():
    """:meth:`search` without :meth:`aopen` raises ``RuntimeError`` with
    a message that points the caller at the lifecycle API."""
    client = BraveSearchClient(api_key="k")

    with pytest.raises(RuntimeError) as exc:
        await client.search(q="test")

    msg = str(exc.value)
    assert "aopen" in msg


@pytest.mark.asyncio
async def test_search_uses_shared_client():
    """Multiple :meth:`search` calls reuse the same underlying
    ``httpx.AsyncClient`` (the cold-TLS fix — synthesis §6 row 17)."""
    client = BraveSearchClient(api_key="k", max_retries=0)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"web": {"results": []}}

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response

        await client.aopen()
        try:
            shared_id = id(client._client)
            assert client._client is not None

            await client.search(q="first")
            assert id(client._client) == shared_id

            await client.search(q="second")
            assert id(client._client) == shared_id

            await client.search(q="third")
            assert id(client._client) == shared_id

            # And the underlying ``get`` was called three times on the
            # same instance — no per-call client construction.
            assert mock_get.call_count == 3
        finally:
            await client.aclose()


@pytest.mark.asyncio
async def test_async_context_manager():
    """``async with BraveSearchClient(...)`` opens on entry and closes
    on exit."""
    async with BraveSearchClient(api_key="k") as client:
        assert client._opened is True
        assert isinstance(client._client, httpx.AsyncClient)
        underlying = client._client

    # After exit: closed.
    assert client._client is None
    assert client._opened is False
    assert underlying.is_closed is True


@pytest.mark.asyncio
async def test_async_context_manager_search_works():
    """End-to-end: ``async with`` enables :meth:`search` immediately."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"web": {"results": []}}

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response

        async with BraveSearchClient(api_key="k", max_retries=0) as client:
            result = await client.search(q="hello")
            assert result["meta"]["query"] == "hello"
            assert mock_get.call_count == 1
