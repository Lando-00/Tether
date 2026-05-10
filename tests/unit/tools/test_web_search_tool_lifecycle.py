"""Lifecycle + SecretsProvider integration tests for :class:`WebSearchTool`.

p4-brave-client-lifecycle (synthesis §4 Phase 4 step 41 + step 44; §6 row
17): :class:`WebSearchTool` now owns a long-lived
:class:`BraveSearchClient` opened in :meth:`startup` and closed in
:meth:`shutdown`. The legacy ``_get_client`` per-call helper is gone;
``BRAVE_API_KEY`` flows through a :class:`SecretsProvider` (default
:class:`EnvFileSecretsProvider` from connector spec §3.5).

Covers:

* :meth:`startup` constructs + opens a :class:`BraveSearchClient` when
  ``BRAVE_API_KEY`` is present.
* :meth:`startup` logs a warning + leaves ``_client`` ``None`` when the
  secret is missing — REQUIRED=False fallback so the tool stays in the
  registry.
* :meth:`run` returns ``{"error": ...}`` after a missing-key startup.
* :meth:`run` delegates to ``self._client.search(...)`` with the
  expected params when the client is initialised.
* :meth:`shutdown` closes the client when present and is a no-op when
  ``_client`` is ``None``.
* The :class:`SecretsProvider` injection point is honoured — a fake
  provider supplied via ``__init__`` is consulted instead of the
  default :class:`EnvFileSecretsProvider`.
"""
from __future__ import annotations

import logging
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from tether.core.secrets import SecretsProvider
from tether.tools.brave_client import BraveSearchClient
from tether.tools.web_search_tool import WebSearchInputs, WebSearchTool


# ---------------------------------------------------------------------------
# Fake SecretsProvider
# ---------------------------------------------------------------------------


class _FakeSecretsProvider(SecretsProvider):
    """In-memory :class:`SecretsProvider` for deterministic tests.

    ``EnvFileSecretsProvider`` reads from ``os.environ`` (and the
    on-disk ``data/secrets/`` fallback) which makes test isolation
    fragile under parallel runs. The fake provider closes that loop.
    """

    def __init__(self, store: Optional[dict] = None) -> None:
        self._store = dict(store or {})

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        self._store[key] = value

    def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None


# ---------------------------------------------------------------------------
# startup() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_with_secret_constructs_client():
    """When ``BRAVE_API_KEY`` is present, ``startup`` constructs and
    opens a :class:`BraveSearchClient`."""
    secrets = _FakeSecretsProvider({"BRAVE_API_KEY": "sekrit-token"})
    tool = WebSearchTool(secrets=secrets)

    assert tool._client is None  # not yet started

    await tool.startup()
    try:
        assert isinstance(tool._client, BraveSearchClient)
        assert tool._client.api_key == "sekrit-token"
        assert tool._client._opened is True
    finally:
        await tool.shutdown()


# ---------------------------------------------------------------------------
# startup() — missing-key fallback (REQUIRED=False policy)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_startup_missing_key_logs_warning(caplog):
    """When ``BRAVE_API_KEY`` is absent, ``startup`` logs a warning and
    leaves ``_client = None``. The tool MUST stay in the registry
    (``REQUIRED = False``) so ``/tools`` listings remain accurate."""
    secrets = _FakeSecretsProvider({})  # empty
    tool = WebSearchTool(secrets=secrets)

    with caplog.at_level(logging.WARNING, logger="tether.tools.web_search_tool"):
        await tool.startup()

    assert tool._client is None
    # Warning was emitted and mentions the key name.
    assert any(
        "BRAVE_API_KEY" in rec.message and rec.levelno == logging.WARNING
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_startup_does_not_raise_on_missing_key():
    """Missing key MUST NOT raise — :class:`BaseTool` defaults to
    ``REQUIRED = False`` and the engine startup gather expects optional
    failures to be silent (synthesis §4 Phase 4 step 41)."""
    secrets = _FakeSecretsProvider({})
    tool = WebSearchTool(secrets=secrets)

    # No exception escapes — the call returns normally.
    await tool.startup()
    assert tool._client is None


# ---------------------------------------------------------------------------
# run() — missing-key error dict
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_with_missing_key_returns_error():
    """After a missing-key ``startup``, ``run`` returns a structured
    error dict — the model gets a clear message rather than a crash."""
    secrets = _FakeSecretsProvider({})
    tool = WebSearchTool(secrets=secrets)
    await tool.startup()

    result = await tool.run(WebSearchInputs(query="anything"))

    assert isinstance(result, dict)
    assert "error" in result
    assert "BRAVE_API_KEY" in result["error"]


@pytest.mark.asyncio
async def test_invoke_with_missing_key_returns_error():
    """Same fallback via the public ``invoke`` API (covers the Style A
    Pydantic dispatch path)."""
    secrets = _FakeSecretsProvider({})
    tool = WebSearchTool(secrets=secrets)
    await tool.startup()

    result = await tool.invoke({"query": "anything"})

    assert "error" in result
    assert "BRAVE_API_KEY" in result["error"]


# ---------------------------------------------------------------------------
# run() — happy path delegates to shared client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_with_client_calls_search():
    """When ``_client`` is set, ``run`` delegates to
    ``self._client.search(...)`` with the validated inputs mapped to
    Brave's ``q``/``count``/``country``/``search_lang``/``freshness``
    params."""
    tool = WebSearchTool(secrets=_FakeSecretsProvider({}))

    mock_client = AsyncMock(spec=BraveSearchClient)
    mock_client.search.return_value = {
        "results": [],
        "meta": {"engine": "brave", "query": "test", "took_ms": 12},
    }
    tool._client = mock_client  # bypass startup for this targeted test

    inputs = WebSearchInputs(
        query="climate models",
        count=7,
        country="gb",
        search_lang="en",
        freshness="pw",
    )
    result = await tool.run(inputs)

    assert result == mock_client.search.return_value
    mock_client.search.assert_awaited_once_with(
        q="climate models",
        count=7,
        country="gb",
        search_lang="en",
        freshness="pw",
    )


# ---------------------------------------------------------------------------
# shutdown()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_closes_client():
    """When a client is present, ``shutdown`` calls
    ``client.aclose`` and unsets ``_client``."""
    tool = WebSearchTool(secrets=_FakeSecretsProvider({}))
    mock_client = AsyncMock(spec=BraveSearchClient)
    tool._client = mock_client

    await tool.shutdown()

    mock_client.aclose.assert_awaited_once()
    assert tool._client is None


@pytest.mark.asyncio
async def test_shutdown_no_client_no_error():
    """``shutdown`` is a no-op when ``_client`` was never set (e.g.,
    after a missing-key ``startup``). Must not raise."""
    tool = WebSearchTool(secrets=_FakeSecretsProvider({}))
    assert tool._client is None

    # Idempotent + no-op — must not raise.
    await tool.shutdown()
    await tool.shutdown()


@pytest.mark.asyncio
async def test_shutdown_after_real_startup_idempotent():
    """End-to-end lifecycle: startup → shutdown → shutdown is safe."""
    secrets = _FakeSecretsProvider({"BRAVE_API_KEY": "k"})
    tool = WebSearchTool(secrets=secrets)

    await tool.startup()
    await tool.shutdown()
    assert tool._client is None
    await tool.shutdown()  # idempotent


# ---------------------------------------------------------------------------
# SecretsProvider injection
# ---------------------------------------------------------------------------


def test_secrets_injection_default_is_envfile():
    """No-arg construction wires the default
    :class:`EnvFileSecretsProvider` per connector spec §3.5."""
    from tether.core.secrets import EnvFileSecretsProvider

    tool = WebSearchTool()
    assert isinstance(tool._secrets, EnvFileSecretsProvider)


def test_secrets_injection_uses_provided():
    """When a :class:`SecretsProvider` is supplied via ``__init__``,
    the tool uses it instead of the default."""
    fake = _FakeSecretsProvider({"BRAVE_API_KEY": "from-fake"})
    tool = WebSearchTool(secrets=fake)
    assert tool._secrets is fake


@pytest.mark.asyncio
async def test_secrets_injection_lookups_brave_api_key():
    """Verify the injected provider is the one consulted for
    ``BRAVE_API_KEY`` during ``startup``."""

    calls = []

    class _RecordingProvider(_FakeSecretsProvider):
        def get(self, key: str) -> Optional[str]:
            calls.append(key)
            return super().get(key)

    secrets = _RecordingProvider({"BRAVE_API_KEY": "looked-up"})
    tool = WebSearchTool(secrets=secrets)

    await tool.startup()
    try:
        assert calls == ["BRAVE_API_KEY"]
        assert tool._client is not None
        assert tool._client.api_key == "looked-up"
    finally:
        await tool.shutdown()
