"""Contract tests for the model-catalog convention.

Every provider must be able to enumerate the models it can serve through
:meth:`ModelProvider.list_model_info`, which is the single source of truth
behind ``GET /api/v1/models/details`` and the CLI model picker.

The contract (documented on the ABC) is:

1. return every model the provider can serve;
2. return ``[]`` when it has none — "no models" is a valid answer;
3. never raise — a provider that cannot enumerate degrades to ``[]``.

Rule 3 matters because :meth:`Engine.list_model_info` merges catalogs from
several providers: one provider blowing up must not blank out the others.
"""
from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest

from tether.core.interfaces import ModelProvider
from tether.engine import Engine
from tether.providers.dummy.provider import DummyProvider
from tether.providers.geniex.provider import GenieXProvider
from tether.providers.nexa.provider import NexaProvider
from tether.providers.ollama.provider import OllamaProvider
from tether.providers.types import ModelDetails


def _concrete_providers() -> list[tuple[str, ModelProvider]]:
    """One instance of every shipped provider that constructs without I/O.

    MLCProvider is covered separately: importing it requires the CodeLinaro
    wheels, which are absent on non-Adreno machines.
    """
    return [
        ("dummy", DummyProvider()),
        ("nexa", NexaProvider()),
        (
            "geniex",
            GenieXProvider(
                base_url="http://127.0.0.1:1",
                model_id="org/repo:Q4_0",
            ),
        ),
        (
            "ollama",
            OllamaProvider(
                base_url="http://127.0.0.1:1",
                models=["qwen3:8b"],
            ),
        ),
    ]


@pytest.mark.parametrize("name,provider", _concrete_providers(), ids=lambda v: v if isinstance(v, str) else "")
def test_catalog_never_raises_and_is_typed(name: str, provider: ModelProvider) -> None:
    """Rule 1 + 3: enumerating a provider yields typed rows, never an error."""
    catalog = provider.list_model_info()
    assert isinstance(catalog, list)
    assert all(isinstance(row, ModelDetails) for row in catalog)


@pytest.mark.parametrize("name,provider", _concrete_providers(), ids=lambda v: v if isinstance(v, str) else "")
def test_catalog_agrees_with_list_models(name: str, provider: ModelProvider) -> None:
    """The name-only projection must not disagree with the rich catalog."""
    catalog_ids = [row.id for row in provider.list_model_info()]
    try:
        names = list(provider.list_models())
    except NotImplementedError:
        # A stub may refuse the legacy call; the catalog must still be empty
        # rather than raising (asserted above).
        assert catalog_ids == []
        return
    assert catalog_ids == names


def test_stub_provider_reports_empty_catalog_not_an_error() -> None:
    """Rule 2: a provider with nothing to offer is enumerable, not fatal."""
    assert NexaProvider().list_model_info() == []


def test_catalog_rows_leave_provider_id_unstamped() -> None:
    """Providers do not know their registry key; the Engine stamps it."""
    for _, provider in _concrete_providers():
        for row in provider.list_model_info():
            assert row.provider_id == "_unwrapped_"


class _ExplodingProvider(ModelProvider):
    """Third-party provider that violates the never-raise rule."""

    kind = "exploding"  # type: ignore[assignment]

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        yield "x"

    def list_models(self) -> List[str]:
        raise RuntimeError("catalog backend is down")

    def unload_model(self, model_name: str) -> bool:
        return False

    def get_context_window(self, model_name: str) -> int:
        return 4096


def test_default_impl_absorbs_a_failing_list_models() -> None:
    """The ABC default honours the contract even when list_models() raises."""
    assert _ExplodingProvider().list_model_info() == []


class _BadContextWindowProvider(DummyProvider):
    """One unreadable model must not blank out the whole catalog."""

    def get_context_window(self, model_name: str) -> int:
        if model_name == "dummy-model-2":
            raise OSError("config file is corrupt")
        return 2048

    # Fall back to the shared implementation rather than Dummy's override.
    list_model_info = ModelProvider.list_model_info


def test_one_unreadable_model_does_not_blank_the_catalog() -> None:
    catalog = _BadContextWindowProvider().list_model_info()
    assert [row.id for row in catalog] == ["dummy-model-1", "dummy-model-2"]
    # The unreadable model still appears, with a conservative window.
    assert catalog[1].context_window == 4096


# ---------------------------------------------------------------------------
# Engine merge behaviour
# ---------------------------------------------------------------------------


def _engine_with(providers: dict[str, ModelProvider]) -> Engine:
    from unittest.mock import AsyncMock

    from tether.protocol.parsers.sliding import SlidingParser

    return Engine(
        providers=providers,
        default_provider_id=next(iter(providers)),
        parser=SlidingParser(),
        session_store=AsyncMock(),
        tools={},
        system_prompt="You are a helpful assistant.",
    )


def test_engine_stamps_provider_id_on_every_row() -> None:
    engine = _engine_with({"d1": DummyProvider()})
    assert {row.provider_id for row in engine.list_model_info()} == {"d1"}


def test_engine_isolates_a_misbehaving_provider() -> None:
    """A provider that violates the contract must not blank out the rest."""

    class _Rogue(_ExplodingProvider):
        # Bypass the ABC's safety net to simulate a provider that overrode
        # list_model_info and raises from it directly.
        def list_model_info(self) -> List[ModelDetails]:
            raise RuntimeError("catalog backend is down")

    engine = _engine_with({"good": DummyProvider(), "bad": _Rogue()})

    rows = engine.list_model_info()
    assert [row.provider_id for row in rows] == ["good", "good"]

    # And the failure is reported rather than silently swallowed.
    health = engine.list_provider_health()
    assert health["bad"]["error"] is not None
    assert "catalog backend is down" in health["bad"]["error"]
    assert health["good"]["error"] is None
