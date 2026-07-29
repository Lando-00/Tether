"""Contract: every concrete provider tolerates the optional inference hints.

``Engine.chat`` forwards ``reasoning_effort`` to ``ModelProvider.stream`` when
the caller supplies one. HTTP validates the value against
``/models/details`` first, but direct library callers do not, so a provider
that does not advertise support must still ACCEPT and ignore the argument
rather than raising ``TypeError`` mid-stream.

Regression guard for the GenieX consolidation merge, where the trunk's
``GenieXProvider.stream`` briefly lost the parameter.
"""
from __future__ import annotations

import inspect

import pytest

from tether.providers.dummy.provider import DummyProvider
from tether.providers.geniex.provider import GenieXProvider
from tether.providers.nexa.provider import NexaProvider
from tether.providers.ollama.provider import OllamaProvider

_PROVIDER_CLASSES = [DummyProvider, GenieXProvider, NexaProvider, OllamaProvider]


@pytest.mark.parametrize(
    "provider_cls", _PROVIDER_CLASSES, ids=lambda c: c.__name__
)
def test_stream_accepts_reasoning_effort(provider_cls) -> None:
    params = inspect.signature(provider_cls.stream).parameters
    assert "reasoning_effort" in params, (
        f"{provider_cls.__name__}.stream must accept reasoning_effort "
        "(accept-and-ignore is fine) so Engine.chat can forward it"
    )
    assert params["reasoning_effort"].default is None


@pytest.mark.parametrize(
    "provider_cls", _PROVIDER_CLASSES, ids=lambda c: c.__name__
)
def test_stream_accepts_request_id(provider_cls) -> None:
    params = inspect.signature(provider_cls.stream).parameters
    assert "request_id" in params
    assert params["request_id"].default is None


@pytest.mark.anyio
async def test_geniex_stream_ignores_reasoning_effort() -> None:
    """GenieX advertises no reasoning support; the hint must not reach the wire."""
    import httpx

    sent: list[dict] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        sent.append(_json.loads(request.content))
        body = (
            'data:{"choices":[{"delta":{"content":"ok"}}]}\n\n'
            "data:[DONE]\n\n"
        )
        return httpx.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url="http://test"
    )
    provider = GenieXProvider(
        base_url="http://test", model_id="org/model:Q4_0", http_client=client
    )

    chunks = [
        chunk
        async for chunk in provider.stream(
            "org/model:Q4_0",
            [{"role": "user", "content": "hi"}],
            reasoning_effort="high",
        )
    ]

    assert chunks == ["ok"]
    assert sent, "provider did not issue a completion request"
    assert "reasoning_effort" not in sent[0]
    # Marker-only contract: native tool fields are never sent either.
    for forbidden in ("tools", "tool_choice", "functions"):
        assert forbidden not in sent[0]
    await client.aclose()


@pytest.fixture
def anyio_backend():
    return "asyncio"
