import json

import pytest

from tests.fixtures.fake_research_provider import FakeResearchProvider
from tether.protocol.orchestration.notebook_prompts import (
    EXTRACTOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
)


def _messages(system_prompt: str | None) -> list[dict]:
    messages: list[dict] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": "test"})
    return messages


async def _collect(provider: FakeResearchProvider, messages: list[dict]) -> list[str]:
    return [
        chunk
        async for chunk in provider.stream(
            "fake-research-model",
            messages,
            tools=None,
            request_id="test-request",
        )
    ]


def test_detect_phase_planner() -> None:
    provider = FakeResearchProvider()

    assert provider._detect_phase(_messages(PLANNER_SYSTEM_PROMPT)) == "planner"


def test_detect_phase_extractor() -> None:
    provider = FakeResearchProvider()

    assert provider._detect_phase(_messages(EXTRACTOR_SYSTEM_PROMPT)) == "extractor"


def test_detect_phase_synthesizer() -> None:
    provider = FakeResearchProvider()

    assert provider._detect_phase(_messages(SYNTHESIZER_SYSTEM_PROMPT)) == "synthesizer"


def test_detect_phase_unknown_returns_unknown() -> None:
    provider = FakeResearchProvider()

    assert provider._detect_phase(_messages(None)) == "unknown"


@pytest.mark.anyio
async def test_stream_planner_emits_canned_dict() -> None:
    provider = FakeResearchProvider()
    expected = {"key_elements": ["q1"]}
    provider.set_planner_response(expected)

    chunks = await _collect(provider, _messages(PLANNER_SYSTEM_PROMPT))

    assert json.loads("".join(chunks)) == expected


@pytest.mark.anyio
async def test_stream_extractor_FIFO() -> None:
    provider = FakeResearchProvider()
    responses = [
        {"facts": [{"text": "f1"}], "follow_up_queries": []},
        {"facts": [{"text": "f2"}], "follow_up_queries": []},
        {"facts": [{"text": "f3"}], "follow_up_queries": []},
    ]
    provider.set_extractor_responses(responses)

    for expected in responses:
        chunks = await _collect(provider, _messages(EXTRACTOR_SYSTEM_PROMPT))
        assert json.loads("".join(chunks)) == expected

    assert await _collect(provider, _messages(EXTRACTOR_SYSTEM_PROMPT)) == []


@pytest.mark.anyio
async def test_stream_synthesizer_string_chunks() -> None:
    provider = FakeResearchProvider()
    provider.set_synthesizer_response(["Hello ", "world", "!"])

    chunks = await _collect(provider, _messages(SYNTHESIZER_SYSTEM_PROMPT))

    assert chunks == ["Hello ", "world", "!"]


@pytest.mark.anyio
async def test_stream_synthesizer_string_single() -> None:
    provider = FakeResearchProvider()
    provider.set_synthesizer_response("Hello world")

    chunks = await _collect(provider, _messages(SYNTHESIZER_SYSTEM_PROMPT))

    assert chunks == ["Hello world"]


@pytest.mark.anyio
async def test_chunk_size_splits_output() -> None:
    provider = FakeResearchProvider(chunk_size=4)
    expected = {"key_elements": ["x"]}
    provider.set_planner_response(expected)

    chunks = await _collect(provider, _messages(PLANNER_SYSTEM_PROMPT))

    assert all(len(chunk) <= 4 for chunk in chunks)
    assert json.loads("".join(chunks)) == expected


@pytest.mark.anyio
async def test_raise_on_planner_raises_inside_generator() -> None:
    provider = FakeResearchProvider()
    provider.raise_on_planner(ValueError("boom"))

    with pytest.raises(ValueError, match="boom"):
        async for _ in provider.stream("fake-research-model", _messages(PLANNER_SYSTEM_PROMPT)):
            pass

    assert await _collect(provider, _messages(PLANNER_SYSTEM_PROMPT)) == []


@pytest.mark.anyio
async def test_raise_on_extractor_clears_after_one_fire() -> None:
    provider = FakeResearchProvider()
    r1 = {"facts": [{"text": "f1"}], "follow_up_queries": []}
    r2 = {"facts": [{"text": "f2"}], "follow_up_queries": []}
    provider.set_extractor_responses([r1, r2])
    provider.raise_on_extractor(RuntimeError("x"))

    with pytest.raises(RuntimeError, match="x"):
        async for _ in provider.stream("fake-research-model", _messages(EXTRACTOR_SYSTEM_PROMPT)):
            pass

    assert json.loads("".join(await _collect(provider, _messages(EXTRACTOR_SYSTEM_PROMPT)))) == r1
    assert json.loads("".join(await _collect(provider, _messages(EXTRACTOR_SYSTEM_PROMPT)))) == r2


@pytest.mark.anyio
async def test_unknown_phase_yields_nothing() -> None:
    provider = FakeResearchProvider()

    assert await _collect(provider, _messages(None)) == []
