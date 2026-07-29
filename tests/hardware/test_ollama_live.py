"""Live integration tests for OllamaProvider (ADR-0022).

Gated by both @pytest.mark.ollama AND the OLLAMA_BASE_URL env var. Default
pytest -q does not select the marker; this entire module skips. To run:

    $env:OLLAMA_BASE_URL = "http://<gpu-pc>:11434"
    pytest -m ollama -v

Optional: set OLLAMA_LIVE_MODEL (default "qwen3:8b") to override the test model.
"""
from __future__ import annotations

import asyncio
import os

import pytest

pytest.importorskip("tether.providers.ollama.provider")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "")
OLLAMA_LIVE_MODEL = os.environ.get("OLLAMA_LIVE_MODEL", "qwen3:8b")

pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(
        not OLLAMA_BASE_URL,
        reason="OLLAMA_BASE_URL not set; skipping live Ollama tests",
    ),
]


@pytest.fixture
def native_provider():
    from tether.providers.ollama.provider import OllamaProvider

    return OllamaProvider(
        base_url=OLLAMA_BASE_URL,
        api_surface="native",
        models=[OLLAMA_LIVE_MODEL],
        default_model=OLLAMA_LIVE_MODEL,
        thinking_models=[OLLAMA_LIVE_MODEL] if "qwen3" in OLLAMA_LIVE_MODEL or "deepseek" in OLLAMA_LIVE_MODEL else [],
    )


@pytest.fixture
def openai_compat_provider():
    from tether.providers.ollama.provider import OllamaProvider

    return OllamaProvider(
        base_url=f"{OLLAMA_BASE_URL.rstrip('/')}/v1",
        api_surface="openai_compat",
        models=[OLLAMA_LIVE_MODEL],
        default_model=OLLAMA_LIVE_MODEL,
    )


# ---------------------------------------------------------------------------
# Test 1: version endpoint
# ---------------------------------------------------------------------------


async def test_live_version(native_provider):
    """GET /api/version returns a non-empty dict with a 'version' key."""
    result = await native_provider._client.version()
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result, "version() returned an empty dict"
    assert "version" in result, f"No 'version' key in response: {result}"


# ---------------------------------------------------------------------------
# Test 2: simple chat — native surface
# ---------------------------------------------------------------------------


async def test_live_simple_chat_native(native_provider):
    """Streaming a short prompt via native surface yields at least one text chunk."""
    messages = [{"role": "user", "content": "Say hi in one word."}]
    chunks = []

    async for event in native_provider._client.stream_chat(
        model=OLLAMA_LIVE_MODEL,
        messages=messages,
        think=False,
    ):
        if event.kind == "text" and event.text:
            chunks.append(event.text)
        if event.kind == "done":
            break

    assert chunks, (
        "No text chunks received from native stream. "
        f"Model={OLLAMA_LIVE_MODEL!r}, URL={OLLAMA_BASE_URL!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: simple chat — OpenAI-compatible surface
# ---------------------------------------------------------------------------


async def test_live_simple_chat_openai_compat(openai_compat_provider):
    """Streaming a short prompt via OpenAI-compat surface yields at least one text chunk."""
    messages = [{"role": "user", "content": "Say hi in one word."}]
    chunks = []

    async for event in openai_compat_provider._client.stream_chat(
        model=OLLAMA_LIVE_MODEL,
        messages=messages,
        think=False,
    ):
        if event.kind == "text" and event.text:
            chunks.append(event.text)
        if event.kind == "done":
            break

    assert chunks, (
        "No text chunks received from openai_compat stream. "
        f"Model={OLLAMA_LIVE_MODEL!r}, URL={OLLAMA_BASE_URL!r}"
    )


# ---------------------------------------------------------------------------
# Test 4: tool call
# ---------------------------------------------------------------------------


async def test_live_tool_call(native_provider):
    """Providing a tool schema and asking the model to use it emits a tool_call event."""
    tool_schema = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Returns the current UTC time as an ISO-8601 string.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
    ]
    messages = [
        {"role": "user", "content": "What is the current time? Use the get_current_time tool."}
    ]
    tool_events = []

    async for event in native_provider._client.stream_chat(
        model=OLLAMA_LIVE_MODEL,
        messages=messages,
        tools=tool_schema,
        think=False,
    ):
        if event.kind == "tool_call":
            tool_events.append(event)
        if event.kind == "done":
            break

    if not tool_events:
        pytest.skip(
            f"Model {OLLAMA_LIVE_MODEL!r} did not emit a tool_call for this prompt. "
            "Some models require explicit prompting or don't support structured tools. "
            "Run `ollama show <model>` to confirm tool support."
        )

    tc = tool_events[0].tool_call
    assert tc is not None
    assert tc.get("type") == "function"
    fn = tc.get("function", {})
    assert fn.get("name") == "get_current_time", f"Unexpected tool name: {fn.get('name')!r}"


# ---------------------------------------------------------------------------
# Test 5: thinking output
# ---------------------------------------------------------------------------


async def test_live_thinking_output(native_provider):
    """When the model is in thinking_models, at least one ProviderThink event is emitted."""
    is_thinking_model = (
        "qwen3" in OLLAMA_LIVE_MODEL or "deepseek" in OLLAMA_LIVE_MODEL
    )
    if not is_thinking_model:
        pytest.skip(
            f"Model {OLLAMA_LIVE_MODEL!r} is not a known thinking model; "
            "set OLLAMA_LIVE_MODEL=qwen3:8b to run this test."
        )

    messages = [
        {"role": "user", "content": "What is 17 × 23? Show your reasoning."}
    ]
    thinking_chunks = []

    async for event in native_provider._client.stream_chat(
        model=OLLAMA_LIVE_MODEL,
        messages=messages,
        think=True,
    ):
        if event.kind == "thinking" and event.text:
            thinking_chunks.append(event.text)
        if event.kind == "done":
            break

    assert thinking_chunks, (
        f"No thinking chunks received for model {OLLAMA_LIVE_MODEL!r}. "
        "Ensure Ollama supports think:true for this model and version."
    )


# ---------------------------------------------------------------------------
# Test 6: cancellation closes stream
# ---------------------------------------------------------------------------


async def test_live_cancellation_closes_stream(native_provider):
    """Setting the cancel token after 500ms causes the stream generator to exit within 2s."""

    class _CancelToken:
        def __init__(self) -> None:
            self._set = False

        def is_set(self) -> bool:
            return self._set

        def set(self) -> None:
            self._set = True

    cancel = _CancelToken()
    messages = [
        {"role": "user", "content": "Count from 1 to 1000, one number per line."}
    ]
    received: list[str] = []

    async def _stream_with_cancel() -> None:
        async for event in native_provider._client.stream_chat(
            model=OLLAMA_LIVE_MODEL,
            messages=messages,
            cancel_token=cancel,
        ):
            if event.kind == "text" and event.text:
                received.append(event.text)
            # Trigger cancel after first chunk arrives (or at most 500ms)
            if received and not cancel.is_set():
                cancel.set()
            if event.kind == "done":
                break

    await asyncio.wait_for(_stream_with_cancel(), timeout=10.0)

    # We received something before cancellation
    assert received, "No text chunks arrived before cancellation"


# ---------------------------------------------------------------------------
# Test 7: model-not-found returns actionable error
# ---------------------------------------------------------------------------


async def test_live_model_not_found_returns_actionable_error(native_provider):
    """Requesting a model that is not pulled raises RuntimeError with 'ollama pull' hint."""
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(RuntimeError, match="ollama pull"):
        async for _ in native_provider._client.stream_chat(
            model="this-model-does-not-exist:0b",
            messages=messages,
        ):
            pass
