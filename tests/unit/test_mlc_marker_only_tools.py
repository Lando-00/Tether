"""Regression test for the ≥3-tool deadlock mitigation in MLCProvider.

Background
----------
With Tether's :class:`SlidingParser` detecting tool calls from the
``<<function_call>>`` marker in the text stream, MLC's native
structured-tool path (``tools=`` / ``tool_choice=`` / ``delta.tool_calls``)
is unused. But passing ``tools=`` to MLC's CodeLinaro build with the
Qwen3 ``use_function_calling`` conv template deadlocks at ≥3 tool
schemas — the async stream opens but yields zero chunks.

:class:`MLCProvider` has a ``marker_only_tools`` constructor flag that
suppresses ``tools=``/``tool_choice=`` at the engine boundary while
still letting the parser detect calls from text. This module locks in
both behaviours so future refactors can't silently regress.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

# MLCProvider imports mlc_llm at module scope. Those are the Qualcomm
# CodeLinaro Adreno wheels, installed out-of-band and only available on
# the Snapdragon target, so skip rather than fail collection elsewhere.
pytest.importorskip("mlc_llm")

from tether.providers.mlc.provider import MLCProvider


class _FakeChoice:
    def __init__(self, content: str) -> None:
        from types import SimpleNamespace

        self.delta = SimpleNamespace(content=content, tool_calls=None)
        self.finish_reason = None


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


async def _fake_stream(*, recorded: dict, **kwargs: Any):
    """Capture the kwargs MLC was called with; yield one canned chunk so
    the provider's ``async for`` advances."""
    recorded.update(kwargs)
    yield _FakeResponse("OK.")


class _FakeChatCompletions:
    def __init__(self, recorded: dict) -> None:
        self._recorded = recorded

    async def create(self, **kwargs: Any):
        recorded = self._recorded
        return _fake_stream(recorded=recorded, **kwargs)


class _FakeEngine:
    def __init__(self, recorded: dict) -> None:
        from types import SimpleNamespace

        self.chat = SimpleNamespace(completions=_FakeChatCompletions(recorded))

    def _abort(self, _rid):  # pragma: no cover — cancel path
        return None


def _patch_engine(monkeypatch, recorded: dict) -> None:
    """Replace ``_ensure_engine`` so ``MLCProvider.stream`` doesn't try
    to load a real MLC engine in this unit test."""

    async def fake_ensure(self, model_name: str):  # noqa: ARG001
        return _FakeEngine(recorded)

    monkeypatch.setattr(MLCProvider, "_ensure_engine", fake_ensure)


def _example_tools() -> List[Dict[str, Any]]:
    """Three tool schemas — the threshold at which the real deadlock fires.
    Shapes mirror what ``BaseTool.auto_schema`` emits."""
    return [
        {
            "type": "function",
            "function": {
                "name": "time",
                "description": "Get the current time.",
                "parameters": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get the weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Default mode (marker_only_tools=True) — the safe path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_suppresses_native_tools(tmp_path, monkeypatch):
    """When ``marker_only_tools`` defaults to True the provider must NOT
    forward ``tools=``/``tool_choice=`` to MLC, regardless of what the
    caller passed in — that's the whole point of the mitigation."""
    (tmp_path / "libs").mkdir()
    provider = MLCProvider(models_root=str(tmp_path), device="auto")
    assert provider.marker_only_tools is True

    recorded: dict = {}
    _patch_engine(monkeypatch, recorded)

    chunks = []
    async for chunk in provider.stream(
        model_name="Qwen3-4B-q4f16_1-MLC",
        messages=[{"role": "user", "content": "hi"}],
        tools=_example_tools(),
    ):
        chunks.append(chunk)

    assert chunks == ["OK."], "fake engine should have yielded its canned chunk"
    assert recorded["tools"] is None, "tools= must be suppressed at the MLC boundary"
    assert recorded["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_default_suppresses_even_with_no_tools(tmp_path, monkeypatch):
    """Same suppression must hold when the caller doesn't pass tools at all
    — ``tool_choice`` should still be ``"none"`` rather than leaking ``"auto"``."""
    (tmp_path / "libs").mkdir()
    provider = MLCProvider(models_root=str(tmp_path), device="auto")

    recorded: dict = {}
    _patch_engine(monkeypatch, recorded)

    async for _ in provider.stream(
        model_name="Qwen3-4B-q4f16_1-MLC",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
    ):
        pass

    assert recorded["tools"] is None
    assert recorded["tool_choice"] == "none"


# ---------------------------------------------------------------------------
# Diagnostic mode (marker_only_tools=False) — for benchmarking the native path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diagnostic_mode_forwards_native_tools(tmp_path, monkeypatch):
    """Flipping the flag to False restores the legacy native-tools path
    that lets operators measure / reproduce the upstream deadlock."""
    (tmp_path / "libs").mkdir()
    provider = MLCProvider(
        models_root=str(tmp_path), device="auto", marker_only_tools=False
    )
    assert provider.marker_only_tools is False

    recorded: dict = {}
    _patch_engine(monkeypatch, recorded)

    tools = _example_tools()
    async for _ in provider.stream(
        model_name="Qwen3-4B-q4f16_1-MLC",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
    ):
        pass

    assert recorded["tools"] == tools, "diagnostic mode must forward tools= verbatim"
    assert recorded["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_diagnostic_mode_tool_choice_none_when_tools_empty(
    tmp_path, monkeypatch
):
    """``tool_choice="auto"`` is only correct when at least one tool is
    being advertised. With no tools we expect ``"none"`` to mirror MLC's
    own contract."""
    (tmp_path / "libs").mkdir()
    provider = MLCProvider(
        models_root=str(tmp_path), device="auto", marker_only_tools=False
    )

    recorded: dict = {}
    _patch_engine(monkeypatch, recorded)

    async for _ in provider.stream(
        model_name="Qwen3-4B-q4f16_1-MLC",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
    ):
        pass

    assert recorded["tools"] is None
    assert recorded["tool_choice"] == "none"
