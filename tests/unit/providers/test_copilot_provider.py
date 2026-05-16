from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from tether.providers.copilot.provider import CopilotProvider
from tether.providers.types import ProviderText, ProviderThink


class _PermissionRequestResult:
    def __init__(self, *, kind: str) -> None:
        self.kind = kind


class _PermissionHandler:
    @staticmethod
    def approve_all(_request: Any, _invocation: Any) -> _PermissionRequestResult:
        return _PermissionRequestResult(kind="approved")


class _FakeSession:
    def __init__(self, events: list[Any], *, error: BaseException | None = None) -> None:
        self.events = events
        self.error = error
        self.handler: Callable[[Any], None] | None = None
        self.disconnected = False
        self.unsubscribed = False
        self.prompts: list[str] = []

    def on(self, handler: Callable[[Any], None]) -> Callable[[], None]:
        self.handler = handler

        def unsubscribe() -> None:
            self.unsubscribed = True

        return unsubscribe

    async def send_and_wait(self, prompt: str) -> None:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        assert self.handler is not None
        for event in self.events:
            self.handler(event)

    async def disconnect(self) -> None:
        self.disconnected = True


class _FakeClient:
    events: list[Any] = []
    error: BaseException | None = None
    instances: list["_FakeClient"] = []

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.started = False
        self.stopped = False
        self.session_kwargs: dict[str, Any] = {}
        self.session: _FakeSession | None = None
        self.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def create_session(self, **kwargs: Any) -> _FakeSession:
        self.session_kwargs = kwargs
        self.session = _FakeSession(self.events, error=self.error)
        return self.session


@pytest.fixture(autouse=True)
def fake_copilot_sdk(monkeypatch: pytest.MonkeyPatch):
    _FakeClient.events = []
    _FakeClient.error = None
    _FakeClient.instances = []

    copilot_mod = types.ModuleType("copilot")
    copilot_mod.CopilotClient = _FakeClient
    session_mod = types.ModuleType("copilot.session")
    session_mod.PermissionRequestResult = _PermissionRequestResult
    session_mod.PermissionHandler = _PermissionHandler

    monkeypatch.setitem(sys.modules, "copilot", copilot_mod)
    monkeypatch.setitem(sys.modules, "copilot.session", session_mod)
    yield


def _event(event_type: str, **data: Any) -> SimpleNamespace:
    return SimpleNamespace(type=SimpleNamespace(value=event_type), data=SimpleNamespace(**data))


async def _collect(async_iterable: Any) -> list[Any]:
    return [item async for item in async_iterable]


def _run(coro: Any) -> Any:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


def test_stream_yields_message_deltas_and_cleans_up() -> None:
    _FakeClient.events = [
        _event("assistant.message_delta", delta_content="hello "),
        _event("assistant.reasoning_delta", delta_content="hidden"),
        _event("assistant.message_delta", delta_content="world"),
    ]
    provider = CopilotProvider(model="gpt-5")

    chunks = _run(
        _collect(
            provider.stream(
                "gpt-5",
                [{"role": "user", "content": "Say hi"}],
            )
        )
    )

    assert chunks == ["hello ", "world"]
    client = _FakeClient.instances[-1]
    assert client.started is True
    assert client.stopped is True
    assert client.session is not None
    assert client.session.disconnected is True
    assert client.session.unsubscribed is True
    assert client.session.prompts == ["user: Say hi"]


def test_stream_typed_maps_text_and_reasoning() -> None:
    _FakeClient.events = [
        _event("assistant.reasoning_delta", delta_content="thinking"),
        _event("assistant.message_delta", delta_content="answer"),
    ]
    provider = CopilotProvider(model="gpt-5")

    events = _run(
        _collect(
            provider.stream_typed(
                model_name="gpt-5",
                messages=[{"role": "user", "content": "Question"}],
            )
        )
    )

    assert isinstance(events[0], ProviderThink)
    assert events[0].text == "thinking"
    assert isinstance(events[1], ProviderText)
    assert events[1].text == "answer"


def test_permission_handler_denies_tools_by_default() -> None:
    provider = CopilotProvider(model="gpt-5", enable_builtin_tools=False)

    _ = _run(
        _collect(
            provider.stream(
                "gpt-5",
                [{"role": "user", "content": "Hi"}],
            )
        )
    )

    client = _FakeClient.instances[-1]
    handler = client.session_kwargs["on_permission_request"]
    result = handler(object(), {})
    assert result.kind == "denied-by-rules"


def test_permission_handler_can_approve_when_enabled() -> None:
    provider = CopilotProvider(model="gpt-5", enable_builtin_tools=True)

    _ = _run(
        _collect(
            provider.stream(
                "gpt-5",
                [{"role": "user", "content": "Hi"}],
            )
        )
    )

    client = _FakeClient.instances[-1]
    handler = client.session_kwargs["on_permission_request"]
    result = handler(object(), {})
    assert result.kind == "approved"


def test_stream_errors_still_cleanup() -> None:
    _FakeClient.error = RuntimeError("boom")
    provider = CopilotProvider(model="gpt-5")

    with pytest.raises(RuntimeError, match="boom"):
        _ = _run(
            _collect(
                provider.stream(
                    "gpt-5",
                    [{"role": "user", "content": "Hi"}],
                )
            )
        )

    client = _FakeClient.instances[-1]
    assert client.stopped is True
    assert client.session is not None
    assert client.session.disconnected is True


def test_construct_without_importing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "copilot", raising=False)
    monkeypatch.delitem(sys.modules, "copilot.session", raising=False)

    provider = CopilotProvider(model="gpt-5", models=["claude-sonnet-4.5"])

    assert provider.kind == "copilot"
    assert provider.list_models() == ["gpt-5", "claude-sonnet-4.5"]
    assert provider.get_context_window("gpt-5") == 128_000
    assert provider.unload_model("gpt-5") is False


def test_dotted_path_factory_loads_without_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "copilot", raising=False)
    monkeypatch.delitem(sys.modules, "copilot.session", raising=False)

    from tether.core.factory import load

    provider = load(
        "tether.providers.copilot.provider.CopilotProvider",
        model="gpt-5",
    )

    assert isinstance(provider, CopilotProvider)
    assert provider.list_models() == ["gpt-5"]


def test_missing_sdk_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "copilot", raising=False)
    monkeypatch.delitem(sys.modules, "copilot.session", raising=False)
    provider = CopilotProvider(model="gpt-5")

    with pytest.raises(RuntimeError, match=r"pip install -e"):
        _ = _run(
            _collect(
                provider.stream(
                    "gpt-5",
                    [{"role": "user", "content": "Hi"}],
                )
            )
        )


def test_capabilities_are_safe_defaults() -> None:
    caps = CopilotProvider().capabilities

    assert caps.streaming is True
    assert caps.tools_native is False
    assert caps.tools_marker is False
    assert caps.thinking_channel is True
    assert caps.cancel_inflight is True
    assert caps.multi_model is True
    assert caps.warm_up_required is False

