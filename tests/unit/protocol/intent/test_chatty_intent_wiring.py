from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest

from tests.golden.conftest import MinimalMemoryStore
from tether.core.interfaces import ModelProvider, Tool
from tether.core.types import OrchestratorConfig, ToolExecutionContext
from tether.protocol.intent.classifier import (
    ConfirmIntentClassifier,
    NullConfirmIntentClassifier,
)
from tether.protocol.intent.regex_classifier import RegexConfirmIntentClassifier
from tether.protocol.orchestration.chatty import ChattyAgentOrchestrator
from tether.protocol.orchestration.tool_runner import ToolRunner
from tether.protocol.parsers.sliding import SlidingParser


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _OneToolThenDoneProvider(ModelProvider):
    def __init__(self) -> None:
        self._calls = 0

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        self._calls += 1
        if self._calls == 1:
            yield (
                "Long enough preamble before the tool call to flush parser. "
                '<<function_call>> {"name": "recorder", "arguments": {}}'
            )
        else:
            yield "All done with the request, no further tool needed."

    def list_models(self) -> List[str]:
        return ["scripted"]

    def unload_model(self, model_name: str) -> bool:
        return True

    def get_context_window(self, model_name: str) -> int:
        return 4096


class _RecorderTool(Tool):
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "recorder"

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "recorder",
                "description": "record context",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    async def invoke(
        self,
        args: Dict[str, Any],
        *,
        context: Optional[ToolExecutionContext] = None,
    ) -> Any:
        self.calls.append({"args": args, "context": context})
        return {"ok": True}


class _AlwaysTrueClassifier(ConfirmIntentClassifier):
    def classify(self, last_user_message: str) -> bool:
        return True


def _config() -> OrchestratorConfig:
    return OrchestratorConfig(
        max_tool_loops=2,
        auto_reload_on_fatal_error=False,
        save_thinking=False,
        include_thinking_in_history=False,
    )


def _build_orch(
    *,
    tool: Optional[_RecorderTool] = None,
    confirm_intent_classifier: Optional[ConfirmIntentClassifier] = None,
) -> ChattyAgentOrchestrator:
    tools: Dict[str, Tool] = {}
    if tool is not None:
        tools[tool.name] = tool
    kwargs: Dict[str, Any] = {}
    if confirm_intent_classifier is not None:
        kwargs["confirm_intent_classifier"] = confirm_intent_classifier
    return ChattyAgentOrchestrator(
        provider=_OneToolThenDoneProvider(),
        parser=SlidingParser(),
        store=MinimalMemoryStore(),
        tools=tools,
        system_prompt="sys",
        config=_config(),
        tool_runner=ToolRunner(tools, timeout_sec=5),
        **kwargs,
    )


def test_chatty_accepts_confirm_intent_classifier_kwarg_and_stores_it() -> None:
    classifier = _AlwaysTrueClassifier()

    orch = _build_orch(confirm_intent_classifier=classifier)

    assert orch._confirm_intent_classifier is classifier


def test_chatty_default_confirm_intent_classifier_is_null() -> None:
    orch = _build_orch()

    assert isinstance(orch._confirm_intent_classifier, NullConfirmIntentClassifier)


@pytest.mark.anyio
async def test_chatty_regex_classifier_sets_user_confirmed_send_true() -> None:
    tool = _RecorderTool()
    orch = _build_orch(
        tool=tool,
        confirm_intent_classifier=RegexConfirmIntentClassifier(),
    )

    async for _ in orch.run(
        session_id="sid-intent",
        prompt="yes please",
        model_name="scripted",
    ):
        pass

    assert len(tool.calls) == 1
    context = tool.calls[0]["context"]
    assert isinstance(context, ToolExecutionContext)
    assert context.last_user_message == "yes please"
    assert context.user_confirmed_send is True
