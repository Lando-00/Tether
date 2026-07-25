from __future__ import annotations

import io
import json
from unittest.mock import Mock

import pytest
from rich.console import Console

from tether.cli import main as cli_main


class StreamResponse:
    def __init__(self) -> None:
        self.status_code = 200

    def __enter__(self) -> "StreamResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        events = [
            {"type": "message_start", "session_id": "s1", "turn_id": "t", "seq": 0},
            {"type": "text_delta", "text": "ok"},
            {"type": "message_stop", "stop_reason": "complete"},
        ]
        for event in events:
            yield json.dumps(event).encode("utf-8")


@pytest.fixture
def cli_harness(monkeypatch: pytest.MonkeyPatch) -> tuple[io.StringIO, Mock]:
    stream = io.StringIO()
    monkeypatch.setattr(
        cli_main,
        "console",
        Console(file=stream, force_terminal=False, width=120),
    )
    monkeypatch.setattr(cli_main, "select_model", lambda _model_name, _provider=None, **kw: ("model-a", None))
    monkeypatch.setattr(cli_main, "manage_sessions", lambda: ("s1", "resume"))
    monkeypatch.setattr(cli_main, "get_session_history", lambda _session_id: [])
    monkeypatch.setattr(cli_main, "display_history", lambda _history: None)
    post = Mock(return_value=StreamResponse())
    monkeypatch.setattr(cli_main.requests, "post", post)
    return stream, post


def test_chat_payload_includes_mode() -> None:
    assert cli_main._chat_payload(
        session_id="s1",
        prompt="hello",
        model_name="m",
        mode=cli_main.ChatMode.research,
    ) == {
        "session_id": "s1",
        "prompt": "hello",
        "model_name": "m",
        "mode": "research",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("chat", cli_main.ChatMode.chat), ("Research", cli_main.ChatMode.research)],
)
def test_parse_chat_mode(raw: str, expected: cli_main.ChatMode) -> None:
    assert cli_main._parse_chat_mode(raw) is expected


def test_parse_chat_mode_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="chat, research"):
        cli_main._parse_chat_mode("banana")


def test_chat_command_sends_startup_research_mode(
    cli_harness: tuple[io.StringIO, Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stream, post = cli_harness
    prompts = iter(["hello", "\\exit"])
    monkeypatch.setattr(cli_main, "ptk_prompt", lambda *args, **kwargs: next(prompts))

    cli_main.main(
        model_name="model-a",
        api_url="http://testserver/api/v1",
        debug=False,
        show_thinking=True,
        mode=cli_main.ChatMode.research,
    )

    assert post.call_args.kwargs["json"]["mode"] == "research"


def test_chat_command_interactive_mode_switch_to_research(
    cli_harness: tuple[io.StringIO, Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream, post = cli_harness
    prompts = iter(["\\research", "hello", "\\exit"])
    monkeypatch.setattr(cli_main, "ptk_prompt", lambda *args, **kwargs: next(prompts))

    cli_main.main(
        model_name="model-a",
        api_url="http://testserver/api/v1",
        debug=False,
        show_thinking=True,
        mode=cli_main.ChatMode.chat,
    )

    assert post.call_args.kwargs["json"]["mode"] == "research"
    assert "Mode switched to research" in stream.getvalue()


def test_chat_command_mode_toggle_back_to_chat(
    cli_harness: tuple[io.StringIO, Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stream, post = cli_harness
    prompts = iter(["\\mode", "hello", "\\exit"])
    monkeypatch.setattr(cli_main, "ptk_prompt", lambda *args, **kwargs: next(prompts))

    cli_main.main(
        model_name="model-a",
        api_url="http://testserver/api/v1",
        debug=False,
        show_thinking=True,
        mode=cli_main.ChatMode.research,
    )

    assert post.call_args.kwargs["json"]["mode"] == "chat"


def test_models_command_is_not_treated_as_mode_toggle(
    cli_harness: tuple[io.StringIO, Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stream, post = cli_harness
    prompts = iter(["\\models", "hello", "\\exit"])
    selected: list[object] = []

    def select_model(model_name, _provider=None, **kw):
        selected.append(model_name)
        return ("model-a", None)

    monkeypatch.setattr(cli_main, "select_model", select_model)
    monkeypatch.setattr(cli_main, "ptk_prompt", lambda *args, **kwargs: next(prompts))

    cli_main.main(
        model_name="model-a",
        api_url="http://testserver/api/v1",
        debug=False,
        show_thinking=True,
        mode=cli_main.ChatMode.chat,
    )

    # Initial startup uses the provided model; \models calls select_model(None).
    assert selected == ["model-a", None]
    assert post.call_args.kwargs["json"]["mode"] == "chat"


class ClarificationStreamResponse(StreamResponse):
    def iter_lines(self):
        events = [
            {"type": "message_start", "session_id": "s1", "turn_id": "t", "seq": 0},
            {
                "type": "notebook_clarification_requested",
                "reason": "ambiguous_correction",
                "message": "Which earlier term should this correction replace?",
                "candidates": ["Irelend"],
            },
            {"type": "message_stop", "stop_reason": "complete"},
        ]
        for event in events:
            yield json.dumps(event).encode("utf-8")


def test_research_clarification_is_rendered_to_the_user(
    cli_harness: tuple[io.StringIO, Mock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream, post = cli_harness
    post.return_value = ClarificationStreamResponse()
    prompts = iter(["Ireland*", "\\exit"])
    monkeypatch.setattr(cli_main, "ptk_prompt", lambda *args, **kwargs: next(prompts))

    cli_main.main(
        model_name="model-a",
        api_url="http://testserver/api/v1",
        debug=False,
        show_thinking=True,
        mode=cli_main.ChatMode.research,
    )

    output = stream.getvalue()
    assert "Which earlier term" in output
    assert "Irelend" in output
