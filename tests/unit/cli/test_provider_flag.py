"""Tests for the --provider / -P flag and \\providers slash command.

ADR-0021 Phase 2.C — CLI disambiguation.
"""

from __future__ import annotations

import io
from unittest.mock import Mock

import pytest
import requests as _requests
from rich.console import Console
from rich.table import Table
from typer.testing import CliRunner

from tether.cli import main as cli_main

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class MockResponse:
    """Minimal requests.Response stand-in that also works as a stream context manager."""

    def __init__(self, body, status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code
        self.text = str(body)
        self.url = "http://testserver"

    def json(self):
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            exc = _requests.HTTPError(f"{self.status_code} Error")
            exc.response = self  # type: ignore[attr-defined]
            raise exc

    # --- context-manager + streaming support (for /chat/stream) ---
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_lines(self):
        return iter([])


def _model_detail(
    model_id: str,
    provider_id: str,
    *,
    is_default: bool = False,
    source: str = "remote",
    provider_kind: str = "copilot",
) -> dict:
    return {
        "id": model_id,
        "provider_id": provider_id,
        "provider_kind": provider_kind,
        "source": source,
        "context_window": 8192,
        "supports_thinking": False,
        "supports_reasoning_effort": False,
        "reasoning_efforts": [],
        "is_default": is_default,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def quiet_console(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Replace cli_main.console with a non-interactive stream console."""
    stream = io.StringIO()
    monkeypatch.setattr(
        cli_main,
        "console",
        Console(file=stream, force_terminal=False, width=120),
    )
    monkeypatch.setattr(cli_main, "API_BASE_URL", "http://127.0.0.1:8080/api/v1")
    return stream


# ---------------------------------------------------------------------------
# test_provider_flag_passed_to_post_body
# ---------------------------------------------------------------------------


def test_provider_flag_passed_to_post_body(
    monkeypatch: pytest.MonkeyPatch,
    quiet_console: io.StringIO,
) -> None:
    """When --provider is given, provider_id MUST appear in the POST body.

    Tests the _chat_payload helper directly rather than the full main()
    loop (whose session management differs across branches).
    """
    from tether.cli.main import ChatMode

    body = cli_main._chat_payload(
        session_id="sess-abc",
        prompt="hello",
        model_name="gpt-5",
        mode=ChatMode.chat,
        provider_id="copilot-gpt5",
    )
    assert body.get("provider_id") == "copilot-gpt5", f"provider_id missing or wrong in payload: {body}"


# ---------------------------------------------------------------------------
# test_no_provider_flag_omits_field_from_body
# ---------------------------------------------------------------------------


def test_no_provider_flag_omits_field_from_body(
    monkeypatch: pytest.MonkeyPatch,
    quiet_console: io.StringIO,
) -> None:
    """When --provider is absent, provider_id must NOT be in the POST body.

    Tests the _chat_payload helper directly.
    """
    from tether.cli.main import ChatMode

    body = cli_main._chat_payload(
        session_id="sess-xyz",
        prompt="hello",
        model_name="gpt-5",
        mode=ChatMode.chat,
        provider_id=None,
    )
    assert "provider_id" not in body, f"provider_id should be absent from payload when provider is None; got: {body}"


# ---------------------------------------------------------------------------
# test_providers_slash_command_renders_table
# ---------------------------------------------------------------------------


def test_providers_slash_command_renders_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """\\providers handler produces a Rich Table with one row per provider."""
    readyz_data = {
        "status": "ok",
        "providers": {
            "mlc-local": {
                "healthy": True,
                "kind": "mlc",
                "source": "local",
                "error": None,
            },
            "copilot-gpt5": {
                "healthy": True,
                "kind": "copilot",
                "source": "remote",
                "error": None,
            },
            "copilot-aux": {
                "healthy": False,
                "kind": "copilot",
                "source": "remote",
                "error": "AuthError: token unset",
            },
        },
        "default_provider_id": "mlc-local",
    }

    monkeypatch.setattr(cli_main, "API_BASE_URL", "http://127.0.0.1:8080/api/v1")
    monkeypatch.setattr(
        cli_main.requests,
        "get",
        lambda url, **kw: MockResponse(readyz_data),
    )

    table = cli_main.get_providers_table()

    assert table is not None, "get_providers_table() returned None; expected a Rich Table"
    assert isinstance(table, Table)
    assert table.row_count == 3, f"Expected 3 provider rows; got {table.row_count}"


# ---------------------------------------------------------------------------
# test_ambiguous_model_without_provider_lists_options
# ---------------------------------------------------------------------------


def test_ambiguous_model_without_provider_lists_options(
    monkeypatch: pytest.MonkeyPatch,
    quiet_console: io.StringIO,
) -> None:
    """Ambiguous --model without --provider must surface the choice (not silently pick)."""
    details = [
        _model_detail("gpt-5", "copilot-primary"),
        _model_detail("gpt-5", "copilot-secondary"),
    ]
    monkeypatch.setattr(cli_main, "get_available_model_details", lambda: details)
    # get_provider_health is called inside _interactive_model_select for the
    # default marker; return empty data so it doesn't make network calls.
    monkeypatch.setattr(cli_main, "get_provider_health", lambda: ({}, None))

    # Mock Prompt.ask to choose option "1" (the first displayed row).
    mock_prompt = Mock()
    mock_prompt.ask = Mock(return_value="1")
    monkeypatch.setattr(cli_main, "Prompt", mock_prompt)

    # Should NOT raise; instead drop into the interactive selector and
    # return the user's choice.
    name, pid = cli_main.select_model("gpt-5", provider=None)

    assert name == "gpt-5"
    # Rows are sorted by (provider_id, id), so "copilot-primary" comes first.
    assert pid == "copilot-primary", f"Expected first sorted provider ('copilot-primary'); got {pid!r}"
    # Verify the ambiguity warning was printed.
    output = quiet_console.getvalue()
    assert "ambiguous" in output.lower() or "provider" in output.lower(), (
        f"Expected an ambiguity message in console output; got: {output!r}"
    )


# ---------------------------------------------------------------------------
# test_ambiguous_model_with_provider_disambiguates
# ---------------------------------------------------------------------------


def test_ambiguous_model_with_provider_disambiguates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambiguous --model with matching --provider resolves without prompting."""
    details = [
        _model_detail("gpt-5", "copilot-primary"),
        _model_detail("gpt-5", "copilot-secondary"),
    ]
    monkeypatch.setattr(cli_main, "get_available_model_details", lambda: details)

    # Prompt.ask must NOT be called — if it is, the test will fail on the
    # Mock.ask assertion below.
    prompt_spy = Mock()
    monkeypatch.setattr(cli_main, "Prompt", prompt_spy)

    name, pid = cli_main.select_model("gpt-5", provider="copilot-primary")

    assert name == "gpt-5"
    assert pid == "copilot-primary", f"Expected 'copilot-primary'; got {pid!r}"
    prompt_spy.ask.assert_not_called()


def test_provider_flag_rejects_model_owned_by_another_provider(
    monkeypatch: pytest.MonkeyPatch,
    quiet_console: io.StringIO,
) -> None:
    details = [
        _model_detail("mlc-model", "mlc"),
        _model_detail("geniex-model", "geniex"),
    ]
    monkeypatch.setattr(cli_main, "get_available_model_details", lambda: details)

    with pytest.raises(cli_main.typer.Exit):
        cli_main.select_model("geniex-model", provider="mlc")

    assert "not available on provider 'mlc'" in quiet_console.getvalue()


def test_provider_flag_filters_interactive_model_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    details = [
        _model_detail("mlc-model", "mlc"),
        _model_detail("geniex-model", "geniex"),
    ]
    monkeypatch.setattr(cli_main, "get_available_model_details", lambda: details)
    monkeypatch.setattr(cli_main, "get_provider_health", lambda: ({}, None))
    prompt_spy = Mock()
    prompt_spy.ask = Mock(return_value="1")
    monkeypatch.setattr(cli_main, "Prompt", prompt_spy)

    model_name, provider_id = cli_main.select_model(None, provider="geniex")

    assert (model_name, provider_id) == ("geniex-model", "geniex")


def test_unconstrained_interactive_selection_includes_all_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an explicit provider, the selector continues to show all rows."""
    details = [
        _model_detail("mlc-model", "mlc"),
        _model_detail("geniex-model", "geniex"),
    ]
    monkeypatch.setattr(cli_main, "get_available_model_details", lambda: details)
    monkeypatch.setattr(cli_main, "get_provider_health", lambda: ({}, None))
    prompt_spy = Mock()
    prompt_spy.ask = Mock(side_effect=["1", "2"])
    monkeypatch.setattr(cli_main, "Prompt", prompt_spy)

    first_model, first_provider = cli_main.select_model(None, provider=None)
    second_model, second_provider = cli_main.select_model(None, provider=None)

    assert (first_model, first_provider) == ("geniex-model", "geniex")
    assert (second_model, second_provider) == ("mlc-model", "mlc")


def test_interactive_selector_defaults_to_configured_provider_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    details = [
        _model_detail("first", "aaa"),
        _model_detail("server-default", "zzz", is_default=True),
    ]
    monkeypatch.setattr(cli_main, "get_available_model_details", lambda: details)
    monkeypatch.setattr(
        cli_main,
        "get_provider_health",
        lambda: ({}, "zzz"),
    )
    prompt_spy = Mock()
    prompt_spy.ask = Mock(return_value="2")
    monkeypatch.setattr(cli_main, "Prompt", prompt_spy)

    model_name, provider_id = cli_main.select_model(None)

    assert (model_name, provider_id) == ("server-default", "zzz")
    prompt_spy.ask.assert_called_with("Select model #", default="2")


def test_interactive_selector_falls_back_to_default_provider_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    details = [
        _model_detail("geniex-model", "aaa", is_default=True),
        _model_detail("mlc-model", "zzz"),
    ]
    monkeypatch.setattr(cli_main, "get_available_model_details", lambda: details)
    monkeypatch.setattr(
        cli_main,
        "get_provider_health",
        lambda: ({}, "zzz"),
    )
    prompt_spy = Mock()
    prompt_spy.ask = Mock(return_value="2")
    monkeypatch.setattr(cli_main, "Prompt", prompt_spy)

    model_name, provider_id = cli_main.select_model(None)

    assert (model_name, provider_id) == ("mlc-model", "zzz")
    prompt_spy.ask.assert_called_with("Select model #", default="2")
