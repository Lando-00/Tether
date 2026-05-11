from __future__ import annotations

import io
from unittest.mock import Mock

import pytest
import requests
from rich.console import Console
from typer.testing import CliRunner

from tether.cli import main as cli_main

API_ROOT = "http://testserver"
API_BASE = f"{API_ROOT}/api/v1"


class MockResponse:
    def __init__(self, body, status_code: int = 200, text: str | None = None) -> None:
        self._body = body
        self.status_code = status_code
        self.text = text if text is not None else str(body)

    def json(self):
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return
        exc = requests.HTTPError(f"{self.status_code} Error")
        exc.response = self
        raise exc


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def console_output(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    stream = io.StringIO()
    monkeypatch.setattr(
        cli_main,
        "console",
        Console(file=stream, force_terminal=False, width=120),
    )
    monkeypatch.setattr(cli_main, "API_BASE_URL", "http://127.0.0.1:8080/api/v1")
    return stream


def output(result, stream: io.StringIO) -> str:
    return result.output + stream.getvalue()


def qr_prompt(payload: str) -> dict:
    return {"kind": "qr_code", "payload": payload, "expires_at": None, "extra": {}}


def ready_result() -> dict:
    return {"state": "ready", "detail": None, "next_prompt": None}


def health_response() -> MockResponse:
    return MockResponse(
        [{"id": "whatsapp", "health": {"state": "ready", "detail": "connected"}}]
    )


def test_connect_whatsapp_happy_path(
    runner: CliRunner,
    console_output: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock(side_effect=[MockResponse(qr_prompt("2@abc")), MockResponse(ready_result())])
    get = Mock(return_value=health_response())
    monkeypatch.setattr(cli_main.requests, "post", post)
    monkeypatch.setattr(cli_main.requests, "get", get)

    result = runner.invoke(
        cli_main.app,
        ["connect", "whatsapp", "--api-url", API_ROOT, "--timeout", "60", "--qr-format", "raw"],
    )

    out = output(result, console_output)
    assert result.exit_code == 0
    assert "Logged in to WhatsApp" in out
    assert "Health: ready (connected)" in out
    assert post.call_args_list[0].args[0] == f"{API_BASE}/connectors/whatsapp/login/begin"
    complete_call = post.call_args_list[1]
    assert complete_call.args[0] == f"{API_BASE}/connectors/whatsapp/login/complete"
    assert 0 < complete_call.kwargs["json"]["payload"]["timeout_sec"] <= 30


def test_connect_whatsapp_qr_rotated(
    runner: CliRunner,
    console_output: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock(
        side_effect=[
            MockResponse(qr_prompt("QR1")),
            MockResponse(
                {
                    "state": "authenticating",
                    "detail": "qr_rotated",
                    "next_prompt": qr_prompt("QR2"),
                }
            ),
            MockResponse(ready_result()),
        ]
    )
    monkeypatch.setattr(cli_main.requests, "post", post)
    monkeypatch.setattr(cli_main.requests, "get", Mock(return_value=health_response()))

    result = runner.invoke(
        cli_main.app,
        ["connect", "whatsapp", "--api-url", API_ROOT, "--timeout", "60", "--qr-format", "raw"],
    )

    out = output(result, console_output)
    assert result.exit_code == 0
    assert out.count("QR code for WhatsApp") == 2
    assert "QR1" in out
    assert "QR2" in out


def test_connect_whatsapp_logged_out_during_pair(
    runner: CliRunner,
    console_output: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock(
        side_effect=[
            MockResponse(qr_prompt("2@abc")),
            MockResponse({"state": "logged_out", "detail": "logged_out_during_pair"}),
        ]
    )
    monkeypatch.setattr(cli_main.requests, "post", post)

    result = runner.invoke(
        cli_main.app,
        ["connect", "whatsapp", "--api-url", API_ROOT, "--timeout", "60", "--qr-format", "raw"],
    )

    assert result.exit_code == 2
    assert "Unpaired" in output(result, console_output)


def test_connect_whatsapp_error_state(
    runner: CliRunner,
    console_output: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock(
        side_effect=[
            MockResponse(qr_prompt("2@abc")),
            MockResponse({"state": "error", "detail": "adapter failed"}),
        ]
    )
    monkeypatch.setattr(cli_main.requests, "post", post)

    result = runner.invoke(
        cli_main.app,
        ["connect", "whatsapp", "--api-url", API_ROOT, "--timeout", "60", "--qr-format", "raw"],
    )

    out = output(result, console_output)
    assert result.exit_code == 3
    assert "adapter failed" in out


def test_connect_whatsapp_qr_scan_timeout(
    runner: CliRunner,
    console_output: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock(
        side_effect=[
            MockResponse(qr_prompt("2@abc")),
            MockResponse(
                {
                    "state": "authenticating",
                    "detail": "qr_scan_timeout",
                    "next_prompt": None,
                }
            ),
        ]
    )
    monkeypatch.setattr(cli_main.requests, "post", post)

    result = runner.invoke(
        cli_main.app,
        ["connect", "whatsapp", "--api-url", API_ROOT, "--timeout", "60", "--qr-format", "raw"],
    )

    out = output(result, console_output)
    assert result.exit_code == 4
    assert "QR expired without scan" in out


def test_connect_whatsapp_server_unreachable(
    runner: CliRunner,
    console_output: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_main.requests,
        "post",
        Mock(side_effect=requests.ConnectionError("connection refused")),
    )

    result = runner.invoke(
        cli_main.app,
        ["connect", "whatsapp", "--api-url", API_ROOT, "--timeout", "60", "--qr-format", "raw"],
    )

    out = output(result, console_output)
    assert result.exit_code == 1
    assert "Connection error" in out
    assert "Could not begin WhatsApp login" in out


def test_logout_whatsapp_happy_path(
    runner: CliRunner,
    console_output: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock(
        return_value=MockResponse({"ok": True, "id": "whatsapp", "state": "logged_out"})
    )
    monkeypatch.setattr(cli_main.requests, "post", post)

    result = runner.invoke(cli_main.app, ["logout", "whatsapp", "--api-url", API_ROOT])

    out = output(result, console_output)
    assert result.exit_code == 0
    assert "Logged out from WhatsApp." in out
    assert post.call_args.args[0] == f"{API_BASE}/connectors/whatsapp/logout"


def test_logout_whatsapp_failure(
    runner: CliRunner,
    console_output: io.StringIO,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    post = Mock(
        return_value=MockResponse({"detail": "Connector not found: 'whatsapp'"}, status_code=404)
    )
    monkeypatch.setattr(cli_main.requests, "post", post)

    result = runner.invoke(cli_main.app, ["logout", "whatsapp", "--api-url", API_ROOT])

    out = output(result, console_output)
    assert result.exit_code == 1
    assert "Connector not found" in out
