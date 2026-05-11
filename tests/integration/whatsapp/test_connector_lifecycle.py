"""WhatsApp connector lifecycle integration tests."""
from __future__ import annotations

import asyncio
import sys

import pytest

from tests.fixtures.mock_whatsapp_adapter import MockWhatsAppClientAdapter
from tether.connectors.types import ConnectorState
from tether.connectors.whatsapp.adapter import Contact, PairStatus
from tether.connectors.whatsapp.connector import WhatsAppConnector
from tether.core.errors import ConnectorNotConfiguredError

pytest.importorskip(
    "tether.connectors.whatsapp.tools",
    reason="WhatsApp tool classes are produced by wa-IMP-D",
)

CANONICAL_TOOL_NAMES = {
    "whatsapp_prepare_send",
    "whatsapp_confirm_send",
    "whatsapp_list_unread",
    "whatsapp_get_thread",
    "whatsapp_inbox_mark_seen",
    "whatsapp_mark_platform_read",
    "whatsapp_send_media",
    "whatsapp_get_contacts",
    "whatsapp_resolve_contact",
}


def _connector(
    tmp_path,
    mock: MockWhatsAppClientAdapter | None = None,
) -> tuple[WhatsAppConnector, MockWhatsAppClientAdapter]:
    adapter = mock or MockWhatsAppClientAdapter()
    return (
        WhatsAppConnector(
            data_dir=tmp_path,
            adapter_factory=lambda: adapter,
        ),
        adapter,
    )


def _write_creds(tmp_path) -> None:
    auth_dir = tmp_path / "auth"
    auth_dir.mkdir(parents=True, exist_ok=True)
    (auth_dir / "neonize.db").write_bytes(b"mock-creds")


async def test_unconfigured_state_initially(tmp_path) -> None:
    connector, _ = _connector(tmp_path)

    status = await connector.auth_status()

    assert status.state is ConnectorState.UNCONFIGURED
    await connector.stop()


async def test_begin_login_returns_qr_prompt(tmp_path) -> None:
    mock = MockWhatsAppClientAdapter()
    mock.qr_bytes = b"2@test,example,qr"
    connector, _ = _connector(tmp_path, mock)

    try:
        prompt = await connector.begin_login()

        assert prompt.kind == "qr_code"
        assert prompt.payload == "2@test,example,qr"
        assert prompt.expires_at is not None
    finally:
        await connector.stop()


async def test_complete_login_paired_transitions_to_ready(tmp_path) -> None:
    mock = MockWhatsAppClientAdapter()
    mock.contacts = [
        Contact(
            jid="447911000000@s.whatsapp.net",
            name="Alice",
            e164="+447911000000",
            is_business=False,
        )
    ]
    connector, _ = _connector(tmp_path, mock)

    try:
        await connector.begin_login()
        mock.next_pair_status = PairStatus.PAIRED

        result = await connector.complete_login(payload={"timeout_sec": 0.1})

        assert result.state is ConnectorState.READY
        assert connector.state is ConnectorState.READY
        assert (await connector.auth_status()).state is ConnectorState.READY
    finally:
        await connector.stop()


async def test_complete_login_logged_out_during_pair(tmp_path) -> None:
    connector, mock = _connector(tmp_path)

    try:
        await connector.begin_login()
        mock.next_pair_status = PairStatus.LOGGED_OUT

        result = await connector.complete_login(payload={"timeout_sec": 0.1})

        assert result.state is ConnectorState.LOGGED_OUT
        assert result.detail == "logged_out_during_pair"
        assert connector.state is ConnectorState.LOGGED_OUT
    finally:
        await connector.stop()


async def test_complete_login_qr_rotated(tmp_path) -> None:
    connector, mock = _connector(tmp_path)

    try:
        mock.qr_bytes = b"2@first,example,qr"
        await connector.begin_login()
        mock.next_pair_status = PairStatus.QR_ROTATED
        mock.qr_bytes = b"2@rotated,example,qr"

        result = await connector.complete_login(payload={"timeout_sec": 0.1})

        assert result.state is ConnectorState.AUTHENTICATING
        assert result.next_prompt is not None
        assert result.next_prompt.payload == "2@rotated,example,qr"
    finally:
        await connector.stop()


async def test_complete_login_timeout(tmp_path) -> None:
    connector, mock = _connector(tmp_path)

    try:
        await connector.begin_login()
        mock.next_pair_status = PairStatus.TIMEOUT

        result = await connector.complete_login(payload={"timeout_sec": 0.1})

        assert result.state is ConnectorState.AUTHENTICATING
        assert result.detail == "qr_scan_timeout"
    finally:
        await connector.stop()


async def test_tools_returned_when_started(tmp_path) -> None:
    connector, _ = _connector(tmp_path)

    try:
        pre_start_names = set(connector.tools())
        assert pre_start_names in (set(), CANONICAL_TOOL_NAMES)

        _write_creds(tmp_path)
        await connector.start()

        assert set(connector.tools()) == CANONICAL_TOOL_NAMES
    finally:
        await connector.stop()


async def test_logout_transitions_to_logged_out(tmp_path) -> None:
    connector, _ = _connector(tmp_path)
    _write_creds(tmp_path)

    await connector.start()
    await connector.logout()

    assert connector.state is ConnectorState.LOGGED_OUT
    assert (await connector.auth_status()).state is ConnectorState.LOGGED_OUT
    with pytest.raises(ConnectorNotConfiguredError):
        await connector.tools()["whatsapp_list_unread"].invoke({})


async def test_auth_status_pre_start_no_creds_returns_unconfigured(tmp_path) -> None:
    before = {
        name for name in sys.modules if name == "neonize" or name.startswith("neonize.")
    }
    connector = WhatsAppConnector(data_dir=tmp_path)

    status = await connector.auth_status()

    after = {
        name for name in sys.modules if name == "neonize" or name.startswith("neonize.")
    }
    assert status.state is ConnectorState.UNCONFIGURED
    assert after == before


async def test_health_returns_state_snapshot(tmp_path) -> None:
    connector, mock = _connector(tmp_path)
    _write_creds(tmp_path)

    try:
        await connector.start()

        health = await connector.health()
        assert health.state is ConnectorState.DEGRADED
        assert health.detail == "connecting_on_resume"

        mock._state = "ready"
        await asyncio.sleep(1.1)

        health = await connector.health()
        assert health.state is ConnectorState.READY
        assert health.detail is None
    finally:
        await connector.stop()
