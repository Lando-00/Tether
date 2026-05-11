from __future__ import annotations

import pytest

from tests.fixtures.mock_whatsapp_adapter import MockWhatsAppClientAdapter
from tether.connectors.types import ConnectorState
from tether.connectors.whatsapp.adapter import Contact
from tether.connectors.whatsapp.tools import (
    WhatsAppGetContactsTool,
    WhatsAppResolveContactTool,
)
from tether.core.errors import ConnectorNotConfiguredError


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _ready() -> ConnectorState:
    return ConnectorState.READY


def _unconfigured() -> ConnectorState:
    return ConnectorState.UNCONFIGURED


def _contacts() -> list[Contact]:
    return [
        Contact(
            jid="15551234567@s.whatsapp.net",
            name="Ada Lovelace",
            e164="+15551234567",
            is_business=False,
        ),
        Contact(
            jid="15557654321@s.whatsapp.net",
            name="Grace Hopper",
            e164="+15557654321",
            is_business=True,
        ),
        Contact(
            jid="15550001111@s.whatsapp.net",
            name="Ada Byron",
            e164="+15550001111",
            is_business=False,
        ),
    ]


def _get_contacts_tool(
    *,
    contacts: list[Contact] | None = None,
    state_provider=_ready,
) -> WhatsAppGetContactsTool:
    return WhatsAppGetContactsTool(
        adapter=MockWhatsAppClientAdapter(),
        contacts_cache=contacts if contacts is not None else _contacts(),
        state_provider=state_provider,
    )


def _resolve_tool(
    *,
    contacts: list[Contact] | None = None,
    state_provider=_ready,
) -> WhatsAppResolveContactTool:
    return WhatsAppResolveContactTool(
        adapter=MockWhatsAppClientAdapter(),
        contacts_cache=contacts if contacts is not None else _contacts(),
        state_provider=state_provider,
    )


@pytest.mark.anyio
async def test_get_contacts_happy_path_matches_name_substring() -> None:
    tool = _get_contacts_tool()

    result = await tool.run(query="Grace")

    assert result == {
        "contacts": [
            {
                "jid": "15557654321@s.whatsapp.net",
                "name": "Grace Hopper",
                "e164": "+15557654321",
                "is_business": True,
            }
        ],
        "count": 1,
        "query": "Grace",
    }


@pytest.mark.anyio
async def test_get_contacts_matches_e164_substring() -> None:
    tool = _get_contacts_tool()

    result = await tool.run(query="54321")

    assert result["count"] == 1
    assert result["contacts"][0]["name"] == "Grace Hopper"


@pytest.mark.anyio
async def test_get_contacts_honors_limit() -> None:
    tool = _get_contacts_tool()

    result = await tool.run(query="Ada", limit=1)

    assert result["count"] == 1
    assert result["contacts"][0]["name"] == "Ada Lovelace"


@pytest.mark.anyio
async def test_get_contacts_returns_empty_matches() -> None:
    tool = _get_contacts_tool()

    result = await tool.run(query="nobody")

    assert result == {"contacts": [], "count": 0, "query": "nobody"}


@pytest.mark.anyio
async def test_get_contacts_refuses_when_not_ready() -> None:
    tool = _get_contacts_tool(state_provider=_unconfigured)

    with pytest.raises(ConnectorNotConfiguredError):
        await tool.run(query="Ada")


@pytest.mark.anyio
async def test_get_contacts_rejects_empty_query() -> None:
    tool = _get_contacts_tool()

    result = await tool.run(query=" ")

    assert result == {"error": "query must be non-empty"}


@pytest.mark.anyio
@pytest.mark.parametrize("limit", [0, 101])
async def test_get_contacts_rejects_invalid_limit(limit: int) -> None:
    tool = _get_contacts_tool()

    result = await tool.run(query="Ada", limit=limit)

    assert result == {"error": f"limit must be in [1, 100], got {limit}"}


@pytest.mark.anyio
async def test_resolve_contact_happy_path_name() -> None:
    contacts = [
        Contact(
            jid="15551234567@s.whatsapp.net",
            name="Ada Lovelace",
            e164="+15551234567",
            is_business=False,
        )
    ]
    tool = _resolve_tool(contacts=contacts)

    result = await tool.run(value="Ada")

    assert result == {"jid": "15551234567@s.whatsapp.net"}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("+15551234567", "15551234567@s.whatsapp.net"),
        ("15551234567@c.us", "15551234567@s.whatsapp.net"),
        ("120363000000000000@g.us", "120363000000000000@g.us"),
    ],
)
async def test_resolve_contact_happy_path_direct_values(
    value: str,
    expected: str,
) -> None:
    tool = _resolve_tool()

    result = await tool.run(value=value)

    assert result == {"jid": expected}


@pytest.mark.anyio
async def test_resolve_contact_ambiguous_name_returns_candidates() -> None:
    tool = _resolve_tool()

    result = await tool.run(value="Ada")

    assert result["error"] == "ambiguous"
    assert [candidate["name"] for candidate in result["candidates"]] == [
        "Ada Lovelace",
        "Ada Byron",
    ]


@pytest.mark.anyio
async def test_resolve_contact_unresolvable_value() -> None:
    tool = _resolve_tool()

    result = await tool.run(value="nobody")

    assert result == {"error": "could not resolve 'nobody'"}


@pytest.mark.anyio
async def test_resolve_contact_refuses_when_not_ready() -> None:
    tool = _resolve_tool(state_provider=_unconfigured)

    with pytest.raises(ConnectorNotConfiguredError):
        await tool.run(value="Ada")


@pytest.mark.anyio
async def test_resolve_contact_rejects_empty_value() -> None:
    tool = _resolve_tool()

    result = await tool.run(value=" ")

    assert result == {"error": "value must be non-empty"}
