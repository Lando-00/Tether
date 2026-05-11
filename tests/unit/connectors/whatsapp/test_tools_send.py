from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.fixtures.mock_whatsapp_adapter import MockWhatsAppClientAdapter
from tether.connectors.types import ConnectorState
from tether.connectors.whatsapp.adapter import Contact, SendResult
from tether.connectors.whatsapp.draft_store import DraftStore
from tether.connectors.whatsapp.target_resolver import resolve_target
from tether.connectors.whatsapp.tools import (
    WhatsAppConfirmSendTool,
    WhatsAppPrepareSendTool,
    WhatsAppSendMediaTool,
)
from tether.core.errors import ConnectorNotConfiguredError
from tether.core.types import ToolExecutionContext


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
    ]


def _resolver(contacts: list[Contact]):
    return lambda value: resolve_target(value, contacts)


def _context(confirmed: bool) -> ToolExecutionContext:
    return ToolExecutionContext(
        session_id="session-1",
        turn_id="turn-1",
        user_confirmed_send=confirmed,
    )


class _FailingTextAdapter(MockWhatsAppClientAdapter):
    async def send_text(self, to_jid: str, text: str) -> SendResult:
        _ = (to_jid, text)
        raise RuntimeError("send failed")


def _prepare_tool(
    *,
    store: DraftStore | None = None,
    state_provider=_ready,
    adapter: MockWhatsAppClientAdapter | None = None,
) -> WhatsAppPrepareSendTool:
    return WhatsAppPrepareSendTool(
        adapter=adapter or MockWhatsAppClientAdapter(),
        draft_store=store or DraftStore(),
        target_resolver_fn=_resolver(_contacts()),
        state_provider=state_provider,
    )


def _confirm_tool(
    *,
    adapter: MockWhatsAppClientAdapter | None = None,
    store: DraftStore | None = None,
    state_provider=_ready,
) -> WhatsAppConfirmSendTool:
    return WhatsAppConfirmSendTool(
        adapter=adapter or MockWhatsAppClientAdapter(),
        draft_store=store or DraftStore(),
        state_provider=state_provider,
    )


def _media_tool(
    *,
    adapter: MockWhatsAppClientAdapter | None = None,
    store: DraftStore | None = None,
    state_provider=_ready,
) -> WhatsAppSendMediaTool:
    return WhatsAppSendMediaTool(
        adapter=adapter or MockWhatsAppClientAdapter(),
        draft_store=store or DraftStore(),
        target_resolver_fn=_resolver(_contacts()),
        state_provider=state_provider,
    )


@pytest.mark.anyio
async def test_prepare_send_happy_path_builds_draft() -> None:
    store = DraftStore(ttl_sec=123)
    tool = _prepare_tool(store=store)

    result = await tool.run(to="Ada", text="hello")

    assert result["preview"] == "hello"
    assert result["resolved_target"] == "15551234567@s.whatsapp.net"
    assert result["expires_in_sec"] == 123
    draft = store.peek(result["draft_id"])
    assert draft is not None
    assert draft.body == "hello"


@pytest.mark.anyio
async def test_prepare_send_two_calls_create_different_draft_ids() -> None:
    store = DraftStore()
    tool = _prepare_tool(store=store)

    first = await tool.run(to="Ada", text="first")
    second = await tool.run(to="Ada", text="second")

    assert first["draft_id"] != second["draft_id"]
    assert store.size() == 2


@pytest.mark.anyio
async def test_prepare_send_refuses_when_not_ready() -> None:
    tool = _prepare_tool(state_provider=_unconfigured)

    with pytest.raises(ConnectorNotConfiguredError):
        await tool.run(to="Ada", text="hello")


@pytest.mark.anyio
async def test_prepare_send_rejects_empty_text() -> None:
    tool = _prepare_tool()

    result = await tool.run(to="Ada", text="   ")

    assert result == {"error": "text must be non-empty"}


@pytest.mark.anyio
async def test_prepare_send_rejects_unresolved_target() -> None:
    tool = _prepare_tool()

    result = await tool.run(to="nobody", text="hello")

    assert "could not resolve target" in result["error"]


@pytest.mark.anyio
async def test_confirm_send_refuses_without_context_and_preserves_draft() -> None:
    store = DraftStore()
    draft = store.prepare(target_jid="15551234567@s.whatsapp.net", body="hello")
    tool = _confirm_tool(store=store)

    result = await tool.run(draft.draft_id, context=None)

    assert result["sent"] is False
    assert "user_confirmed_send is False" in result["reason"]
    assert store.peek(draft.draft_id) == draft


@pytest.mark.anyio
async def test_confirm_send_refuses_when_context_flag_false() -> None:
    store = DraftStore()
    draft = store.prepare(target_jid="15551234567@s.whatsapp.net", body="hello")
    tool = _confirm_tool(store=store)

    result = await tool.run(draft.draft_id, context=_context(False))

    assert result["sent"] is False
    assert store.peek(draft.draft_id) == draft


@pytest.mark.anyio
async def test_confirm_send_succeeds_when_context_flag_true() -> None:
    adapter = MockWhatsAppClientAdapter()
    store = DraftStore()
    draft = store.prepare(target_jid="15551234567@s.whatsapp.net", body="hello")
    tool = _confirm_tool(adapter=adapter, store=store)

    result = await tool.run(draft.draft_id, context=_context(True))

    assert result == {
        "sent": True,
        "message_id": "mock-text-1",
        "to_jid": "15551234567@s.whatsapp.net",
    }
    assert adapter.sent_texts[0][0:2] == ("15551234567@s.whatsapp.net", "hello")
    assert store.peek(draft.draft_id) is None


@pytest.mark.anyio
async def test_confirm_send_preserves_draft_on_send_failure() -> None:
    adapter = _FailingTextAdapter()
    store = DraftStore()
    draft = store.prepare(target_jid="15551234567@s.whatsapp.net", body="hello")
    tool = _confirm_tool(adapter=adapter, store=store)

    result = await tool.run(draft.draft_id, context=_context(True))

    assert result["sent"] is False
    assert "draft preserved" in result["reason"]
    assert store.peek(draft.draft_id) == draft


@pytest.mark.anyio
async def test_confirm_send_refuses_when_not_ready() -> None:
    store = DraftStore()
    draft = store.prepare(target_jid="15551234567@s.whatsapp.net", body="hello")
    tool = _confirm_tool(store=store, state_provider=_unconfigured)

    with pytest.raises(ConnectorNotConfiguredError):
        await tool.run(draft.draft_id, context=_context(True))


@pytest.mark.anyio
async def test_confirm_send_media_draft_dispatches_file_bytes() -> None:
    adapter = MockWhatsAppClientAdapter()
    store = DraftStore()
    media_path = Path(__file__).resolve()
    media = _media_tool(adapter=adapter, store=store)
    draft_result = await media.run(
        to="Ada",
        file_path=str(media_path),
        mime="text/plain",
        caption="source file",
    )
    confirm = _confirm_tool(adapter=adapter, store=store)

    result = await confirm.run(draft_result["draft_id"], context=_context(True))

    assert result["sent"] is True
    assert adapter.sent_media[0]["to_jid"] == "15551234567@s.whatsapp.net"
    assert adapter.sent_media[0]["body"] == media_path.read_bytes()
    assert adapter.sent_media[0]["mime"] == "text/plain"
    assert adapter.sent_media[0]["caption"] == "source file"


@pytest.mark.anyio
async def test_confirm_send_stale_draft_returns_not_found_or_expired() -> None:
    store = DraftStore(ttl_sec=0.01)
    draft = store.prepare(target_jid="15551234567@s.whatsapp.net", body="hello")
    time.sleep(0.03)
    tool = _confirm_tool(store=store)

    result = await tool.run(draft.draft_id, context=_context(True))

    assert result["sent"] is False
    assert "not found or expired" in result["reason"]


@pytest.mark.anyio
async def test_send_media_happy_path_builds_draft_only() -> None:
    adapter = MockWhatsAppClientAdapter()
    store = DraftStore()
    media_path = Path(__file__).resolve()
    tool = _media_tool(adapter=adapter, store=store)

    result = await tool.run(
        to="Ada",
        file_path=str(media_path),
        mime="text/x-python",
        caption="caption",
    )

    assert result["resolved_target"] == "15551234567@s.whatsapp.net"
    assert result["preview"].startswith("<text/x-python from test_tools_send.py")
    assert adapter.outbox == []
    draft = store.peek(result["draft_id"])
    assert draft is not None
    assert draft.media_path == str(media_path)
    assert draft.media_mime == "text/x-python"
    assert draft.media_caption == "caption"


@pytest.mark.anyio
async def test_send_media_two_calls_create_different_draft_ids() -> None:
    store = DraftStore()
    media_path = Path(__file__).resolve()
    tool = _media_tool(store=store)

    first = await tool.run(to="Ada", file_path=str(media_path))
    second = await tool.run(to="Ada", file_path=str(media_path))

    assert first["draft_id"] != second["draft_id"]
    assert store.size() == 2


@pytest.mark.anyio
async def test_send_media_refuses_when_not_ready() -> None:
    tool = _media_tool(state_provider=_unconfigured)

    with pytest.raises(ConnectorNotConfiguredError):
        await tool.run(to="Ada", file_path=str(Path(__file__).resolve()))


@pytest.mark.anyio
async def test_send_media_rejects_relative_path() -> None:
    tool = _media_tool()

    result = await tool.run(to="Ada", file_path="relative.txt")

    assert "file_path must be absolute" in result["error"]


@pytest.mark.anyio
async def test_send_media_rejects_parent_traversal() -> None:
    tool = _media_tool()

    result = await tool.run(
        to="Ada",
        file_path="C:/..//Windows/System32/drivers/etc/hosts",
    )

    assert "error" in result


@pytest.mark.anyio
async def test_send_media_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.jpg"
    target.write_bytes(b"jpg")
    link = tmp_path / "link.jpg"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    tool = _media_tool()

    result = await tool.run(to="Ada", file_path=str(link))

    assert "symlinks not allowed" in result["error"]


@pytest.mark.anyio
async def test_send_media_rejects_unc_path() -> None:
    tool = _media_tool()

    result = await tool.run(to="Ada", file_path="//server/share/file.jpg")

    assert result == {"error": "UNC paths not allowed"}


@pytest.mark.anyio
async def test_send_media_rejects_nonexistent() -> None:
    tool = _media_tool()
    missing = Path(__file__).resolve().with_name("missing-media-for-test.bin")

    result = await tool.run(to="Ada", file_path=str(missing))

    assert "cannot resolve file_path" in result["error"]


@pytest.mark.anyio
async def test_send_media_rejects_unresolved_target() -> None:
    tool = _media_tool()

    result = await tool.run(to="nobody", file_path=str(Path(__file__).resolve()))

    assert "could not resolve target" in result["error"]


def test_confirm_send_schema_excludes_context() -> None:
    tool = _confirm_tool()
    params = tool.schema["function"]["parameters"]

    assert "draft_id" in params["properties"]
    assert "draft_id" in params["required"]
    assert "context" not in params["properties"]
