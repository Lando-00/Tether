"""Experimental GitHub Copilot SDK-backed model provider."""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from importlib import import_module
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional

from tether.core.interfaces import ModelProvider
from tether.providers.types import (
    ModelDetails,
    ProviderCapabilities,
    ProviderEvent,
    ProviderText,
    ProviderThink,
)

_DONE = object()

# Default reasoning effort values surfaced for Copilot models that
# advertise reasoning_effort support (see ``reasoning_effort_models``).
# Matches the public-preview SDK enum; clients are encouraged to query
# /models/details rather than hard-coding this list.
DEFAULT_REASONING_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high")


class CopilotProvider(ModelProvider):
    """ModelProvider adapter for the public-preview GitHub Copilot SDK.

    This provider intentionally treats the SDK as a streaming text source. The
    Copilot SDK's own tool/agent loop is denied by default so Tether's
    orchestrator, tool audit, and connector send-safety remain authoritative.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-5",
        models: Optional[List[str]] = None,
        context_window: int = 128_000,
        context_windows: Optional[Dict[str, int]] = None,
        github_token_env: Optional[str] = "COPILOT_GITHUB_TOKEN",
        github_token: Optional[str] = None,
        use_logged_in_user: bool = True,
        external_server_url: Optional[str] = None,
        cli_path: Optional[str] = None,
        copilot_home: Optional[str] = None,
        client_config: Optional[Dict[str, Any]] = None,
        provider: Optional[Dict[str, Any]] = None,
        enable_builtin_tools: bool = False,
        reasoning_effort_models: Optional[List[str]] = None,
        reasoning_efforts: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.models = tuple(dict.fromkeys([model, *(models or [])]))
        self.context_window = context_window
        self.context_windows = dict(context_windows or {})
        self.github_token_env = github_token_env
        self.github_token = github_token
        self.use_logged_in_user = use_logged_in_user
        self.external_server_url = external_server_url
        self.cli_path = cli_path
        self.copilot_home = copilot_home
        self.client_config = dict(client_config or {})
        self.provider = dict(provider or {})
        self.enable_builtin_tools = enable_builtin_tools
        # When ``reasoning_effort_models`` is None, every configured model
        # is assumed to support reasoning_effort (Copilot SDK lists this as
        # a per-model capability; users override here when the SDK adds
        # non-reasoning models). Explicit empty list = none support it.
        self._reasoning_effort_models = (
            None
            if reasoning_effort_models is None
            else tuple(reasoning_effort_models)
        )
        self.reasoning_efforts = tuple(
            reasoning_efforts
            if reasoning_efforts is not None
            else DEFAULT_REASONING_EFFORTS
        )

    @property
    def kind(self) -> str:
        return "copilot"

    @property
    def source(self) -> str:
        return "remote"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=True,
            tools_native=False,
            tools_marker=False,
            thinking_channel=True,
            cancel_inflight=True,
            multi_model=True,
            warm_up_required=False,
        )

    def _model_supports_reasoning_effort(self, model_name: str) -> bool:
        if self._reasoning_effort_models is None:
            return True
        return model_name in self._reasoning_effort_models

    def default_model(self) -> Optional[str]:
        return self.model

    def list_model_info(self) -> List[ModelDetails]:
        """Override the ABC default to advertise reasoning_effort support.

        Copilot SDK accepts ``reasoning_effort`` on :meth:`create_session`
        for compatible models. ``reasoning_effort_models`` constructor
        arg narrows this when needed; the default treats every
        configured model as supporting it (matches today's SDK shape).
        """
        caps = self.capabilities
        efforts = list(self.reasoning_efforts)
        details: List[ModelDetails] = []
        for name in self.models:
            supports = self._model_supports_reasoning_effort(name)
            details.append(
                ModelDetails(
                    id=name,
                    provider_kind="copilot",
                    source="remote",
                    context_window=int(
                        self.context_windows.get(name, self.context_window)
                    ),
                    supports_thinking=caps.thinking_channel,
                    supports_reasoning_effort=supports,
                    reasoning_efforts=efforts if supports else None,
                    is_default=(name == self.model),
                )
            )
        return details

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[str | List[Dict[str, Any]], None]:
        """Stream assistant text chunks from a short-lived Copilot SDK session."""
        async for kind, text in self._run_copilot_turn(
            model_name=model_name,
            messages=messages,
            request_id=request_id,
            reasoning_effort=reasoning_effort,
        ):
            if kind == "text":
                yield text

    async def stream_typed(
        self,
        *,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        cancel_token: Optional[Any] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncIterator[ProviderEvent]:
        """Stream typed provider events from Copilot SDK session events."""
        async for kind, text in self._run_copilot_turn(
            model_name=model_name,
            messages=messages,
            request_id=request_id,
            reasoning_effort=reasoning_effort,
        ):
            if cancel_token is not None and self._cancelled(cancel_token):
                break
            if kind == "think":
                yield ProviderThink(text=text)
            else:
                yield ProviderText(text=text)

    def list_models(self) -> List[str]:
        """Return statically configured models.

        The Copilot SDK exposes model listing asynchronously through its client
        lifecycle. Tether's current provider ABC keeps this method synchronous,
        so the initial provider avoids hidden event-loop blocking.
        """
        return list(self.models)

    def unload_model(self, model_name: str) -> bool:
        """Copilot SDK does not expose model unload semantics."""
        return False

    def get_context_window(self, model_name: str) -> int:
        return int(self.context_windows.get(model_name, self.context_window))

    async def warm_up(self, model_name: str) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def _run_copilot_turn(
        self,
        *,
        model_name: str,
        messages: List[Dict[str, Any]],
        request_id: Optional[str],
        reasoning_effort: Optional[str] = None,
    ) -> AsyncIterator[tuple[str, str]]:
        copilot_mod, session_mod = self._load_sdk()
        client = self._build_client(copilot_mod)
        session = None
        unsubscribe = None
        send_task: Optional[asyncio.Task[Any]] = None
        queue: asyncio.Queue[Any] = asyncio.Queue()
        idle = asyncio.Event()
        loop = asyncio.get_running_loop()

        def enqueue(item: Any) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def on_event(event: Any) -> None:
            event_type = self._event_type(event)
            if event_type == "session.idle":
                idle.set()
                return
            delta_kind = self._delta_kind(event_type)
            if delta_kind is None:
                return
            text = self._delta_content(getattr(event, "data", None))
            if text:
                enqueue((delta_kind, text))

        def on_send_done(task: asyncio.Task[Any]) -> None:
            with suppress(asyncio.CancelledError):
                exc = task.exception()
                enqueue(exc if exc is not None else _DONE)

        try:
            await client.start()
            session_kwargs: Dict[str, Any] = dict(
                on_permission_request=self._permission_handler(session_mod),
                model=model_name or self.model,
                streaming=True,
                provider=self.provider or None,
            )
            effective_model = model_name or self.model
            if reasoning_effort is not None and self._model_supports_reasoning_effort(
                effective_model
            ):
                session_kwargs["reasoning_effort"] = reasoning_effort
            session = await client.create_session(**session_kwargs)
            unsubscribe = session.on(on_event)
            prompt = self._messages_to_prompt(messages)

            send_and_wait = getattr(session, "send_and_wait", None)
            if send_and_wait is not None:
                send_task = asyncio.create_task(send_and_wait(prompt))
            else:
                send_task = asyncio.create_task(self._send_and_wait_idle(session, prompt, idle))
            send_task.add_done_callback(on_send_done)

            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            if send_task is not None and not send_task.done():
                send_task.cancel()
                with suppress(asyncio.CancelledError):
                    await send_task
            if callable(unsubscribe):
                unsubscribe()
            if session is not None:
                await session.disconnect()
            await client.stop()

    async def _send_and_wait_idle(self, session: Any, prompt: str, idle: asyncio.Event) -> None:
        await session.send(prompt)
        await idle.wait()

    def _load_sdk(self) -> tuple[Any, Any]:
        try:
            return import_module("copilot"), import_module("copilot.session")
        except ImportError as exc:
            raise RuntimeError(
                "CopilotProvider requires the optional dependency "
                "`github-copilot-sdk`. Install it with `pip install -e "
                "\".[copilot]\"` or select a different provider."
            ) from exc

    def _build_client(self, copilot_mod: Any) -> Any:
        client_cls = getattr(copilot_mod, "CopilotClient")
        if self.external_server_url:
            external_cls = getattr(copilot_mod, "ExternalServerConfig", None)
            if external_cls is None:
                raise RuntimeError(
                    "Installed github-copilot-sdk does not expose ExternalServerConfig."
                )
            return client_cls(external_cls(url=self.external_server_url))

        options = dict(self.client_config)
        token = self._resolve_github_token()
        if token:
            options["github_token"] = token
        options["use_logged_in_user"] = self.use_logged_in_user
        if self.cli_path:
            options["cli_path"] = self.cli_path
        if self.copilot_home:
            options["copilot_home"] = self.copilot_home

        if not options:
            return client_cls()

        try:
            return client_cls(options)
        except TypeError:
            subprocess_cls = getattr(copilot_mod, "SubprocessConfig", None)
            if subprocess_cls is None:
                raise
            return client_cls(subprocess_cls(**options))

    def _resolve_github_token(self) -> Optional[str]:
        if self.github_token:
            return self.github_token
        if not self.github_token_env:
            return None
        token = os.environ.get(self.github_token_env)
        if token:
            return token
        if not self.use_logged_in_user:
            raise RuntimeError(
                f"{self.github_token_env} is not set and use_logged_in_user=False."
            )
        return None

    def _permission_handler(self, session_mod: Any) -> Any:
        if self.enable_builtin_tools:
            return session_mod.PermissionHandler.approve_all

        result_cls = getattr(session_mod, "PermissionRequestResult", None)

        def deny_all(_request: Any, _invocation: Any) -> Any:
            if result_cls is None:
                return {"kind": "denied-by-rules"}
            return result_cls(kind="denied-by-rules")

        return deny_all

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if content is None:
                content = ""
            parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    @staticmethod
    def _event_type(event: Any) -> str:
        raw = getattr(event, "type", "")
        value = getattr(raw, "value", raw)
        return str(value)

    @staticmethod
    def _delta_kind(event_type: str) -> Optional[str]:
        normalized = event_type.lower()
        if normalized == "assistant.message_delta" or normalized.endswith(
            "assistant_message_delta"
        ):
            return "text"
        if normalized == "assistant.reasoning_delta" or normalized.endswith(
            "assistant_reasoning_delta"
        ):
            return "think"
        return None

    @staticmethod
    def _delta_content(data: Any) -> str:
        if data is None:
            return ""
        if isinstance(data, dict):
            value = data.get("delta_content", data.get("deltaContent", ""))
        else:
            value = getattr(data, "delta_content", None)
            if value is None:
                value = getattr(data, "deltaContent", "")
        return value or ""

    @staticmethod
    def _cancelled(cancel_token: Any) -> bool:
        cancelled = getattr(cancel_token, "cancelled", None)
        if callable(cancelled):
            return bool(cancelled())
        is_cancelled = getattr(cancel_token, "is_cancelled", None)
        if callable(is_cancelled):
            return bool(is_cancelled())
        return False


__all__ = ["CopilotProvider"]

