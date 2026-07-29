"""OllamaProvider — Tether ModelProvider for Ollama servers (ADR-0022 Phase 2.A).

The provider is HTTP-only — no MLC / hardware imports apply.  It wraps an
:class:`OllamaClientBase` implementation (native NDJSON or
OpenAI-compatible SSE) selected by the ``api_surface`` constructor arg.
Phase 2.A ships the native client end-to-end; the openai-compat client is
lazy-imported so Phase 2.B can drop its module in without touching this
file.

Wire-protocol details, error mapping, capabilities and storage layout are
ratified in:

    docs/adr/0022-ollama-provider.md
    docs/adr/0022-contract-stubs.md  (§5, §6, §8)
"""
from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Sequence
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Literal, Optional

import httpx
import structlog

from tether.core.interfaces import ModelProvider
from tether.providers.ollama.client import (
    OllamaClientBase,
    OllamaNativeClient,
)
from tether.providers.types import (
    ModelDetails,
    ProviderCapabilities,
    ProviderEvent,
    ProviderText,
    ProviderThink,
    ProviderToolCall,
)
from tether.security.outbound import assert_safe_url

_log = structlog.get_logger(__name__)


class OllamaProvider(ModelProvider):
    """HTTP-based :class:`ModelProvider` for Ollama (native NDJSON + OpenAI-compat).

    Constructed by :func:`tether.core.factory.load` via ``ProviderSpec.args``.
    Does NOT import ``mlc_llm`` — ADR-0016 isolation rule does not apply.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_surface: Literal["native", "openai_compat"] = "native",
        models: Optional[List[str]] = None,
        default_model: Optional[str] = None,
        thinking_models: Sequence[str] = (),
        context_windows: Optional[Dict[str, int]] = None,
        api_key_env: Optional[str] = None,
        discover_at_startup: bool = False,
        timeout_seconds: float = 600.0,
        connect_timeout_seconds: float = 10.0,
        keep_alive: Optional[Any] = None,
        default_options: Optional[Dict[str, Any]] = None,
        url_validator: Optional[Callable[[str], None]] = None,
    ) -> None:
        # --- URL validation (loud failure at construct time) ------------------
        validator = url_validator or (lambda u: assert_safe_url(u, settings=None))
        try:
            validator(base_url)
        except Exception as exc:
            raise ValueError(
                f"OllamaProvider: invalid base_url {base_url!r}: {exc}"
            ) from exc

        if api_surface not in ("native", "openai_compat"):
            raise ValueError(
                f"OllamaProvider: unknown api_surface {api_surface!r}; "
                f"expected 'native' or 'openai_compat'"
            )

        self.base_url: str = base_url.rstrip("/")
        self.api_surface: Literal["native", "openai_compat"] = api_surface
        # Code-review follow-up (ADR-0022 P3.2): for the OpenAI-compatible
        # surface the user is documented to pass base_url already ending in
        # ``/v1`` (matching how the existing OpenAI client works). When they
        # forget the suffix, the client would POST to ``/chat/completions``
        # which Ollama doesn't recognise. Auto-append the suffix with a
        # one-time INFO log so the deployment "just works" but the operator
        # is informed of the normalisation.
        if (
            self.api_surface == "openai_compat"
            and not self.base_url.endswith("/v1")
            and "/v1/" not in self.base_url
        ):
            _log.info(
                "ollama.openai_compat.base_url_normalised",
                from_url=self.base_url,
                to_url=self.base_url + "/v1",
            )
            self.base_url = self.base_url + "/v1"
        self.models: List[str] = list(models or [])
        self.default_model_name: Optional[str] = default_model
        self._thinking_models: frozenset[str] = frozenset(thinking_models)
        self.context_windows: Dict[str, int] = dict(context_windows or {})
        self._context_window_cache: Dict[str, int] = {}
        self._discovered_models: List[str] = []
        self.api_key_env: Optional[str] = api_key_env
        self.discover_at_startup: bool = discover_at_startup
        self.timeout_seconds: float = float(timeout_seconds)
        self.connect_timeout_seconds: float = float(connect_timeout_seconds)
        self.keep_alive: Optional[Any] = keep_alive
        self.default_options: Dict[str, Any] = dict(default_options or {})
        self.url_validator = url_validator

        if self._thinking_models and api_surface == "openai_compat":
            _log.warning(
                "ollama.thinking_models_ignored_for_openai_compat",
                thinking_models=sorted(self._thinking_models),
            )

        # Shared httpx client — owned here, closed in aclose() via the inner
        # client's aclose() chain. Optional bearer-token auth is wired here
        # so both surface implementations inherit it transparently.
        timeout_cfg = httpx.Timeout(
            connect=connect_timeout_seconds,
            read=timeout_seconds,
            write=30.0,
            pool=5.0,
        )
        headers: Dict[str, str] = {}
        if api_key_env:
            tok = os.environ.get(api_key_env, "")
            if tok:
                headers["Authorization"] = f"Bearer {tok}"

        self._http: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout_cfg,
            headers=headers,
        )

        self._client: OllamaClientBase = self._build_client()

    # ------------------------------------------------------------------
    # Client selection (lazy import for the openai-compat surface so Phase
    # 2.B's module is not required at import time of this file).
    # ------------------------------------------------------------------

    def _build_client(self) -> OllamaClientBase:
        if self.api_surface == "native":
            return OllamaNativeClient(
                self.base_url,
                timeout=self.timeout_seconds,
                connect_timeout=self.connect_timeout_seconds,
                http_client=self._http,
            )
        if self.api_surface == "openai_compat":
            try:
                from tether.providers.ollama.openai_client import (  # type: ignore[import-not-found]
                    OllamaOpenAICompatClient,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "api_surface='openai_compat' requires "
                    "tether.providers.ollama.openai_client (Phase 2.B). "
                    "Use api_surface='native' or install the missing module."
                ) from exc
            return OllamaOpenAICompatClient(
                self.base_url,
                timeout=self.timeout_seconds,
                connect_timeout=self.connect_timeout_seconds,
                http_client=self._http,
            )
        raise ValueError(f"unknown api_surface: {self.api_surface!r}")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def kind(self) -> str:
        return "ollama"

    @property
    def source(self) -> str:
        # Ollama runs on the operator's LAN / localhost; classify as local
        # so CLIs render the right hint. The ABC's str return type is kept
        # for back-compat; ModelDetails uses the narrower Literal.
        return "local"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=True,
            tools_native=True,
            tools_marker=False,
            thinking_channel=bool(self._thinking_models),
            cancel_inflight=True,
            multi_model=True,
            # The Ollama server holds model state, so warming buys the client
            # nothing on the inference path.
            warm_up_required=False,
            # Code-review follow-up (ADR-0022 P3.2): we still want a
            # connectivity probe at engine startup so an unreachable Ollama
            # lands in ``_provider_start_failures`` (degraded mode) instead of
            # 503ing only at first-request time. ``warm_up_on_startup`` is the
            # flag the engine's degraded-mode loop iterates, and it is the
            # correct one here because the probe is a cheap HTTP call — unlike
            # MLC, whose warm_up would load model weights onto the GPU.
            warm_up_on_startup=True,
        )

    def default_model(self) -> Optional[str]:
        if self.default_model_name:
            return self.default_model_name
        if self.models:
            return self.models[0]
        return None

    def list_models(self) -> List[str]:
        # Config-driven list, merged with anything `warm_up` discovered.
        if self._discovered_models:
            seen = set(self.models)
            merged = list(self.models)
            for name in self._discovered_models:
                if name not in seen:
                    merged.append(name)
                    seen.add(name)
            return merged
        return list(self.models)

    def list_model_info(self) -> List[ModelDetails]:
        default = self.default_model()
        names = self.list_models()
        details: List[ModelDetails] = []
        for name in names:
            ctx = (
                self._context_window_cache.get(name)
                or self.context_windows.get(name)
                or 4096
            )
            details.append(
                ModelDetails(
                    id=name,
                    provider_id="_unwrapped_",
                    provider_kind="ollama",
                    source="local",
                    context_window=int(ctx),
                    supports_thinking=name in self._thinking_models,
                    supports_reasoning_effort=False,
                    reasoning_efforts=None,
                    is_default=(default is not None and name == default),
                )
            )
        return details

    def get_context_window(self, model_name: str) -> int:
        cached = self._context_window_cache.get(model_name)
        if cached:
            return int(cached)
        return int(self.context_windows.get(model_name, 4096))

    def unload_model(self, model_name: str) -> bool:
        # Ollama auto-unloads via keep_alive; Tether has no explicit unload
        # for remote providers in Phase 1 (per ADR-0022 / contract §6).
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def warm_up(self, model_name: str) -> None:
        try:
            await self._client.version()
        except RuntimeError:
            # client.version() already wraps httpx errors into RuntimeError
            # with a friendly base_url-aware message; re-raise unchanged so
            # Engine's degraded-mode startup can surface it.
            raise
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Ollama unreachable at {self.base_url}: {exc}"
            ) from exc

        if self.discover_at_startup:
            try:
                raw = await self._client.list_models()
            except Exception as exc:  # pragma: no cover - defensive
                _log.warning("ollama.discover.failed", error=str(exc))
                raw = []
            discovered = [
                m.get("name") or m.get("id")
                for m in raw
                if isinstance(m, dict) and (m.get("name") or m.get("id"))
            ]
            self._discovered_models = [n for n in discovered if isinstance(n, str)]
            if not self._discovered_models:
                _log.warning("ollama.discover.empty", base_url=self.base_url)

        if model_name:
            try:
                info = await self._client.show_model(model_name)
            except RuntimeError as exc:
                _log.info(
                    "ollama.show_model.skipped",
                    model=model_name,
                    error=str(exc),
                )
                info = None
            if isinstance(info, dict):
                ctx = _extract_context_length(info)
                if ctx:
                    self._context_window_cache[model_name] = ctx

    async def aclose(self) -> None:
        # The inner client owns the httpx client only if it created it;
        # in our wiring the provider supplied the shared client so we close
        # it directly here AND call the inner aclose() for forward-compat
        # with clients that may hold extra resources.
        try:
            await self._client.aclose()
        finally:
            try:
                await self._http.aclose()
            except Exception:  # pragma: no cover - defensive
                pass

    # ------------------------------------------------------------------
    # Streaming — legacy stream() (str + list[dict] yields)
    # ------------------------------------------------------------------

    async def stream(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        request_id: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[str | List[Dict[str, Any]], None]:
        del reasoning_effort  # advertised unsupported per ADR-0022
        think = (
            model_name in self._thinking_models and self.api_surface == "native"
        )
        options = dict(self.default_options) if self.default_options else None
        translated_messages = _translate_messages_for_ollama(messages)
        translated_tools = _translate_tools_for_ollama(tools)

        async for event in self._client.stream_chat(
            model=model_name,
            messages=translated_messages,
            tools=translated_tools,
            think=think,
            options=options,
            keep_alive=self.keep_alive,
        ):
            if event.kind == "text" and event.text:
                yield event.text
            elif event.kind == "thinking" and event.text:
                # legacy stream() exposes text channel only; thinking text
                # is forwarded as plain str so callers that don't branch on
                # the v2 typed path still see it.
                yield event.text
            elif event.kind == "tool_call" and event.tool_call is not None:
                yield [event.tool_call]
            elif event.kind == "done":
                return

    # ------------------------------------------------------------------
    # Streaming — v2 typed (ProviderEvent yields)
    # ------------------------------------------------------------------

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
        del max_output_tokens, reasoning_effort  # advertised unsupported
        think = (
            model_name in self._thinking_models and self.api_surface == "native"
        )
        options = dict(self.default_options) if self.default_options else None
        translated_messages = _translate_messages_for_ollama(messages)
        translated_tools = _translate_tools_for_ollama(tools)

        async for event in self._client.stream_chat(
            model=model_name,
            messages=translated_messages,
            tools=translated_tools,
            think=think,
            options=options,
            keep_alive=self.keep_alive,
            cancel_token=cancel_token,
        ):
            if event.kind == "text" and event.text:
                yield ProviderText(text=event.text)
            elif event.kind == "thinking" and event.text:
                yield ProviderThink(text=event.text)
            elif event.kind == "tool_call" and event.tool_call is not None:
                tc = event.tool_call
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments", "{}")
                args_dict: Dict[str, Any]
                if isinstance(raw_args, dict):
                    args_dict = raw_args
                else:
                    try:
                        parsed = json.loads(raw_args) if raw_args else {}
                        args_dict = parsed if isinstance(parsed, dict) else {
                            "_raw": parsed
                        }
                    except json.JSONDecodeError:
                        args_dict = {"_raw": str(raw_args)}
                yield ProviderToolCall(
                    tool_call_id=tc.get("id")
                    or f"call-{uuid.uuid4().hex[:12]}",
                    name=fn.get("name", ""),
                    arguments=args_dict,
                )
            elif event.kind == "done":
                return


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------


def _translate_messages_for_ollama(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Pass through messages, translating Tether's ``tool_result`` rows to Ollama ``tool``."""
    out: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool_result":
            new = dict(msg)
            new["role"] = "tool"
            out.append(new)
        else:
            out.append(msg)
    return out


def _translate_tools_for_ollama(
    tools: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Tether's ``BaseTool.auto_schema`` already emits OpenAI-style tool dicts.

    Ollama accepts the same ``{"type": "function", "function": {...}}``
    envelope. If a tool dict arrives without that envelope, wrap it.
    """
    if not tools:
        return None
    out: List[Dict[str, Any]] = []
    for t in tools:
        if (
            isinstance(t, dict)
            and t.get("type") == "function"
            and "function" in t
        ):
            out.append(t)
            continue
        # Bare-schema shape ({"name": ..., "description": ..., "parameters": ...})
        if isinstance(t, dict) and "name" in t:
            out.append({"type": "function", "function": t})
            continue
        out.append(t)  # passthrough; let Ollama complain if malformed
    return out


def _extract_context_length(show_response: Dict[str, Any]) -> Optional[int]:
    """Best-effort extraction of the model's context length from /api/show."""
    # Ollama /api/show responses vary by model family; the canonical key is
    # ``model_info.<family>.context_length`` (e.g. ``llama.context_length``)
    # but ``parameters`` and top-level ``context_length`` are also seen.
    if not isinstance(show_response, dict):
        return None
    info = show_response.get("model_info")
    if isinstance(info, dict):
        for key, val in info.items():
            if isinstance(key, str) and key.endswith(".context_length"):
                try:
                    return int(val)
                except (TypeError, ValueError):
                    continue
        if isinstance(info.get("context_length"), int):
            return int(info["context_length"])
    if isinstance(show_response.get("context_length"), int):
        return int(show_response["context_length"])
    return None


__all__ = ["OllamaProvider"]
