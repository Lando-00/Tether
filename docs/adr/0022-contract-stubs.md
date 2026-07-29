# ADR-0022: Contract Stubs — Operational Reference for Phase-2 Sub-Agents

> **Purpose**: copy-from-me reference for the three Phase-2 implementation tracks
> (client layer, provider layer, engine wiring + live tests). Everything below is
> binding unless explicitly marked "recommendation" or "open decision".
>
> Parent ADR: [0022-ollama-provider.md](./0022-ollama-provider.md)
> Foundation ADR: [0021-multi-provider-registry.md](./0021-multi-provider-registry.md)

---

## 1. Settings + `pyproject.toml`

### `[ollama]` optional dependency

```toml
# pyproject.toml  [project.optional-dependencies]
ollama = [
    "httpx>=0.27,<0.29",   # matches brave_client.py pin; long-lived streaming client
]
```

`respx` (the httpx mock library) belongs in `[dev]` only — **do not** add it as a
top-level dependency:

```toml
dev = [
    "pytest>=7",
    "pytest-asyncio",
    "respx>=0.21",     # ADD: httpx mock for ollama + brave client tests
    "ruff",
    "mypy",
    "build",
]
```

> Note: `httpx` is already in `[project.dependencies]` (core dep). The `[ollama]`
> extra pins it more tightly. Installing `pip install -e ".[ollama]"` will not
> introduce a new package — it will constrain the already-installed version.

### `@pytest.mark.ollama` marker

Add to `pyproject.toml` `[tool.pytest.ini_options].markers`:

```toml
markers = [
    # ... existing markers ...
    "ollama: tests that require a live Ollama server (OLLAMA_BASE_URL must be set); default-off",
]
```

Update `addopts` to exclude the new marker from default runs:

```toml
addopts = "--strict-markers --strict-config -m \"not network and not hardware and not e2e and not docs and not ollama\""
```

### `config/default.yml` — registry entry (commented out)

```yaml
providers:
  # ── Ollama provider (native NDJSON surface, LAN GPU PC) ──────────────────
  # Uncomment and set base_url to enable. The provider_id key ("ollama_lan")
  # is the routing key callers pass as `provider_id`; it is arbitrary but
  # must be unique within model_registry.
  #
  # model_registry:
  #   ollama_lan:
  #     impl: tether.providers.ollama.provider.OllamaProvider
  #     args:
  #       base_url: "http://192.168.1.50:11434"
  #       api_surface: "native"           # "native" (default) | "openai_compat"
  #       models:
  #         - "qwen3:8b"
  #         - "llama3.1:8b"
  #       default_model: "qwen3:8b"
  #       thinking_models:
  #         - "qwen3:8b"
  #       context_windows:
  #         qwen3:8b: 40960
  #         llama3.1:8b: 131072
  #       discover_at_startup: false
  #       timeout_seconds: 600.0
  #       connect_timeout_seconds: 10.0
  #       keep_alive: null               # null = Ollama default (5 min); -1 = never unload
  #       default_options: {}            # e.g. {temperature: 0.7, top_p: 0.9}
  #
  # ── Ollama provider (OpenAI-compatible SSE surface) ──────────────────────
  #   ollama_compat:
  #     impl: tether.providers.ollama.provider.OllamaProvider
  #     args:
  #       base_url: "http://192.168.1.50:11434"
  #       api_surface: "openai_compat"
  #       api_key_env: null              # set to env-var name if Ollama requires a Bearer token
  #       models:
  #         - "qwen3:8b"
  #       default_model: "qwen3:8b"
  #       thinking_models: []            # must be empty; openai_compat ignores think:true
  #       timeout_seconds: 600.0
  #       connect_timeout_seconds: 10.0
```

---

## 2. `OllamaClientBase` — Protocol definition

**Location**: `src/tether/providers/ollama/client_base.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class OllamaStreamEvent:
    """Normalised event emitted by both client implementations.

    Clients translate provider-specific wire formats into this shape;
    OllamaProvider translates these into ProviderEvent (ProviderText /
    ProviderThink / ProviderToolCall) for the orchestrator.
    """

    kind: Literal["text", "thinking", "tool_call", "done"]
    text: str = ""
    # For kind="tool_call": a single MLC-style tool-call dict ready to be
    # wrapped in a list and passed to _native_tool_call_from_chunk.
    # Shape: {"id": str, "type": "function", "function": {"name": str, "arguments": str}}
    tool_call: dict | None = None
    stop_reason: str | None = None  # for kind="done"


@runtime_checkable
class OllamaClientProtocol(Protocol):
    """Structural interface shared by OllamaNativeClient and OllamaOpenAICompatClient.

    Both clients receive the same httpx.AsyncClient instance from OllamaProvider.
    Neither client calls aclose() on the shared client; OllamaProvider.aclose() owns
    the lifecycle.
    """

    async def version(self) -> dict:
        """GET /api/version → {"version": "0.x.y", ...}"""
        ...

    async def list_models(self) -> list[dict]:
        """GET /api/tags (native) or GET /v1/models (compat).
        Returns list of model dicts, each with at least a "name" key.
        """
        ...

    async def show_model(self, model: str) -> dict:
        """POST /api/show (native) or GET /v1/models/<model> (compat).
        Returns model detail dict (may include context_length, parameters, etc.).
        """
        ...

    async def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        think: bool = False,
        options: dict | None = None,
        cancel_token: Any | None = None,
    ) -> AsyncIterator[OllamaStreamEvent]:
        """Stream a chat completion, yielding OllamaStreamEvent objects.

        cancel_token: any object with a truthy .is_set() method (e.g. CancelToken).
        When is_set() returns True, the iterator should exit cleanly at the next
        chunk boundary without raising.
        """
        ...

    async def aclose(self) -> None:
        """No-op: lifecycle owned by OllamaProvider. Present for Protocol completeness."""
        ...
```

---

## 3. `OllamaNativeClient` — request/response examples

**Location**: `src/tether/providers/ollama/native_client.py`

### Example A — plain chat

**Request** (`POST /api/chat`):
```json
{
  "model": "qwen3:8b",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": true,
  "options": {}
}
```

**NDJSON response lines**:
```
{"model":"qwen3:8b","created_at":"2025-07-14T10:00:00Z","message":{"role":"assistant","content":"Hi"},"done":false}
{"model":"qwen3:8b","created_at":"2025-07-14T10:00:00Z","message":{"role":"assistant","content":" there!"},"done":false}
{"model":"qwen3:8b","created_at":"2025-07-14T10:00:00Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","total_duration":123456}
```

**Emitted `OllamaStreamEvent`s**:
```python
OllamaStreamEvent(kind="text", text="Hi")
OllamaStreamEvent(kind="text", text=" there!")
OllamaStreamEvent(kind="done", stop_reason="stop")
```

---

### Example B — chat with tools and a tool_call response

**Request** (`POST /api/chat`):
```json
{
  "model": "qwen3:8b",
  "messages": [{"role": "user", "content": "What's the weather in London?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "stream": true
}
```

**NDJSON response** (Ollama emits tool_call in final non-streaming message):
```
{"model":"qwen3:8b","created_at":"2025-07-14T10:00:01Z","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"location":"London","unit":"celsius"}}}]},"done":true,"done_reason":"stop"}
```

**Emitted `OllamaStreamEvent`s**:
```python
OllamaStreamEvent(
    kind="tool_call",
    tool_call={
        "id": "a1b2c3d4e5f6",          # synthesised — Ollama omits id
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "London", "unit": "celsius"}',
        },
    },
)
OllamaStreamEvent(kind="done", stop_reason="stop")
```

---

### Example C — chat with `think: true` (thinking + text)

**Request** (`POST /api/chat`):
```json
{
  "model": "qwen3:8b",
  "messages": [{"role": "user", "content": "Explain quantum entanglement."}],
  "stream": true,
  "think": true
}
```

**NDJSON response**:
```
{"model":"qwen3:8b","message":{"role":"assistant","thinking":"Let me reason through this carefully...","content":""},"done":false}
{"model":"qwen3:8b","message":{"role":"assistant","thinking":"","content":"Quantum entanglement is"},"done":false}
{"model":"qwen3:8b","message":{"role":"assistant","thinking":"","content":" a phenomenon where"},"done":false}
{"model":"qwen3:8b","message":{"role":"assistant","thinking":"","content":""},"done":true,"done_reason":"stop"}
```

**Emitted `OllamaStreamEvent`s**:
```python
OllamaStreamEvent(kind="thinking", text="Let me reason through this carefully...")
OllamaStreamEvent(kind="text", text="Quantum entanglement is")
OllamaStreamEvent(kind="text", text=" a phenomenon where")
OllamaStreamEvent(kind="done", stop_reason="stop")
```

> **Implementation note**: a chunk may have both non-empty `thinking` AND non-empty
> `content`. Emit the `thinking` event first, then the `text` event. Both may be
> empty strings in intermediate chunks — skip those silently.

---

## 4. `OllamaOpenAICompatClient` — request/response examples

**Location**: `src/tether/providers/ollama/openai_compat_client.py`

### Example A — plain chat (SSE)

**Request** (`POST /v1/chat/completions`):
```json
{
  "model": "qwen3:8b",
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": true
}
```

**SSE response lines**:
```
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}],"model":"qwen3:8b"}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" there!"}}],"model":"qwen3:8b"}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"model":"qwen3:8b"}

data: [DONE]

```

**Emitted `OllamaStreamEvent`s**:
```python
OllamaStreamEvent(kind="text", text="Hi")
OllamaStreamEvent(kind="text", text=" there!")
OllamaStreamEvent(kind="done", stop_reason="stop")
```

> `[DONE]` sentinel: when `line == "data: [DONE]"`, break the loop and flush any
> pending `_OAIToolCallBuffer`. Do NOT attempt `json.loads("[DONE]")`.

---

### Example B — chat with tools (SSE streaming deltas)

**SSE response lines** (arguments arrive across three events):
```
data: {"id":"chatcmpl-xyz","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]}}],"model":"qwen3:8b"}

data: {"id":"chatcmpl-xyz","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"location\":"}}]}}],"model":"qwen3:8b"}

data: {"id":"chatcmpl-xyz","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"London\"}"}}]}}],"model":"qwen3:8b"}

data: {"id":"chatcmpl-xyz","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"model":"qwen3:8b"}

data: [DONE]

```

**Buffer state after each event**:
```
event 1: buf[0] = {"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}
event 2: buf[0]["function"]["arguments"] = "{\"location\":"
event 3: buf[0]["function"]["arguments"] = "{\"location\":\"London\"}"
finish_reason → flush buffer
```

**Emitted `OllamaStreamEvent`s** (only on flush):
```python
OllamaStreamEvent(
    kind="tool_call",
    tool_call={
        "id": "call_abc",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"location":"London"}',
        },
    },
)
OllamaStreamEvent(kind="done", stop_reason="tool_calls")
```

---

### Example C — no tool support (`thinking` via OpenAI compat — N/A)

The OpenAI-compatible surface does not support `think: true`. If `thinking_models` is
non-empty and `api_surface="openai_compat"`, the provider logs a WARNING at `__init__`
and never sends `think: true` in requests. No special SSE framing for thinking exists
in this path.

---

## 5. `OllamaProvider` constructor signature

**Location**: `src/tether/providers/ollama/provider.py`

```python
from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from typing import Any, Literal

import httpx
import structlog

from tether.core.interfaces import ModelProvider
from tether.providers.types import (
    ModelInfo,
    ProviderCapabilities,
    ProviderEvent,
    ProviderText,
    ProviderThink,
    ProviderToolCall,
)
from tether.security.outbound import assert_safe_url

_log = structlog.get_logger(__name__)


class OllamaProvider(ModelProvider):
    """HTTP provider for Ollama (native NDJSON + OpenAI-compatible SSE surfaces).

    ADR-0022. Constructed by tether.core.factory.load() via ProviderSpec.args.
    Does NOT import mlc_llm — ADR-0016 isolation rule does not apply.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_surface: Literal["native", "openai_compat"] = "native",
        models: list[str] | None = None,
        default_model: str | None = None,
        thinking_models: Sequence[str] = (),
        context_windows: dict[str, int] | None = None,
        api_key_env: str | None = None,
        discover_at_startup: bool = False,
        timeout_seconds: float = 600.0,
        connect_timeout_seconds: float = 10.0,
        keep_alive: str | int | None = None,
        default_options: dict | None = None,
        # Injected by factory.load(); tests can override with a custom checker.
        url_validator: Callable[[str], None] | None = None,
    ) -> None:
        # --- URL validation (loud failure at construct time) ---
        _validator = url_validator or (lambda u: assert_safe_url(u, settings=None))
        try:
            _validator(base_url)
        except Exception as exc:
            raise ValueError(
                f"OllamaProvider: invalid base_url {base_url!r}: {exc}"
            ) from exc

        self._base_url: str = base_url.rstrip("/")
        self._api_surface: Literal["native", "openai_compat"] = api_surface
        self._models: list[str] = list(models or [])
        self._default_model: str | None = default_model
        self._thinking_models: frozenset[str] = frozenset(thinking_models)
        self._context_windows: dict[str, int] = dict(context_windows or {})
        self._api_key_env: str | None = api_key_env
        self._discover_at_startup: bool = discover_at_startup
        self._keep_alive: str | int | None = keep_alive
        self._default_options: dict = dict(default_options or {})

        # Warn early if thinking_models configured for openai_compat
        if self._thinking_models and api_surface == "openai_compat":
            _log.warning(
                "ollama.thinking_models_ignored_for_openai_compat",
                thinking_models=list(self._thinking_models),
            )

        # Shared httpx client — one instance, owned here, closed in aclose()
        timeout = httpx.Timeout(
            connect=connect_timeout_seconds,
            read=timeout_seconds,
            write=30.0,
            pool=5.0,
        )
        headers: dict[str, str] = {}
        if api_key_env:
            import os
            key = os.environ.get(api_key_env, "")
            if key:
                headers["Authorization"] = f"Bearer {key}"

        self._http: httpx.AsyncClient = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers=headers,
        )

        # Internal client (selected by api_surface)
        if api_surface == "native":
            from tether.providers.ollama.native_client import OllamaNativeClient
            self._client = OllamaNativeClient(self._http)
        else:
            from tether.providers.ollama.openai_compat_client import OllamaOpenAICompatClient
            self._client = OllamaOpenAICompatClient(self._http)
```

### Storage layout

| Attribute | Type | Purpose |
|---|---|---|
| `self._base_url` | `str` | Validated base URL, trailing slash stripped |
| `self._api_surface` | `Literal["native","openai_compat"]` | Selects internal client |
| `self._models` | `list[str]` | Config-driven model whitelist; merged with discovered models |
| `self._default_model` | `str \| None` | Explicit default; falls back to `_models[0]` if None |
| `self._thinking_models` | `frozenset[str]` | Models for which `think: true` is sent |
| `self._context_windows` | `dict[str, int]` | Config-driven context window overrides |
| `self._http` | `httpx.AsyncClient` | Shared client, owned by provider |
| `self._client` | `OllamaClientProtocol` | Internal client impl (native or compat) |
| `self._keep_alive` | `str \| int \| None` | Forwarded to `/api/chat` as top-level key |
| `self._default_options` | `dict` | Forwarded to `/api/chat` `options` field |

---

## 6. ABC method implementations (spec)

### `kind` property
Returns `"ollama"`. No logic.

### `source` property  
Returns `f"ollama+{self._api_surface}://{self._base_url}"`. Used by Engine health
reporting and log correlation.

### `capabilities` property
```python
return ProviderCapabilities(
    streaming=True,
    tools_native=True,
    tools_marker=False,
    thinking_channel=bool(self._thinking_models),  # True only if any thinking models configured
    cancel_inflight=True,
    multi_model=True,
    warm_up_required=False,
)
```

### `default_model() -> str`
Return `self._default_model` if set, else `self._models[0]` if the list is non-empty,
else raise `RuntimeError("OllamaProvider has no models configured")`.

### `list_models() -> list[str]`
Return `list(self._models)`. Pure list copy; no network call.

### `list_model_info() -> list[ModelInfo]`
Return one `ModelInfo` per model in `self._models`. Populate `metadata` with:
- `"context_window"`: `self._context_windows.get(name, 4096)` (4096 is the fallback
  default; operators should configure explicit values)
- `"thinking_channel"`: `name in self._thinking_models`
- `"supports_reasoning_effort"`: `False`
- `"api_surface"`: `self._api_surface`
- `"provider_id"`: `"_unwrapped_"` (Engine will replace with registry key per ADR-0021)

### `get_context_window(model_name: str) -> int`
Return `self._context_windows.get(model_name, 4096)`. Does not raise for unknown
models (returns conservative default).

### `unload_model(model_name: str) -> bool`
Return `False`. Ollama auto-unloads models via `keep_alive`; there is no explicit
unload API exposed by Tether for remote providers in Phase 1.

### `warm_up(model_name: str) -> None`
```
1. Attempt GET /api/version via self._client.version().
   On httpx.ConnectError / httpx.ConnectTimeout:
     raise RuntimeError(f"could not reach Ollama at {self._base_url}; is the server running?")
2. If self._discover_at_startup:
     raw_models = await self._client.list_models()
     discovered = [m["name"] for m in raw_models]
     # Merge: union, preserving order (config-listed first)
     existing = set(self._models)
     for name in discovered:
         if name not in existing:
             self._models.append(name)
             existing.add(name)
3. Return None.
```

### `aclose() -> None`
```python
await self._http.aclose()
```

### `stream(model_name, messages, tools, *, request_id) -> AsyncGenerator[str | list[dict], None]`

The legacy `stream()` path consumed by `ChattyAgentOrchestrator`:

```
think = model_name in self._thinking_models and self._api_surface == "native"
options = {**self._default_options}
if self._keep_alive is not None:
    options["keep_alive"] = self._keep_alive  # NOTE: keep_alive is top-level in /api/chat, not in options — adjust in client

async for event in self._client.stream_chat(
    model=model_name,
    messages=messages,
    tools=tools,
    think=think,
    options=options if options else None,
):
    if event.kind == "text":
        yield event.text
    elif event.kind == "thinking":
        yield event.text  # thinking text as plain str on legacy path
    elif event.kind == "tool_call" and event.tool_call is not None:
        yield [event.tool_call]   # list-of-one-dict → _native_tool_call_from_chunk
    elif event.kind == "done":
        return
```

> **Critical**: the `yield [event.tool_call]` wraps the single tool-call dict in a
> list, matching what `_native_tool_call_from_chunk(chunk: List[Dict])` expects.

### `stream_typed(*, model_name, messages, tools, request_id, max_output_tokens, cancel_token) -> AsyncIterator[ProviderEvent]`

The v2 typed path:

```
think = model_name in self._thinking_models and self._api_surface == "native"
options = {**self._default_options}

async for event in self._client.stream_chat(
    model=model_name, messages=messages, tools=tools,
    think=think, options=options or None, cancel_token=cancel_token,
):
    if event.kind == "text" and event.text:
        yield ProviderText(text=event.text)
    elif event.kind == "thinking" and event.text:
        yield ProviderThink(text=event.text)
    elif event.kind == "tool_call" and event.tool_call is not None:
        tc = event.tool_call
        fn = tc.get("function", {})
        raw_args = fn.get("arguments", "{}")
        try:
            args_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args_dict = {"_raw": raw_args}
        yield ProviderToolCall(
            tool_call_id=tc.get("id") or f"call-{uuid.uuid4().hex[:12]}",
            name=fn.get("name", ""),
            arguments=args_dict,
        )
    elif event.kind == "done":
        return
```

---

## 7. Tool-call translation function

**Location**: `src/tether/providers/ollama/translation.py`
(Pure functions; tests can import and call directly without constructing a provider.)

```python
from __future__ import annotations

import json
import uuid


def translate_native_tool_calls(ollama_tool_calls: list[dict]) -> list[dict]:
    """Translate Ollama /api/chat message.tool_calls to MLC-style list-of-dicts.

    Input shape (Ollama native, per-item):
        {"id"?: str, "function": {"name": str, "arguments": dict}}

    Output shape (MLC-style, per-item — consumed by _native_tool_call_from_chunk):
        {"id": str, "type": "function", "function": {"name": str, "arguments": str}}

    Rules:
    - "id" is preserved if present; synthesised as uuid.uuid4().hex[:12] if absent.
    - "arguments" dict is serialised to a JSON string (chatty.py handles both
      str and dict, but str is canonical for the MLC path).
    - "type" is always set to "function" (Ollama omits it).

    >>> translate_native_tool_calls([{"function": {"name": "f", "arguments": {"x": 1}}}])
    [{'id': '...', 'type': 'function', 'function': {'name': 'f', 'arguments': '{"x": 1}'}}]
    """
    result: list[dict] = []
    for tc in ollama_tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        raw_args = fn.get("arguments", {})
        if isinstance(raw_args, dict):
            args_str = json.dumps(raw_args)
        elif isinstance(raw_args, str):
            args_str = raw_args
        else:
            args_str = json.dumps(raw_args)
        tc_id: str = tc.get("id") or uuid.uuid4().hex[:12]
        result.append(
            {
                "id": tc_id,
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            }
        )
    return result


class OAIToolCallBuffer:
    """Buffer for OpenAI-compatible streaming tool-call delta fragments.

    Accumulates per-index fragments and flushes a complete MLC-style list
    when the stream ends or a finish_reason is received.

    Usage:
        buf = OAIToolCallBuffer()
        for chunk in sse_stream:
            if chunk["tool_calls"]:
                buf.feed(chunk["tool_calls"])
        result = buf.flush()   # → list[dict] in MLC-style shape
    """

    def __init__(self) -> None:
        self._buf: dict[int, dict] = {}

    def feed(self, tool_call_deltas: list[dict]) -> None:
        """Accumulate delta fragments.

        Each delta dict may contain: index (int, required), id (str, optional
        — only in first delta for that index), type (str), function.name (str,
        may be partial), function.arguments (str, may be partial).
        """
        for tc in tool_call_deltas:
            idx: int = tc.get("index", 0)
            if idx not in self._buf:
                self._buf[idx] = {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            entry = self._buf[idx]
            if tc.get("id"):
                entry["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                # name arrives in first delta only; concatenation is idempotent
                entry["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                # arguments arrive across multiple deltas — concatenate
                entry["function"]["arguments"] += fn["arguments"]

    def flush(self) -> list[dict]:
        """Return complete MLC-style tool-call list, sorted by index.

        Assigns synthetic IDs (uuid.uuid4().hex[:12]) for entries where
        the server omitted 'id'. Clears the buffer.
        """
        result: list[dict] = []
        for idx in sorted(self._buf):
            entry = self._buf[idx]
            if not entry["id"]:
                entry["id"] = uuid.uuid4().hex[:12]
            result.append(entry)
        self._buf.clear()
        return result

    @property
    def is_empty(self) -> bool:
        return not self._buf
```

---

## 8. Error-code → exception mapping

| Ollama signal | Exception class | Message template | Raised from |
|---|---|---|---|
| `httpx.ConnectError` / `httpx.ConnectTimeout` at `warm_up` | `RuntimeError` | `"could not reach Ollama at {base_url}; is the server running?"` | `warm_up()` |
| HTTP 404 body contains `"model 'X' not found"` | `RuntimeError` | `"model '{name}' not pulled on Ollama; run \`ollama pull {name}\` on the server"` | `stream()` / `stream_typed()` — raised before any chunk |
| `httpx.TimeoutException` during stream | `RuntimeError` | `"Ollama timed out after {timeout_seconds}s; increase timeout_seconds in provider config"` | `stream()` / `stream_typed()` |
| NDJSON line is not valid JSON | _(no exception)_ | WARNING log: `"ollama.ndjson.malformed"` with `line=` | `OllamaNativeClient.stream_chat()` — continue |
| SSE line missing `data:` prefix or not valid JSON | _(no exception)_ | WARNING log: `"ollama.sse.malformed"` with `line=` | `OllamaOpenAICompatClient.stream_chat()` — continue |
| Cancel token set mid-stream | _(no exception)_ | DEBUG log: `"ollama.stream.cancelled"` | `stream_chat()` — clean return |
| `base_url` fails scheme / host check | `ValueError` | `"OllamaProvider: invalid base_url '{url}': {reason}"` | `__init__()` |
| `thinking_models` non-empty + `api_surface="openai_compat"` | _(no exception)_ | WARNING log: `"ollama.thinking_models_ignored_for_openai_compat"` | `__init__()` |
| HTTP 5xx from Ollama during non-streaming call | `RuntimeError` | `"Ollama server error {status} at {url}"` | relevant client method |
| `list_models()` returns empty after `discover_at_startup` | _(no exception)_ | WARNING log: `"ollama.discover.empty"` | `warm_up()` — continue with config list |

---

## 9. CLI / config / yaml examples

Two full `model_registry` entries for `config/default.yml`. Both commented out.

```yaml
providers:
  # The existing single-provider `model:` key is unchanged.
  model:
    impl: tether.providers.mlc.provider.MLCProvider
    args:
      models_root: models
      device: auto
      max_tokens: 1024
      marker_only_tools: true

  # ── ADR-0021 multi-provider registry (Phase 12+) ─────────────────────────
  # Each key is the provider_id; callers pass provider_id= to route requests.
  # Both entries below are commented out; uncomment to enable.
  #
  # model_registry:
  #
  #   # Entry 1: Ollama on LAN GPU PC, native NDJSON surface (default)
  #   ollama_lan:
  #     impl: tether.providers.ollama.provider.OllamaProvider
  #     args:
  #       base_url: "http://192.168.1.50:11434"  # LAN IP of GPU PC
  #       api_surface: "native"                   # NDJSON /api/chat (default)
  #       models:
  #         - "qwen3:8b"                           # must be pulled on GPU PC
  #         - "llama3.1:8b"
  #       default_model: "qwen3:8b"
  #       thinking_models:
  #         - "qwen3:8b"                           # enables think:true for this model
  #       context_windows:
  #         qwen3:8b: 40960
  #         llama3.1:8b: 131072
  #       discover_at_startup: false               # set true to auto-discover pulled models
  #       timeout_seconds: 600.0                   # read timeout (long — LLM streams)
  #       connect_timeout_seconds: 10.0            # connect timeout (LAN should be fast)
  #       keep_alive: null                         # null=Ollama default 5min; -1=never unload
  #       default_options: {}                      # e.g. {temperature: 0.7}
  #
  #   # Entry 2: same Ollama server, OpenAI-compatible SSE surface
  #   ollama_compat:
  #     impl: tether.providers.ollama.provider.OllamaProvider
  #     args:
  #       base_url: "http://192.168.1.50:11434"
  #       api_surface: "openai_compat"              # SSE /v1/chat/completions
  #       models:
  #         - "qwen3:8b"
  #       default_model: "qwen3:8b"
  #       thinking_models: []                       # must be empty for openai_compat
  #       api_key_env: null                         # set to env var name if Ollama requires auth
  #       timeout_seconds: 600.0
  #       connect_timeout_seconds: 10.0
```

---

## 10. Test ordering / dependencies

Phase-2 sub-agent tracks map to the following files. Merge order for Phase 3.1:

| Order | Track | Test file(s) | Source file(s) | Dependency |
|---|---|---|---|---|
| 1 | **A — client layer** | `tests/unit/providers/test_ollama_client.py`, `test_ollama_openai_client.py` | `src/tether/providers/ollama/client_base.py`, `native_client.py`, `openai_compat_client.py`, `translation.py` | None — pure unit tests with `respx` mocks |
| 2 | **B — provider layer** | `tests/unit/providers/test_ollama_provider.py` | `src/tether/providers/ollama/provider.py` | Depends on Track A (client layer must be present) |
| 3 | **C — engine wiring** | `tests/integration/test_ollama_engine_wiring.py` | No new source; wires existing `Engine` + `ProviderSpec` | Depends on Tracks A + B; requires ADR-0021 Engine changes to be merged |
| 4 | **D — live tests** | `tests/hardware/test_ollama_live.py` | No new source | Depends on Tracks A–C; requires `OLLAMA_BASE_URL` env var and GPU PC online |

**Module layout** (all new files in `src/tether/providers/ollama/`):

```
src/tether/providers/ollama/
├── __init__.py
├── client_base.py        # OllamaClientProtocol, OllamaStreamEvent
├── native_client.py      # OllamaNativeClient
├── openai_compat_client.py  # OllamaOpenAICompatClient, OAIToolCallBuffer
├── provider.py           # OllamaProvider (ModelProvider subclass)
└── translation.py        # translate_native_tool_calls, OAIToolCallBuffer (re-exported)
```

> `translation.py` is deliberately separate from the client files so Phase-2 Track B
> tests (`test_ollama_provider.py`) can import and unit-test the translation functions
> in isolation, without instantiating any HTTP client.

**`conftest.py` fixture** (add to `tests/hardware/conftest.py` or top-level `conftest.py`):

```python
import os
import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "ollama: tests requiring a live Ollama server (OLLAMA_BASE_URL env var)",
    )

@pytest.fixture(scope="session")
def ollama_base_url() -> str:
    url = os.environ.get("OLLAMA_BASE_URL", "")
    if not url:
        pytest.skip("OLLAMA_BASE_URL not set; skipping live Ollama tests")
    return url