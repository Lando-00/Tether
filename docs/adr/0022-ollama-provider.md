# ADR-0022: Ollama Provider — native + OpenAI-compatible HTTP surfaces

- **Status**: Proposed
- **Date**: 2025-07-14 (Phase 13)
- **Related ADRs**:
  - ADR-0021 — multi-provider registry (foundation this ADR lands on; defines `ProviderSpec`, `ModelDetails`, `_provider_start_failures`, registry key semantics)
  - ADR-0016 — MLC isolation rule (confirmed N/A for HTTP providers; `OllamaProvider` has no `mlc_llm` import, no GC or daemon-thread invariants apply)
  - ADR-0003 — GC-disabled daemon-thread shutdown (N/A for HTTP; cited for completeness — `OllamaProvider` uses `httpx.AsyncClient.aclose()`, not daemon threads)
  - ADR-0011 — outbound URL allowlist (`assert_safe_url` applies at construct time; see §8)

---

## 1. Context

Tether runs on a Snapdragon X Elite laptop (on-device MLC inference). Users also own
GPU workstations — in this case, an RTX 4060 Ti 16 GB PC — where Ollama can serve
larger or quantisation-complementary models. Calling that PC over LAN gives Tether
access to a second, GPU-accelerated inference surface without shipping MLC wheels for
x86-64.

Phase 12 (ADR-0021) delivered the multi-provider registry: `Engine` now owns a
`dict[str, ModelProvider]` and routes requests by `provider_id`. Adding a new provider
is a purely **additive** change — no changes to `Engine`, no changes to existing
providers, no changes to the orchestrator's legacy `stream()` path.

Ollama supports two HTTP surfaces:

- **`/api/chat`** — native NDJSON streaming; supports `think: true` for reasoning
  models; `arguments` in tool-call responses are already parsed dicts.
- **`/v1/chat/completions`** — OpenAI-compatible SSE; useful when the caller already
  has OpenAI-shaped tooling or wants to verify compatibility parity.

Both surfaces are supported by a single `OllamaProvider` class, selected at
construction via `api_surface: Literal["native", "openai_compat"]`.

---

## 2. Decision

The following decisions are **locked** for Phase 13 and must not be relitigated by
Phase-2 sub-agents. Each is a single sentence.

1. **API surface**: Same `OllamaProvider` class; `api_surface="native"` (default)
   uses `/api/chat` NDJSON; `api_surface="openai_compat"` uses `/v1/chat/completions`
   SSE. The switch selects the internal client at construction time.
2. **Worktree**: `D:\Dev\Tether-ollama-provider`, branch `feature/ollama-provider`,
   based on `feature/copilot-sdk-provider`.
3. **Live testing**: scaffolded under `@pytest.mark.ollama` + `OLLAMA_BASE_URL` env
   var; skipped in default `pytest -q`; run manually against the GPU PC.
4. **Default config**: new `providers.model_registry` entry is a `ProviderSpec`
   (ADR-0021 shape); constructor args pass through `ProviderSpec.args`.
5. **Model whitelist**: config-driven `models: list[str]`; optional
   `discover_at_startup: true` calls `/api/tags` during `warm_up` and merges results.
6. **Thinking-channel models**: `thinking_models: list[str]` whitelist; provider sets
   `think: true` on `/api/chat` requests only for models in this list.
7. **`reasoning_effort`**: unsupported in Phase 1; `list_model_info` advertises
   `supports_reasoning_effort=False`; provider ignores any non-`None` value forwarded
   by the orchestrator.
8. **Tool-call translation**: provider translates Ollama's `message.tool_calls` into
   the MLC-style `[{"id", "type", "function": {"name", "arguments": str}}]` list that
   `_native_tool_call_from_chunk` already consumes; synthesises a `tool_call_id` via
   `uuid.uuid4().hex[:12]` when Ollama omits `id`.
9. **Capabilities**: `streaming=True, tools_native=True, tools_marker=False,
   thinking_channel=True, cancel_inflight=True, multi_model=True,
   warm_up_required=False`.
10. **Lifecycle**: `httpx.AsyncClient` constructed in `__init__`; closed in `aclose`;
    `warm_up(model_name)` performs `GET /api/version` and optional model-list pull;
    connection failure at `warm_up` raises with the failing URL so `Engine` captures it
    in `_provider_start_failures` (degraded mode per ADR-0021).

---

## 3. HTTP client design

Two internal client classes share a common structural interface defined as a
`Protocol` (not an ABC). **Rationale**: `Protocol` enables structural duck-typing in
tests without requiring inheritance from a base class, keeping test doubles trivial.

```python
# src/tether/providers/ollama/client_base.py
class OllamaClientProtocol(Protocol):
    async def version(self) -> dict: ...
    async def list_models(self) -> list[dict]: ...
    async def show_model(self, model: str) -> dict: ...
    async def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        think: bool = False,
        options: dict | None = None,
        cancel_token: Any | None = None,
    ) -> AsyncIterator[OllamaStreamEvent]: ...
    async def aclose(self) -> None: ...