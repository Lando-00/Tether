# Briefing for the Refactor Agent

> **Purpose.** A peer-agent handoff. While you've been planning the
> major Tether refactor, a parallel research pass produced six docs in
> `docs/research/` covering hardware, runtime, models, NPU options,
> and orchestration patterns. This briefing condenses what *the
> refactor plan needs to know* from that research, with explicit
> extension points to leave clean.
>
> **Status of the research:** complete; all docs in `docs/research/`
> are linked from `docs/research/README.md`.
> **Status:** Historical planning snapshot. The refactor has landed; use
> [`architecture.md`](./architecture.md) and
> [`refactor/synthesis-2026-05.md`](./refactor/synthesis-2026-05.md) for the
> current architecture and decision digest. Forward-looking language below is
> retained as historical rationale.

## 1. Hard constraints the refactor must respect

These are facts about the environment, not negotiable.

| Constraint | Why it matters |
|------------|----------------|
| Hardware: **Surface Pro 11, Snapdragon X Elite, 16 GB unified RAM** | Hard cap on the model + KV-cache + scratch budget. Treat **9 GB** as the practical model RAM ceiling; the rest is OS, browser, dev tools, working set. |
| Backend: **Adreno X1 GPU via OpenCL** (`*-adreno.dll`), NOT Hexagon NPU | Older docs say "NPU"; that's a misnomer. Don't perpetuate it. NPU is a *future* path (see §4 below). |
| Runtime: **Qualcomm CodeLinaro MLC-LLM `2025.06.r1`**, NOT upstream `mlc-ai/mlc-llm` | Wheels named `mlc_llm_adreno_cpu_clml_2025.06.r1` / `tvm_adreno_cpu_clml_2025.06.r1`. `pip show mlc-llm` returns "not found"; use the full name. |
| Python: **x64 3.12 under Prism emulation** (Qualcomm only ships `cp312-cp312-win_amd64`) | Don't try to switch to native ARM64 Python — `import mlc_llm` will break. `platform.machine()` returns `'ARM64'` even from x64 Python under emulation; don't be misled by it. |
| Conda env name: **`mlc-venv2`** | Activate hooks call `vswhere.exe` and pollute stdout. Activate the environment and use `python`, or invoke `$env:CONDA_PREFIX\python.exe` directly when capturing structured output. Never assume a contributor-specific conda base path. |
| **Practical context windows are bounded by KV-cache RAM, not by `context_window_size`** | A model with 32 KV heads (no GQA) at 40 960 ctx wants ~15 GB of KV alone — over budget. The refactor must not assume `context_window_size` is usable end-to-end. See `scripts/research/estimate_ram.py`. |
| `prefill_chunk_size ≤ 256` is a yellow flag on Adreno | Triggered the OpenCL shutdown hang documented in [`runbooks/shutdown-hang-fix-summary.md`](./runbooks/shutdown-hang-fix-summary.md). Avoid for new candidates. |
| **Never re-enable Python GC in the engine-shutdown thread** | The mitigation in `tether_service/providers/mlc/provider.py` (no GC, daemon thread, bounded `terminate()`, immediate ref clear) is load-bearing. Keep that pattern. |

Source: [`docs/research/05_mlc_llm_versioning.md`](research/05_mlc_llm_versioning.md),
[`docs/research/02_adreno_mlc_landscape.md`](research/02_adreno_mlc_landscape.md),
[`docs/research/04_candidate_shortlist.md`](research/04_candidate_shortlist.md).

## 2. Architectural shape the refactor should aim for

The current `tether_service/` already follows a good MCP-ish layering;
the refactor should **double down** on that and explicitly preserve
two pluggability seams.

### Seam A — `ModelProvider` must be a real, swappable boundary

Today: only `MLCProvider` (and a `dummy/`) implements `core/interfaces.py::ModelProvider`.
After research: a second provider — **`NexaProvider`** for the
Snapdragon X Elite NPU — is now a concrete next-milestone, not a
"someday." The refactor plan should leave the seam clean enough that
adding `NexaProvider` later is a contained change, not a
cross-cutting one.

**What the seam needs to handle:**

- Different **model formats**: MLC dist directories with
  `mlc-chat-config.json` + `params_shard_*.bin` *vs* GGUF / NEXA files
  on disk. The provider — not the orchestrator — should resolve
  "model name" → "loadable artifact". A `Provider.list_models()` /
  `resolve_model(name)` shape decouples this.
- Different **streaming shapes**: MLC yields raw text chunks; NexaSDK
  emits OpenAI-style tool-call deltas. The provider should normalize
  to a single internal event type (`ContentDelta | ToolCallDelta |
  Done | Error`) the orchestrator consumes, OR each provider ships
  its own parser.
- Different **engine lifecycles**: MLC's `AsyncMLCEngine` has a
  bounded-shutdown story; NexaSDK's `LLM.from_(...)` has its own. The
  refactor's lifecycle/cleanup design should not assume the MLC
  shape; encapsulate it inside the provider.
- Different **licensing**: NPU components in NexaSDK require a token
  (`NEXA_TOKEN` env var). The provider should read its own secrets,
  the way `WebSearchTool` reads `BRAVE_API_KEY` today.
- Different **context window math**: see Seam C below.

**Concrete asks for the plan:**

1. Keep `ModelProvider` as the single ABC providers implement. Resist
   the temptation to break out separate ABCs for "streaming model"
   vs "tool-calling model" — both are streaming-with-tools today.
2. Move all MLC-specific knowledge (dist root, libs dir, base-key
   matching, shutdown bounded-terminate) into the `mlc/` provider
   subdir. The factory should not import from `mlc/` except via the
   ABC.
3. Add a placeholder `tether_service/providers/nexa/` directory with
   a stub `NexaProvider` that raises `NotImplementedError`. Wires up
   the import path now so adding it later is a one-class change.

### Seam B — Orchestrator must be a strategy, not a god-object

Today: `protocol/orchestration/orchestrator.py` implements one
strategy — a chatty-agent ReAct loop. After research:
[`docs/research/06_context_strategies.md`](research/06_context_strategies.md)
documents a **Notebook-of-atomic-facts** orchestrator (Ralph-for-reading
/ GraphReader pattern) that gets dramatically more out of small
practical contexts on multi-step tasks.

**What the seam needs to handle:**

- Different **state containers**: chatty-agent stores state in chat
  history; Notebook stores it in a separate fact list outside the
  LLM context. Both need persistence in the session store.
- Different **call patterns**: chatty-agent does
  `stream(messages_with_history)` with the full trace; Notebook does
  several focused `stream(plan_prompt)`, `stream(extract_prompt)`,
  `stream(synthesize_prompt)` calls each with its own minimal
  context.
- Different **stopping conditions**: chatty-agent stops when no more
  tool calls; Notebook stops when notebook reaches "enough facts" or
  the queue empties.
- Different **per-request resource profiles**: Notebook is heavier
  (more LLM calls per user turn) but each call is smaller — useful
  on small-context models, wasteful on long-context cloud models.

**Concrete asks for the plan:**

1. Define an **`Orchestrator` ABC** alongside `ModelProvider`,
   `StreamParser`, and `SessionStore` in `core/interfaces.py`. The
   minimum surface is `async run(session_id, user_message,
   tools) -> AsyncIterator[Event]`, where `Event` is the same
   NDJSON-shaped type the HTTP layer already streams.
2. Make orchestrator **selection per-request**, not just per-deploy.
   A field on the chat-stream request body (or a routing rule based
   on a user-supplied tag like `"mode": "research"`) lets us A/B the
   strategies without redeploying.
3. Generalize `SessionStore` so it can persist arbitrary
   **per-session JSON state**, not just the linear message log. A
   `notebook` would be one consumer; a future "task list", "TODO
   tree", or other agentic state would also use it. Could be as
   simple as a `session_state(session_id, key, value_json)` table.
4. Keep `ChattyAgentOrchestrator` (the renamed current code) as the
   default. **Add** `NotebookOrchestrator` as a second concrete
   class once the ABC is in place — even as a skeleton — so the
   strategy registry has more than one entry from day one. Empty
   strategies have a way of staying empty.

### Seam C — Context-window math should be first-class

The current `MLCProvider.get_context_window(model_name)` returns the
*advertised* `context_window_size`. After research, the *practical*
ceiling is what matters — and it's a function of the model config
**and** the RAM budget.

**Concrete asks for the plan:**

1. Add a method like
   `Provider.get_practical_context_window(model_name, ram_budget_gb)`
   to the ABC. Implementation lives in the provider, but the
   *interface* is shared so the orchestrator can reason about
   budgets generically.
2. The MLC provider's implementation can mirror the math in
   `scripts/research/estimate_ram.py` (weights + KV + scratch). For
   the future `NexaProvider`, NPU runtimes have their own constraints
   — they answer the same question, differently.
3. Surface this in the API: a `/api/v1/models` response should
   include both `advertised_ctx` and `practical_ctx_at_9gb` per
   model. Keeps the operator (us, today) honest about what's loadable.

### Seam D — Function-calling protocol must not be hard-coded

Today: orchestrator + `SlidingParser` looks for the literal
`<<function_call>>` text marker emitted by the system prompt. That
shape is *one* convention. NexaSDK speaks **OpenAI tool-call deltas**.
Future providers may speak something else.

**Concrete asks for the plan:**

1. Each provider declares its own **parser** (already an interface —
   `StreamParser` in `core/interfaces.py`). Don't bind the
   orchestrator to `SlidingParser` directly; resolve it from the
   factory by provider, or from an explicit config key.
2. The orchestrator consumes a normalized `ToolCallDelta` event
   regardless of how the parser produced it. That keeps the loop
   logic stable across providers.
3. The `system.prompt` in `config/default.yml` that bakes
   `<<function_call>> {…}` into the conversation is **MLC-specific**.
   Move it next to the parser, or under a `providers.mlc.system_prompt`
   key, so swapping providers swaps the steering text alongside.

## 3. What to leave as concrete extension points

If the plan does only the items above as ABCs, here are the
*half-built* hooks I'd recommend wiring in even before the matching
implementation lands:

- **`tether_service/providers/nexa/`** — empty package + stub
  `NexaProvider` that raises `NotImplementedError`. Listed in
  `default.yml` under a commented-out `providers.model.impl`. Keeps
  the import path real so adding it later is "fill in the methods,"
  not "add a new module + edit the factory."
- **`tether_service/protocol/orchestration/notebook.py`** — empty
  package + stub `NotebookOrchestrator`. Same reasoning.
- **A `session_state` table or JSON column** in `SqliteSessionStore`
  — even if no orchestrator uses it on day one. Adding a column
  later is fine; designing the migration story for "we forgot to
  reserve any per-session JSON" is annoying.
- **A `mode` field on the chat-stream request body** — defaulted to
  `"chat"`, with `"research"` reserved. Even if `"research"` raises
  `501 Not Implemented` at first, the API contract is set.
- **`get_practical_context_window`** on `ModelProvider` — even if
  v1 just returns `min(advertised, 8192)` as a coarse estimate.
  Wires the concept into the API surface immediately; the math
  refines later.

## 4. The forward-looking integrations these seams unlock

These are *not* day-one work for the refactor. They are *what we
want to be cheap to add later*. Plan accordingly.

| Future feature | Seam(s) that make it cheap | Reference |
|----------------|----------------------------|-----------|
| **NexaSDK NPU provider** — Snapdragon X Elite NPU via `pip install nexaai`. Native Windows ARM64. OpenAI-compat function calling. Supports Qwen3, Qwen3-VL, OmniNeural-4B. | Seam A (`ModelProvider`), Seam D (parser per provider) | [`docs/research/03_npu_hexagon_landscape.md`](research/03_npu_hexagon_landscape.md) §1 |
| **Notebook / Graph-Reader research mode** — small practical contexts handle long multi-step research workflows by storing state in atomic facts outside the LLM context. | Seam B (`Orchestrator` ABC), Seam C (practical-ctx math), `session_state` storage | [`docs/research/06_context_strategies.md`](research/06_context_strategies.md) |
| **New candidate models** (Qwen3-4B-q4f16_1, Hermes-3-Llama-3.2-3B, Qwen3-8B, etc.) — drop-in replacements / upgrades for the current fleet. | None — works on the existing MLC seam, but Seam C is the difference between "loads" and "loads usefully." | [`docs/research/04_candidate_shortlist.md`](research/04_candidate_shortlist.md) |
| **Newer MLC-LLM CodeLinaro release** — when Qualcomm cuts `2025.07.x` (or whatever the next tag is). | Pinning the wheel hash via `direct_url.json`; recompiling all `dist/libs/*-adreno.dll` after upgrade. | [`docs/research/05_mlc_llm_versioning.md`](research/05_mlc_llm_versioning.md) §"When to recheck" |

## 5. Anti-patterns to actively avoid in the refactor

- **Don't switch off the Qualcomm CodeLinaro fork** even if upstream
  `mlc-ai/mlc-llm` has shinier features. Upstream lacks the
  Adreno-OpenCL Windows-on-ARM build path. Stay on the fork; track
  upstream as a watch list.
- **Don't try to make `import mlc_llm` work under native ARM64
  Python.** Wheels are x64-only. Fixing the apparent arch mismatch
  breaks the import.
- **Don't bake the `<<function_call>>` marker into orchestrator
  internals.** It's a steering convention, not a protocol.
- **Don't assume the model's advertised `context_window_size` is
  loadable.** Phi-3.5-mini at 40 960 ctx wants 15 GB of KV. Always
  go through `get_practical_context_window`.
- **Don't introduce `mode="auto"` orchestrator selection in v1.**
  Get the explicit modes working first; route automatically later.
  Auto-routing is a quality knob, not an architecture decision.
- **Don't treat Tier-S/A model picks as load-bearing for the
  refactor.** Models are swappable; the architecture is the thing.
  But do leave room for a 3 B / 4 B / 8 B fleet (one of each size
  class running serially within the 9 GB budget).

## 6. Where to look in the research

| Topic | Doc |
|-------|-----|
| Index + TL;DR | [`docs/research/README.md`](research/README.md) |
| Current model inventory | [`docs/research/01_current_inventory.md`](research/01_current_inventory.md) + [`_notes.md`](research/01_current_inventory_notes.md) |
| What changed in MLC ecosystem | [`docs/research/02_adreno_mlc_landscape.md`](research/02_adreno_mlc_landscape.md) |
| HF `mlc-ai` catalog (auto-generated, 298 repos ≤ 9 GB) | [`docs/research/02_mlc_ai_hf_catalog.md`](research/02_mlc_ai_hf_catalog.md) |
| NPU options + the NexaSDK update | [`docs/research/03_npu_hexagon_landscape.md`](research/03_npu_hexagon_landscape.md) |
| Tiered candidate models | [`docs/research/04_candidate_shortlist.md`](research/04_candidate_shortlist.md) |
| Runtime version audit (we're on `2025.06.r1`) | [`docs/research/05_mlc_llm_versioning.md`](research/05_mlc_llm_versioning.md) |
| Ralph / Graph-Reader / Notebook orchestration | [`docs/research/06_context_strategies.md`](research/06_context_strategies.md) |
| Reusable scripts | `scripts/research/{inventory_models,hf_mlc_catalog,estimate_ram,jfrog_clo_catalog}.py` |

## 7. One-paragraph mission for the refactor plan

Tether is and will remain a **small-RAM, single-user, local-first
inference service on Snapdragon X Elite**. The refactor should
preserve the strengths the current code already has (clean MCP
layering, config-driven DI, NDJSON streaming, sessioned tool
loops) and explicitly leave room for two near-term additions: a
**second `ModelProvider` for the NPU** (NexaSDK), and a
**second `Orchestrator` strategy** for research-style multi-step
tasks (Notebook / Graph-Reader). Anything in the new architecture
that makes either of those a "rewrite" rather than a "new file" is
a planning bug. Anything that hard-codes today's MLC-specific
shapes (the `<<function_call>>` marker, the `dist/`-based model
discovery, the conv-template-tied system prompt, the *advertised*
context window) propagates technical debt past the one chance we
have to clean it up. **This is the moment to set those seams
correctly.**
