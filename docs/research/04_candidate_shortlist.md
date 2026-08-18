# Candidate Model Shortlist — Tether on Snapdragon X Elite (16 GB)

> **Status:** Historical MLC model-selection snapshot from 2026. References
> to a "default" below mean the default model within the MLC research fleet,
> not Tether's current default provider (GenieX).
>
> **Audience:** future Tether-team-of-one, day 1 after the refactor lands,
> picking models to download and experiment with.
> **Methodology:** filtered the [`mlc-ai`
> HuggingFace catalog](./02_mlc_ai_hf_catalog.md) by quantization
> (`q4f16_0/1`, `q4f32_1`, `q3f16_1`, `q0f16`), weight size (≤ 9 GB on
> disk), recency, and family diversity. Then sanity-checked memory
> footprint with `scripts/research/estimate_ram.py` so we don't recommend
> something that fits on disk but blows up at runtime.
> **Backed by:** [01 inventory](./01_current_inventory.md),
> [02 ecosystem](./02_adreno_mlc_landscape.md),
> [03 NPU landscape](./03_npu_hexagon_landscape.md).

## How to read this list

Each candidate has a **practical context window** column, which is *not*
necessarily the same as the model's advertised context. KV-cache cost
scales with `2 × num_layers × num_kv_heads × head_dim × ctx × 2 bytes`,
and for models without grouped-query attention (GQA) the cost can rival
or exceed the weight footprint at large contexts. The practical column
is the largest context that keeps **weights + KV + ~256 MB scratch ≤ 9
GB**, computed for `q4f16_*` weights.

A worked example from our existing fleet: Phi-3.5-mini-instruct
*advertises* 40 960 ctx, but its KV cache at that context is **15 GB**
because it has no GQA (32 KV heads × 32 layers). At our 9 GB cap, the
practical ceiling is closer to 16-20k tokens.

The "post-DL action" column tells you whether you can `pip-and-go` from
HuggingFace or whether you'd need to compile an Adreno DLL yourself
(see [02_adreno_mlc_landscape.md](./02_adreno_mlc_landscape.md#mlc_llm-compile-for-adreno-on-this-device)).

---

## Tier S — drop-in replacements / upgrades for current fleet

Each existing model gets a head-to-head challenger. These are the first
ones to actually pull down post-refactor.

### Replacement for **Qwen3-4B-q4f16_0-MLC** (current healthy default)

| Repo | Params | Quant | Weights | Practical ctx | Why try it |
|------|-------:|-------|--------:|--------------:|------------|
| **[mlc-ai/Qwen3-4B-q4f16_1-MLC](https://huggingface.co/mlc-ai/Qwen3-4B-q4f16_1-MLC)** | 4.0 B | q4f16_1 | 2.11 GB | 40 k | Refreshed 2026-04-18; q4f16_1 has slightly better quant scaling than q4f16_0 used today. Same conv template family. |
| [mlc-ai/Qwen3.5-4B-q4f16_1-MLC](https://huggingface.co/mlc-ai/Qwen3.5-4B-q4f16_1-MLC) | 4.0 B | q4f16_1 | 2.20 GB | ~32 k | Brand new (2026-04-22). Direct successor architecture (`qwen3_5` registered in MLC). Carries the GatedDeltaNet hybrid lineage. May need MLC-LLM `>=` the wheel that landed [#3473](https://github.com/mlc-ai/mlc-llm/pull/3473). |
| [mlc-ai/Qwen3-1.7B-q4f16_1-MLC](https://huggingface.co/mlc-ai/Qwen3-1.7B-q4f16_1-MLC) | 1.7 B | q4f16_1 | 0.90 GB | full 40 k easily | Tiny variant — lets us run two engines simultaneously (e.g. one for chat, one as a tool-call router) within budget. |

### Replacement for **Qwen2.5-7B-q4f16_0-MLC** (the fragile one)

The 7B at `prefill_chunk_size=256` is the model that prompted the
shutdown-hang fix. Try a clean 7B variant before throwing more energy
at the workaround.

| Repo | Params | Quant | Weights | Practical ctx | Why try it |
|------|-------:|-------|--------:|--------------:|------------|
| **[mlc-ai/Qwen2.5-7B-Instruct-q4f16_1-MLC](https://huggingface.co/mlc-ai/Qwen2.5-7B-Instruct-q4f16_1-MLC)** | 7.0 B | q4f16_1 | 3.99 GB | 32 k (4 KV heads, GQA wins) | Same family but `q4f16_1` and the official `Instruct` config — almost certainly *not* `prefill_chunk_size=256`. **Verify the new config's prefill chunk before declaring victory.** |
| [mlc-ai/Qwen3-8B-q4f16_1-MLC](https://huggingface.co/mlc-ai/Qwen3-8B-q4f16_1-MLC) | 8.0 B | q4f16_1 | 4.29 GB | 32 k | Step up to Qwen 3 architecture at the 7B class. Same conv template family Tether already understands. |
| [mlc-ai/Hermes-3-Llama-3.1-8B-q4f16_1-MLC](https://huggingface.co/mlc-ai/Hermes-3-Llama-3.1-8B-q4f16_1-MLC) | 8.0 B | q4f16_1 | 4.21 GB | 32 k | Tool-calling specialist. 30k+ downloads. If we want maximum function-calling reliability with the existing `<<function_call>>` parser, this is probably the best 8B. |

### Replacement for **Phi-3.5-mini-instruct-q4f16_1-MLC**

Two reasons to replace: (a) no GQA → KV-cache bottleneck at full ctx,
(b) `use_function_calling: false` in its conv template.

| Repo | Params | Quant | Weights | Practical ctx | Why try it |
|------|-------:|-------|--------:|--------------:|------------|
| **[mlc-ai/Phi-4-mini-instruct-q4f16_1-MLC](https://huggingface.co/mlc-ai/Phi-4-mini-instruct-q4f16_1-MLC)** | ~3.8 B | q4f16_1 | 2.01 GB | check config first | The natural successor. Newer Phi-4 architecture. Confirm the published `mlc-chat-config.json` uses GQA — if it does, this jumps to a default daily driver. |
| [mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC](https://huggingface.co/mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC) | 3.0 B | q4f16_1 | 1.68 GB | ~64 k (good GQA: 24/8) | Excellent KV-cache efficiency. Llama 3.2 family is well-trodden in the MLC ecosystem (60k+ downloads on the 1B, 6k on the 3B). |
| [mlc-ai/Hermes-3-Llama-3.2-3B-q4f16_1-MLC](https://huggingface.co/mlc-ai/Hermes-3-Llama-3.2-3B-q4f16_1-MLC) | 3.0 B | q4f16_1 | 1.68 GB | ~64 k | Hermes 3 finetune of Llama 3.2 3B. Function-calling-tuned at a small footprint — combines the best traits we want for Tether. |

### Replacement / fix for **gemma-3-4b-it-q4f16_1-MLC** (currently broken — no DLL)

Three options:
1. **Compile our own** Adreno DLL for the existing 4B-it weights using
   `mlc_llm compile --device opencl` (Gemma 3 architecture *is* now
   registered in MLC mainline as `gemma3` / `gemma3_text`).
2. **Wait for `mlc-ai/gemma-3-...-MLC` repos** — only one stub exists
   at time of writing.
3. **Drop Gemma 3 for now** and use Gemma 2 instead (mature, multiple
   variants in the catalog):

| Repo | Params | Quant | Weights | Practical ctx | Why try it |
|------|-------:|-------|--------:|--------------:|------------|
| [mlc-ai/gemma-2-2b-it-q4f16_1-MLC](https://huggingface.co/mlc-ai/gemma-2-2b-it-q4f16_1-MLC) | 2.0 B | q4f16_1 | 1.37 GB | full (sliding window 4096) | Battle-tested, 15k+ downloads. Sliding-window attention caps real KV usage. |
| [mlc-ai/gemma-2-9b-it-q4f16_1-MLC](https://huggingface.co/mlc-ai/gemma-2-9b-it-q4f16_1-MLC) | 9.0 B | q4f16_1 | 4.84 GB | full (sliding window) | Larger Gemma 2. Watch for [#3184](https://github.com/mlc-ai/mlc-llm/issues/3184) (sliding-window kernel issues) on Adreno. |

---

## Tier A — new families worth a slot in the experiment lineup

Different architectures / training data than the Qwen/Phi/Llama trio
we know.

| Repo | Params | Quant | Weights | Practical ctx | Why interesting |
|------|-------:|-------|--------:|--------------:|------------------|
| [mlc-ai/Mistral-7B-Instruct-v0.3-q4f16_1-MLC](https://huggingface.co/mlc-ai/Mistral-7B-Instruct-v0.3-q4f16_1-MLC) | 7.0 B | q4f16_1 | 3.80 GB | 32 k (8 KV heads) | The Mistral 7B reference baseline. 8k downloads. Useful as a sanity-check yardstick for any benchmarks. |
| [mlc-ai/Ministral-3-3B-Reasoning-2512-q4f16_1-MLC](https://huggingface.co/mlc-ai/Ministral-3-3B-Reasoning-2512-q4f16_1-MLC) | 3.0 B | q4f16_1 | 1.80 GB | ~32 k | Mistral's "reasoning" small model from Feb 2026. Worth seeing how the reasoning trace interacts with Tether's tool-call parser (likely needs Qwen3-style `<think>` stripping in `SqliteSessionStore.get_history`). |
| [mlc-ai/OLMo-2-1124-7B-Instruct-q4f16_1-MLC](https://huggingface.co/mlc-ai/OLMo-2-1124-7B-Instruct-q4f16_1-MLC) | 7.0 B | q4f16_1 | 3.82 GB | check config | AllenAI OLMo 2. Fully open (data + code + weights). Useful as a reference point for "model behaviour we can fully audit". Brand-new in catalog (2026-04-22). |
| [mlc-ai/DeepSeek-R1-Distill-Qwen-7B-q4f16_1-MLC](https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Qwen-7B-q4f16_1-MLC) | 7.0 B | q4f16_1 | 3.99 GB | 32 k | R1-distilled reasoning model on top of Qwen 7B. The reasoning trace is verbose — interesting for evaluating "how much does Tether's history layer truncate before context blows up". |
| [mlc-ai/DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC](https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC) | 8.0 B | q4f16_1 | 4.21 GB | 32 k (Llama 3 GQA) | R1 distill on Llama 3 8B. Same KV-economic profile as Llama 3.1 8B. |
| [mlc-ai/Qwen2.5-Coder-7B-Instruct-q4f16_1-MLC](https://huggingface.co/mlc-ai/Qwen2.5-Coder-7B-Instruct-q4f16_1-MLC) | 7.0 B | q4f16_1 | 3.99 GB | 32 k | If we ever want to add a "code helper" tool-running mode, this is the obvious dedicated model. |

---

## Tier B — small / very-small (≤ 2 B)

Useful for fast experiments, side-models, agentic routing, or always-on
tool dispatchers that can run alongside a heavier chat model.

| Repo | Params | Quant | Weights | Notes |
|------|-------:|-------|--------:|-------|
| [mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC](https://huggingface.co/mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC) | 1.0 B | q4f16_1 | 0.65 GB | The most-downloaded MLC repo (60k+). Tiny, polite, decent at simple instruction following. |
| [mlc-ai/Qwen3-0.6B-q4f16_1-MLC](https://huggingface.co/mlc-ai/Qwen3-0.6B-q4f16_1-MLC) | 0.6 B | q4f16_1 | 0.31 GB | Smallest Qwen 3. Lower latency than 1.7B by a noticeable margin. Useful as a "draft" model in speculative decoding (subject to [#3290](https://github.com/mlc-ai/mlc-llm/issues/3290) on Adreno). |
| [mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC](https://huggingface.co/mlc-ai/Qwen2.5-1.5B-Instruct-q4f16_1-MLC) | 1.5 B | q4f16_1 | 0.81 GB | Very popular (22k DLs). Worth a spot in the lineup. |
| [mlc-ai/SmolLM2-1.7B-Instruct-q4f16_1-MLC](https://huggingface.co/mlc-ai/SmolLM2-1.7B-Instruct-q4f16_1-MLC)\* | 1.7 B | q4f16_1 | ~1 GB | HuggingFace-trained small model series. Useful diversity vs the Qwen/Llama mainstream. *(check exact repo name in catalog — base "SmolLM2" family entry)* |
| [mlc-ai/OLMo-2-0425-1B-Instruct-q4f16_1-MLC](https://huggingface.co/mlc-ai/OLMo-2-0425-1B-Instruct-q4f16_1-MLC) | 1.0 B | q4f16_1 | 0.78 GB | Tiny OLMo 2. Good "open auditable baseline" companion to the 7B variant. |
| [mlc-ai/Qwen3.5-0.8B-q4f16_1-MLC](https://huggingface.co/mlc-ai/Qwen3.5-0.8B-q4f16_1-MLC) | 0.8 B | q4f16_1 | 0.39 GB | Latest Qwen 3.5 in the smallest size — super cheap to test the architecture. |

---

## Tier C — stretch (≥ 9 B, only at the edge of the budget)

Available because their `q4f16_1` weights stay just under 9 GB. Usable
**only with a small context window** because of KV-cache pressure.

| Repo | Params | Quant | Weights | Realistic ctx in 9 GB | Tradeoff |
|------|-------:|-------|--------:|----------------------:|----------|
| [mlc-ai/Qwen3-14B-q4f16_1-MLC](https://huggingface.co/mlc-ai/Qwen3-14B-q4f16_1-MLC) | 14 B | q4f16_1 | 7.74 GB | ~4-8 k | Strong reasoning at the edge of budget. Will spill at any large context. |
| [mlc-ai/Qwen2.5-14B-Instruct-q4f16_1-MLC](https://huggingface.co/mlc-ai/Qwen2.5-14B-Instruct-q4f16_1-MLC) | 14 B | q4f16_1 | 7.74 GB | ~4-8 k | Mature 14B. Same caveat. |
| [mlc-ai/DeepSeek-R1-Distill-Qwen-14B-q4f16_1-MLC](https://huggingface.co/mlc-ai/DeepSeek-R1-Distill-Qwen-14B-q4f16_1-MLC) | 14 B | q4f16_1 | 7.74 GB | ~4-8 k | Reasoning-heavy. Reasoning eats context, which we don't have much of at this size — possibly counterproductive. |
| [mlc-ai/Qwen3.5-9B-q4f16_1-MLC](https://huggingface.co/mlc-ai/Qwen3.5-9B-q4f16_1-MLC) | 9 B | q4f16_1 | 4.69 GB | ~24 k | Best of the stretch tier — newer architecture, more context room than the 14Bs. **Worth promoting to Tier A** if it actually loads cleanly. |

---

## Quick-start pick (if you want one paragraph of guidance)

After the refactor lands, **download three models** and call it a day
for the first round:

1. **`mlc-ai/Qwen3-4B-q4f16_1-MLC`** — direct upgrade to the current
   default. Same architecture family, refreshed quantization.
2. **`mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC`** *or*
   **`mlc-ai/Hermes-3-Llama-3.2-3B-q4f16_1-MLC`** — the GQA-friendly
   small model that gives Tether breathing room at large contexts and
   (in the Hermes case) explicit tool-calling tuning.
3. **`mlc-ai/Qwen3-8B-q4f16_1-MLC`** — the upgrade for "I want a
   smarter answer" sessions, in place of the fragile Qwen2.5-7B.

That's a 4 B + 3 B + 8 B set that comfortably co-exists in 16 GB so
long as we only load one at a time and pick a reasonable
`context_window_size` per use case.

---

## Things to verify per candidate before downloading

For each candidate above, glance at the published
`mlc-chat-config.json` on HuggingFace and confirm:

- [ ] `prefill_chunk_size ≥ 1024` (avoid the Qwen2.5-7B-style driver
      fragility — see
      [02_adreno_mlc_landscape.md](./02_adreno_mlc_landscape.md#mlc_llm-compile-for-adreno-on-this-device)).
- [ ] `num_key_value_heads < num_attention_heads` (proxy for GQA and
      thus reasonable KV-cache scaling).
- [ ] `conv_template.use_function_calling` is `true` if we want
      Tether's `<<function_call>>` orchestrator to lean on it. (If
      `false`, the system-prompt steering still works but is less
      reliable — like Phi-3.5-mini today.)
- [ ] A pre-compiled adreno DLL is **not** part of the `mlc-ai/...-MLC`
      repo content (these repos ship Vulkan/CUDA libs at most). You
      will have to compile one for Adreno yourself — see the recipe in
      [02_adreno_mlc_landscape.md](./02_adreno_mlc_landscape.md#mlc_llm-compile-for-adreno-on-this-device).
      Plan for this step before assuming "download = run".
- [ ] If using `scripts/research/estimate_ram.py` against the model's
      `mlc-chat-config.json`, that the **practical context window**
      stays comfortably below 9 GB total runtime footprint with at
      least ~1 GB of headroom.
