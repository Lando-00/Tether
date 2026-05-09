# Adreno GPU + MLC-LLM Landscape (May 2026 snapshot)

> **Audience:** future Tether-team-of-one picking up the project after the
> refactor.
> **Goal:** explain what the MLC ecosystem looks like today on Snapdragon
> X Elite hardware, and what changed since Tether was last touched.
> **Companion data:**
> [`02_mlc_ai_hf_catalog.md`](./02_mlc_ai_hf_catalog.md) /
> [`02_mlc_ai_hf_catalog.json`](./02_mlc_ai_hf_catalog.json) (auto-generated
> by `scripts/research/hf_mlc_catalog.py`).

## TL;DR

- **MLC-LLM is still the right tool** for Tether's GPU-execution path on
  Snapdragon X Elite. There is no public alternative that's both
  (a) as ergonomic from Python and (b) ships pre-compiled artifacts.
- Mainline MLC-LLM does **not** officially advertise Windows-on-ARM /
  Adreno X1 as a supported target; the published support matrix lists
  Linux/Win on **Vulkan/CUDA/ROCm** and Android on **OpenCL/Adreno**.
  The Snapdragon-X-Elite-on-Windows path we use today (`*-adreno.dll`)
  is effectively running the *Android Adreno OpenCL* code path on
  Windows-on-ARM. It works, but we are off the official support matrix —
  filed as
  [issue #2617](https://github.com/mlc-ai/mlc-llm/issues/2617) (still open
  since July 2024).
- The set of **supported model architectures** has grown substantially.
  Highlights now in `mlc_llm/model/model.py`:
  Llama 4, Gemma 3, Qwen 3 (incl. MoE), Qwen 3.5 (incl. GatedDeltaNet
  hybrid), OLMo 2, DeepSeek v2 / v3, Ministral 3, Nemotron, MiniCPM, plus
  embedding architectures (`bert`, `qwen3-embedding`) and vision-language
  models (`phi3_v`, `llava`, `gemma3` multimodal).
- The pre-compiled `mlc-ai/*-MLC` HuggingFace organisation has **400+**
  repos. After filtering to ≤ 9 GB on-disk, **298** are realistic
  candidates for Tether on this device. Some of the freshest entries are
  Qwen3.5 (April 2026), OLMo 2 0425/1124, and Ministral 3.
- **KV-cache cost at large contexts is the real budget constraint**, not
  the weight footprint. Phi-3.5-mini at its advertised 40 960 ctx needs
  ~15 GB of KV cache because it has no GQA (32 KV heads × 32 layers). Any
  shortlist must consider a *practical* context window per model, not the
  config's published maximum.

## What changed in MLC-LLM since Tether's last update

### 1. New model architectures landed

Pulled from the architecture registry on `main` (`python/mlc_llm/model/model.py`):

| Architecture | Notes |
|--------------|-------|
| `llama4` | Llama 4 family supported. |
| `gemma3`, `gemma3_text` | Gemma 3 (text-only and the multimodal variant). |
| `qwen3`, `qwen3_moe` | Qwen 3 dense + Qwen 3 MoE. |
| `qwen3_5`, `qwen3_5_text` | Qwen 3.5, including the GatedDeltaNet hybrid added in [#3449](https://github.com/mlc-ai/mlc-llm/pull/3449) (March 2026). |
| `qwen3-embedding` | Embedding model (could power local retrieval/tools later). |
| `olmo`, `olmo2` | AllenAI OLMo / OLMo 2 — fully open models with training data. |
| `ministral3` | Mistral's "small" 3-series (Ministral). |
| `deepseek_v2`, `deepseek_v3` | DeepSeek V2 / V3. |
| `nemotron` | NVIDIA Nemotron family. |
| `phi3_v` | Phi-3 vision-language model. |
| `minicpm` | MiniCPM (small, strong-on-tool-calling family). |
| `bert`, `bert-bge` | Encoder/embedding models — would let Tether do local retrieval without a separate runtime. |

### 2. WebGPU subgroup support

[#3431 (April 2026)](https://github.com/mlc-ai/mlc-llm/pull/3431) added
`--enable-subgroups` for WebGPU. Not directly relevant to Tether's
Adreno path, but signals that the project is actively pushing on browser
deployment.

### 3. Qwen 3 reasoning quirks

[#3484 (April 2026)](https://github.com/mlc-ai/mlc-llm/pull/3484) — "Strip
reasoning in history for Qwen 3 chat" — important if we move to a
reasoning-capable Qwen3/3.5 variant. Tether's
`SqliteSessionStore.get_history` needs to be aware of `<think>...</think>`
content otherwise the model can confuse itself in multi-turn loops.

### 4. CI and tooling refresh

[#3486](https://github.com/mlc-ai/mlc-llm/pull/3486) migrated to GitHub
Actions + ruff. Tooling-level only, mentioned because the lack of
versioned releases (`v0.1.dev0` is still the only release tag) means
"latest" effectively tracks the nightly wheels at
<https://mlc.ai/wheels>. We should pin a specific wheel hash when we
upgrade post-refactor so the inference behaviour is reproducible.

## Snapdragon X Elite specifically

### Status of official support

[Issue #2617](https://github.com/mlc-ai/mlc-llm/issues/2617) (Jul 2024)
explicitly requests Windows-on-ARM Snapdragon X Elite support. **Still
open.** The blocker called out in the thread is Vulkan SDK availability
for ARM64 Windows. Until that lands officially, the realistic paths are:

1. **OpenCL/Adreno via the Android code path** (what we do today —
   `*-adreno.dll`).
2. **Vulkan when it becomes available** — Qualcomm has confirmed native
   Vulkan support in their Adreno X1 driver, but the LunarG SDK
   distribution for ARM64 Windows is the missing piece for a clean MLC
   build.

We should keep the Adreno OpenCL path as the working one and revisit
Vulkan-on-X-Elite when the official ARM64 Vulkan SDK distribution is
unblocked.

### `mlc_llm compile` for Adreno on this device

The compile command pattern Tether used originally still works:

```powershell
# convert weights
mlc_llm convert_weight ./hf_cache/<MODEL> --quantization q4f16_1 \
    -o dist/<MODEL>-q4f16_1-MLC

# generate config (pick conv-template appropriate to the model family)
mlc_llm gen_config ./hf_cache/<MODEL> --quantization q4f16_1 \
    --conv-template <template> --context-window-size <ctx> \
    -o dist/<MODEL>-q4f16_1-MLC

# compile to Adreno OpenCL .dll for Windows-on-ARM
mlc_llm compile dist/<MODEL>-q4f16_1-MLC/mlc-chat-config.json \
    --device opencl \
    -o dist/libs/<MODEL>-q4f16_1-adreno.dll
```

Watch points if we end up doing this for new candidates:

- The Adreno backend has had recurring bugs around small batch / sliding
  window — see [#1936](https://github.com/mlc-ai/mlc-llm/issues/1936)
  (`tvm.dlight.gpu.LowBatchGEMV`) and
  [#3184](https://github.com/mlc-ai/mlc-llm/issues/3184) (sliding window
  not supported by the BeginForward kernel). When something fails to
  compile or run, those threads are usually instructive.
- [#3290 (July 2025)](https://github.com/mlc-ai/mlc-llm/issues/3290) —
  speculative decoding fails on Android OpenCL with
  `CL_INVALID_BUFFER_SIZE`. Avoid speculative-decoding configurations on
  Adreno for now.
- [#2088 (April 2024)](https://github.com/mlc-ai/mlc-llm/issues/2088) —
  `CL_INVALID_WORK_GROUP_SIZE` errors on Adreno can be sensitive to
  `prefill_chunk_size`. This is the exact class of issue we already hit
  with Qwen2.5-7B (`prefill_chunk_size=256` → driver fragility →
  shutdown-hang work in `tether_service/providers/mlc/provider.py`).
  **Default to candidates whose published configs use `prefill_chunk_size
  ≥ 1024`.**

## The `mlc-ai` HuggingFace catalog as it stands today

`scripts/research/hf_mlc_catalog.py` snapshots the org. As of this
research pass:

- **Total repos in `mlc-ai`:** 400.
- **Repos with weight shards ≤ 9 GB on disk (or unknown size):** 298.
- **Top families by downloads:** Llama 3.2, Qwen 2.5, Hermes 3 (Llama 3.1
  finetune), Phi-3-mini, Gemma 2, Mistral, Phi-3.5-mini, Qwen 3,
  Qwen 2.5-Coder, Gemma.
- **Most-recently published families** (last_modified in 2026):
  - Qwen 3 (`mlc-ai/Qwen3-*`) — 17 repos, refreshed 2026-04-18.
  - Qwen 3.5 (`mlc-ai/Qwen3.5-*`) — 9 repos, brand-new (2026-04-22, still
    0 downloads at time of capture).
  - Ministral 3 (`mlc-ai/Ministral-3-*`) — 6 repos.
  - OLMo 2 (`mlc-ai/OLMo-2-1124-*` and `mlc-ai/OLMo-2-0425-*`).
- **Quantization spread per family:** most well-known families publish
  at minimum `q4f16_1` and `q4f32_1`. `q3f16_1` is only available for a
  handful of large families (Llama 3 / 3.1, Hermes 3, Mistral 7B,
  Gemma 2 9B). `q0f16` (unquantized fp16) appears across the small
  models (≤ 3 B params); useful when we want to compare *quantization
  loss* but expensive in memory.
- **Notable absences in the catalog** despite being on the architecture
  registry:
  - Llama 4 — supported in code, no `mlc-ai` upload yet.
  - Gemma 3 (text-only or multimodal) — only one stub repo
    (`mlc-ai/gemma3`) which is much newer than the rest of the catalog.
  - Most Phi-4 variants beyond `Phi-4-mini-instruct`.
  - DeepSeek V2 Lite is the only DeepSeek base in the catalog (besides
    R1 distillates).

These gaps matter because they mean if we want those models we'll have
to compile them ourselves with `mlc_llm compile --device opencl`,
which is doable (the architectures are registered) but adds friction
versus the `pip-and-go` path of Tier-S/A candidates that already have
pre-compiled MLC repos.

## Implications for the post-refactor experiment plan

1. **Stay on Adreno-OpenCL for now.** It's the only path where we have
   working artefacts and a community trail to debug from. Watch
   [#2617](https://github.com/mlc-ai/mlc-llm/issues/2617) and the
   ARM64 Vulkan SDK story; switch when those mature.
2. **Always benchmark with a *practical* context window**, not the model's
   advertised maximum. The KV-cache cost dominates the budget at 32k+
   contexts for any model without GQA.
3. **Prefer candidates with published `mlc-ai/...-MLC` repos** for the
   first round of experiments — zero compile risk, faster iteration.
4. **Watch the `prefill_chunk_size` field** before downloading any new
   model. Anything ≤ 256 is a yellow flag based on the Qwen2.5-7B
   experience.
5. **Pin the MLC-LLM nightly wheel** we use, since there are no real
   releases. Track that pin in a CI-checked file post-refactor.
