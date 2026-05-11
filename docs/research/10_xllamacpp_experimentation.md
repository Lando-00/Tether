# XLLamaCPP / Vulkan Experimentation — Findings

**Date:** May 2026
**Hardware:** Surface Pro 11, Snapdragon X Elite (X1E80100), Adreno X1-85 GPU, 16 GB UMA RAM
**Runtime under test:** `xllamacpp 2026.5.9093` (Vulkan wheel) on native ARM64 Python 3.12.10
**Build target reported by the wheel:** `Windows ARM64`, commit `1e5ad35`
**Adreno Vulkan driver:** Qualcomm proprietary, build 9876292e35, 2025-05-30 (Vulkan 1.3.295)
**Baseline for comparison:** Tether's existing MLC-LLM stack on the same machine — `Qwen3-4B-q4f16_1-MLC` via OpenCL → Adreno (40 tok/s prefill, 18-20 tok/s decode, 30 s engine load, 117 s cold warmup).

---

## TL;DR

**Not viable for Tether today.**

- ✅ Install worked first try via the published Vulkan wheel index.
- ✅ Native ARM64 wheel exists (added 3 days before this experiment — see xllamacpp issue #137).
- ✅ Device detection picks up the Adreno X1 GPU cleanly.
- ✅ Engine load time is 10× better than MLC (3 s vs 30 s).
- ❌ **Vulkan / Adreno** backend produces garbage tokens for every prompt size.
  Initial config crashed with `vk::CommandBuffer::end: Incomplete`;
  workaround (`n_ubatch=128`, `flash_attn_type=0`, `n_ctx=2048`) stopped the
  crash but every output is pure token salad like
  `F0%-AC*B>7<-;D,-` for "Reply with the word: OK".
- ❌ **CPU ARM64** backend produces coherent output for short prompts only
  (≤ ~50 tokens). Above that threshold it also produces gibberish, regardless
  of model source (tested both `Qwen/Qwen3-4B-GGUF` and `unsloth/Qwen3-4B-GGUF`),
  regardless of whether we go through `handle_chat_completions` (chat template)
  or `handle_completions` (raw prompt), and regardless of the
  `no_extra_bufts` knob that disables ARM REPACK.

Even when output IS coherent, decode tok/s is roughly half MLC's: 9 vs 18-20.
The wheel is bleeding-edge new (literally 3 days old) and several upstream
Vulkan-Adreno issues are still tracked open / recently closed on llama.cpp.

**Recommendation:** don't build `XLlamaCppProvider` for Tether yet. Watch
xllamacpp + llama.cpp for fixes. Revisit in 3-6 months.

---

## What we tested

### Install path

```powershell
# inside a fresh native ARM64 venv (Python 3.12.10, Windows ARM64)
pip install xllamacpp --index-url https://xorbitsai.github.io/xllamacpp/whl/vulkan
```

pip selected `xllamacpp-2026.5.9093-cp310-abi3-win_arm64.whl` (25.1 MB).
The wheel is `abi3`-stable so it's Python 3.10+ compatible. `BUILD_INFO`
reports `{'build_number': 1, 'commit': '1e5ad35', 'compiler': 'Clang 20.1.6',
'build_target': 'Windows ARM64'}`.

`xlc.get_device_info()` correctly reports both backends:

| name    | description                       | mem free  | fp16 | UMA |
|---------|-----------------------------------|-----------|------|-----|
| Vulkan0 | Qualcomm(R) Adreno(TM) X1-85 GPU  | 12.6 GB   | yes  | yes |
| CPU     | Snapdragon(R) X 12-core X1E80100  | 2.4 GB free of 16 GB | — | — |

`xlc.get_system_info()` confirms `NEON = 1`, `ARM_FMA = 1`,
`LLAMAFILE = 1`, `REPACK = 1` — the native ARM64 ops are linked in.

### Models tried

- `Qwen/Qwen3-4B-GGUF` `Qwen3-4B-Q4_K_M.gguf` (Qwen's official upload,
  ~2.38 GB, metadata says it's an AWQ-derived Q4_K_M).
- `unsloth/Qwen3-4B-GGUF` `Qwen3-4B-Q4_K_M.gguf` (community direct quant,
  ~2.38 GB, metadata says `general.name = "Qwen3-4B"` — distinct lineage).

Both produce identical patterns: clean output for small prompts, gibberish
above ~50-100 prompt tokens.

### Bench prompts (cloned from `scripts/bench_model.py`)

- `tiny` — "Reply with just the word: OK" (9 prompt tokens, max 16 out)
- `medium` — TCP/UDP question (17 prompt tokens, max 200 out)
- `long-context` — 173-token Apple paragraph + question (233 prompt tokens, max 64 out)
- `tool-call` — Tether-style system prompt instructing `<<function_call>>`
  format + user time question (100 prompt tokens, max 128 out)

---

## Results

### Run A — CPU only (`--n-gpu-layers 0`)

Loaded in **2.7 s**, warmup **0.85 s**. Honest token counts (`cache_prompt: False`
to defeat KV reuse, defaults sampler):

| Prompt          | Prompt tok | Resp tok | TTFT (s) | Prefill tok/s | Decode tok/s | Output |
|-----------------|-----------:|---------:|---------:|--------------:|-------------:|:------:|
| `tiny`          | 9          | 2        | 0.68     | 13.3          | 3.1          | ✅ "OK" |
| `medium`        | 17         | 113      | 0.76     | 22.5          | 9.2          | ✅ coherent TCP/UDP explanation |
| `long-context`  | 233        | 65       | 127.89   | 1.8           | 9.4          | ❌ gibberish |
| `tool-call`     | 100        | 77       | 52.51    | 1.9           | 5.5          | ❌ gibberish |

The prefill rate collapse on `long-context` and `tool-call` (22 → 1.8 tok/s)
correlates exactly with the output corruption. Same numbers reproduce on
unsloth's GGUF and with `no_extra_bufts=True` (REPACK disabled).

### Run B — Vulkan / Adreno GPU (`--n-gpu-layers -1`)

First attempt (default `n_ubatch=512`, `n_ctx=4096`, `flash_attn=auto`)
hard-crashed with access violation (`0xc0000005`) inside `xllamacpp.pyd` at
offset `0x239a58` during `llama_kv_cache` allocation; the engine printed
`llama_init_from_model: failed to initialize the context: vk::CommandBuffer::end: Incomplete` then segfaulted.

Workaround config that doesn't crash: `n_ubatch=128`, `n_ctx=2048`,
`flash_attn_type=0`. Loads in 6.5 s, warmup 6.87 s. With that:

| Prompt          | Prompt tok | Resp tok | TTFT (s) | Prefill tok/s | Decode tok/s | Output |
|-----------------|-----------:|---------:|---------:|--------------:|-------------:|:------:|
| `tiny`          | 9          | 17       | 6.42     | 1.4           | 7.5          | ❌ `F0%-AC*B>7<-;D,-` |
| `medium`        | 17         | 201      | 6.75     | 2.5           | 6.9          | ❌ `-87+<#/A"'2/1<>'D#EA1-0+)"A9/C6A)…` |
| `long-context`  | 233        | 65       | 115.71   | 2.0           | 6.6          | ❌ similar |
| `tool-call`     | 100        | 129      | 58.09    | 1.7           | 6.8          | ❌ similar |

GPU prefill (1.4-2.5 tok/s) is actually 5-10× **slower** than CPU prefill on
short prompts. Every response — across all prompt sizes — is uniform-looking
gibberish in the same character set, suggesting a wrong logit/softmax pipeline
rather than a tokenizer or chat-template issue. This is consistent with
known llama.cpp+Adreno Vulkan kernel bugs (see llama.cpp issue #8455 series).

### Run C — Raw `handle_completions` (no chat template)

Sanity check to rule out chat-template rendering:

| Prompt              | Prompt chars | Output      |
|---------------------|--------------|-------------|
| `"The capital of France is"` | 24 | ✅ ` Paris. Which of the following is correct` |
| `"Explain the difference between TCP and UDP:\n"` | 44 | ✅ Coherent 500-char explanation |
| 697-char Apple-founder prompt | 697 | ❌ `G招勝感觉自己我不知道, . Managedersetclr ...` |

The same threshold reproduces on the raw API. The bug is not in chat-template
rendering — it's in the prefill / attention pipeline somewhere above
~50-100 input tokens on this specific Q4_K_M + Windows ARM64 + Qwen3
combination.

---

## What's blocked / not yet tested

- **Phase 4 (tool-calling format probe)** — blocked. With outputs unreliable
  on any prompt that includes a system message or > 50 tokens of context,
  there's no point measuring whether the model emits `<<function_call>>`
  cleanly; we'd just be measuring noise.
- **Phase 5 (x64 Prism control)** — blocked. The bug repros across both
  backends (Vulkan + CPU), so testing the same wheel under Prism wouldn't
  change the verdict.

---

## Comparison to current MLC baseline

| Dimension                | MLC q4f16_1 (Adreno OpenCL)     | xllamacpp Q4_K_M (best case) |
|--------------------------|---------------------------------|------------------------------|
| Wheel install            | Conda + CodeLinaro pre-builds   | One `pip install`, native ARM64 |
| Per-model setup          | Custom DLL compile per model    | Just `wget some.gguf` |
| Engine load time         | ~30 s                           | **~3 s** ✅ |
| Cold first-prefill       | ~117 s                          | < 1 s on small prompts ✅ |
| Decode tok/s             | 18-20                           | 9-10 (when output is valid) |
| Prefill tok/s            | 30-300                          | 1-22 (when output is valid) |
| Output correctness       | ✅ Across all prompt sizes      | ❌ Gibbers above ~50 input tokens |
| Tool-call marker support | ✅ Works via system prompt      | ⏸ Not measurable today |
| GPU acceleration         | ✅ OpenCL on Adreno             | ❌ Vulkan on Adreno is broken |

For Tether's primary use case (multi-turn chat with system prompt + tools),
**MLC is the only viable runtime today**. xllamacpp wins on developer
experience (model selection, install) but doesn't produce usable output on
the workloads that matter.

---

## Why it might get better

xllamacpp issue **#137** ("BLD: Vulkan for Windows ARM64") was filed
2026-04-30 by `iwr-redmond` (the same commenter who replied on the Nexa
issue thread) and closed **2026-05-08** — 3 days before this experiment.
We are testing literally-just-shipped code. The expected timeline for the
Vulkan path to mature is weeks-to-months. The CPU long-prompt corruption
appears to be a separate bug that wasn't tracked yet at the time of writing.

Upstream issues / context worth watching:
- xorbitsai/xllamacpp#137 (closed, but tracks Vulkan ARM64 follow-ups)
- ggml-org/llama.cpp#8455 (older Vulkan Snapdragon X bring-up issues; closed)
- Subsequent llama.cpp Vulkan issues mentioning Adreno

---

## Recommendation for Tether

1. **Do not build `XLlamaCppProvider` skeleton now.** Tether's MLC path is
   working end-to-end (see `scripts/dev/library_smoke.py` +
   `scripts/dev/tool_call_smoke.py` in the Tether repo).
2. **Keep the experiment workspace** at
   `D:\Dev\TetherWorkspace\xllamacpp-experiment\` so we can re-run the bench
   against future xllamacpp versions without re-setting up.
3. **Re-evaluate in 3-6 months.** Re-run `bench.py` against the same prompts
   on the next xllamacpp release; if outputs are clean for the full prompt
   set and decode is at least 15 tok/s, revisit the integration plan.
4. **When (if) we do integrate**, the path is HTTP-via-`Server`-class:
   xllamacpp ships an OpenAI-compatible `Server.handle_chat_completions`,
   so an `XLlamaCppProvider` would look like an HTTP-style adapter that
   speaks Tether's `ModelProvider` interface and forwards messages /
   tools = None / parses streaming chunks for the `<<function_call>>`
   marker — same pattern as the Nexa stub we already have, but actually
   usable.

---

## Reproducibility

Experiment workspace contents (kept under
`D:\Dev\TetherWorkspace\xllamacpp-experiment\`):

```
.venv_arm64/                       # native ARM64 Python 3.12 venv
models/
  Qwen3-4B-Q4_K_M.gguf             # Qwen official (AWQ-derived)
  Qwen3-4B-Q4_K_M-unsloth.gguf     # unsloth direct quant
bench.py                            # the 4-prompt cold/warm harness
diag_raw_vs_chat.py                 # bypass chat template diagnostic
diag_raw_vs_chat.log                # captured outputs
bench_cpu.json / .md                # first CPU bench (broken token counts)
bench_cpu_nc.json / .md             # CPU with cache_prompt=False (clean numbers)
bench_cpu_unsloth.json / .md        # unsloth model, CPU
bench_cpu_norepack.json / .md       # CPU + no_extra_bufts (REPACK off)
bench_gpu_tiny.json / .md           # Vulkan with crash-workaround config
```

To reproduce the CPU bench (the one that produces real numbers):

```powershell
.\.venv_arm64\Scripts\python.exe bench.py `
    --model models\Qwen3-4B-Q4_K_M-unsloth.gguf `
    --label arm64-cpu --n-gpu-layers 0 `
    --out-json bench_cpu.json --out-md bench_cpu.md
```

To reproduce the GPU crash-workaround config:

```powershell
.\.venv_arm64\Scripts\python.exe bench.py `
    --model models\Qwen3-4B-Q4_K_M-unsloth.gguf `
    --label arm64-vulkan --n-ctx 2048 --n-ubatch 128 --skip-cold `
    --out-json bench_gpu.json --out-md bench_gpu.md
```

The harness writes the bench_*.json captures for direct comparison if a
future xllamacpp release improves things.
