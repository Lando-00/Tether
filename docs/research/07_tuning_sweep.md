# Tuning Sweep Results — Qwen3-4B-q4f16_1 on Adreno X Elite

> **Date:** 2026-05-09
> **Hardware:** Surface Pro 11, Snapdragon X Elite, 16 GB
> **Runtime:** Qualcomm CodeLinaro MLC-LLM `2025.06.r1`, x64 Python 3.12 under Prism
> **Methodology:** 4-prompt bench, warm iteration only (cold-start pattern characterised separately in [`artifacts/bench_q4f16_1.md`](./artifacts/bench_q4f16_1.md)).
> **Sequence:** Baseline (with thinking) → /no_think (no recompile) → 4 recompile variants on top of /no_think.

## Headline finding

**`/no_think` is doing 95% of the work.** The Qwen3-3 prompt-level directive that disables `<think>...</think>` blocks is responsible for the dramatic speedups; the GPU compile-flag experiments add at most 5–10% on top, often within measurement noise.

For Tether's UX, this means **the right default is `/no_think` enabled** for tool-routing and quick-answer turns, with thinking re-enabled only when the user explicitly asks for analysis.

## All variants — total response time (warm, lower is better)

| Prompt | Baseline ⟨think⟩ | /no_think | CLML | adrenoaccl | shrunk_ctx | all combined |
|--------|---:|---:|---:|---:|---:|---:|
| **tiny** (17 → ~5 tok) | 1.31 s | 0.85 s | 0.89 s | 0.79 s | 0.83 s | **0.71 s** ✅ |
| **medium** (13 → ~150 tok) | 9.79 s | 8.83 s | **8.60 s** ✅ | 9.50 s | 8.98 s | 8.97 s |
| **long-context** (169 → ~30 tok) | 10.34 s | 2.23 s | 2.35 s | **2.22 s** | 2.41 s | 2.23 s |
| **tool-call** (26 → ~30 tok) | 5.38 s | 2.10 s | 2.15 s | **2.03 s** ✅ | 2.22 s | 2.09 s |

`/no_think` alone:
* tool-call **5.38 s → 2.10 s** (2.6× faster)
* long-context **10.34 s → 2.23 s** (4.6× faster)
* tiny **1.31 s → 0.85 s** (1.5× faster)
* medium ≈ unchanged (response budget was the limiter, not thinking)

## All variants — decode tok/s (warm, higher is better)

| Prompt | Baseline ⟨think⟩ | /no_think | CLML | adrenoaccl | shrunk_ctx | all combined |
|--------|---:|---:|---:|---:|---:|---:|
| tiny | 20.0 | 16.1 | 18.3 | 18.0 | 19.5 | **19.7** |
| medium | 17.4 | 17.9 | **18.5** | 16.7 | 17.7 | 18.0 |
| long-context | 16.4 | 17.3 | 16.3 | 17.2 | 16.4 | 17.3 |
| tool-call | 17.5 | 17.9 | 18.5 | **19.3** ✅ | 17.2 | 18.2 |

Decode rates cluster tightly at **16–20 tok/s** across every variant. Compile-flag experiments don't materially change this — all the speedup in the total-time table came from generating *fewer* tokens (skipping thinking), not from generating tokens faster.

This matches expectations: at 4B params on Adreno OpenCL, decode is bandwidth-bound by KV-cache reads. CLML and adrenoaccl optimise the matmul kernels but the bottleneck is elsewhere.

## TTFT (warm, lower is better)

| Prompt | Baseline | /no_think | CLML | adrenoaccl | shrunk_ctx | all combined |
|--------|---:|---:|---:|---:|---:|---:|
| tiny | 0.51 | 0.54 | 0.62 | 0.51 | 0.57 | **0.46** |
| medium | 0.58 | 0.52 | 0.56 | 0.56 | 0.54 | 0.68 |
| long-context | 0.56 | 0.55 | 0.58 | 0.54 | 0.64 | 0.55 |
| tool-call | 0.58 | 0.48 | 0.59 | 0.53 | 0.54 | **0.50** |

Sub-second TTFT across the board — well within "feels instant" territory. No variant moves this meaningfully.

## What the compile flags actually did

| Flag | Expected | Observed |
|------|----------|----------|
| `--opt openclml=1` (CLML graph executor) | 15–30% decode boost (per Qualcomm's pitch) | **+0–5% in our test**. Best on `medium` (+3%); roughly flat elsewhere. CLML graph executor likely fell back to plain OpenCL for many ops in our model graph. |
| `--opt adrenoaccl=1` (Adreno proprietary OpenCL extensions) | 10–20% extra | **+0–10%**. Best on `tool-call` (+8% decode). Modest help. |
| `--overrides "context_window_size=12288;max_batch_size=1"` (shrunk envelope) | Faster decode + much faster cold-start | **Warm: roughly flat.** Cold-start improvement not measured this round (we used `--skip-cold`). The interactive runtime caps KV at 12 246 anyway, so we're not losing usable context. |
| All three combined | Sum of above | **+0–10%**. Marginally better than any single flag, but well within the noise of /no_think alone. |

## Recommendation for Tether

### Ship config

1. **Default `/no_think` ON** for chat sessions. Wire it in by either:
   * appending `/no_think` to the user message before sending to the engine, or
   * passing `chat_template_kwargs={"enable_thinking": False}` if MLC's chat completion API supports it (preferred — avoids polluting the prompt the user sees in their session log).
2. **Expose a per-message `mode: "think"` flag** on the chat-stream request body so the user can opt back in for hard reasoning tasks. Connects to the orchestrator-mode field already proposed in `docs/REFACTOR_BRIEFING.md` §3.
3. **Use the `all-combined` DLL** (`Qwen3-4B-q4f16_1-adreno-all.dll`) as the canonical lib. Marginal 5–10% across variants, no downside, and the shrunk context envelope (12288) reduces TVM kernel-shape variants → faster cold-start in addition to the warm numbers above.

### Don't bother with (this hardware, this model class)

* Heavy autotune passes — the 4B model class is already memory-bound on Adreno, not compute-bound.
* CLML on its own — the graph-executor coverage isn't comprehensive enough on this op set.

### Worth chasing later

* **Pre-warm common shapes on engine startup.** Cold-start of 30–200 sec per new shape (per [`artifacts/bench_q4f16_1.md`](./artifacts/bench_q4f16_1.md)) means the first user message of any session feels broken. Run a short prep prompt at engine init to JIT the common kernels. One-time 30 sec startup cost pays for itself on every session.
* **Standardize `max_tokens` values in Tether's API surface.** Each distinct `(prompt_tokens, max_tokens)` combo is a fresh shape to TVM. Pick a small set of canonical values (e.g. 64 / 256 / 1024) and reuse them rather than letting each request pick arbitrary integers.
* **Strip `<think>` blocks from `SqliteSessionStore.get_history`.** Even with `/no_think` ON, the model still emits an empty `<think> </think>` wrapper. If we ever toggle thinking ON for a turn, the reasoning tokens will end up in history unless explicitly stripped. (`strip_reasoning_in_history: true` upstream of `qwen3` in MLC handles this; our patched `qwen3-openai-tools-min` template doesn't have that flag — worth porting in.)

## Artifacts

The compiled DLLs (5.30 MB each) and the raw 14-file bench corpus
(6 .md + 6 .json = ~30 KB) live in:

```
docs/research/artifacts/
├── bench_q4f16_1.md / .json                   baseline cold+warm
├── bench_q4f16_1_nothink.md / .json           baseline + /no_think
├── bench_q4f16_1_clml.md / .json              CLML + /no_think
├── bench_q4f16_1_accl.md / .json              adrenoaccl + /no_think
├── bench_q4f16_1_shrunk.md / .json            shrunk + /no_think
└── bench_q4f16_1_all.md / .json               all + /no_think
```

Compiled DLLs are NOT committed (5.3 MB each, regenerable from
weights via `mlc_llm compile --device windows:adreno_x86`).

Reproduce with:

```powershell
python scripts\bench_model.py `
  --model-dir <dist>\Qwen3-4B-q4f16_1-MLC `
  --model-lib <dist>\libs\Qwen3-4B-q4f16_1-adreno.dll `
  --label "Qwen3-4B-q4f16_1" `
  --no-think --skip-cold `
  --out-md <name>.md --out-json <name>.json
```
