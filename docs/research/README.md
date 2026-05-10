# Research — Model Candidates for Tether on Snapdragon X Elite

This folder is the output of a research-only pass conducted while
Tether was paused for a major refactor. The goal was to refresh our
view of the MLC-LLM model ecosystem after ~18 months and produce an
**actionable shortlist of models to try once the refactor lands**.

> Hardware target: **Microsoft Surface Pro 11, Snapdragon X Elite,
> 16 GB unified memory.**
> Backend (today): **Adreno X1 GPU via OpenCL** (`*-adreno.dll` libs in
> `models/libs/`). NPU/Hexagon is *not* used today despite older docs
> using the word "NPU" loosely.

## TL;DR (one paragraph)

After the refactor, **download three models** and run them through
Tether before deciding what else to keep:

1. **`mlc-ai/Qwen3-4B-q4f16_1-MLC`** — direct upgrade to the current
   default Qwen3-4B. Same conv template family, refreshed quantization,
   newest MLC build (April 2026).
2. **`mlc-ai/Hermes-3-Llama-3.2-3B-q4f16_1-MLC`** *(or
   `mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC` if no tool-call tuning
   needed)* — strong GQA means very cheap KV cache at long contexts;
   Hermes 3 finetune is explicitly tool-calling tuned.
3. **`mlc-ai/Qwen3-8B-q4f16_1-MLC`** — drop-in upgrade for the
   currently fragile `Qwen2.5-7B-q4f16_0-MLC`. Newer arch, 4.3 GB
   weights, well behaved KV growth.

Stay on the **Adreno OpenCL** path for the first round of post-refactor
experiments. **NPU/Hexagon is now a real, near-term option** thanks to
**NexaSDK** (Qualcomm-endorsed, native Windows ARM64, Python SDK with
streaming + OpenAI-compatible function calling, supports Qwen3 and
multimodal models on the X Elite NPU) — see
`03_npu_hexagon_landscape.md` for the integration shape. Concretely
the natural follow-up is "stand up a `NexaProvider` and run Qwen3
head-to-head on GPU vs NPU."

## What's in this folder

| File | What it is |
|------|------------|
| [`01_current_inventory.md`](./01_current_inventory.md) | Auto-generated snapshot of `models/` — current models, their configs, on-disk size, and which ones have a matching compiled `.dll`. Regenerate via `python scripts/research/inventory_models.py --out docs/research/01_current_inventory.md`. |
| [`01_current_inventory_notes.md`](./01_current_inventory_notes.md) | Companion narrative: per-model operational lessons (the Qwen2.5-7B shutdown-hang story, the broken Gemma 3 setup, the Adreno-vs-NPU naming clarification). |
| [`02_adreno_mlc_landscape.md`](./02_adreno_mlc_landscape.md) | What changed in MLC-LLM since Tether was last updated — new model architectures supported, Adreno / Snapdragon X Elite specific notes, known issues. The "what's mlc-llm doing today" doc. |
| [`02_mlc_ai_hf_catalog.md`](./02_mlc_ai_hf_catalog.md) / [`.json`](./02_mlc_ai_hf_catalog.json) | Auto-generated catalog of **every pre-compiled MLC model under the [mlc-ai](https://huggingface.co/mlc-ai) HF org** that fits in 9 GB on disk. 298 repos, grouped by family, sorted by popularity. Regenerate via `python scripts/research/hf_mlc_catalog.py --max-gb 9 --out-md docs/research/02_mlc_ai_hf_catalog.md --out-json docs/research/02_mlc_ai_hf_catalog.json --verbose`. |
| [`03_npu_hexagon_landscape.md`](./03_npu_hexagon_landscape.md) | Survey of the NPU stacks (NexaSDK, QNN SDK, Genie SDK, ONNX Runtime QNN EP, llama.cpp `ggml-hexagon`). Updated May 2026: **NexaSDK** is now the practical NPU entry point — productized, Qualcomm-endorsed, native Windows ARM64 with a Python SDK that maps cleanly onto Tether's `ModelProvider` interface. NPU follow-up is now concrete, not "someday". |
| [`04_candidate_shortlist.md`](./04_candidate_shortlist.md) | **The headline doc.** Tier S/A/B/C list of candidates with HF links, weight sizes, and notes on practical context windows. Read this if you only have five minutes. |
| [`05_mlc_llm_versioning.md`](./05_mlc_llm_versioning.md) | **Runtime version audit.** What MLC-LLM/TVM version is actually installed, where it came from (Qualcomm's CodeLinaro distribution, `2025.06.r1`), and the recheck/upgrade recipe. Confirms `mlc-venv2` is on the latest published release. |
| [`05_codelinaro_catalog.md`](./05_codelinaro_catalog.md) / [`.json`](./05_codelinaro_catalog.json) | Auto-generated listing of every file in the CodeLinaro `clo-472-adreno-opensource-ai/mlc-llm` repo. Regenerate via `python scripts/research/jfrog_clo_catalog.py --repo clo-472-adreno-opensource-ai --path mlc-llm --out-md docs/research/05_codelinaro_catalog.md --out-json docs/research/05_codelinaro_catalog.json`. |
| [`06_context_strategies.md`](./06_context_strategies.md) | Architecture/orchestration research: **Ralph loops** (Geoffrey Huntley) and **Graph-Reader / Ralph-for-reading** (Steve Hanov, `smhanov/laconic`). Counters Tether's practical-context-window squeeze on small-RAM hardware by replacing the chatty-agent ReAct loop with a Notebook-of-atomic-facts pattern. Sketches a `NotebookOrchestrator` that would slot into the existing `MLCProvider`/`NexaProvider` interface. |
| [`07_tuning_sweep.md`](./07_tuning_sweep.md) | **Live-tested tuning findings on Qwen3-4B-q4f16_1 / Adreno X Elite.** The Qwen3 `/no_think` directive is the highest-leverage win (2.6×–4.6× on tool-call / long-context paths). Compile-flag tweaks (`openclml=1`, `adrenoaccl=1`) only add 0–10%, often within noise. Backed by 6 raw bench reports under [`artifacts/`](./artifacts/). |
| [`08_clml_compilation.md`](./08_clml_compilation.md) | **Why CLML compile flags don't help LLMs in our stack** — pattern matchers in `tvm/relax/backend/adreno/clml.py` require static `tir.IntImm(1)` for the batch dim, but MLC's Relax serving pipeline keeps `batch_size` symbolic. Verified via `--debug-dump`: zero CLML annotations across all phases. Recommendation: drop `--opt openclml=1` / `adrenoaccl=1` from any compile pipeline. |
| [`09_nexa_npu_experimentation.md`](./09_nexa_npu_experimentation.md) | **NexaSDK NPU experiment (May 2026).** Native ARM64 CLI v0.2.73 installs and runs; NPU inference is **blocked upstream** by a Qualcomm/Nexa license-server outage following their March 2026 acquisition. Includes recheck recipe + an HTTP-targeting `NexaProvider` skeleton at [`reference/nexa_provider_skeleton.py`](./reference/nexa_provider_skeleton.py). |

## What's in `scripts/`

Standalone tools that DO touch `models/` and the running `mlc_llm`
runtime (different from `scripts/research/` below — those are
read-only research helpers):

| Script | Purpose |
|--------|---------|
| [`setup_model.py`](../../scripts/setup_model.py) | End-to-end pipeline for adding a new MLC model: download from HF, optionally patch the conv_template (e.g. graft `qwen3-openai-tools-min` onto a freshly-pulled vanilla qwen3 build), compile to an Adreno DLL via `mlc_llm compile --device windows:adreno_x86`, and optionally run a smoke test. Reusable for future model adds. |
| [`bench_model.py`](../../scripts/bench_model.py) | 4-prompt cold/warm bench harness for any compiled MLC model. Has `--no-think` (Qwen3 `/no_think` directive), `--skip-cold` (faster reruns once shapes are JIT'd), `--out-md` / `--out-json` for comparable runs across models. The data behind [`07_tuning_sweep.md`](./07_tuning_sweep.md). |

## What's in `scripts/research/`

Standalone Python scripts (no Tether runtime imports — they work
mid-refactor):

| Script | Purpose |
|--------|---------|
| [`inventory_models.py`](../../scripts/research/inventory_models.py) | Walk `models/`, parse each `mlc-chat-config.json`, cross-reference `models/libs/*.dll`, emit a Markdown report. |
| [`hf_mlc_catalog.py`](../../scripts/research/hf_mlc_catalog.py) | Public HF Hub API → enumerate `mlc-ai/*` repos, parse names, total `.bin` shard sizes, filter by GB cap, emit JSON + Markdown. No auth needed. |
| [`estimate_ram.py`](../../scripts/research/estimate_ram.py) | Estimate runtime footprint = weights + KV cache + scratch. Accepts `--model-config` (reads an `mlc-chat-config.json`) or fully manual params. Has a `--sweep-ctx` mode for "what's the practical context window in N GB". |
| [`jfrog_clo_catalog.py`](../../scripts/research/jfrog_clo_catalog.py) | Walk any Qualcomm CodeLinaro Artifactory repo via the JFrog REST API. Emits per-file size, checksums, timestamps. Used to track new MLC-LLM releases on `clo-472-adreno-opensource-ai`. |

These scripts are intentionally minimal and live outside `tether_service/`
so they're stable across the refactor. They are also the reproducible
recipe — re-running them in 6 months will produce a fresh snapshot of
the ecosystem without relying on this Markdown going stale.

## How the research was scoped (recap)

- **Backend:** primary Adreno GPU / OpenCL, with a side survey of the
  NPU/Hexagon options for a future track.
- **Tool calling:** **not** required as a hard filter. Mix of
  function-calling specialists (Hermes 3, Qwen 3 with the
  `qwen3-openai-tools-min` template) and general chat models.
- **Quantizations covered:** `q4f16_0`, `q4f16_1`, `q4f32_1`,
  `q3f16_1`, `q0f16` (and any other useful ones surfaced by the
  catalog).
- **RAM budget:** ≤ **9 GB** on a 16 GB machine, with KV cache and
  ~256 MB activation reserve included in the budget.
- **Depth:** catalog-only — no downloads, no benchmarks (yet). The
  scripts are in place to make benchmarking easy when the refactor
  unblocks it.

## Re-running this research later

```powershell
# 1. Refresh the local inventory
python scripts\research\inventory_models.py --out docs\research\01_current_inventory.md

# 2. Refresh the HF catalog (takes ~2-3 min for ~400 repos)
$env:PYTHONIOENCODING = 'utf-8'
python scripts\research\hf_mlc_catalog.py `
  --max-gb 9 --verbose `
  --out-json docs\research\02_mlc_ai_hf_catalog.json `
  --out-md   docs\research\02_mlc_ai_hf_catalog.md

# 3. Sanity-check a candidate's RAM footprint by sweeping context sizes
python scripts\research\estimate_ram.py `
  --model-config models\Qwen3-4B-q4f16_0-MLC\mlc-chat-config.json `
  --sweep-ctx 4096,8192,16384,32768,40960

# 4. Recheck whether a newer MLC-LLM build dropped on CodeLinaro
python scripts\research\jfrog_clo_catalog.py `
  --repo clo-472-adreno-opensource-ai `
  --path mlc-llm `
  --out-md   docs\research\05_codelinaro_catalog.md `
  --out-json docs\research\05_codelinaro_catalog.json

# 5. Bench an existing compiled model (warm-only, fast iteration)
python scripts\bench_model.py `
  --model-dir models\Qwen3-4B-q4f16_1-MLC `
  --model-lib models\libs\Qwen3-4B-q4f16_1-adreno.dll `
  --label "Qwen3-4B-q4f16_1 (recheck)" `
  --no-think --skip-cold `
  --out-md docs\research\artifacts\bench_q4f16_1_recheck.md `
  --out-json docs\research\artifacts\bench_q4f16_1_recheck.json

# 6. Add a brand-new candidate model from the shortlist
python scripts\setup_model.py `
  --hf-repo mlc-ai/Hermes-3-Llama-3.2-3B-q4f16_1-MLC `
  --dist models `
  --conv-template-from models\Qwen3-4B-q4f16_0-MLC `
  --verify
```

The hand-written narrative documents (`01_current_inventory_notes.md`,
`02_adreno_mlc_landscape.md`, `03_npu_hexagon_landscape.md`,
`04_candidate_shortlist.md`, `07`-`09`) are *not* auto-regenerated.
Update them manually when the underlying ecosystem changes meaningfully
— start with the auto-generated catalog, then revise the narrative.
