# Hexagon NPU Landscape (May 2026 snapshot)

> **Status:** light-touch survey. The goal is *not* to dive deep — it's
> to know whether an NPU track is worth a *follow-up* project after
> Tether's GPU-on-Adreno path is back online and stable.
>
> **TL;DR (revised May 2026):** **NexaSDK** has emerged as a productized,
> Qualcomm-endorsed NPU runtime for Snapdragon X Elite that ticks every
> box Tether needs: native Windows ARM64, Python SDK with streaming,
> OpenAI-compatible API + function calling, and Day-0 model support on
> the families we care about (Qwen3, Granite 4, multimodal). Tether's
> existing MLC-Adreno-GPU path remains the right *immediate* target
> (it's working and has no migration cost), but the NPU follow-up is
> now a real, near-term option — not a "someday" project.
>
> _Original (March 2026) framing kept below for traceability; updated
> sections are marked **\[updated 2026-05\]**._

## ⭐ Update (May 2026): NexaSDK is the practical NPU path now

Three things changed since this doc was first written:

1. **NexaSDK** ([github.com/NexaAI/nexa-sdk](https://github.com/NexaAI/nexa-sdk)
   — also mirrored at `github.com/qualcomm/nexa-sdk`) shipped a
   **native Windows ARM64 build for Qualcomm NPU** (`nexa-cli_windows_arm64.exe`),
   plus a `pip install nexaai` Python SDK that works on the same
   Windows ARM64 target.
2. **Qualcomm featured NexaSDK three times in their official developer
   blogs** during late 2025: OmniNeural-4B on Hexagon NPU
   ([Sep 2025](https://www.qualcomm.com/developer/blog/2025/09/omnineural-4b-nexaml-qualcomm-hexagon-npu)),
   Day-0 Granite 4.0 NPU support
   ([Oct 2025](https://www.qualcomm.com/developer/blog/2025/10/granite-4-0-to-the-edge-on-device-ai-for-real-world-performance)),
   and Snapdragon mobile integration
   ([Nov 2025](https://www.qualcomm.com/developer/blog/2025/11/nexa-ai-for-android-simple-way-to-bring-on-device-ai-to-smartphones-with-snapdragon)).
   The repo is hosted in Qualcomm's GitHub org. This is effectively the
   blessed "easy button" NPU stack now.
3. The community signal is real: r/Surface thread
   ["Surface X Elite users — your NPU can now run real local AI models
   with NexaSDK"](https://old.reddit.com/r/Surface/comments/1onlrw5/surface_x_elite_users_your_npu_can_now_run_real/)
   describes the workflow as "download → token → `nexa infer`."

### What NexaSDK gets right for Tether

| Tether requirement | NexaSDK status |
|--------------------|----------------|
| Native Windows ARM64 (no Prism emulation) | ✅ `nexa-cli_windows_arm64.exe` ships precompiled |
| Python integration | ✅ `pip install nexaai`, streaming `llm.generate_stream()` |
| Streaming token output | ✅ generator-based, identical shape to Tether's `MLCProvider.stream` |
| OpenAI-compatible API | ✅ explicitly listed in the feature comparison table |
| Function calling | ✅ explicitly listed alongside the OpenAI API |
| Model swap at runtime | ✅ `LLM.from_(model=…)` per request |
| Same model family we already use (Qwen3) | ✅ `NexaAI/Qwen3-*-GGUF` and `NexaAI/Qwen3-VL-*` |
| Multimodal (future-proofing) | ✅ VLM, ASR, OCR, TTS all in one SDK |
| KV-cache reuse / sessions | partial — needs verification at the `GenerationConfig` level |
| Tool calling parsed as a stream | needs verification — depends on how the OpenAI compat layer emits tool calls |

### What still needs care

- **NPU components are dual-licensed.** CPU/GPU pieces are Apache 2.0;
  NPU pieces require a license key. **Free for personal use**
  (Tether's situation — one device, personal project), but commercial
  use requires contacting `hello@nexa.ai`. The README also notes the
  validation service is deprecated and a public token is bundled in
  the repo, but treating that as forever-future-proof would be naive.
- **Different model format.** NexaSDK uses **GGUF** (llama.cpp format)
  + a proprietary **NEXA** format. Existing `dist/<...>-MLC` model
  artifacts don't translate — we'd be re-downloading models from the
  `NexaAI/*-GGUF` HuggingFace org or Nexa's own model hub.
- **Verify session-level KV reuse.** Tether reuses the same engine
  across messages within a session, expecting the model's KV cache to
  persist. NexaSDK's docs are thinner on this than MLC's; needs a
  spike before committing.
- **Verify function-call streaming shape.** Tether's
  `SlidingParser` looks for `<<function_call>>` markers in the stream.
  NexaSDK's "OpenAI-compatible function calling" likely emits OpenAI
  tool-call deltas, which is a *different* protocol than
  Tether's marker-in-text format. The orchestrator and parser would
  need to learn the new shape (or we'd inject the old marker pattern
  via system prompt).

### Suggested follow-up (not for the immediate post-refactor work)

- Build a **`NexaProvider`** implementing
  `tether_service/core/interfaces.py::ModelProvider`, parallel to the
  existing `MLCProvider`.
- Pull `NexaAI/Qwen3-1.7B-GGUF` (CPU) and `NexaAI/OmniNeural-4B`
  (NPU) as the first test models — directly comparable to the Qwen3-4B
  we already run on Adreno.
- Benchmark tok/s, time-to-first-token, RAM, and energy against
  the existing MLC-Adreno path on the same prompts.
- Resolve the function-calling shape — either teach the orchestrator
  to read OpenAI tool-call deltas, or steer NexaSDK into the
  `<<function_call>>` text format via system prompt.

If those four go well, NexaSDK becomes a real second `ModelProvider`
in the refactored codebase; if not, MLC remains the default and we've
documented why.

---

## (Original framing, kept for traceability)

Tether's existing MLC-on-Adreno-OpenCL path remains the right primary
target for the immediate post-refactor work. NPU deployment on
Snapdragon X Elite is now actually feasible (multiple stacks support
it), but **none of them is a drop-in replacement for MLC-LLM's
ergonomics today**. Recommendation: revisit in a future dedicated
project.

> _As of May 2026, the "none of them is a drop-in replacement"
> statement is no longer accurate — see the NexaSDK section above._
>
> _The rest of this doc surveys the lower-level NPU stacks (QNN SDK,
> Genie, ONNX Runtime QNN EP, llama.cpp Hexagon backend), which are
> still the foundation NexaSDK sits on top of, and remain useful
> reference if NexaSDK ever stops being the right answer._

## Why "NPU" was a misnomer in older Tether docs

Several README / instruction snippets in the project mention "NPU
execution". Confirmed from `dist/libs/`: every compiled artifact is
`*-adreno.dll`, which targets the **Adreno X1 GPU via OpenCL**, not the
Hexagon NPU. Adreno and Hexagon are different blocks on the Snapdragon
SoC. Different toolchain, different precision support, different
performance profile.

Going forward, refer to the existing path as "Adreno GPU / OpenCL" and
reserve "NPU" / "Hexagon" / "HTP" for what's described below.

## Snapdragon X Elite NPU at a glance

- Hexagon NPU on Snapdragon X Elite is rated ~45 TOPS (INT8). The same
  block is referred to as **HTP** (Hexagon Tensor Processor) in
  Qualcomm's SDK docs.
- Native NPU precision is **INT8 / INT16 / FP16** (FP16 only on newer
  chips — X Elite supports it). **No INT4 path on the NPU** — quantization
  has to use formats compatible with the QNN SDK (e.g. block-wise INT8 or
  INT4 with scales unpacked to INT8 at load time).
- Unified memory means RAM budget on the NPU is the same 16 GB system
  pool — moving a model to the NPU does not free RAM, it just changes
  who reads it.

## Stacks that can target the Hexagon NPU today

### 1. NexaSDK — \[updated 2026-05\] the practical, productized option

- Repo: <https://github.com/NexaAI/nexa-sdk> (also mirrored under the
  `qualcomm` GitHub org). Documentation: <https://docs.nexa.ai>.
- Sits on top of `ggml` (CPU/GPU paths) and Qualcomm's NPU runtime
  (NPU path). One SDK that fans out across CPU, GPU, and NPU on the
  same device.
- **Windows ARM64 native** — both the CLI (`nexa-cli_windows_arm64.exe`)
  and the Python SDK (`pip install nexaai`) install cleanly on the
  Surface Pro 11 setup we have. No Prism emulation involved.
- **Models supported on the Snapdragon-X-Elite NPU path**: Qwen3 and
  Qwen3-VL (LLM + multimodal), `NexaAI/OmniNeural-4B` (Qualcomm's
  multimodal agent flagship), Granite 4.0, Parakeet (ASR),
  DeepSeek-OCR, Gemma 3n (vision). Bigger LLMs and other modalities
  fall back to CPU/GPU on the same device.
- **Format**: GGUF (llama.cpp lineage) plus a proprietary `.nexa`
  format for the NPU artefacts.
- **API surface**: streaming `llm.generate_stream(prompt, GenerationConfig)`,
  OpenAI-compatible REST API, function calling explicitly listed.
- **License**: dual. Apache 2.0 for CPU/GPU components; NPU
  components free for personal use (with a license token), commercial
  use needs contact with Nexa.
- **Why this matters for Tether**: it's the first NPU stack where the
  integration cost looks like *swap one provider class*, not *write a
  new runtime*. Ranks above QNN SDK / Genie for our use case.

### 2. Qualcomm AI Engine Direct (QNN SDK) — the foundation

- Native C/C++ SDK from Qualcomm. Everything below sits on top of it.
- Supported OS: Android, Linux, Windows.
- Snapdragon X Elite is officially in the supported chipset list.
- Qualcomm distributes [`qai_hub_models`](https://github.com/qualcomm/ai-hub-models)
  (Python package `qai_hub_models`) for model export / profile / inference.
- **Important caveat for our setup**: the package's README explicitly
  states it requires **AMDx64 Python on Windows** — Windows ARM64 Python
  is not supported. So even though we run on Snapdragon X Elite, we'd
  use x64 Python under emulation for the SDK. That's a different Python
  environment from the one we use for MLC today.
- Pre-quantized LLMs distributed by Qualcomm via AI Hub: Llama 2/3
  variants, Phi-2, Phi-3, Mistral, Qwen 2 / 2.5 (some). The list grows
  but lags the open ecosystem by months.

### 3. Qualcomm Genie SDK (LLM-specific layer on QNN)

- Genie is the "make LLMs work" layer Qualcomm built on top of QNN. It
  is the path Qualcomm uses internally for their Copilot+ LLM demos.
- Genie expects models pre-converted to a specific `.bin` artifact + a
  `genie-config.json`. Conversion happens via QNN tools and is *not*
  trivial — requires the QAIRT (Qualcomm AI Runtime) SDK and
  per-architecture work.
- Pros: best NPU performance numbers Qualcomm has shown publicly.
- Cons: Windows-x64 toolchain, Qualcomm-account login required for AI
  Hub Workbench, model conversion is more involved than MLC's
  one-liner.

### 4. ONNX Runtime + QNN Execution Provider (`onnxruntime-qnn`)

- Possibly the most pragmatic NPU path for a Python service like
  Tether.
- Pre-built `onnxruntime-qnn` wheels on PyPI: **Windows ARM64 for
  inference**, **Windows x64 for quantization**.
- Backends inside the QNN EP: `cpu`, `gpu`, `htp` (NPU, the default),
  `saver` (debug). Lots of tunables: `htp_performance_mode`,
  `htp_graph_finalization_optimization_mode`, `vtcm_mb`, etc.
- Workflow: get a HuggingFace model → convert to ONNX → quantize on x64
  via the QNN tools → ship the quantized ONNX to the ARM64 device →
  load with `onnxruntime-qnn`'s `QNNExecutionProvider`.
- Suitable for pure decoder-only LLMs. Some op support gaps still being
  closed (search the ORT issue tracker for "QNN").

### 5. llama.cpp's `ggml-hexagon` backend

- llama.cpp added a Hexagon backend (`ggml-hexagon`) in mid-to-late 2025.
- Open issues we found targeting our exact hardware:
  - [#19871](https://github.com/ggml-org/llama.cpp/issues/19871)
    (Feb 2026) — Windows ARM64, Snapdragon Adreno X1-85, Phi-2 Q4_0:
    `GGML_ASSERT (device) failed`. Status: closed not-planned, but
    instructive about what is and isn't working.
  - [#18139](https://github.com/ggml-org/llama.cpp/issues/18139)
    (Dec 2025) — "Hexagon backend cannot achieve the same performance
    as QNN SDK." Translation: backend works, but is currently slower
    than going direct via QNN.
  - [#18075](https://github.com/ggml-org/llama.cpp/issues/18075)
    (Dec 2025) — `GET_ROWS`, `SET_ROWS`, `FLASH_ATTN` only work on CPU
    in the Hexagon backend, falling back from NPU.
- **Practical signal:** the Hexagon backend in llama.cpp is real and
  shipping, but it's still maturing and does not yet hit the ceiling
  of what QNN can do natively. If we ever want a GGUF-based NPU path,
  this is the one to track.

### 6. MLC-LLM mainline — *not* yet on Hexagon

- [mlc-ai/mlc-llm#1689](https://github.com/mlc-ai/mlc-llm/issues/1689)
  (Jan 2024, open) — "run the LLM model on the Qualcomm Hexagon NPU".
- [mlc-ai/mlc-llm#2673](https://github.com/mlc-ai/mlc-llm/issues/2673)
  (Jul 2024, open) — "Addition of Qualcomm NPU devices inference."
- [mlc-ai/mlc-llm#2688](https://github.com/mlc-ai/mlc-llm/issues/2688)
  (Jul 2024, open) — Qualcomm AI100 hardware question.
- The TVM `USE_HEXAGON` build flag exists, but MLC-LLM's compiled-model
  story for Hexagon is not productised. **Mainline MLC-LLM does not
  ship a Hexagon NPU path today.**

## Tooling vs Tether's needs

Tether's runtime needs are: a **Python-callable, async, streaming**
chat completion API, with **session-level KV-cache reuse**, **tool
calling parsing in the stream**, and a way to **swap models at runtime**.

| Stack                      | Streaming Python API | KV reuse | Multi-model | Effort to integrate |
|----------------------------|----------------------|----------|-------------|---------------------|
| MLC-LLM (Adreno OpenCL)    | ✅ today (what we use) | ✅       | ✅          | none — already done |
| **NexaSDK (NPU/GPU/CPU)**  | ✅ `pip install nexaai`, native ARM64 | needs verification | ✅ `LLM.from_(model=…)` | **low — write a `NexaProvider`** |
| ONNX Runtime + QNN EP      | ✅ via ORT iterator   | partial  | requires per-model session | moderate (rewrite the provider layer) |
| llama.cpp + Hexagon        | ✅ via llama.cpp HTTP / Python bindings | ✅       | ✅          | high (different model format, GGUF, would need to re-add tool-call parsing) |
| QNN/Genie direct           | C++/Python wrapper to write | ✅ in Genie | needs explicit context switching | very high — basically a new provider |
| Qualcomm AI Hub (cloud)    | mostly an offline export tool | n/a   | n/a         | doesn't fit a local server |

The cheapest path to "try the NPU" from Tether's current architecture
is **a `NexaProvider` written against the `nexaai` Python SDK**,
sitting alongside the existing `MLCProvider`. That keeps everything
else (sessions, parsing, orchestration, tools) intact while opening
the NPU door. **\[updated 2026-05\]** ONNX-Runtime-QNN remains a
viable lower-level fallback, but NexaSDK now offers the same surface
with substantially less plumbing.

## Concrete next steps (NOT for this round, but for the follow-up project) \[updated 2026-05\]

If/when we revisit the NPU track:

1. **Install NexaSDK in `mlc-venv2` (or a fresh sibling env).** The
   Python SDK is pip-installable and works alongside the existing
   `mlc_llm` install since they don't share native libs.
2. **Pull `NexaAI/Qwen3-1.7B-GGUF`** as the first head-to-head test
   — directly comparable to the Qwen3-4B already running on Adreno.
3. **Pull `NexaAI/OmniNeural-4B`** to exercise the actual NPU path
   (the smaller Qwen variants run on CPU/GPU; OmniNeural is the
   flagship NPU-only model).
4. **Build a thin `NexaProvider`** implementing
   `tether_service/core/interfaces.py::ModelProvider`, wrapping
   `nexaai.LLM.from_(...)` + `generate_stream(...)`. Match the
   chunk-yielding shape of `MLCProvider.stream`.
5. **Verify session-level KV reuse.** NexaSDK's docs are thin on this
   versus MLC's; confirm whether re-calling `generate_stream` on the
   same `LLM` instance preserves the KV cache, or whether we have to
   manage rolling-prompt context manually.
6. **Resolve function calling.** NexaSDK exposes "OpenAI-compatible
   tool calling" — that's a *different* protocol than Tether's
   `<<function_call>>` text marker. Either:
   - Teach `Orchestrator` + `SlidingParser` to also accept OpenAI
     tool-call deltas; or
   - Steer NexaSDK into emitting the marker form via system prompt
     (less reliable, but no parser changes).
7. **Benchmark** end-to-end latency and tokens/sec against the same
   model on the existing MLC-Adreno path. Track energy too — NPU is
   advertised as 9× more energy-efficient and 2× faster than CPU/GPU
   on the same prompts (per the r/Surface post and Qualcomm's own
   blogs).

If steps 4-7 go well, NexaSDK becomes a real second `ModelProvider` in
the refactored codebase, and we end up with a multi-backend Tether
that can pick GPU or NPU per session. If they don't, MLC-Adreno
remains the default and we've learned exactly where the NPU path
breaks.

## Recommendation \[updated 2026-05\]

**Stay on MLC-Adreno-OpenCL for the immediate post-refactor work** —
that's still the working path, and the model research in
`04_candidate_shortlist.md` is built around it. **But the NPU
follow-up is no longer "someday" — it's now a concrete sub-project
with a clear shape**, and NexaSDK is the entry point.

Specifically: after the new architecture is humming on the GPU and
we've done the first round of model experiments, the natural next
milestone is "stand up `NexaProvider` and run Qwen3 head-to-head on
GPU vs NPU for the same prompts." That's a one-week-ish spike, not
a multi-month research project.
