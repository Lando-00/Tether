# MLC-LLM Version Audit & Upgrade Path

> **Companion to** the model-research pass (`docs/research/01-04`).
> **Bottom line up front:** Tether's `mlc-venv2` env is **already on the
> latest published Qualcomm CodeLinaro release** (`2025.06.r1`,
> 2025-11-13). No upgrade work is needed today. This doc explains what
> "current" actually means here, and gives a recipe for spotting the
> next release whenever it lands.

## What stack we're actually on

The `*-adreno.dll` libs in `dist/libs/` aren't from upstream
[`mlc-ai/mlc-llm`](https://github.com/mlc-ai/mlc-llm) — they're from
**Qualcomm Technologies' Adreno-tuned distribution of MLC-LLM**,
hosted on Qualcomm's CodeLinaro Artifactory under
[`clo-472-adreno-opensource-ai/mlc-llm/`](https://codelinaro.jfrog.io/ui/native/clo-472-adreno-opensource-ai/mlc-llm/).
The Feb 2025 Qualcomm developer blog post
["Harnessing Qualcomm Adreno GPU for Generative AI: Open-Source
Approach"](https://www.qualcomm.com/developer/blog/2025/02/harnessing-qualcomm-adreno-gpu-generative-ai-open-source-approach)
is the entry-point document; it walks through host setup, SDK install,
and per-model compilation. Per the blog:

> Qualcomm Technologies has improved the performance of OpenCL backend
> for Adreno GPU significantly. Majority of these enhancements are
> contributed back to community already. Through TVM, we offer an
> open-source solution for vision (or non-generative AI) models with
> incredible performance. Now, we are proud to reiterate Qualcomm
> Technologies' commitment and dedication towards open source-based
> solution for generative AI models on Adreno GPU through MLC/TVM.

So the value-add of the Qualcomm fork is:

1. **Tested wheels** for Snapdragon X Elite / X Plus on Windows and
   Snapdragon 8 Gen 1/2/3/Elite on Android-or-Linux.
2. **CLML acceleration** — the
   `mlc_llm_adreno_cpu_clml_*` / `tvm_adreno_cpu_clml_*` variants
   bundle Qualcomm's Compute Library for Machine Learning (CLML),
   which speeds up the OpenCL backend on Adreno well beyond stock
   OpenCL.
3. **Reference build artefacts** for `mlc_llm-utils-*` (CLI binaries
   and runtime DLLs) per platform.
4. **Public documentation bundle** (`mlc-llm-docs-2025.06.r1.zip`).

## Current `mlc-venv2` snapshot

Everything below was captured by invoking the env's Python directly
(`$env:CONDA_PREFIX\python.exe` after activating `mlc-venv2`) — the env's
activate hooks call `vswhere.exe` and pollute stdout, so `conda run`
is not safe for structured capture.

### Python and platform

| Field | Value |
|-------|-------|
| `python --version` | `Python 3.12.11` |
| `platform.platform()` | `Windows-11-10.0.26200-SP0` |
| `platform.machine()` | `ARM64` (the *hardware* — Surface Pro 11) |
| `platform.architecture()` | `('64bit', 'WindowsPE')` |

### Installed Qualcomm CodeLinaro wheels

| Package | Version | Wheel tag | Origin |
|---------|---------|-----------|--------|
| `mlc_llm_adreno_cpu_clml_2025.06.r1` | `0.1.dev0` | `cp312-cp312-win_amd64` | local file under `%USERPROFILE%\Downloads\` (SHA-256 `c1ad7c79…`) |
| `tvm_adreno_cpu_clml_2025.06.r1`     | `0.20.dev0` | `cp312-cp312-win_amd64` | bundled with the same release |

Site-packages directory mtimes show install on **2025-09-28**.

### TVM build fingerprint (the highest-signal version marker)

`tvm.support.libinfo()` reports:

| Key | Value |
|-----|-------|
| `GIT_COMMIT_HASH` | `0bf585f6d826ef9aad1090053260b0b34bb2fe6d` |
| `GIT_COMMIT_TIME` | `2025-05-22 17:18:40 +0530` |
| `LLVM_VERSION` | `20.1.5` |
| `USE_OPENCL` | `<upstream-ci-root>/CLML-SDK/clml-sdk-5-win-x86` |
| `USE_CLML` | `<upstream-ci-root>/CLML-SDK/clml-sdk-5-win-x86` |
| `USE_CLML_GRAPH_EXECUTOR` | `<upstream-ci-root>/CLML-SDK/clml-sdk-5-win-x86` |
| `USE_VULKAN` / `USE_HEXAGON` / `USE_METAL` / `USE_CUDA` | `OFF` |
| `TVM_CXX_COMPILER_PATH` | MSVC 14.40.33807 (VS 2022 Pro), x64 host |

This confirms:

- TVM was cut at commit `0bf585f6` from **22 May 2025** — the snapshot
  Qualcomm took to build the `2025.06.r1` artefacts.
- Backend compilation pulled in **CLML SDK 5** on Windows x86 (x64) —
  that's the secret sauce making the Adreno OpenCL path fast.
- Vulkan and Hexagon (NPU) backends are deliberately *off*. Confirms
  the
  [`03_npu_hexagon_landscape.md`](./03_npu_hexagon_landscape.md)
  finding: this stack does not target the NPU.

### Why it's an x64 wheel on an ARM64 device

Surprising at first glance, but consistent:

- Qualcomm's CodeLinaro release ships **only `cp312-cp312-win_amd64`
  Python wheels** for Windows. There is no `win_arm64` wheel for the
  Python `mlc_llm` package.
- So Tether's Python is **x64 running under Prism (Windows-on-ARM
  emulation)**, not native ARM64 Python.
- The `mlc_llm-utils-win-arm64-clml-2025.06.r1.zip` archive in the
  catalog (below) provides native ARM64 *CLI binaries* (`mlc_cli_chat.exe`
  + DLLs), but Tether uses the Python integration — the CLI utilities
  are not in our path.

This is by design (per the Feb 2025 blog), not a misconfiguration.
Don't try to switch to native ARM64 Python — it would break the
`mlc_llm` import.

## CodeLinaro `2025.06.r1` — full release contents

Captured by `scripts/research/jfrog_clo_catalog.py` (raw data in
[`05_codelinaro_catalog.json`](./05_codelinaro_catalog.json), table in
[`05_codelinaro_catalog.md`](./05_codelinaro_catalog.md)).

| File | Purpose | Size | What we use |
|------|---------|-----:|:-----------:|
| `mlc-llm-docs-2025.06.r1.zip` | Documentation bundle | 7.3 MB | optional |
| `mlc_llm-utils-linux-arm64-2025.06.r1.tar.bz2` | Linux ARM64 CLI binaries | 42.5 MB | — |
| `mlc_llm-utils-win-arm64-2025.06.r1.zip` | Windows ARM64 CLI binaries (no CLML) | 6.7 MB | — |
| `mlc_llm-utils-win-arm64-clml-2025.06.r1.zip` | Windows ARM64 CLI binaries (with CLML) | 6.8 MB | — |
| `mlc_llm-utils-win-x86-2025.06.r1.zip` | Windows x64 CLI binaries (no CLML) | 7.0 MB | — |
| `mlc_llm-utils-win-x86-clml-2025.06.r1.zip` | Windows x64 CLI binaries (with CLML) | 7.1 MB | — |
| `mlc_llm_adreno_cpu_2025.06.r1-0.1.dev0-cp312-cp312-manylinux_2_28_x86_64.whl` | Linux x64, no CLML | 62.4 MB | — |
| `mlc_llm_adreno_cpu_clml_2025_06_r1-0.1.dev0-cp312-cp312-win_amd64.whl` | **Windows x64, with CLML** | 5.4 MB | ✅ installed |
| `mlc_llm_adreno_cuda_2025.06.r1-0.1.dev0-cp312-cp312-manylinux_2_28_x86_64.whl` | Linux x64, CUDA | 62.4 MB | — |
| `tvm_adreno_cpu_clml_2025_06_r1-0.20.dev0-cp312-cp312-win_amd64.whl` | **TVM Windows x64, with CLML** | 41.9 MB | ✅ installed |
| `tvm_adreno_mlc_cpu_2025.06.r1-0.20.dev0-cp312-cp312-manylinux_2_28_x86_64.whl` | TVM Linux x64 | 164.3 MB | — |
| `tvm_adreno_mlc_cuda_2025.06.r1-0.20.dev0-cp312-cp312-manylinux_2_28_x86_64.whl` | TVM Linux x64 + CUDA | 164.4 MB | — |

All 12 files were uploaded together on **2025-11-13**.

### Diff: local install vs CodeLinaro

| What | Local | Catalog | Match |
|------|-------|---------|:-----:|
| `mlc_llm` wheel SHA-256 | `c1ad7c795e782ec876b96f66439206456cd0c8ae9e5d8d6a587b40275b08478e` | `c1ad7c795e78…` | ✅ |
| `mlc_llm` package name | `mlc_llm_adreno_cpu_clml_2025.06.r1` | (from same wheel) | ✅ |
| `tvm` package name | `tvm_adreno_cpu_clml_2025.06.r1` | (from same wheel) | ✅ |
| Python tag | `cp312-cp312-win_amd64` | only Windows wheel offered | ✅ |
| TVM commit | `0bf585f6 / 2025-05-22` | (cut for `2025.06.r1`) | ✅ |

**Conclusion:** `mlc-venv2` is bit-for-bit on the latest CodeLinaro
release. There is no newer release to upgrade to.

## What's the Qualcomm-recommended setup?

The Feb 2025 blog (slightly out of date with respect to `2025.06.r1`
filenames, but the workflow is unchanged):

```powershell
# Host preparation (per Qualcomm blog — Windows path)
conda create -n mlc-venv -c conda-forge `
  "llvmdev=15" "cmake>=3.24" git rust numpy==1.26.4 decorator psutil `
  typing_extensions scipy attrs git-lfs python=3.12 onnx clang_win-64
conda activate mlc-venv

pip install torch==2.2.0 torchvision==0.18.0 torchaudio==2.3.0

# SDK installation — newer release than the blog text uses
pip install tvm_adreno_cpu_clml_2025_06_r1-0.20.dev0-cp312-cp312-win_amd64.whl
pip install mlc_llm_adreno_cpu_clml_2025_06_r1-0.1.dev0-cp312-cp312-win_amd64.whl

# Verify
python -c "import tvm; print(tvm.__path__)"
python -c "import mlc_llm; print(mlc_llm.__path__)"
```

Tether's existing `mlc-venv2` was set up with this same recipe (env
name diverges only in the digit — `mlc-venv2` vs `mlc-venv`). The
torch version installed in our env is newer (`torch 2.8.0`,
`numpy 1.26.4` matched), but the core mlc-llm/tvm pair is the
release-pinned x64+CLML wheel. **No deviation that needs correcting.**

## Compiling new model libs (recipe lifted verbatim from the blog)

If we ever want to recover the broken `gemma-3-4b-it` model — or
compile any new candidate from the
[shortlist](./04_candidate_shortlist.md) that doesn't have a published
adreno DLL — the workflow is:

```powershell
# 1. Generate config
python -m mlc_llm gen_config `
  ./hf_cache/<MODEL> `
  --quantization q4f16_1 `
  --conv-template <template> `
  --prefill-chunk-size 2048 `
  -o ./dist/<MODEL>-q4f16_1-MLC

# 2. Quantize parameters
python -m mlc_llm convert_weight `
  ./hf_cache/<MODEL> `
  --quantization q4f16_1 `
  -o ./dist/<MODEL>-q4f16_1-MLC

# 3. Compile to Adreno DLL  — note the device target
python -m mlc_llm compile `
  ./dist/<MODEL>-q4f16_1-MLC/mlc-chat-config.json `
  --device windows:adreno_x86 `
  -o ./dist/libs/<MODEL>-q4f16_1-adreno.dll
```

The key flag is `--device windows:adreno_x86`. That string targets the
CLML-accelerated OpenCL backend Qualcomm built into TVM. Linux/Android
hosts use `--device android:adreno-so` and produce `.so` instead.

> Reminder from
> [`02_adreno_mlc_landscape.md`](./02_adreno_mlc_landscape.md):
> set `--prefill-chunk-size 2048` (not `256`). Smaller chunks have
> historically caused the OpenCL driver fragility / shutdown-hang
> documented in [`../runbooks/shutdown-hang-fix-summary.md`](../runbooks/shutdown-hang-fix-summary.md).

## Performance baselines (from the Qualcomm blog, X Elite column)

For sanity-checking any Tether benchmarks we run later:

| Model | X Elite decode (tok/s) | X Elite TTFT (s) for 256 prompt |
|-------|-----------------------:|-------------------------------:|
| Llama-2-7b-chat-hf | 20 | 2.45 |
| Meta-Llama-3-8B-Instruct | 17 | 2.7 |
| gemma-2b-it | 41 | 0.77 |
| Mistral-7B-Instruct-v0.2 | 16.3 | 3.3 |
| phi-2 | 42.8 | 1.25 |
| Phi-3-mini-4k-instruct | 33.5 | 1.5 |
| Qwen-7B-Chat | 15.7 | 2.5 |
| llava-1.5-7b-hf | 21 | 2.15 |

These are Qualcomm's own measurements with `q4f16_0`, no CLML
mention — actual numbers on `2025.06.r1`-with-CLML may be higher.

## Recommendation

1. **Do not upgrade or change the SDK install today.** We're already
   on the only published CodeLinaro release.
2. **Stay on x64 Python under Prism.** The Qualcomm-published wheels
   are x64-only; "fixing" this by switching to ARM64 Python would
   break `import mlc_llm`.
3. **Keep the upstream `mlc-ai/mlc-llm` window open as a watch list,
   not a switch target.** Architectures like Llama 4, Gemma 3,
   Qwen 3.5, OLMo 2 landed upstream after `2025.06.r1` was cut and
   are *not* in our installed `mlc_llm/model/` directory yet. They'll
   become available when Qualcomm cuts `2025.07.x` (or whatever the
   next release is named).

### When to recheck for a new release

```powershell
# 1. Re-run the catalog script — single REST call, no auth, fast
python scripts\research\jfrog_clo_catalog.py `
  --repo clo-472-adreno-opensource-ai `
  --path mlc-llm `
  --out-md   docs\research\05_codelinaro_catalog.md `
  --out-json docs\research\05_codelinaro_catalog.json

# 2. If you see a folder other than 2025.06.r1 in the output,
#    that's a new release. Compare the new wheel SHA-256 against
#    pip's direct_url.json before deciding to upgrade:
Get-Content `
  "$env:CONDA_PREFIX\Lib\site-packages\mlc_llm_adreno_cpu_clml_2025_06_r1-0.1.dev0.dist-info\direct_url.json"
```

If a new release does drop, the upgrade is mechanical:

```powershell
conda activate mlc-venv2
pip uninstall mlc_llm_adreno_cpu_clml_2025.06.r1 tvm_adreno_cpu_clml_2025.06.r1
# Download the new wheels from CodeLinaro …
pip install <new tvm wheel>
pip install <new mlc_llm wheel>
# Verify
python -c "import tvm, mlc_llm; print(tvm.support.libinfo()['GIT_COMMIT_HASH'])"
```

After any such upgrade, **all existing `dist/libs/*-adreno.dll` files
need to be recompiled** because the TVM ABI may have changed. Plan to
recompile against the new TVM/CLML build the same day.

## Source links

- Qualcomm dev blog (entry point): <https://www.qualcomm.com/developer/blog/2025/02/harnessing-qualcomm-adreno-gpu-generative-ai-open-source-approach>
- CodeLinaro UI (JS-rendered, slow): <https://codelinaro.jfrog.io/ui/native/clo-472-adreno-opensource-ai/mlc-llm/>
- CodeLinaro REST (use this in scripts): <https://codelinaro.jfrog.io/artifactory/api/storage/clo-472-adreno-opensource-ai/mlc-llm>
- Upstream MLC-LLM (reference, not our install): <https://github.com/mlc-ai/mlc-llm>
- Upstream model registry (for cross-referencing supported architectures): <https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/model/model.py>
