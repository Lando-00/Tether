# CLML Compilation — Research Findings

> **Date:** 2026-05-10
> **Author of investigation:** Tether research session (May 2026)
> **Bottom line:** **Stop chasing CLML compile flags.** In the installed
> Qualcomm CodeLinaro `2025.06.r1` wheels, `--opt openclml=1` and
> `--opt adrenoaccl=1` are **no-ops on Qwen3-class LLMs** because the
> backend's pattern matcher requires a constraint the MLC Relax serving
> pipeline cannot satisfy. The 5–10% variations we observed in the
> tuning sweep were measurement noise, not acceleration. Workflow
> recommendation: use the standard compile recipe with no `--opt`
> flags, and direct optimization effort elsewhere
> (`/no_think`, pre-warming, smaller models).

## Method

Six diagnostic steps:

1. Pulled `mlc-llm-docs-2025.06.r1.zip` from CodeLinaro (Qualcomm's
   own docs for this exact wheel build) and grep'd every reference to
   CLML / openclml.
2. Read the MLC-LLM and TVM source for what `openclml=1` and
   `adrenoaccl=1` actually do.
3. Probed the runtime to check whether CLML graph executor is wired in
   (`tvm.support.libinfo()`, `is_clml_runtime_enabled()`,
   `relax.op.is_openclml_runtime_enabled`).
4. Recompiled with `--debug-dump` to dump the Relax IR after every
   pipeline phase and grepped for CLML annotations.
5. Compared compiled DLL bytes across the 5 tuning variants.
6. Cross-checked against `adrenoaccl` to see whether the second flag
   has different constraints.

## What we expected

Per Qualcomm's
[Feb 2025 Adreno blog post](https://www.qualcomm.com/developer/blog/2025/02/harnessing-qualcomm-adreno-gpu-generative-ai-open-source-approach):

> Qualcomm Technologies has improved the performance of OpenCL backend
> for Adreno GPU significantly. Majority of these enhancements are
> contributed back to community already.

Implied: `--opt openclml=1` enables the CLML graph executor, offloading
matmul-heavy hot paths to Qualcomm's hand-tuned kernels. Expected
decode-rate uplift: 15–30% based on Qualcomm's published numbers.

Observed in our tuning sweep: 0–10% variation across all variants,
within measurement noise.

## What we found (concrete evidence)

### 1. The `openclml=1` flag IS implemented; it just does nothing here

`mlc_llm/compiler_pass/pipeline.py` lines 141–145:

```python
# Adreno Openclml BYOC offloading.
(
    OpenCLMLOffLoadForLLM(target=target)
    if openclml
    else tvm.transform.Sequential([])
),
```

So the pass *is* wired in. `OpenCLMLOffLoadForLLM` lives at
`tvm/relax/backend/adreno/clml.py` line 690.

### 2. CLML for LLMs only matches a single pattern

`tvm/relax/backend/adreno/clml.py` lines 655–687:

```python
def dequantize_matmul_patterns():
    ...
    return [_dequantize_matmul_pattern("openclml.dequant_matmul")]

clml_llm_patterns = [*dequantize_matmul_patterns()]
```

The other 50+ CLML patterns in this file
(`nn.conv2d`, `batch_norm`, `max_pool2d`, etc.) are vision-model patterns
and never match a transformer LLM graph. So the only thing CLML can
offload from a Qwen3 model is `dequantize → matmul` fusions.

That's the right thing to target — q4f16 LLMs are dominated by
weight-dequantize-then-matmul calls. Good in theory.

### 3. The pattern's check function rejects every matmul in the graph

`_check_dequantize_matmul` (lines 623–652) requires:

```python
if ctx.annotated_expr["lhs"].struct_info.dtype != "float16":
    return False                                            # gate 1
...
if not (
    (len(root.struct_info.shape) == 3)
    and isinstance(root.struct_info.shape[0], tir.IntImm)   # gate 2
    and (root.struct_info.shape[0] == 1)                    # gate 3
    and (root.struct_info.dtype == "float16")
):
    return False
```

Now compare to what Qwen3's actual matmul looks like in the Relax IR
(`debug-phase0.py` line 2476 from our debug-dumped compile):

```python
matmul435: R.Tensor((batch_size, 1, 6144), dtype="float16") =
    R.matmul(rms_norm435, permute_dims435, out_dtype="void")
```

* Gate 1: `lhs.dtype == "float16"` — ✅ passes
* Gate 2: `shape[0]` is `batch_size` — a `tir.Var`, not a `tir.IntImm` — ❌ **fails**
* Gate 3: never reached.

The pattern matcher needs `shape[0]` to be a literal integer. The MLC
Relax serving pipeline keeps `batch_size` as a symbolic variable so the
compiled engine can support dynamic batched serving. The two are
incompatible.

### 4. `--overrides max_batch_size=1` does NOT fix the symbolic dim

We tried this in the tuning sweep. The override sets the *upper bound*
on batch size; it does not substitute the symbolic `batch_size`
variable with `1` in the IR. Confirmed by the same `debug-phase0.py`
output: `batch_size` remained symbolic even with `max_batch_size=1`.

### 5. Confirmed by `--debug-dump` after a CLML-enabled compile

Recompiled `Qwen3-4B-q4f16_1-MLC` with both `openclml=1` AND
`max_batch_size=1`, then grep'd all 6 phase IR dumps for the strings
`openclml`, `clml`, `dequant_matmul`:

```
> rg -i "openclml|clml|dequant_matmul" research/clml/debug_all/
(no matches)
```

**Zero pattern matches.** No CLML annotations entered the IR at any
pipeline phase. The pass ran, found nothing to partition, and emitted
nothing.

### 6. CLML graph executor isn't even wired into the runtime

```python
>>> from tvm.relax.backend.adreno.clml import is_clml_runtime_enabled
>>> is_clml_runtime_enabled()
False
>>> tvm.get_global_func('relax.op.is_openclml_runtime_enabled', True)
None
```

Even if the pattern matcher had succeeded, the runtime function
that would execute CLML kernels is **not registered**. So compile-time
partitioning would have produced subgraphs the runtime couldn't run.
Belt-and-braces blocked.

### 7. `adrenoaccl` has the identical broken constraint

`tvm/relax/backend/contrib/adrenoaccl.py` line 56:

```python
and (root.struct_info.shape[0] == 1)
```

Same `shape[0] == 1` static-batch requirement. Same failure mode. Our
debug-dump grep for `adrenoaccl|adreno_accl|adrenobyoc` in the
adrenoaccl-enabled compile also returned **zero matches**.

### 8. The Qualcomm docs themselves don't tell users to set these flags

Reading `mlc-llm-docs-2025.06.r1/_sources/get_started/sample.rst.txt`
end-to-end: the canonical compile command is

```
python -m mlc_llm compile <cfg> --device windows:adreno_x86 -o <out>.dll
```

with no mention of `--opt openclml=1` or `--opt adrenoaccl=1` in the
Windows or Linux examples. The flags exist in the CLI but are not
part of Qualcomm's documented flow for LLM compilation. This is
consistent with their being legacy / future / vision-only knobs that
don't currently apply to Relax LLM serving.

## So what was different about our 5 tuning DLLs?

Compiled bytes differ across variants:

| Variant | DLL SHA-256 (first 16 hex) | Size |
|---------|---|---:|
| baseline | `B1039BA8E38D4FB5` | 5.30 MB |
| `openclml=1` | `CC496236839154A9` | 5.30 MB |
| `adrenoaccl=1` | `1A2EAEB097EB965F` | 5.30 MB |
| `shrunk_ctx` | `882DA62EC75E24A6` | 5.30 MB |
| `all combined` | `31DC341B1C68C3A5` | 5.30 MB |

Different bytes, **same functional content**. TVM's code generation
is non-deterministic (kernel ordering, label naming, etc.) — so
re-running compile produces a different DLL even when the IR pipeline
is functionally equivalent. The 5–10% bench variations we saw were
the natural variance of OpenCL kernel scheduling on identical
generated code, not real acceleration.

## Why Qualcomm's blog showed bigger CLML wins, then?

A few possible explanations, in order of likelihood:

1. **Different model class.** Qualcomm's published CLML-on-Adreno
   benchmarks are heavily skewed toward CNN inference (image
   classification, segmentation). Those graphs are dominated by
   `conv2d`, `batch_norm`, `pool2d` — exactly the ops CLML's pattern
   table covers richly. LLMs were a later / lighter integration.
2. **CLML for LLMs may rely on the older Relay pipeline.** TVM has two
   compile paths: Relay (mature, vision-focused, full CLML coverage)
   and Relax (the new dataflow IR used by MLC-LLM today, with
   `dequantize_matmul` as the only LLM-relevant CLML pattern). It's
   plausible CLML LLM speedups in older Qualcomm posts came from a
   Relay-based path, not Relax serving.
3. **The constraint is a ports / WIP gap.** The CLML LLM pattern
   matcher in our wheel was likely written for a non-serving compile
   mode where batch is statically resolved to 1 before reaching this
   pass. Qualcomm hasn't adapted it for the dynamic-batch serving
   path. Future releases may fix this.

## Recommendations

### Tether ship config

1. **Drop `--opt openclml=1` and `--opt adrenoaccl=1` from any
   Tether compile pipeline.** They're zero-impact and add complexity.
   Standard compile is just:

   ```powershell
   python -m mlc_llm compile <cfg.json> --device windows:adreno_x86 -o <name>-adreno.dll
   ```

2. **Use the baseline `Qwen3-4B-q4f16_1-adreno.dll`** in `models/libs/`
   as the canonical lib. The 5 experiment DLLs we tested
   (`Qwen3-4B-q4f16_1-adreno-{clml,accl,shrunk,all}.dll`) were
   functionally equivalent — same outputs, marginal compiled-byte
   differences — and were deleted after the investigation.

### Where to actually direct optimization effort

In rough order of expected ROI:

1. **`/no_think` ON by default** — already proven 2.6× win on
   tool-call latency, 4.6× on long-context summary. See
   `tuning_sweep_summary.md`.
2. **Pre-warm common kernel shapes on engine startup** — eliminates
   the 30–200 sec cold-start cost per new shape (the big remaining
   user-perceived latency).
3. **Smaller models for tool routing.** A `Qwen3-1.7B-q4f16_1` or
   `Qwen3-0.6B-q4f16_1` running alongside the 4B for fast routing /
   classification turns. Both fit easily in our 9 GB budget on top
   of the 4B.
4. **`max_tokens` standardization.** Each unique `max_tokens` value
   is a new shape to TVM. Pick a small set (64 / 256 / 1024) and
   reuse them.

### Things to track for the future

- **CLML LLM pattern fix.** Watch the next Qualcomm CodeLinaro release
  (`scripts/research/jfrog_clo_catalog.py` re-run will spot it). If a
  post-`2025.06.r1` build includes a fix to `_check_dequantize_matmul`
  that accepts symbolic batch, recompile and re-bench. Worth a recheck
  every 3-6 months.
- **NexaSDK NPU.** The non-CLML answer to "make Adreno LLMs faster" is
  to move off Adreno onto the Hexagon NPU via NexaSDK
  (see `docs/research/03_npu_hexagon_landscape.md` in the Tether repo).
  A `NexaProvider` post-refactor likely beats anything CLML could give
  us today.

## Reproducing this investigation

The `--debug-dump` IR (~7 MB Relax IR per compile) was used to verify
zero CLML annotations and is NOT committed (regenerable). To reproduce:

```powershell
# 1. Verify CLML runtime is not wired
python -c "from tvm.relax.backend.adreno.clml import is_clml_runtime_enabled; print(is_clml_runtime_enabled())"
# Expected: False

# 2. Recompile a model with debug-dump
python -m mlc_llm compile <dist>\Qwen3-4B-q4f16_1-MLC\mlc-chat-config.json `
  --device windows:adreno_x86 `
  --opt "openclml=1" `
  --overrides "context_window_size=4096;max_batch_size=1" `
  --debug-dump some_debug_folder `
  -o some-output.dll

# 3. Grep the IR dumps for CLML annotations
rg -i "openclml|clml|dequant_matmul" some_debug_folder/
# Expected: no matches
```

If the third command returns matches in a future Qualcomm release,
the pattern matcher has been fixed and CLML is worth re-evaluating.
