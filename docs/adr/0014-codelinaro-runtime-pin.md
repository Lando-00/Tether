# ADR-0014: Pin Qualcomm CodeLinaro MLC-LLM `2025.06.r1` runtime

- **Status**: Accepted (Phase 0A locked)
- **Date**: 2026-05 (refactor close)
- **Synthesis digest**: [`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md)

## Context

Tether's MLC-LLM runtime is a Qualcomm-vendored fork distributed via CodeLinaro JFrog
(`mlc_llm_adreno_cpu_clml_*` + `tvm_adreno_cpu_clml_*` wheels), not upstream
`mlc-ai/mlc-llm`. Upstream wheels do NOT contain the CLML-accelerated OpenCL backend
nor the Snapdragon X Elite Adreno code paths. A `pip install --upgrade mlc-llm` from
PyPI silently swaps in incompatible binaries — the import succeeds but inference either
falls back to CPU or errors at engine initialisation.

The frozen working version is `2025.06.r1` (cp312-cp312-win_amd64).

## Decision

Pin the runtime explicitly:

- `environment-mlc-venv2.yml` (current dev env) and `environment-tether.yml` (fresh validation env) reference the CodeLinaro wheel filenames; their parent commit messages reference the pinned
  pinned URL; the version string `2025.06.r1` appears verbatim so drift is obvious.
- `mlc-venv2` conda env documentation and any developer runbook must call out
  **"do NOT `pip upgrade mlc_llm`"** with rationale.
- The shutdown patterns documented in ADR-0003 (GC-disable + daemon thread) are validated
  only against `2025.06.r1`; a new r-suffix release requires re-validation before the
  pin is moved.
- Future re-pins go through a new ADR that explicitly supersedes ADR-0014.

## Consequences

### Positive

- Reproducible runtime; version drift is detected at install time rather than at
  inference time.
- The GC-disable + daemon-thread shutdown patterns (ADR-0003) are tested only against
  `2025.06.r1`; pinning prevents undefined behaviour on other versions.
- Clear upgrade path: new ADR = new pin = new validation run.

### Negative

- Manual upgrade overhead when Qualcomm publishes a new CodeLinaro release.
- Developers cannot use standard `pip install --upgrade` workflows and must be explicitly
  warned.

## Alternatives considered

- **Track upstream `mlc-ai/mlc-llm`**: would lose CLML/Adreno acceleration; all
  orchestration code paths assume the Qualcomm fork's engine API surface.
- **Pin only major version, not r-suffix**: insufficient — patch releases have changed
  shutdown timing semantics (the Qwen2.5-7B hang was `2025.06.r0` vs `r1`-specific).
- **Container image with baked-in wheels**: viable but out of scope for this single-user
  local deployment; deferred to a future ADR if containerisation is adopted.

## References

- `environment-mlc-venv2.yml` + `environment-tether.yml`
- `scripts/setup_fresh_env.ps1` (idempotent bootstrap)
- `docs/runbooks/fresh-env-setup.md`
- Synthesis digest: [`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md)
- `docs/runbooks/shutdown-hang-fix-summary.md` (the GC-disable fix is r1-specific)
- ADR-0003: GC-disabled daemon-thread shutdown for OpenCL
