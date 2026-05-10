# ADR-0017: Practical context window — Seam C (deferred but ratified)

- **Status**: Accepted in principle (implementation deferred — see status note below)
- **Date**: 2026-05
- **Synthesis citation**: §12.4

## Status note

**Implementation deferred.** As of HEAD `0e01b85`, `ModelProvider` does NOT yet expose a
`get_practical_context_window(model_name, ram_budget_gb)` method, and the orchestrator does
NOT yet clamp history to this window. Tracked as a Phase 3 follow-up. Once the method lands
in `ModelProvider` and the orchestrator uses it, flip this status to "Accepted (Phase 3)".

This ADR ratifies the decision and documents the approved design so that the implementation
can proceed without reopening the architecture discussion.

## Context

Models advertise large context windows (e.g., Phi-3.5-mini-instruct: 40,960 tokens;
Qwen3-4B: 40,960 tokens). Loading them at the full advertised context can OOM the
Snapdragon X Elite's ~9 GB practical RAM ceiling because the KV cache grows linearly with
context length:

```
kv_bytes_per_token = num_kv_layers * 2 * num_kv_heads * head_dim * dtype_size
total_kv_bytes     = kv_bytes_per_token * ctx_len
```

For Phi-3.5-mini (32 KV heads, no GQA, 96 layers, head_dim=96, fp16) this is roughly
600 MB per 1,000 tokens. At 40,960 tokens the KV cache alone exceeds 24 GB — far beyond
the hardware budget. A naive `min(advertised_ctx, hard_cap)` is insufficient because
different models have different KV-cache footprints per token.

Without this seam, the orchestrator passes unbounded history to the provider and either
the engine silently truncates (losing turns with no user feedback) or the device OOMs.

## Decision

`ModelProvider` (in `tether.core.interfaces`) exposes a new ABC method:

```python
def get_practical_context_window(
    self,
    model_name: str,
    ram_budget_gb: float,
) -> int:
    """Return the largest safe context length (in tokens) for this model
    given the available RAM budget."""
```

Each provider implements its own RAM-aware calculation:

- **`MLCProvider`**: read `num_hidden_layers`, `num_attention_heads`,
  `num_key_value_heads`, `hidden_size` from `mlc-chat-config.json`; compute KV bytes
  per token; clamp `ctx_len` so that `weights_bytes + kv_bytes + scratch_bytes ≤ ram_budget_gb * 1e9`.
- **v1 fallback** (acceptable interim for Phase 3 deliverable): `min(advertised_ctx, 8192)`.
- **Future providers** (`NexaProvider`, `OllamaProvider`): own math — NPU memory footprint
  differs from OpenCL/Adreno.

The orchestrator clamps `len(history_tokens)` to the value returned by this method before
assembling the prompt sent to the provider. It does NOT use the advertised context window.

`Settings.runtime.ram_budget_gb` (default `9.0`) is the budget input surfaced to the
provider.

## Consequences

### Positive

- Hard ceiling against OOM crashes on resource-constrained hardware; the crash mode is
  "context truncated" (logged, user-visible) not "device OOM" (process killed, silent data
  loss).
- Provider-owned math means each backend is honest about its own memory costs; the
  orchestrator remains provider-agnostic.
- Configurable budget (`ram_budget_gb`) allows the same codebase to run on devices with
  different RAM ceilings without code changes.

### Negative

- Adds an ABC method, increasing the `ModelProvider` contract surface that all future
  providers must implement.
- Models with low practical context windows can frustrate users with long sessions — but a
  clear "context truncated to fit RAM" log is preferable to a silent OOM.

### Trade-offs accepted

- The clamp may leave advertised capacity unused on machines with more RAM. Mitigation: the
  budget is user-configurable; a developer on a 32 GB device can raise it.

## Alternatives considered

- **Static per-model cap in YAML**: brittle — every new model requires a config edit; does
  not account for KV-cache differences between models with and without GQA.
- **Crash-then-retry-with-smaller-ctx**: poor UX; OOM crashes can wedge the Adreno GPU
  driver requiring a device reboot.
- **Delegate entirely to the engine** (let MLC silently truncate): loses observability;
  users have no indication that context was dropped.

## References

- `_synthesis.md` §12.4
- `docs/research/06_context_strategies.md`
- `docs/architecture.md` §3 Seam C row
- `src/tether/core/interfaces.py` — `ModelProvider` ABC (method to be added)
- (Future) `src/tether/providers/mlc/provider.py::get_practical_context_window`
