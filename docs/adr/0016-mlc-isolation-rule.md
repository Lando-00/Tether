# ADR-0016: MLC isolation rule — no imports from `mlc/` outside factory

- **Status**: Accepted
- **Date**: 2026-05 (refactor close)
- **Synthesis digest**: [`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md)

## Context

`mlc_llm` (the CodeLinaro fork) imports `tvm` at module load time and constructs runtime
wrappers that touch the Adreno OpenCL device driver. **Importing
`tether.providers.mlc.provider` for any reason** — a stray `isinstance()` check elsewhere,
a registry that lazy-imports all providers to enumerate schemas, a test file that imports
the wrong symbol — instantiates GPU state as a side effect of the import, not of any
explicit `engine.load()` call.

This was the root cause documented in synthesis §6 bug #7 ("import-time side effects"):
unit tests that touched provider code were initialising the GPU, making the test suite
non-deterministic on machines without MLC wheels and slow on machines that did have them.
It also meant that `import tether` in a library context pulled in GPU initialisation, which
broke the library-first goal of ADR-0001.

## Decision

`tether.providers.mlc.*` may be imported ONLY by:

1. `tether.core.factory.load(impl, ...)` — the dotted-path resolver that is called at
   `Engine` construction time, only when `impl` resolves to an MLC class.
2. Direct test-side imports under `tests/hardware/` — these are opt-in via the
   `-m hardware` marker and are never run in CI without a real device.

All other modules MUST refer to providers through the `ModelProvider` ABC defined in
`tether.core.interfaces`. Static type-only imports (`if TYPE_CHECKING: from
tether.providers.mlc.provider import MLCProvider`) are permitted because `TYPE_CHECKING`
is `False` at runtime.

Until a CI import-graph check is added, this ADR is the authoritative convention; any
violation found in review MUST be treated as a bug.

## Consequences

### Positive

- `import tether` is GPU-free; `Engine.from_settings(...)` is the only call that opens
  GPU state.
- A developer without MLC wheels installed can `import tether`, run all non-hardware unit
  tests, and serve a dummy provider. Only `pip install tether[mlc]` activates the heavy
  path.
- Test suite startup time drops because GPU initialisation does not happen at collection
  time.

### Negative

- Future provider implementations (Nexa, Ollama, etc.) must follow the same isolation
  rule — it is a convention, not an enforced import graph check (yet).
- Import-order violations are caught only at hardware-test time unless a CI linting step
  is added (tracked as a follow-up; see References).

## Alternatives considered

- **Make MLC import lazy via `__getattr__` on the package**: harder to reason about, and
  breaks the `from tether.providers.mlc.provider import MLCProvider` ergonomics required
  inside the factory itself.
- **Move MLC into a separate installable package `tether-mlc`**: would create a circular
  install story — the factory needs to know that dotted paths exist before the extra package
  is installed.
- **Require MLC wheels in all environments**: defeats the library-first goal (ADR-0001)
  and makes CI prohibitively expensive.

## References

- Synthesis digest: [`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md)
- `src/tether/core/factory.py::load` — the sole authorised import site
- `src/tether/core/interfaces.py` — `ModelProvider` ABC
- ADR-0001: Library-first composition root
- ADR-0014: CodeLinaro runtime pin (the wheel being isolated)
