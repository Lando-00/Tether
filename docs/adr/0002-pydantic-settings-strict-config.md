# ADR-0002: Pydantic Settings + `StrictModel` as the typed config seam

- **Status**: Accepted (Phase 2 of refactor)
- **Date**: 2026-05 (Phase 2)
- **Synthesis citation**: §2.3, §3.1, §3.2, §4 Phase 2 step 17

## Context

Pre-refactor, configuration was a loosely-typed `dict` produced by an ad-hoc YAML loader.
Settings were re-read from disk and `os.environ` from inside the orchestrator and tool runner
(synthesis §6 bugs #8, #11 — `load_settings()` called per chat turn). Two documented
features — `testing.yml` and `TETHER_IGNORE_DEV_CONFIG` — had zero callers (§6 bug #10). New
sub-systems (connectors, inbox, observability, security) need their own typed sub-models that
the orchestrator must not invent fields against.

## Decision

Adopt **Pydantic v2 Settings** as the single typed configuration seam:

- `class Settings(BaseSettings)` with `SettingsConfigDict(env_prefix="TETHER__",
  env_nested_delimiter="__", extra="forbid")`. `extra="forbid"` is non-negotiable — typos
  fail loudly.
- **Override priority**: `defaults < default.yml < overlay.yml < env vars < explicit args`.
- **Pure loader**: `load_settings(*, default_yaml=None, overlay_yaml=None, env=None) ->
  Settings`. Tests pass explicit dicts; the entrypoint passes `os.environ`. The loader does
  no I/O beyond reading the YAML paths it is given.
- **Typed sub-models** (each is a `StrictModel`-style `BaseModel(extra="forbid")`):
  `HttpSettings`, `ProvidersSettings`, `ToolsSettings`, `LimitsSettings`, `ContextSettings`,
  `SystemSettings`, `SecuritySettings`, `ObservabilitySettings`, `StorageSettings`,
  `OrchestratorSettings`, `ConnectorsSettings`, `InboxSettings`, plus future sub-models.
- **Settings are built once, at the composition root**, and stashed on `app.state.settings`
  / passed to `Engine.from_settings(settings)`. The orchestrator and tool runner take a
  narrowed `OrchestratorConfig` slice in their constructors — they never call
  `load_settings()` themselves.
- **Delete `testing.yml` + `TETHER_IGNORE_DEV_CONFIG`**: documented features with zero
  callers; tests pass `Settings(...)` directly.

## Consequences

### Positive
- One source of truth: every config value has a typed home. Typos in YAML or env vars
  fail at startup, not silently at first use.
- The orchestrator becomes pure: zero `os.environ` reads, zero per-turn YAML loads.
- New sub-systems (connectors, inbox, observability) extend `Settings` by adding a
  sub-model — no string-keyed dict coupling.
- IDE autocomplete + mypy across all config consumers.

### Negative
- Adding a new config knob requires touching the Pydantic model (acceptable; that is the
  point).
- The single-source rule means writing tests that need a particular config shape requires
  constructing a `Settings(...)` with explicit nested objects.

### Trade-offs accepted
- Boilerplate: every sub-model needs explicit fields. We accept it as the cost of
  `extra="forbid"`.

## Alternatives considered

- **Continue with raw `dict`s** — rejected: indirection bugs (#8, #11) are a direct
  consequence of untyped config flowing through the orchestrator.
- **`dataclasses` + manual env parsing** — rejected: no nested-key env support; reinvents
  Pydantic v2's `env_nested_delimiter`.
- **`dynaconf` / `omegaconf`** — rejected: too much surface area for a single-process
  service; we'd still want Pydantic validation downstream.

## References

- `files/investigations/_synthesis.md` §2.3, §3.1, §3.2, §4 Phase 2 steps 17–19, §6 bugs #8, #10, #11, §12.6
- `src/tether/config/settings.py`, `src/tether/config/default.yml`
