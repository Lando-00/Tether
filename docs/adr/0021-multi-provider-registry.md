# ADR-0021: Multi-provider registry (per-request routing, degraded-mode lifecycle)

- **Status**: Accepted
- **Date**: 2026-06 (Phase 12, multi-provider rollout)
- **Supersedes (partially)**: ADR-0001 §"single `provider` on Engine"
- **Related**: ADR-0007 (strategy registry pattern — orchestrators), ADR-0016
  (MLC isolation rule), ADR-0003 (GC-disabled daemon shutdown), ADR-0009
  (SqliteInbox shared DB)

## Context

Today `Engine` holds exactly one `ModelProvider` (`self.provider`).
`Engine.from_settings` reads `providers.model` (singular) and constructs it via
`tether.core.factory.load`. The HTTP, CLI, and library entry points all assume
that one provider serves every model.

Two recent commits on `feature/copilot-sdk-provider` prepared the ground for
multi-provider:

- **`3d0509b`** added `CopilotProvider` — a remote, SDK-backed provider that
  overrides `source="remote"`, supports `reasoning_effort`, and runs entirely
  out-of-process from MLC.
- **`4fc7e82`** added `GET /api/v1/models/details` and the `reasoning_effort`
  request-validation slice on `/chat/stream`, plus the
  `ModelProvider.default_model()` / `list_model_info()` seams on the ABC.

Users now want one Tether server to expose **both** the local MLC stack and a
remote Copilot SDK at the same time, and route per request. The current
one-provider invariant blocks this. The Copilot provider is also a useful
fallback when MLC fails to load (no wheel, no device, GPU OOM), provided
*the rest of the server keeps serving*.

The shape of the design is heavily constrained by existing invariants:

- **ADR-0016** forbids importing `tether.providers.mlc.*` outside
  `core.factory.load`. The registry MUST keep all provider construction inside
  `factory.load`; no eager imports at package load.
- **ADR-0003 / `daemon_thread_call`** is load-bearing for the MLC OpenCL
  shutdown path. The watchdog already takes a `List[Any]` and filters by
  `isinstance(p, HardwareLifecycle)`; multi-provider is closer to ready here
  than at the Engine layer.
- **StrictModel** (`extra='forbid'`, frozen) on every Pydantic settings model:
  new fields must be explicit.
- **Per-request observability** (`RequestIdMiddleware` + structlog contextvars)
  must keep working unchanged.
- **SqliteInbox / ConnectorRegistry / connector drains** are independent of
  the model provider. Model-provider degraded startup MUST NOT cancel
  connector drains or close the inbox.

## Decision

Adopt a typed provider registry on `Engine`, with explicit per-request
routing, degraded-mode startup, and a one-cycle deprecation alias for the
singular `providers.model` shape.

### 1. Routing API (locked)

`StreamRequest` gains an optional field `provider_id: Optional[str]`. When
supplied, the selected provider must be healthy and advertise the requested
model. Unknown provider_id → 422; known-but-unhealthy provider_id → 503; a
model owned by another provider → 422.

When omitted, the Engine routes only if exactly one healthy provider advertises
the raw model name. Unknown names and duplicate ownership return 422 rather
than relying on registry order or silently falling back to the configured
default. The provider_id is a stable identifier set by config (registry key),
NOT the provider's `.kind` property — multiple providers of the same kind are
permitted (e.g. two Copilot accounts under different ids).

### 2. Failure isolation: degraded mode (locked)

`Engine.from_settings` constructs every entry in
`settings.providers.model_registry`. Each `factory.load(spec.impl, **spec.args)`
runs in its own try/except. Construction failures are captured per-id,
logged at ERROR with the `provider_id` bound, and the offending entry is
recorded in `engine._provider_start_failures: Dict[str, str]` (id → repr of
exception). Successful providers are stored in `engine.providers: Dict[str,
ModelProvider]`.

The configured `default_model_provider` remains immutable when it fails; it
is never replaced by another provider behind the caller's back. The engine is
considered "up" as long as at least one provider is healthy.
`/readyz` reports per-provider health additively; the store check remains
the gating signal for `ready=False`. `/models` and `/models/details` hide
unhealthy providers' models. `/chat/stream` with a failed `provider_id`
returns 503.

`Engine.__aenter__` performs degraded startup of providers BEFORE store /
inbox / tools / connectors. A provider's `warm_up()` failure demotes the
provider but does not abort engine startup. Connector drains and the inbox
lifecycle are independent — a degraded provider set has zero effect on
either.

### 3. CLI disambiguation (locked)

- New `--provider / -P` typer Option on the root `cli` callback AND the
  `chat` subcommand. Forwarded as `provider_id` in the POST body.
- The `\models` slash command grows a `provider_id` column and filters models
  to the selected provider when one is supplied.
- New `\providers` slash command lists the configured registry with health
  status.
- When `--model X` is ambiguous across providers (the same model_name exists
  under two provider_ids) and `--provider` was not supplied, the CLI drops
  into the existing `\models` selector with the ambiguous rows pre-filtered.
  A selected provider is never replaced by a different provider's model.

### 4. Back-compat (locked)

`providers.model` (singular) remains a valid config for one release cycle.
On load, the Pydantic validator on `ProvidersSettings` synthesises a
one-entry `model_registry` from it, with id `"default"`, and sets
`default_model_provider="default"`. A deprecation warning is logged on
every load via `warnings.warn(..., DeprecationWarning)`. Setting **both**
`providers.model` and `providers.model_registry` is a `ConfigError` — fail
loud rather than guess. Removing the singular field is tracked as a follow-up
ADR.

### 5. `is_default` semantics (locked)

`ModelProvider.default_model()` keeps its per-provider meaning ("which of
my models do I pre-select for clients?"). When `Engine` builds the merged
`/models/details` list, it does **not** invent a cross-provider winner; it
preserves each provider's `is_default` flag verbatim. The server-wide
default is communicated implicitly via `default_model_provider` in settings;
clients that need a single "pick this first" hint should pick the model
flagged `is_default=true` from the default provider.

**Decision flagged for Phase 2**: we do NOT add a separate
`is_server_default: bool` field on `ModelDetails` in this ADR. Adding one
later is additive and StrictModel-safe; doing it now invents a contract no
client has asked for.

## Lifecycle semantics

`Engine.from_settings` (degraded-mode pseudocode):

---

## Implementation notes

Implemented across commits `8e57a32` through `6cc2e25` on
`feature/copilot-sdk-provider` (Phase 12):

- **`8e57a32`** — `feat(config)`: `ProvidersSettings` multi-provider registry with
  back-compat alias (P1.2)
- **`8437297`** — `feat(types)`: `ModelDetails.provider_id` with sentinel default
- **`22b0ea7`** — `merge: mp-2a-engine` — `Engine` multi-provider dict,
  `from_settings` degraded-mode startup, `HardwareWatchdog` multi-provider list,
  `/readyz` per-provider health block (P2.A)
- **`ee9e672`** — `merge: mp-2b-http` — `provider_id` routing on `/chat/stream`,
  merged `/models` + `/models/details` across providers (P2.B)
- **`ce43cea`** — `merge: mp-2c-cli` — `--provider / -P` flag, `\providers` slash
  command, `\models` `provider_id` column, ambiguous-model fallback (P2.C)
- **`6cc2e25`** — `fix(adr-0021)`: addressed 1 BLOCKING + 2 HIGH + 3 MEDIUM
  findings from the parallel rubber-duck + code-review pass. The BLOCKING fix
  tightened `Engine.__init__` validation so an unhealthy `default_provider_id`
  can no longer reach the legacy `self.provider` shim and crash; the two HIGH
  fixes corrected the CLI `\models` switch (dropping a new `provider_id` when
  only the provider changed) and the `/readyz` `provider` bool (which was
  forced False in the hw-error branch, lying about registry health); the three
  MEDIUM fixes patched `/readyz` error-path response completeness (additive
  schema contract), CLI reasoning-effort capability lookup (now
  `(id, provider_id)`-scoped), and 503 detail message redaction (was leaking
  raw exception text that could contain paths or tokens).
