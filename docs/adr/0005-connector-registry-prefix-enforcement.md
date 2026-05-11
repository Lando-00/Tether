# ADR-0005: `ConnectorRegistry` with mandatory `{connector_id}_` tool prefix

- **Status**: Accepted (Phase 4.5 of refactor)
- **Date**: 2026-05 (Phase 4.5)
- **Synthesis citation**: §3, §10 (especially §10.4–§10.5), §4 Phase 4.5 (steps 48a–48e)

## Context

A parallel planning session produced the **Connectors Spec** (`session-state/.../plan.md`)
defining a `Connector` ABC for future WhatsApp/Gmail/etc. integrations. Each connector
exposes a set of tools (e.g. `whatsapp_send`, `whatsapp_mark_seen`) and may also drain
inbound events (Phase 6.5 / ADR-0009). The orchestrator already aggregates tools into one
flat list — we need a way to register connector-owned tools alongside built-in tools without
collisions, and we need a runtime registry that owns connector lifecycle.

## Decision

Add a **`ConnectorRegistry`** (`src/tether/core/connector_registry.py`) with these
properties:

- **API**: `get(id)`, `all()`, `aggregate_tools()`, `start_connector(id)`,
  `stop_connector(id)`. `aggregate_tools()` returns a flat `dict[str, Tool]` concatenated
  with the built-in `ToolRegistry` tools and passed to the orchestrator. Orchestrator and
  parser do not care which registry a tool came from.
- **Mandatory `{connector_id}_` tool-prefix enforcement** at registry build time. Each
  connector's tools must have `tool.name.startswith(f"{connector.id}_")`; missing prefix is
  a fail-fast `ValueError` at startup. **Collision detection** (against `tools.registry`
  and across connectors) also happens at boot — fail-fast with a clear error.
- **Engine-owned**: wired into `Engine.from_settings()` and `create_asgi_app(engine)`.
  In the FastAPI lifespan startup hook, for each connector whose
  `auth_status == ready`, schedule `await registry.start_connector(id)`. In shutdown,
  cancel drain tasks first, then `await asyncio.wait_for(connector.stop(), timeout=2.0)`
  per connector. The registry's `wait_for(stop(), 2s)` only protects cooperative async
  stops — connectors with native blocking cleanup mirror the `HardwareWatchdog`
  daemon-thread+force-exit pattern themselves (cf. ADR-0003).
- **Filesystem layout**: lazy-create `data/connectors/<id>/` per connector for credential
  storage; SecretsProvider (env → file fallback at `data/secrets/<key>` mode 0600) handles
  the actual secret reads.
- **OAuth callback** is handled by the main FastAPI app at
  `/api/v1/connectors/{id}/oauth/callback` (NOT a sidecar loopback server). OAuth `state`
  parameters live in an in-memory `TTLCache(maxsize=8, ttl=300)` on the registry. After
  successful `complete_login()`, the route handler calls `registry.start_connector(id)` so
  the user gets a working connector without restarting the process.
- **Built-in only, no plugin SDK in v1**: connectors live in `src/tether/connectors/<id>/`
  and ship with the Tether package. A pluggable connector ABC (third-party packages) is
  deferred until two real connectors exist.

## Consequences

### Positive
- Tool name collisions are impossible by construction — a connector cannot accidentally
  shadow `web_search` or another connector's tool.
- The orchestrator stays connector-agnostic: it sees one flat tool list.
- New connectors land as a single sub-package without orchestrator/parser changes.
- OAuth flows work in the main app, no extra ports to coordinate.

### Negative
- Tool names get verbose (`whatsapp_send` not `send`). This is the point — explicit > terse.
- The 2s cooperative stop budget means a misbehaving connector can hold shutdown for that
  long per connector; large numbers of connectors compound.

### Trade-offs accepted
- Built-in-only deployment trades extensibility for v1 simplicity. The seam is in place
  to add a plugin entry point later (parallel to `entry_points("tether.tools")`).

## Alternatives considered

- **No prefix enforcement** — rejected: future connectors will collide with built-in
  tools; debugging name conflicts in the model's tool-call output is painful.
- **Separate `connector_tools` dict passed to orchestrator** — rejected: doubles the
  surface; orchestrator becomes connector-aware.
- **Sidecar OAuth server on a loopback port** — rejected per spec §3.8: extra port,
  firewall friction, cookie domain issues.

## References

- `files/investigations/_synthesis.md` §3, §4 Phase 4.5, §10.4, §10.5, §10.7, §10.8
- Connectors Spec: `session-state/5c8a15fc-.../plan.md` §3.1, §3.3, §3.5, §3.7, §3.8
- `src/tether/core/connector_registry.py`, `src/tether/core/secrets.py`
- `src/tether/adapters/http/routers/connectors.py`

## Implementation status: connector start await (Phase 9 P0-F)

Tribunal §3 P0-07 (A2-F2). Prior to Phase 9 P0-F, `Engine.__aenter__` scheduled `ConnectorRegistry.start_connector(cid)` for each READY-at-config-time connector with `asyncio.create_task` and returned without awaiting them. Failures were silently reaped by `aclose()`'s `gather(*tasks, return_exceptions=True)`. The first `chat()` arriving in the gap could land on a half-initialized connector.

P0-F changes the contract:

1. `__aenter__` now `await`s the gathered start tasks BEFORE returning control to the FastAPI lifespan.
2. Per-failure exceptions are logged at `logger.exception` granularity with the connector id pulled from the task name (`start_connector:<cid>`).
3. Failing connectors are removed from `ConnectorRegistry._connectors` so subsequent tool dispatch sees a deterministic `ConnectorNotConfiguredError` rather than a phantom object.
4. `/readyz` exposes a `connector_start_failures: [cid, ...]` array and degrades `ready=false` when non-empty (response stays HTTP 200; the Tribunal P1-29 status-code posture is a separate change).

The 2 s `stop_all` budget on shutdown is unaffected — start tasks are completed (success or failure) by the time `aclose()` runs, so the cancel-pending block is a no-op for the awaited path.
