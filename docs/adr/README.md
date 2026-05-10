# Architecture Decision Records

Each ADR captures one architectural decision with context, rationale, and consequences.
Decisions are immutable once Accepted; if a decision changes, a new ADR is created that
supersedes the old one.

## Index

| # | Title | Status |
|---|---|---|
| [0001](./0001-library-first-engine.md) | Library-first composition root (`Engine` class) | Accepted |
| [0002](./0002-pydantic-settings-strict-config.md) | Pydantic Settings + `StrictModel` typed config | Accepted |
| [0003](./0003-gc-disabled-shutdown-daemon.md) | GC-disabled daemon-thread shutdown for OpenCL | Accepted |
| [0004](./0004-tool-v2-base-tool-and-decorator.md) | Tool v2: `BaseTool` + `@tool` + `ToolExecutionContext` | Accepted |
| [0005](./0005-connector-registry-prefix-enforcement.md) | `ConnectorRegistry` with mandatory `{connector_id}_` prefix | Accepted |
| [0006](./0006-wire-protocol-v2.md) | Wire protocol v2 (NDJSON) + parser strategy (Seam D) | Accepted |
| [0007](./0007-orchestrator-strategy.md) | Orchestrator strategy ABC (Seam B) | Accepted |
| [0008](./0008-aiosqlite-yoyo-migrations.md) | Persistence: `aiosqlite` + WAL + yoyo-migrations | Accepted |
| [0009](./0009-sqlite-inbox-shared-db.md) | `SqliteInbox` in shared `data/tether.db` (deviates from spec §3.6) | Accepted |
| [0010](./0010-structlog-observability.md) | `structlog` + `RequestId` middleware + optional OTel | Accepted |
| [0011](./0011-outbound-url-allowlist.md) | Outbound URL allowlist + `assert_safe_url` | Accepted |
| [0012](./0012-csrf-cors-trustedhost-middleware-ordering.md) | CSRF + CORS + TrustedHost middleware ordering | Accepted |
| [0013](./0013-src-layout-and-tether-service-alias.md) | `src/` layout + `tether_service` deprecation alias | Accepted |

## Format

ADRs follow a minimal Nygard-style template — see any existing ADR for reference. The binding
contract for the refactor lives in `files/investigations/_synthesis.md` (session-state).
Anything quoted in the **Decision** section of an ADR is paraphrased from that synthesis;
ADRs do not invent new architecture, they ratify it.
