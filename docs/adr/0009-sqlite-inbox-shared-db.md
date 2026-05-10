# ADR-0009: `SqliteInbox` in shared `data/tether.db` (deviates from spec §3.6)

- **Status**: Accepted (Phase 6.5)
- **Date**: 2026-05 (Phase 6.5)
- **Synthesis citation**: §10 (especially §10.8 #5), §11.6, §4 Phase 6.5 (steps 66a–66d)

## Context

The Connectors Spec §3.6 says inbound events should live in `data/inbox.db` — a separate
SQLite file from session/message storage. The synthesis pushed back: separate DB files mean
two DSN settings, two connection lifecycles, two migration tracks, and no shared transaction
boundary when (e.g.) a connector wants to atomically append an inbound event and update
session state. The spec's footnote on §3.4 already concedes "can be same DB; separate file
is fine."

## Decision

**Deviate from Connectors Spec §3.6**: put `inbound_events` as a new table inside the
shared `data/tether.db`, managed by yoyo migration `004_inbox.sql`. This was explicitly
ratified by the user in §11.6 (table row "§10.8 #5 — Single DB with `inbound_events`
table — ✅ ratified").

- **One database file**: `data/tether.db` holds everything. The spec's `inbox.db` filename
  becomes a non-issue; the per-connector filesystem layout (`data/connectors/<id>/` for
  credentials/secrets) is unchanged.
- **`InboundInbox` ABC** in `src/tether/context/inbox_store.py`, with **`SqliteInbox`** impl
  using a shared :class:`AsyncSqliteStore` aiosqlite connection. Methods: `append_many`,
  `list_unread`, `list_recent`, `mark_seen`, `prune_older_than`. `append_many` runs as a
  single SQLite transaction via `executemany`.
- **`inbox_seen` is Tether-side inbox state**, NOT platform read state. `mark_seen` records
  that the inbox showed the event to a downstream consumer (UI, API caller). Connectors that
  want to mark messages read on the upstream platform (WhatsApp, Gmail) expose a separate
  `*_inbox_mark_seen` tool (§10.6 of the synthesis defines the capability tags). A future
  `platform_read` column on `inbound_events` will land via a separate migration when the
  first platform that exposes a read-state API needs it.
- **Schema** (single table `inbound_events`) with `idx_inbound_events_received_at` and
  `idx_inbound_events_unread` (composite on `connector_id, inbox_seen, received_at`)
  indexes. LIKE-based search deferred (no v1 callers need it; FTS5 deferred too).
- **Wired into `Engine.from_settings()`**: when `settings.inbox.enabled` is true, builds a
  `SqliteInbox` against the same DSN as `SqliteSessionStore` and passes it to the
  `ConnectorRegistry`. `start_connector(id)` spawns a per-connector
  :class:`tether.runtime.task_supervisor.SupervisedTask` that iterates
  `connector.inbound_stream()` and persists events via `inbox.append_many`. Per-event
  exceptions are logged + skipped — a single bad event MUST NOT kill the drain task.
- **HTTP routes**: `GET /api/v1/connectors` returns connector health states; `GET
  /api/v1/connectors/{id}/inbox?unread=true&limit=N` returns events;
  `POST /api/v1/connectors/{id}/inbox/mark-seen` flips `inbox_seen`. `/readyz`
  continues to return 200 even when a connector is `unconfigured` / `logged_out` (spec
  §8.4 acceptance).
- **`Engine.aclose()` ordering**: cancel drain tasks → stop connectors → stop tools → stop
  provider → close session store → close inbox. Shutdown is bounded by the connector
  budget per ADR-0005.

## Consequences

### Positive
- One DSN, one connection lifecycle (per store), one migration track, one transaction
  boundary. Simpler shape across all of `SqliteInbox` + `AsyncSqliteStore` + `tool_audit`.
- `prune_older_than` and `mark_seen` can run alongside session writes inside the same
  WAL with no cross-process sync concern.
- Backups are one file.

### Negative
- Mixing inbox-write traffic with session-write traffic in one SQLite file means one
  bad migration rolls back inbox + sessions together. Mitigated by yoyo's forward-only
  discipline and the single-user assumption.
- Slightly larger `data/tether.db` over time; pruning policy is opinionated.

### Trade-offs accepted
- We accept the spec deviation. The spec author explicitly left the door open ("can be
  same DB; separate file is fine"); user has ratified.

## Alternatives considered

- **Honor spec literally with `data/inbox.db`** — rejected per §11.6; doubles the
  lifecycle surface for no benefit in a single-user app.
- **Keep inbox as a side-effect of session messages** — rejected: inbox events have no
  conversational `session_id` until a user surfaces them; they need their own queryable
  index.
- **External queue (e.g. Redis)** — rejected: introduces a process boundary and a runtime
  dependency for a single-user local-first app.

## Final implementation

Phase 6.5 landed via `refactor/p65-implement` (synthesis §4 steps 66a–66h):

- `src/tether/context/_async_sqlite_base.py` — shared `AsyncSqliteStore` base
  (synthesis §13.4 M2).
- `src/tether/context/inbox_store.py` — `InboundInbox` ABC + `SqliteInbox` impl.
- `src/tether/context/migrations/004_inbox.sql` (+ `.rollback.sql`).
- `src/tether/runtime/task_supervisor.py` — `SupervisedTask` (synthesis §13.4 M3).
- `src/tether/core/connector_registry.py` — drain-task wiring in
  `start_connector` / `stop_connector`.
- `src/tether/app/http/routers/connectors.py` — `GET /inbox` + `POST /inbox/mark-seen`.
- `tests/unit/context/test_inbox_store.py` — 20 unit tests on the inbox surface.
- `tests/integration/test_connector_inbox_drain.py` — end-to-end drain coverage.

## References

- `files/investigations/_synthesis.md` §10 (whole), §10.8 #5, §11.6, §4 Phase 6.5
- Connectors Spec: `session-state/5c8a15fc-.../plan.md` §3.4, §3.6
- `src/tether/context/inbox_store.py` (`InboundInbox` ABC + `SqliteInbox`)
- `src/tether/context/migrations/004_inbox.sql`
- ADR-0005 (`ConnectorRegistry`), ADR-0008 (persistence stack)
