# ADR-0009: `SqliteInbox` in shared `data/tether.db` (deviates from spec §3.6)

- **Status**: Accepted (Phase 6.5 of refactor)
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
shared `data/tether.db`, managed by yoyo migration `003_inbox.sql`. This was explicitly
ratified by the user in §11.6 (table row "§10.8 #5 — Single DB with `inbound_events`
table — ✅ ratified").

- **One database file**: `data/tether.db` holds everything. The spec's `inbox.db` filename
  becomes a non-issue; the per-connector filesystem layout (`data/connectors/<id>/` for
  credentials/secrets) is unchanged.
- **`InboundInbox` ABC** in `src/tether/context/inbox_store.py`, with **`SqliteInbox`** impl
  using the same `aiosqlite` connection as `AsyncSqliteStore`. Methods: `append`,
  `append_many`, `list_unread`, `mark_seen`, `search`, `prune_older_than`. `append_many` is
  a single SQLite transaction.
- **`read_at` is Tether-side inbox state**, NOT platform read state. `mark_seen` records
  that the inbox showed the event to a downstream consumer (UI, API caller). Connectors that
  want to mark messages read on the upstream platform (WhatsApp, Gmail) expose a separate
  `*_inbox_mark_seen` tool (§10.6 of the synthesis defines the capability tags).
- **Schema** (single table `inbound_events`) with `idx_inbox_unread` (`read_at IS NULL`)
  and `idx_inbox_received` indexes. LIKE-based `search` is acceptable for v1; FTS5 deferred.
- **Wired into `Engine.from_settings()`**: `prune_older_than(max_age_days)` runs once at
  startup. `ConnectorRegistry.start_connector(id)` spawns a drain task per connector with
  non-`None` `inbound_stream()`; the drain task persists yielded events via
  `inbox.append_many()`. On drain-task error: connector health → `degraded`, exponential
  backoff before re-subscribe.
- **HTTP routes**: `GET /api/v1/connectors` returns connector health states; `GET
  /api/v1/connectors/{id}/inbox?unread=true&limit=20` returns unread events. `/readyz`
  continues to return 200 even when a connector is `unconfigured` / `logged_out` (spec
  §8.4 acceptance).
- **`Engine.aclose()` ordering**: cancel drain tasks → stop connectors → stop provider →
  close store. Shutdown is bounded by the connector budget per ADR-0005.

## Consequences

### Positive
- One DSN, one connection, one migration track, one transaction boundary. Simpler
  lifecycle across all of `SqliteInbox` + `AsyncSqliteStore` + `tool_audit`.
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

## References

- `files/investigations/_synthesis.md` §10 (whole), §10.8 #5, §11.6, §4 Phase 6.5
- Connectors Spec: `session-state/5c8a15fc-.../plan.md` §3.4, §3.6
- `src/tether/context/inbox_store.py` (`InboundInbox` ABC + `SqliteInbox`)
- `src/tether/context/migrations/003_inbox.sql`
- ADR-0005 (`ConnectorRegistry`), ADR-0008 (persistence stack)
