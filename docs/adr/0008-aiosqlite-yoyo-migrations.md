# ADR-0008: Persistence — `aiosqlite` + WAL + yoyo-migrations

- **Status**: Accepted (Phase 6 of refactor)
- **Date**: 2026-05 (Phase 6)
- **Synthesis citation**: §3.6, §4 Phase 6 (steps 59–66)

## Context

The pre-refactor persistence layer mixed several concerns: `SQLAlchemy` was listed in
`requirements.txt` but only `llm_service/` (retiring) used it; the active code path used
raw `sqlite3` synchronously with manual schema management. There was no migration
mechanism, so schema evolution required hand-edited SQL. The default DB path was
`./data/tether.db` relative to the caller's CWD — a library consumer importing Tether in
another project would write into that project's directory. Schema lacked the correlation
IDs (`turn_id`, `seq`, `tool_call_id`) needed by Wire protocol v2 (ADR-0006) and the audit
log needed by observability (ADR-0010).

## Decision

Adopt **`aiosqlite`** as the async store, **WAL** journal mode, and **`yoyo-migrations`**
for forward-only SQL files:

- **Async SQLite via `aiosqlite`** (lazy-imported). One connection per `Engine`, opened in
  WAL mode, `synchronous=NORMAL`. **`SQLAlchemy` is dropped** from `requirements.txt`.
- **`yoyo-migrations`** for forward-only SQL files in
  `src/tether/context/migrations/`. First migration `001_current_schema.sql` is a baseline
  matching today's tables; `002_v2_schema.sql` adds Wire-v2 correlation columns and the
  audit log; `003_inbox.sql` adds the inbox table (ADR-0009).
- **Schema v2** — five core tables + an audit table + a debug view:
  - `sessions`, `turns`, `messages`, `tool_calls`, `raw_events`.
  - `tool_audit` (B5-owned columns: `args_sha256`, `capabilities`, `status`, `error_kind`,
    `duration_ms`, `correlation_id`).
  - `turn_timeline` view backing the debug endpoint.
  - `messages.role` widened to `('system','user','assistant','thinking','tool',
    'tool_result')`.
  - `messages.tool_call_id` FK + partial unique indexes ensure exactly one `tool` row + one
    `tool_result` row per `tool_call_id`.
  - `raw_events(turn_id, seq) UNIQUE` — durable replayable order for SSE `Last-Event-ID`
    reconnect.
  - `tool_calls.status` includes `cancelled` (per §11 R3).
  - `session_state` table (per §12.3 Seam B) for future `NotebookOrchestrator` atomic-fact
    list and per-session JSON state.
- **`AsyncSqliteStore`** (M2) implements the widened `SessionStore` ABC (`turn_id`, `seq`,
  `tool_call_id`, plus `get_state` / `set_state` / `delete_state` for `session_state`).
  Lazy-import gates `aiosqlite` so library consumers without a DB still load.
- **Default DB path**: `platformdirs.user_data_dir("Tether")` when
  `Settings.storage.dsn is None`. Library consumers stop writing into other projects' CWDs.
- **`MemorySessionStore` rewritten** to share serializer/reconstruction logic with
  `AsyncSqliteStore` and pass the same contract suite (B1 + B2). Used by tests; same ABC
  as the real store.
- **Migration data strategy**: green-field rebuild after backup (user-ratified §11.6). The
  existing `data/tether_dev.db` and `mlc_sessions.db` are archived, not migrated.

## Consequences

### Positive
- One async store, no sync→async wrapping; no event-loop blocking on DB I/O.
- `yoyo-migrations` is lighter than alembic (no model graph), suits raw-SQL persistence.
- WAL + write-batching gets us concurrent reads during a chat turn without locking.
- `platformdirs` keeps library consumers from polluting other projects' CWDs.
- `MemorySessionStore` ↔ `AsyncSqliteStore` share a contract suite — tests can pick either
  without divergence.

### Negative
- One extra dependency (`yoyo-migrations`).
- Migration files are append-only; reversing a bad change requires a new forward
  migration. Acceptable for a single-user app.

### Trade-offs accepted
- Concurrent writers across processes are not supported (one Engine = one connection). The
  app is single-process by design.

## Alternatives considered

- **Stay on `sqlite3` sync + ad-hoc schema** — rejected: blocks the event loop; no migration
  path; correlation columns can't land cleanly.
- **`SQLAlchemy` + `alembic`** — rejected: heavy for our raw-SQL needs; `llm_service/` used
  it and that code is retiring.
- **`asyncpg` / Postgres** — rejected: overkill for a single-user local-first app; SQLite
  handles 9 GB of message history without blinking.

## References

- `files/investigations/_synthesis.md` §3.6, §4 Phase 6, §11 R3, §11.6, §12.3 Seam B
- `src/tether/context/sqlite_store.py`, `src/tether/context/memory_store.py`
- `src/tether/context/migrations/00{1,2,3}_*.sql`
- ADR-0006 (Wire v2 correlation IDs), ADR-0009 (inbox table), ADR-0010 (audit log columns)
