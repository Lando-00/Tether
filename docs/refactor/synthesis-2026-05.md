# Tether — Refactor Synthesis Digest (2026-05)

> Curated successor to the session-state `_synthesis.md` that the
> refactor was conducted against. The original lived in
> `~/.copilot/session-state/...` and is not in the repo. This file
> captures the **invariants, deferrals, and locked decisions** that
> survive into mainline, so a fresh clone has a self-contained
> canonical reference.

## §1 Locked decisions

- Library-first composition root via `Engine` (ADR-0001).
- Strict typed Pydantic config (ADR-0002).
- GC-disabled MLC shutdown (ADR-0003).
- Tool v2 with `@tool` decorator + `ToolExecutionContext` (ADR-0004).
- Connector registry with `{connector_id}_` prefix (ADR-0005).
- Wire protocol v2 NDJSON + Seam D parser strategy (ADR-0006).
- Orchestrator strategy ABC (ADR-0007).
- aiosqlite + yoyo migrations (ADR-0008).
- Shared `data/tether.db` for sessions + inbox (ADR-0009).
- structlog + opt-in OTel (ADR-0010; OTel experimental per P0-I).
- Outbound URL allowlist (ADR-0011).
- CSRF + CORS + TrustedHost middleware ordering (ADR-0012).
- `src/` layout + `tether_service` deprecation alias (ADR-0013).
- Pinned CodeLinaro MLC-LLM 2025.06.r1 (ADR-0014).
- Single-user outbound-send doctrine (ADR-0015).
- MLC isolation rule (ADR-0016).
- Seam C `get_practical_context_window` deferred (ADR-0017).

## §2 Extension seams

- **Seam A — ModelProvider**: live; MLC + Nexa stub.
- **Seam B — Orchestrator strategy**: live; ChattyAgent + Notebook stub.
- **Seam C — Practical context window**: deferred per ADR-0017.
- **Seam D — Per-provider parser**: partial; `system.prompt` move and
  `create_parser()` ABC are open follow-ups.

## §3 Open deferrals

- `tether_service` alias removal (target: v0.2.0).
- NDJSON v0 deprecation removal (target: v0.2.0).
- `stream_typed()` v2 cutover (Phase 9 carrying via P0-E guard until done).
- Real `async_span` (M4) — paired with OTel lifetime tracking
  (P0-I follow-up).
- WhatsApp / Gmail connectors (Phases 2a / 2b).

## §4 Phase-9 P0 round 1 fixes (this milestone)

- **P0-A** — Storage DSN single source of truth.
- **P0-B1/B2/B3** — Tool result sandbox; TrustedHost + JSON
  content-type; CSRF token file.
- **P0-C** — Cancel finalize.
- **P0-D** — `hw_reset` GC-disable.
- **P0-E** — Parser list-chunk guard.
- **P0-F** — Connector start await.
- **P0-G** — Test antipattern purge.
- **P0-H** — `async_span` truth-pass.
- **P0-I** — OTel experimental gate.
- **P0-J** — This file + drift gate (Tribunal §3 P0-20).

## §5 Pointers

- Architecture: [`../architecture.md`](../architecture.md)
- ADRs: [`../adr/`](../adr/)
- Runbooks: [`../runbooks/`](../runbooks/)
- Audit verdicts (Phase-9 input): kept private in session-state;
  **summarized** here in §4.
