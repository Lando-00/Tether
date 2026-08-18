# Domain Docs

Tether is a single-context repository.

Before exploring or changing domain behaviour:

1. Read `CONTEXT.md`.
2. Read `docs/adr/` decisions relevant to the change.
3. Read `docs/architecture.md` and
   `docs/refactor/synthesis-2026-05.md` when the change crosses a layer or
   affects a locked invariant.

Use the vocabulary from `CONTEXT.md` in issue titles, specifications, tests,
and implementation notes. If a change conflicts with an ADR or locked
synthesis decision, surface that conflict instead of silently overriding it.
