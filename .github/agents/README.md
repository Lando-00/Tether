# Tether Refactor — Sub-Agent Templates

This directory holds reusable prompt templates the orchestrator (Copilot CLI main
session) fills in and ships to sub-agents. They exist to keep prompts consistent
across launches, to bake in invariants that never change per-launch, and to
survive context compaction.

> Authoritative repository digest:
> [`../../docs/refactor/synthesis-2026-05.md`](../../docs/refactor/synthesis-2026-05.md).
> The original private session-state plan is not required.

## Roles

| Template | When the orchestrator launches it | Default model |
|---|---|---|
| `implementation.agent.md` | Per todo from SQL `todos` table | Per-phase (see §13.2 model matrix) |
| `code-review.agent.md` | After every implementation diff | `claude-haiku-4.5` (cheap; runs every diff) |
| `rubber-duck.agent.md` | Phase 2 / 3 / 5 / 6.5 boundaries; 3× parallel | `claude-opus-4.7-xhigh`, `claude-opus-4.7-1m`, `gpt-5.5` |

## Usage rules (orchestrator-side)

1. **Fill, do not edit**. Templates use `{{var}}` placeholders. Substitute
   values per-launch from the SQL todo + the relevant `_synthesis.md` section.
   Never modify the template file itself for one-off needs — drift defeats the
   purpose. Add new baked-in rules only if they apply to *every* future run.

2. **Cite, do not paraphrase**. When filling `{{scope_files}}` /
   `{{acceptance}}`, copy the exact lines from `_synthesis.md` and the
   connector spec. Cite section numbers (e.g. `§4.5`, spec `§8.1`).

3. **Background, then review**. Implementation sub-agents run as
   `mode: "background"`. Code-review sub-agents run `sync` against the diff.

4. **Branch per todo**. Implementation sub-agents commit to
   `refactor/{{todo_id}}`. Reviewer reads that branch's diff against `main`.

5. **Hardware tests are user-run**. Sub-agents write the test under
   `tests/hardware/` with `@pytest.mark.hardware`, but never invoke
   `pytest -m hardware` themselves. The orchestrator batches those into the
   phase-boundary `[USER VERIFY ON SNAPDRAGON]` block.

6. **Pre-flight env in every run**. Use the provider-appropriate interpreter:
   a python.org ARM64 stdlib venv for GenieX-only work, or the x64 conda env
   under Prism for MLC. Invoke its `python.exe` directly in automation because
   activation does not persist across one-shot shells. See
   [`../copilot-instructions.md`](../copilot-instructions.md).

7. **Backend terminology**. The MLC backend is the Adreno X1 GPU via OpenCL.
   It is **not** an NPU. GenieX is the shipped Hexagon NPU provider.
   The implementation template enforces this in writing.

## Lifecycle (per todo)

```
                  SQL ready-todo query
                          │
                          ▼
          ┌───────────────────────────────┐
          │  Orchestrator fills template  │
          │  → implementation.agent.md    │
          └───────────────┬───────────────┘
                          │ task(mode=background)
                          ▼
              ┌───────────────────────┐
              │  Implementation agent │ → unified diff on refactor/{{todo_id}}
              └───────────┬───────────┘
                          │ completion notification
                          ▼
          ┌───────────────────────────────┐
          │  Orchestrator fills template  │
          │  → code-review.agent.md       │
          └───────────────┬───────────────┘
                          │ task(mode=sync)
                          ▼
              ┌───────────────────────┐
              │   Code-review agent   │ → findings (BLOCKING/REVISION/NIT)
              └───────────┬───────────┘
                          │
                  Orchestrator decides
                  ┌──────┼──────┐
                  ▼      ▼      ▼
              commit  revise  escalate
                          │      │
                       write_  ask_user
                       agent   tool
```

At Phase 2 / 3 / 5 / 6.5 boundaries, the orchestrator additionally launches
**three** rubber-ducks in parallel before requesting the user gate.

## Files

- `README.md` — this file
- `implementation.agent.md` — implementation prompt template
- `code-review.agent.md` — diff review prompt template
- `rubber-duck.agent.md` — architectural critique prompt template
