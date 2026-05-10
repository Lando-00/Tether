# ADR-0004: Tool v2 — `BaseTool` + `@tool` + `ToolExecutionContext`

- **Status**: Accepted (Phase 4 of refactor)
- **Date**: 2026-05 (Phase 4, with `ToolExecutionContext` plumbed in Phase 0/4 per §10)
- **Synthesis citation**: §3.2, §4 Phase 0 step 8, §4 Phase 4 (steps 40–48), §10.5

## Context

The pre-refactor tool layer had four entangled defects:
1. **Three-way contract mismatch** (synthesis §6 bug #4): `Tool` ABC expected `run(args:
   dict)`, `BaseTool.run` expected `**kwargs`, `ToolRunner` called `tool.run(**args)`.
   Implementing the public `Tool` ABC directly was broken.
2. **`auto_schema` silently degraded** `Optional`, `Literal`, `List`, defaults
   (§6 bugs #5, #6); `**kwargs` produced a phantom required string field.
3. **Discovery was YAML-only** via a `tools.registry` list — adding a tool meant editing
   config; no decorator, no entry-point support.
4. **No execution-time context**: tools couldn't see the calling `session_id`, `turn_id`,
   `tool_call_id`, or downstream user-confirmation flags. Connector tools (Phase 4.5+) need
   this.

## Decision

Adopt **Tool v2** with a hybrid authoring model:

- **`BaseTool` v2** (`src/tether/tools/base.py`): hybrid — author writes either an
  `Inputs: ClassVar[type[ToolInputs]]` Pydantic model **OR** annotated `run()` parameters.
  Framework-facing entry is `BaseTool.invoke(args: dict, context:
  Optional[ToolExecutionContext] = None)`; author-facing is `run()`. This resolves the
  three-way contract mismatch.
- **`@tool` decorator + discovery**: `pkgutil` package scan of `tether.tools` plus
  `entry_points("tether.tools")` for third-party tool packages. YAML's residual role
  shrinks to `tools.disabled: [...]` allow-list and per-tool `args:` only. The legacy
  `registry`/`enabled` lists are removed.
- **Schema generation v2**: `auto_schema` handles `Optional`, `Literal`, `List`,
  `Annotated[..., Field(...)]`, defaults; `*args`/`**kwargs` filtered out (fixes phantom
  required string). Output is OpenAI-style `{"type":"function","function":{...}}`.
- **Lifecycle hooks**: `async startup()` / `async shutdown()` non-abstract on `BaseTool`;
  wired into `Engine.__aenter__` / `Engine.aclose()` as `startup_all()` / `shutdown_all()`.
  Tools that share state (e.g. `BraveSearchClient`'s httpx pool) open it once.
- **`ToolExecutionContext` dataclass** (`src/tether/core/types.py`): per-call frozen
  context carrying `session_id`, `turn_id`, `tool_call_id`, `request_id`, plus
  `user_confirmed_send: bool = False`. Threaded `Orchestrator → ToolRunner →
  BaseTool.invoke(args, context=...)`. Existing tools ignore it; connector tools (e.g.
  `*_confirm_send`) consume it. Phase 0 plumbs the type and wiring; the
  `user_confirmed_send` classifier ships with WhatsApp/Gmail in a later session.
- **Capability metadata**: `capabilities: ClassVar[frozenset[str]]` populated from B5's
  canonical `Capability` enum; `cost_hint: Literal["fast","slow","expensive"]`.
  Allowlist enforcement gated by `Settings.security.capability_allowlist.enabled` (default
  `False`).
- **Tool-result safety**: results wrapped as `<<TOOL_RESULT name="..." call_id="...">>...
  <<END_TOOL_RESULT>>` in history. `BaseTool.sanitize_result` opt-in hook (default no-op).

## Consequences

### Positive
- Adding a tool = one file + `@tool` decorator. No YAML edits.
- Schema is faithful to Python types — model sees real enums, defaults, optionals.
- Connector tools have first-class access to per-call context without subclassing the
  orchestrator.
- Tools can hold long-lived clients via `startup`/`shutdown` (e.g. `BraveSearchClient`
  shares one httpx pool).

### Negative
- The dual authoring style (`Inputs` model vs annotated `run`) is a documentation surface;
  the tool-author guide must explain when to choose each.
- `ToolExecutionContext` defaults `user_confirmed_send=False` — connector confirm-send
  tools refuse until a future phase wires up the classifier (acceptable; the safer default).

### Trade-offs accepted
- We accept the small `BaseTool.invoke` shim layer to keep `run()` ergonomic for authors.

## Alternatives considered

- **Pure annotation-only API** (no `Inputs` model) — rejected: complex schemas (nested
  validators, custom error messages) want Pydantic.
- **Pure `Inputs`-only API** — rejected: the simple-tool case (`def run(self, query: str)`)
  is too verbose if it must declare a `ToolInputs` model.
- **Separate context object per tool** — rejected: tools shouldn't depend on a tool-specific
  context type; one frozen `ToolExecutionContext` is enough for everything we know about.

## References

- `files/investigations/_synthesis.md` §3.2, §4 Phase 0 step 8, §4 Phase 4, §6 bugs #4–#6, §10.5
- `src/tether/tools/base.py`, `src/tether/tools/registry.py`
- `src/tether/core/types.py` (`ToolExecutionContext`), `src/tether/core/capabilities.py`
