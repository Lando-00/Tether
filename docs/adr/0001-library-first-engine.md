# ADR-0001: Library-first composition root (`Engine` class)

- **Status**: Accepted (Phase 2 of refactor)
- **Date**: 2026-05 (Phase 2)
- **Synthesis citation**: §2.1, §2.3, §3.1, §4 Phase 2 (steps 21–25)

## Context

Pre-refactor, the only way to use Tether was to spin up the FastAPI server. Business logic,
HTTP routing, and lifecycle were entangled: `tether_service/app/__init__.py` ran
`create_app()` at import time (opening GPU/DB/network as a side effect of `import
tether_service`), `GenerationService` was instantiated by a `ServiceFactory`, and `load_settings()`
was re-read on every chat turn from inside the orchestrator and tool runner. There was no
public Python API for embedding Tether in a non-HTTP process.

## Decision

Adopt a **library-first** architecture with a single composition root, the `Engine` class:

- **One factory**: `Engine.from_settings(settings)` is the *only* place dependencies get
  wired together. No IoC framework; manual factory pattern.
- **Async context manager**: `async with Engine.from_settings(settings) as eng: ...` — async
  only, no sync wrappers. `Engine.__aenter__` runs tool/connector startup;
  `Engine.aclose()` tears everything down (provider → connectors → store).
- **Typed streaming API**: `async for ev in eng.chat(...) -> AsyncIterator[WireEvent]` yields
  Pydantic objects, not bytes. HTTP/SSE/NDJSON are thin transport wrappers.
- **Public surface** (`tether/__init__.py` re-exports): `Engine`, `Settings`, `load_settings`,
  `WireEvent`, `BaseTool`, `Tool`, `ModelProvider`. Lower-level building blocks remain
  importable from their packages directly (the once-planned `tether.advanced` namespace was
  dropped per §11 R15).
- **No import-time side effects**: importing `tether` does NOT touch GPU/DB/network and does
  NOT import `mlc_llm` (verified by a clean-venv test per §11 R8).
- **HTTP layer becomes a thin adapter**: `create_asgi_app(engine)` mounts routers; the
  `__main__` entrypoint wires `engine = Engine.from_settings(...); app =
  create_asgi_app(engine); uvicorn.run(app, ...)`.

## Consequences

### Positive
- A user can `from tether import Engine` and embed Tether in any Python process — CLI, GUI,
  another web app, a notebook — with no FastAPI dependency.
- The composition root is one function call; reasoning about dependency wiring is trivial.
- Lifecycle is deterministic: `__aenter__` → `aclose()` covers tools, connectors, provider,
  and store in defined order.
- Tests can build an `Engine` with explicit `Settings(...)` — no env var gymnastics.

### Negative
- All public API methods are `async`. CLI and notebook users must run inside an event loop.
- Backward compatibility surface is wider: `tether_service` must remain a re-export alias
  for one cycle (see ADR-0013).

### Trade-offs accepted
- We rejected sync wrappers around the async API; the `asyncio.run(...)` boilerplate is the
  caller's responsibility.

## Alternatives considered

- **IoC container (e.g. `dependency_injector`)** — rejected as overkill for a single composition
  root with stable dependency shape.
- **Keep `GenerationService` + `ServiceFactory`** — rejected: `ServiceFactory` was unused dead
  code (synthesis §6 bug #9), and `GenerationService` had no library-first surface.
- **Sync API with optional async** — rejected: doubles the maintenance surface; libraries
  consuming Tether are themselves likely async (FastAPI, Discord bots, etc.).

## References

- `files/investigations/_synthesis.md` §2.1, §2.3, §3.1, §4 Phase 2, §11 R8, §11 R9, §11 R15
- `src/tether/engine.py`, `src/tether/bootstrap.py`, `src/tether/__init__.py`
- `src/tether/adapters/http/api.py` (post-refactor: ~50 LOC create_app)
