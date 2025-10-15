# Copilot Onboarding — Tether
### MLC-LLM Session Service with MCP Architecture
   
**Tether** is a Python/FastAPI service offering session-based, token-streaming access to MLC-LLM models.  
It stores chat history in SQLite, follows a Model-Context-Protocol (MCP) architecture, and supports function-calling tools via NDJSON events.

### Agent Goals
- Deliver code that builds and runs locally on first try  
- Follow MCP layering and SOLID principles (depend on interfaces)  
- Reuse existing abstractions (e.g., parser interface) before adding new ones

   

## Repo & Stack
### Quickstart (PowerShell)
```powershell
conda activate mlc-venv2
pip install -r requirements.txt
# Start API server
python -m llm_service.app         # http://localhost:8090/docs
# Or debug via VS Code: .\run_debug.ps1
```

### Big Picture Architecture
- **CLI** (`cli_chat.py` / `tests/One_shot_test`): one-shot prompts & demo parsers.
- **API** (`llm_service/app.py` ➔ `protocol/api/app_factory.py`): FastAPI server using DI.
- **Model**: `llm_service/model` loads MLC-LLM engine, applies stream patches.
- **Context**: `context/session_store.py` persists sessions in `mlc_sessions.db*` (enable WAL).
- **Protocol**:
  - `core`: interfaces (e.g., `StreamParser`), types, config, logging.
  - `orchestration`: `detector`, `executor`, `emitter` flow in `controller.py`.
  - `service`: `GenerationService` defines streaming pipeline & hooks.
  - `api`: routers, schemas (`protocol/api/schemas.py`), lifespan.
- **Tools**: modules under `tools/`; register each tool in `protocol/orchestration/controller.py`.

### Developer Workflows
- **Bootstrap**: see Quickstart.
- **Run CLI**: `python cli_chat.py --session demo --prompt "Hello"`.
- **Smoke Test**:
```powershell
curl -N http://localhost:8090/api/v1/generate/stream `
  -H "Content-Type: application/json" `
  -d '{"session_id":"demo","prompt":"Say hello"}'
```
- **Tests**: `pytest -q`; compatibility shim in `pytest_shim.py`; one-shot in `tests/One_shot_test`.
- **Lint**: `ruff check .`; `black .`.

### Conventions & Patterns
- **Streaming NDJSON**: output events with `type` fields (`text`, `tool_started`, `tool_complete`).
- **Parser Interface**: implement `StreamParser` in `protocol/core/interfaces.py`, inject into `GenerationService`.
- **Session DB**: local SQLite (`mlc_sessions.db*`); WAL mode recommended for concurrent writes.
- **Adding Tools**: create file in `tools/`; import & register in `controller.py` orchestrator.

### Navigation Tips
- Trace streaming: `protocol/service/generation_service.py` ➔ `orchestration/orchestrator.py` ➔ `emitter.py`.
- Startup & DI: `protocol/api/app_factory.py`, then inspect routers in `protocol/api/routers`.

### Integration Points
- **Native libraries**: see `mlc_llm-utils-win-x86-*/bin` for DLLs and `tvm_runtime.dll`.
- **Environment variables**: `MLC_DIST_PATH`, `MLC_SQLITE_URL`, `MLC_HOST`, `MLC_PORT`.

## Common pitfalls & workarounds
- No models found: ensure `dist/` exists and `MLC_DIST_PATH` points to it.
- OpenCL missing/slow: system falls back to CPU; tune prefill chunk size/KV cache in MLC config.
- SQLite locks: prefer WAL mode (`PRAGMA journal_mode=WAL;`) for concurrent writes.

## Project Layout (where to change things)
```bash
llm_service/
  model/                 # MLC engine load/infer/cache
  context/               # Sessions + message persistence (SQLite)
  protocol/
    core/                # Interfaces, types, config, logging, execution strategies
    orchestration/       # Tool detection, parsing, execution, emission, controllers, adapters
    service/             # ProtocolService, GenerationService (streaming entrypoints)
    api/                 # FastAPI schemas, routers, app factory
tools/                   # Tool registry + examples
cli_chat.py              # CLI entry
# Copilot Onboarding — Tether

Tether is a Python/FastAPI MCP-architected service for session-based streaming with MLC-LLM models.

## Goals for AI Agents
- Generate buildable code on first run (PowerShell on Windows).
- Adhere to MCP layers (model, context, protocol) using interfaces.
- Reuse existing abstractions before extending.

## Key Components
- **CLI**: `cli_chat.py` and `tests/One_shot_test` for one-shot demos.
- **API**: `llm_service/app.py` ➔ `protocol/api/app_factory.py`.
- **Model**: `llm_service/model/mlc_engine.py`, `mlc_stream_patch.py`.
- **Context**: `context/session_store.py` with SQLite WAL (`mlc_sessions.db`).
- **Protocol**:
  - `protocol/core/interfaces.py` (e.g. `StreamParser`).
  - `protocol/orchestration` (`detector.py`, `executor.py`, `emitter.py`, `controller.py`, `orchestrator.py`).
  - `protocol/service/generation_service.py` for streaming pipeline.
  - `protocol/api/routers/` and `schemas.py`.

## Developer Workflows
```powershell
# Setup
conda activate mlc-venv2; pip install -r requirements.txt
# Run API
python -m llm_service.app          # http://localhost:8090/docs
# Debug
.\run_debug.ps1
# Tests & Lint
pytest -q; ruff check .; black .
```

## Conventions & Integration
- NDJSON events: `text`, `tool_started`, `tool_complete`, etc.
- Implement `StreamParser` in `protocol/core/interfaces.py` and inject into `GenerationService`.
- Register new tools in `tools/` and `protocol/orchestration/controller.py`.
- Environment variables: `MLC_DIST_PATH`, `MLC_SQLITE_URL`, `MLC_HOST`, `MLC_PORT`.

## Navigation Tips
- Trace a request: `protocol/service/generation_service.py` → `protocol/orchestration/orchestrator.py` → `protocol/orchestration/emitter.py`.
- DI startup: `protocol/api/app_factory.py` → routers in `protocol/api/routers/`.