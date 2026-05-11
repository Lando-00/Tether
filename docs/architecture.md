# Tether — Architecture

> Post-Phase-8 architecture. This is the **living map** for new contributors and AI agents.
> Locked decisions / phase digest: [`refactor/synthesis-2026-05.md`](./refactor/synthesis-2026-05.md).

### Living doc; see ADR-0010 for current observability status

> **Truth-pass note (Phase 9 P0-H)**: this document is the post-Phase-8
> architecture diagram + invariants.  Items marked **DEFERRED** (M4
> `async_span`, Seam C `get_practical_context_window`, parts of Seam D) are
> documented here so the seams stay visible, but they are *not* implemented
> in the current build.  Treat them as design contracts the next phase has
> to honour, not as features available today.  Tribunal §3 P0-18.

Tether is a single-user, local-first FastAPI service that streams chat completions from
on-device LLMs (currently MLC-LLM on Snapdragon X Elite Adreno GPU) with first-class
function calling, persistent session history, and a connector framework for inbound-read /
outbound-send personal data integrations.

The architecture is **library-first**: HTTP, CLI, and (future) GUI are thin adapters over a
single `Engine` composition root. Anyone can `from tether import Engine` and bypass HTTP.

---

## 1. Layered overview

```mermaid
flowchart TB
    subgraph adapters["Adapters (thin)"]
        HTTP["HTTP — FastAPI<br/>app/http/api.py"]
        CLI["CLI — Typer<br/>tether.cli.main"]
        Lib["Library — from tether import Engine"]
    end

    subgraph engine["Engine (composition root)"]
        Eng["Engine<br/>engine.py"]
    end

    subgraph protocol["Protocol layer"]
        Orch["Orchestrator (strategy)<br/>ChattyAgentOrchestrator"]
        GenSvc["GenerationService<br/>protocol/service/"]
        Parser["SlidingParser<br/>parsers/"]
    end

    subgraph tools["Tool system"]
        Reg["ToolRegistry"]
        Runner["ToolRunner<br/>(+ ToolExecutionContext)"]
        Base["BaseTool<br/>(@tool decorator)"]
        Builtin["TimeTool · WeatherTool<br/>WebSearchTool · …"]
    end

    subgraph connectors["Connector framework"]
        Connector["Connector ABC"]
        ConnReg["ConnectorRegistry<br/>(prefix enforcement)"]
        EchoCon["EchoConnector (fixture)"]
        WhatsApp[":WhatsApp (future)"]:::future
        Gmail[":Gmail (future)"]:::future
    end

    subgraph context["Context (persistence)"]
        SessStore["SqliteSessionStore<br/>chat history"]
    Inbox["SqliteInbox<br/>inbound_events"]
        AuditDB["tool_audit table"]
        AsyncBase["AsyncSqliteStore (M2)"]
    end

    subgraph providers["Model providers (Seam A)"]
        MP["ModelProvider ABC"]
        MLC["MLCProvider<br/>Adreno OpenCL"]
        Nexa[":NexaProvider (stub)"]:::future
        Ollama[":OllamaProvider (reserved)"]:::future
    end

    subgraph hardware["Hardware boundary"]
        Watchdog["HardwareWatchdog"]
        Daemon["daemon_thread_call (M1)"]
    end

    subgraph obssec["Observability + Security"]
        ReqId["RequestIdMiddleware"]
        Otel["OpenTelemetry (optional)"]
        Allowlist["Outbound URL allowlist"]
        CsrfCorsTH["CSRF + CORS + TrustedHost"]
    end

    HTTP --> Eng
    CLI --> Eng
    Lib --> Eng
    Eng --> Orch
    Eng --> Reg
    Eng --> ConnReg
    Eng --> SessStore
    Eng --> MP
    Orch --> GenSvc
    Orch --> Parser
    Orch --> Runner
    Runner --> Base
    Reg --> Base
    Builtin --> Base
    ConnReg --> Connector
    EchoCon --> Connector
    WhatsApp -.-> Connector
    Gmail -.-> Connector
    Connector --> Inbox
    Connector --> Reg
    GenSvc --> MP
    MP --> MLC
    MP -.-> Nexa
    MP -.-> Ollama
    MLC --> Watchdog
    MLC --> Daemon
    SessStore --> AsyncBase
    Inbox --> AsyncBase
    HTTP --> ReqId
    HTTP --> CsrfCorsTH
    Runner --> AuditDB
    Runner -.-> Allowlist

    classDef future stroke-dasharray: 5 5, stroke:#888, color:#888
```

Solid arrows = wired today. Dashed/grayed = scaffolded seams (no runtime impl).

---

## 2. Request lifecycle (a single chat turn)

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant FastAPI as FastAPI<br/>(/api/v1/chat/stream)
    participant Engine
    participant Orch as ChattyAgentOrchestrator
    participant Store as SqliteSessionStore
    participant Provider as MLCProvider
    participant Parser as SlidingParser
    participant Runner as ToolRunner
    participant Tool as Tool (e.g. web_search)

    Client->>FastAPI: POST /chat/stream {session, message}
    FastAPI->>FastAPI: RequestId · CSRF · CORS · TrustedHost
    FastAPI->>Engine: stream(req)
    Engine->>Orch: orchestrate(turn)
    loop max_tool_loops (default 5)
        Orch->>Store: append user/turn message
        Orch->>Store: get_history(session)
        Store-->>Orch: [messages incl. prior tool calls/results]
        Orch->>Provider: stream(messages)
        Provider-->>Orch: chunks (text + EOS)
        Orch->>Parser: feed(chunk)
        Parser-->>Orch: ParseEvent (text · tool_call_detected)
        alt no tool call
            Orch-->>Client: NDJSON: text_delta
        else tool call detected
            Orch-->>Client: NDJSON: tool_call (status=started)
            Orch->>Runner: run(name, args, ctx)
            Note right of Runner: ctx = ToolExecutionContext<br/>{session_id, turn_id,<br/>last_user_message,<br/>user_confirmed_send}
            Runner->>Tool: run(**args, ctx=ctx)
            Tool-->>Runner: result | error
            Runner->>Store: append tool_call + tool_result
            Runner->>Store: insert tool_audit row (256KB cap)
            Runner-->>Orch: ToolOutcome
            Orch-->>Client: NDJSON: tool_result (status=ok|error)
            Note over Orch: continue loop — model sees result next iteration
        end
    end
    Orch-->>Client: NDJSON: message_stop
    Orch->>Store: complete_turn (200ms timeout)
```

**Streaming envelope (v2 default)**: `message_start` → N×(`text_delta` | `tool_call` | `tool_result`) → `message_stop`.
Legacy v0 envelope (`text` / `tool_started` / etc.) available via `Accept: application/x-ndjson; version=0` until deprecation.

**Cancel-path budgets** (synthesis §3.5): tool-grace 250ms, persist budget 200ms on
`_audit_tool_call` and `complete_turn` so a slow store can't block client cancellation.

---

## 3. Extension seams (synthesis §12)

These are the four pre-engineered pluggability points. Each is an ABC plus a default
implementation; adding a second impl is a config change + one new class.

| Seam | Where | What | Status |
|---|---|---|---|
| **A — ModelProvider** | `providers/types.py::ModelProvider` | Swap inference backend. Default: `MLCProvider`. Future: `NexaProvider` (NPU), `OllamaProvider` (reserved). | ✅ ABC live; MLC impl + Nexa stub |
| **B — Orchestrator strategy** | `protocol/orchestration/types.py::Orchestrator` | Swap conversation policy. Default: `ChattyAgentOrchestrator` (ReAct-style chatty loop). Future: Ralph Loop / Notebook of Atomic Facts (small-context strategy). Selected via `mode` field on `StreamRequest` + per-session state. | ✅ ABC live; one impl |
| **C — Practical context window** | `providers/types.py::get_practical_context_window(model_name, ram_budget_gb)` | Per-provider RAM-aware effective context. Used to clamp history before send. | ❌ DEFERRED — synthesis §12.4 work; ABC method not yet on ModelProvider; tracked as a follow-up. |
| **D — Per-provider parser** | `protocol/parsers/types.py::Parser` | Per-provider tool-call detection. MLC: `<<function_call>>` marker (`SlidingParser`). Future: provider-native function-call schemas (Ollama JSON mode, NexaSDK OpenAI-compat). System prompt lives under `providers.<id>.args.system_prompt` so parsers and prompts ship together. | ⚠️ Partial — per-turn factory wired in orchestrator; `Provider.create_parser()` ABC method NOT yet on ModelProvider. `system.prompt` still globally bound (synthesis §12.5 #2 requires moving to providers.mlc.args.system_prompt). Tracked as a follow-up. |

---

## 4. Connector framework (Phase 4.5; foundation for WhatsApp/Gmail)

Connectors are first-class extensions for personal-data integrations. They appear to the
model as **regular tools** (with mandatory `{connector_id}_` prefix) but additionally
publish an **inbound stream** of events (DMs, emails, etc.) to `SqliteInbox`.

```mermaid
flowchart LR
    subgraph lifecycle["Connector lifecycle"]
        direction TB
        S["start()"] --> R["running"]
        R -->|"begin_login()"| L["login_pending"]
        L -->|"complete_login()"| R
        R -->|"logout()"| Empty["unconfigured<br/>(creds deleted)"]
        R -->|"stop()"| Stopped["stopped"]
    end

    subgraph routes["HTTP routes /api/v1/connectors/*"]
        H1["GET .../health"]
        H2["GET .../auth_status"]
        H3["POST .../begin_login"]
        H4["POST .../complete_login"]
        H5["POST .../logout"]
        H6["GET .../oauth/callback"]
    end

    subgraph runtime["Runtime"]
        Drain["Inbound drain task<br/>(SupervisedTask, M3)"]
        Cred["SecretsProvider<br/>(env → file fallback)"]
        Inbox["SqliteInbox<br/>append_many idempotent"]
    end

    R --> Drain
    Drain --> Inbox
    L --> Cred
    routes --> lifecycle
```

**Send-safety pattern** (synthesis §4 footer): every outbound-send tool exposes
`*_prepare_send` (returns a draft id) and `*_confirm_send` (consults
`ToolExecutionContext.user_confirmed_send`). Tools never auto-send; the model is forced
to elicit explicit user confirmation. This is why `ToolExecutionContext` exists — tools
have no turn awareness on their own.

**Test fixture**: `tests/fixtures/echo_connector.py::EchoConnector` exercises every
contract (`list[str]` arg, `Optional[T]` arg, confirm-send tool, inbound stream).

### 4a. WhatsApp connector (Phase 2b)

The WhatsApp connector is the first production connector built on the Phase 4.5 framework.
It is **single-user, WhatsApp Web** (personal account) backed by
[neonize](https://github.com/krypton-byte/neonize) (pinned `==0.3.17.post0`), a Python binding
over the Go `whatsmeow` library. A `WhatsAppClientAdapter` ABC isolates the rest of Tether from
neonize instabilities (Snapdragon X Elite DLL selection bug and event-loop singleton reset on
restart — both patched at the adapter boundary; see ADR-0018 §D1). Once linked, the connector
registers nine tools in the `ToolRegistry` under the mandatory `whatsapp_` prefix:

| Tool | Purpose |
|---|---|
| `whatsapp_prepare_send` | Build a text draft (no send) |
| `whatsapp_confirm_send` | Dispatch a draft after explicit user confirmation |
| `whatsapp_list_unread` | List unread inbound events from Tether's local inbox |
| `whatsapp_get_thread` | Recent messages with a peer (reads `SqliteInbox`) |
| `whatsapp_inbox_mark_seen` | Mark Tether-local inbox events as seen (does not affect WhatsApp UI) |
| `whatsapp_mark_platform_read` | Send WhatsApp read receipts (blue checkmarks visible to the other party) |
| `whatsapp_send_media` | Build a media draft from a file path (image/video/audio/document) |
| `whatsapp_get_contacts` | Search the contact list by name or E.164 substring |
| `whatsapp_resolve_contact` | Resolve a display name / E.164 / JID to canonical JID |

**State machine**: the connector lifecycle follows the standard `Connector` ABC states
(`unconfigured → login_pending → running → stopped`). On `begin_login()` neonize emits a QR
code; the CLI renders it and awaits `complete_login()` (QR scanned on the user's phone). The
full state-transition table is in ADR-0018.

**Send-safety doctrine** (ADR-0019): `whatsapp_prepare_send` / `whatsapp_send_media` return a
draft id without touching the network. `whatsapp_confirm_send` calls
`ToolExecutionContext.user_confirmed_send`; if the flag is `False` the tool refuses and the
model must re-solicit affirmation from the user. The flag is set by the
`ConfirmIntentClassifier` (v1: `RegexConfirmIntentClassifier`, injected via dotted-path config;
default-safe — refuses on ambiguous affirmatives such as "yes but cancel"). This makes
accidental sends structurally impossible regardless of what the model says.

**Install**:

```powershell
pip install -e ".[whatsapp]"
```

**CLI subcommands**:

```powershell
tether-cli connect whatsapp    # QR-pair a new device
tether-cli logout whatsapp     # unlink and delete local credentials
```

Full design: [ADR-0018](./adr/0018-whatsapp-connector-library-and-adapter.md) (library choice,
adapter seam, upstream-defect patches) and
[ADR-0019](./adr/0019-confirm-intent-classifier-seam.md) (ConfirmIntentClassifier ABC and
`RegexConfirmIntentClassifier` v1).

---

## 5. Data layout

```
data/
├── tether.db                        # single SQLite (WAL)
│   ├── sessions                     # session_id, created_at, …
│   ├── messages                     # session_id, turn_id, role, content, …
│   ├── inbound_events               # connector inbound events (Phase 6.5; SqliteInbox)
│   ├── tool_audit                   # tool calls + results (256 KB cap)
│   └── _yoyo_*                      # yoyo-migrations metadata
├── connectors/<id>/                 # per-connector private state
└── secrets/                         # file-based fallback for SecretsProvider

models/                              # MLC model dirs + compiled libs
├── libs/                            # *-adreno.dll, *-cpu.dll
├── Qwen3-4B-q4f16_0-MLC/            # weights + mlc-chat-config.json
└── …
```

`models_root` is configurable via `TETHER_MODELS_DIR` env var; defaults to `models`.
Formerly `dist/` (renamed in Phase 8 step 85 — the old name implied build artifacts).

---

## 6. Concurrency + hardware boundary

```mermaid
flowchart LR
    subgraph asyncio["asyncio event loop"]
        FastAPI
        Orch
        Runner
        Drain["Connector drain tasks"]
    end

    subgraph daemon["Daemon thread (M1: daemon_thread_call)"]
        Term["MLCProvider._terminate_bounded<br/>(GC disabled — load-bearing)"]
        WD["HardwareWatchdog<br/>(periodic libc check)"]
    end

    subgraph os["OS"]
        OpenCL["Adreno X1 GPU<br/>(via OpenCL)"]
    end

    Orch --> FastAPI
    Runner --> Orch
    Drain --> Orch
    Orch -.->|"shutdown"| Term
    Term --> OpenCL
    WD -.->|"watchdog poll"| OpenCL
```

**Critical invariant**: `gc.disable()` inside the MLC shutdown daemon thread is
**load-bearing**. Re-enabling it deadlocks Qwen2.5-7B on Ctrl+C (smaller `prefill_chunk_size`
exposes a destructor hang). Enforced by the comment in `daemon_thread_call` and a
codebase rule. See `docs/runbooks/SHUTDOWN_HANG_FIX_SUMMARY.md` (post-Phase-8).

---

## 7. Module-graph dependency rules

```mermaid
flowchart LR
    A["adapters<br/>app.http · cli"] --> B["engine"]
    B --> C["protocol · tools · connectors"]
    C --> D["context"]
    C --> E["providers"]
    E --> F["runtime<br/>(daemon_call · watchdog · spans)"]
    D --> G["core<br/>(interfaces · settings · registry_validator)"]
    C --> G
    E --> G

    classDef leaf fill:#eef
    class G leaf
```

**Dependency direction is one-way.** Higher layers depend on lower; never the reverse.

- `core` hosts shared **types, ABCs, and registries**. It may import:
  - `protocol.parsers.events` (for `ParserEvent` ABC consumers)
  - `protocol.orchestration.policies` (for `LoopLimitPolicy` / `ToolErrorPolicy` enums consumed by `Settings`)
  - `providers.types` (for `ProviderCapabilities` / `ProviderEvent` ABC consumers)
  - `connectors.base/types` (for `ConnectorRegistry`)

  It MUST NOT import implementations (`protocol.orchestration.chatty`, `connectors.echo`, `providers.mlc.provider`, etc.). The split of contracts (`*.types`, `*.events`, `*.policies`) below the implementation files is what makes this acyclic.

- `runtime` is hardware-adjacent; only `providers` may depend on it.
- `protocol`, `tools`, `connectors` may depend on `context` + `providers` + `core`.
- `engine` wires everything; `adapters` only see `engine` and serialization types.

---

## 8. Six shared modules (synthesis §13.4)

These exist to dedup logic and keep the layering honest:

| ID | Path | Purpose | Used by |
|---|---|---|---|
| **M1** | `runtime/daemon_call.py::daemon_thread_call` | GC-disabled daemon thread for blocking native cleanup | HardwareWatchdog · MLC `_terminate_bounded` · future native-cleanup connectors |
| **M2** | `context/_async_sqlite_base.py::AsyncSqliteStore` | aiosqlite + WAL + yoyo-migrations base | SqliteSessionStore · SqliteInbox |
| **M3** | `runtime/task_supervisor.py::SupervisedTask` | structured-concurrency wrapper for long-running tasks | Connector drain tasks · future Gmail polling |
| **M4** | (deferred) | structlog span helper — **NOT IMPLEMENTED** as of 2026-05. Today every "span" site uses bare `structlog.get_logger().info(name, ...)`. Real `async_span` lifetime tracking lands with the OTel adapter rework (P0-I follow-up; ADR-0010). | n/a |
| **M5** | `core/registry_validator.py::validate_unique_names` | uniqueness + forbidden-name + required-prefix checks at boot | ToolRegistry · ConnectorRegistry |
| **M6** | `config/_strict.py::StrictModel` | Pydantic base with `extra='forbid'` and frozen | every Settings sub-model (~12) |

---

## 9. Frozen / locked decisions (don't relitigate)

Authoritative: synthesis §3 + §10–§13. Quick-reference here:

- **Single-user, local-only.** No multi-tenant, no cloud auth, no plugin SDK.
- **Drop-and-rebuild** chat history at schema cutover (no migration tools needed).
- **`LoopLimitPolicy`** = `EMIT_LIMIT_EVENT` (default).
- **`ToolErrorPolicy`** = `FEED_BACK_TO_MODEL` (tool errors no longer break the loop).
- **Hardware backend** = Adreno X1 GPU via OpenCL — never call it "NPU". (NPU is reserved for the future `NexaProvider`, Seam A.)
- **Runtime** = Qualcomm CodeLinaro MLC-LLM `2025.06.r1` (Prism-emulated x64 Python 3.12).
- **GC-disable in MLC shutdown daemon thread** = load-bearing.
- **`tether_service` import alias** = backward-compat for one release cycle; deletion deferred.
- **Frozen paths**: pre-refactor reference code (`llm_service/`, `legacy/`) is archived at the [`archive/pre-refactor`](https://github.com/Lando-00/Tether/tree/archive/pre-refactor) branch. Reference there if you need to understand pre-refactor behavior; never bring it back into `main`.
- **No forbidden features**: autonomous reply, scheduled jobs, push, voice, Canvas, multi-agent routing, plugin manifests.

### Environment

The refactor was developed in conda env `mlc-venv2`. A fresh env `tether` exists for end-of-refactor validation. Both coexist; `mlc-venv2` is the active dev env, `tether` is created from scratch via `scripts/setup_fresh_env.ps1`. Full step-by-step in [`docs/runbooks/fresh-env-setup.md`](./runbooks/fresh-env-setup.md). Env specs at `environment-mlc-venv2.yml` and `environment-tether.yml`. Both pin CodeLinaro `2025.06.r1` per ADR-0014.

---

## 10. Pointers

- **Locked decisions / phase digest**: [`refactor/synthesis-2026-05.md`](./refactor/synthesis-2026-05.md) — curated successor to the prior (private) refactor-planning document.
- **Per-decision detail**: [`adr/`](./adr/) — Architecture Decision Records.
- **Per-todo state**: SQL `todos` table (session-local).
- **Sub-agent prompt templates**: [`.github/agents/`](../.github/agents/) (`implementation.agent.md`, `code-review.agent.md`).
- **Repository conventions**: [`.github/copilot-instructions.md`](../.github/copilot-instructions.md).
- **Connector contract**: see [ADR-0005](./adr/0005-connector-registry-prefix-enforcement.md) and [ADR-0009](./adr/0009-sqlite-inbox-shared-db.md).
