# ADR-0021 — Phase 2 contract stubs

Operational reference for the multi-provider registry rollout. Phase 2
sub-agents copy from this document; deviations require an ADR amendment.
Companion to `0021-multi-provider-registry.md`.

> **Invariants that bound every signature below**
>
> - ADR-0016: provider construction MUST go through `tether.core.factory.load`.
>   No imports of `tether.providers.mlc.*` or `tether.providers.copilot.*` at
>   package load time. The registry stores **constructed instances**, not
>   classes or dotted paths.
> - ADR-0003: `runtime/daemon_call.py::daemon_thread_call` and
>   `_terminate_bounded` are frozen.
> - StrictModel (`extra='forbid'`, frozen) applies to every Pydantic
>   settings model touched below.

## 1. Settings (`src/tether/config/settings.py`)

### `ProviderSpec` (unchanged)

```python
class ProviderSpec(StrictModel):
    impl: str
    args: dict[str, Any] = Field(default_factory=dict)
```

### `ProvidersSettings` (NEW shape)

```python
class ProvidersSettings(StrictModel):
    """``providers:`` section — multi-provider registry + parser/store.

    Two shapes are accepted on load:

      (NEW)    model_registry: { id: ProviderSpec, ... }
               default_model_provider: str
      (LEGACY) model: ProviderSpec
               -> synthesised into model_registry={"default": <spec>} with
                  default_model_provider="default". DeprecationWarning logged.

    Setting BOTH 'model' and 'model_registry' raises ConfigError.
    """

    # Legacy singular — kept for one release cycle. Validator below
    # promotes it into model_registry. Default None so absence is legal.
    model: Optional[ProviderSpec] = None

    # New multi-provider registry. Key = provider_id (stable, config-set).
    # Default {} so legacy-only configs validate, then the validator
    # rejects the (None, {}) case.
    model_registry: Dict[str, ProviderSpec] = Field(default_factory=dict)

    # Server-wide default provider_id. Required once the validator has
    # finished promoting legacy 'model' into the registry. MUST be a key
    # of the resolved registry.
    default_model_provider: Optional[str] = None

    parser: ProviderSpec
    session_store: ProviderSpec

    @model_validator(mode="after")
    def _promote_legacy_and_validate(self) -> "ProvidersSettings":
        # 1. Reject ambiguous config.
        if self.model is not None and self.model_registry:
            raise ConfigError(
                "providers.model (singular, deprecated) and "
                "providers.model_registry are mutually exclusive. "
                "Remove providers.model."
            )
        # 2. Promote legacy form. Object.__setattr__ used because the
        #    model is frozen; the validator runs before freeze enforcement
        #    in pydantic v2's mode='after' phase.
        if self.model is not None and not self.model_registry:
            warnings.warn(
                "providers.model is deprecated; use providers.model_registry "
                "with default_model_provider. Will be removed in the next "
                "release. (ADR-0021)",
                DeprecationWarning,
                stacklevel=2,
            )
            object.__setattr__(self, "model_registry", {"default": self.model})
            if self.default_model_provider is None:
                object.__setattr__(self, "default_model_provider", "default")
        # 3. Reject empty registry.
        if not self.model_registry:
            raise ConfigError(
                "providers.model_registry is empty (and providers.model "
                "is unset). At least one provider is required."
            )
        # 4. Resolve default_model_provider.
        if self.default_model_provider is None:
            raise ConfigError(
                "providers.default_model_provider is required when "
                "providers.model_registry is set."
            )
        if self.default_model_provider not in self.model_registry:
            raise ConfigError(
                f"providers.default_model_provider="
                f"{self.default_model_provider!r} is not a key of "
                f"providers.model_registry "
                f"(known ids: {sorted(self.model_registry)})."
            )
        return self
```

`ConfigError` is `tether.core.errors.ConfigError`. The validator MUST run as
`mode="after"` so field defaults and types have already been coerced.

## 2. Engine (`src/tether/engine.py`)

### Constructor (replaces today's `provider=` kwarg)

```python
def __init__(
    self,
    *,
    # CHANGED: from `provider: ModelProvider` to a dict + default id.
    providers: Dict[str, ModelProvider],
    default_provider_id: str,
    parser: Optional[StreamParser] = None,
    parser_factory: Optional[Callable[[], StreamParser]] = None,
    session_store: SessionStore,
    tools: Dict[str, Tool],
    system_prompt: str,
    watchdog_mode: WatchdogMode = WatchdogMode.LIBRARY,
    orchestrator_config: Optional[OrchestratorConfig] = None,
    tool_runner: Optional[ToolRunner] = None,
    hw_watchdog: Optional["HardwareWatchdog"] = None,
    connector_registry: Optional["ConnectorRegistry"] = None,
    inbox: Optional["InboundInbox"] = None,
    orchestrator_registry: Optional[Dict[str, str]] = None,
    orchestrator_default_mode: str = "chat",
    # NEW: per-id construction failures captured by from_settings.
    provider_start_failures: Optional[Dict[str, str]] = None,
) -> None: ...
```

Stored state:

```python
self.providers: Dict[str, ModelProvider] = dict(providers)
self.default_provider_id: str = default_provider_id
self._provider_start_failures: Dict[str, str] = dict(provider_start_failures or {})
# Back-compat shim for one cycle. Marked private to discourage new use;
# legacy tests / callers reading `engine.provider` keep working.
self.provider: ModelProvider = self.providers[self.default_provider_id]
```

`self.provider` MUST NOT be removed in this phase. It is the deprecation
alias that lets the 1660 existing tests keep passing. Remove in the same
follow-up that removes `providers.model`.

### New method

```python
def list_provider_health(self) -> Dict[str, Dict[str, Any]]:
    """Per-provider snapshot for /readyz. Cheap, sync, no network.

    Shape:
        {
          "<provider_id>": {
            "healthy": bool,
            "kind":    str,           # provider.kind
            "source":  Literal["local","remote"],
            "error":   Optional[str], # str(exc) if construction failed
                                      # OR if warm_up failed at __aenter__
          },
          ...
        }
    """
```

Entries include both:
- ids in `self.providers` (healthy=True unless flagged unhealthy by warm_up).
- ids in `self._provider_start_failures` (healthy=False, kind="unknown",
  source="unknown", error=<message>).

### Updated method signatures

```python
async def chat(
    self, *,
    session_id: str,
    prompt: str,
    model_name: str,
    mode: Optional[str] = None,
    cancel_token: Optional["CancelToken"] = None,
    reasoning_effort: Optional[str] = None,
    provider_id: Optional[str] = None,   # NEW
) -> AsyncGenerator["WireEvent", None]: ...

async def stream(
    self, *,
    session_id: str,
    prompt: str,
    model_name: str,
    mode: Optional[str] = None,
    cancel_event: Optional[asyncio.Event] = None,
    reasoning_effort: Optional[str] = None,
    provider_id: Optional[str] = None,   # NEW
) -> AsyncGenerator[bytes, None]: ...
```

Resolution rule (both methods):

```python
pid = provider_id or self.default_provider_id
if pid not in self.providers:
    if pid in self._provider_start_failures:
        raise ProviderUnhealthyError(pid, self._provider_start_failures[pid])
    raise UnknownProviderError(pid)
provider = self.providers[pid]
# Pass `provider=` (not `self.provider`) into the orchestrator construction.
```

Both new exceptions live in `tether.core.errors`. The HTTP layer maps
`UnknownProviderError → 422` and `ProviderUnhealthyError → 503`.

### Model introspection

```python
def list_models(self) -> List[str]:
    """Merged union across HEALTHY providers.

    Rule: if the same model_name appears in two providers, both entries
    are included, disambiguated as ``"<provider_id>/<model_name>"`` in the
    legacy list. Non-duplicates remain bare model_names (back-compat).
    This is the ONLY shape change to /api/v1/models — still list[str].
    """

def list_model_info(self) -> List[ModelDetails]:
    """Merged ModelDetails across HEALTHY providers.

    Each provider's list_model_info() result is wrapped via:

        info.model_copy(update={"provider_id": pid})

    No de-dup is performed at this layer — duplicates surface as separate
    rows with distinct provider_id values. Clients pick by (provider_id,
    model id).
    """

def unload_model(self, model_name: str, *, provider_id: Optional[str] = None) -> bool:
    """Route to a specific provider.

    Resolution:
      - if provider_id is given, dispatch directly.
      - else, dispatch to every healthy provider that lists model_name;
        return True if ANY returned True. (Engine-level fan-out is
        cheap because providers' unload_model is sync + idempotent.)
    """
```

### `Engine.from_settings` — degraded-mode pseudocode

```python
@classmethod
def from_settings(cls, settings: Settings, *, watchdog_mode=...) -> "Engine":
    # ... existing parser / store / migrations / inbox / tools setup ...

    providers: Dict[str, ModelProvider] = {}
    failures: Dict[str, str] = {}
    for pid, spec in settings.providers.model_registry.items():
        try:
            providers[pid] = factory.load(spec.impl, **spec.args)
            logger.info("provider.loaded", provider_id=pid, impl=spec.impl)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "provider.load_failed",
                provider_id=pid, impl=spec.impl,
                error_class=type(exc).__name__, error_message=str(exc),
            )
            failures[pid] = f"{type(exc).__name__}: {exc}"

    if not providers:
        raise ConfigError(
            f"All {len(failures)} providers failed to construct: {failures!r}"
        )

    default_pid = settings.providers.default_model_provider
    if default_pid not in providers:
        # Default provider itself failed. Fall back to first healthy id
        # in declaration order. Loud warning — operator must fix config.
        original = default_pid
        default_pid = next(iter(providers))
        logger.warning(
            "provider.default_unhealthy_fallback",
            requested=original, fallback=default_pid,
        )

    hw_watchdog = HardwareWatchdog(
        list(providers.values()), mode=watchdog_mode,
    )

    engine = cls(
        providers=providers,
        default_provider_id=default_pid,
        provider_start_failures=failures,
        # ... rest unchanged ...
    )
    return engine
```

## 3. HardwareWatchdog (`src/tether/runtime/hw_watchdog.py`)

**Constructor: UNCHANGED.** `HardwareWatchdog(providers: List[Any], *,
mode=..., shutdown_budget_sec=...)`. Internal `isinstance(p,
HardwareLifecycle)` filter already does the right thing for a mixed list
(MLC + Copilot + Dummy).

`health_summary()`, `reset_after()`, `shutdown_all()`: unchanged. The
daemon-thread GC invariant (`daemon_thread_call`, `_terminate_bounded`) is
not touched by this phase.

Phase 2 regression test: with `providers=[CopilotProvider(), DummyProvider()]`
(no HardwareLifecycle entries), `health_summary()` returns
`{"providers": [], "overall": "healthy"}` and `shutdown_all()` is a no-op.

## 4. HTTP API

### `StreamRequest` (`app/http/routers/chat.py`)

```python
class StreamRequest(BaseModel):
    session_id: str = Field(..., pattern=r"^[A-Za-z0-9_-]{1,128}$")
    prompt: str = Field(..., min_length=1, max_length=32768)
    model_name: str = Field(..., pattern=r"^[A-Za-z0-9._-]{1,128}$")
    mode: Literal["chat", "research"] = "chat"
    reasoning_effort: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9._-]{1,32}$")
    provider_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional provider routing key. When omitted, the server uses "
            "providers.default_model_provider. Unknown values return 422; "
            "known-but-unhealthy values return 503."
        ),
        # Same alphabet as model_name; provider ids are config-controlled
        # and never user-typed at runtime, but keep the validator tight.
        pattern=r"^[A-Za-z0-9._-]{1,64}$",
    )
```

### `/api/v1/chat/stream` flow

1. Pydantic validates field shape. Unknown shape → 422.
2. Resolve provider_id: `pid = body.provider_id or engine.default_provider_id`.
3. If `pid` not in `engine.providers`:
   - if in `engine._provider_start_failures`: **503**, body
     `{"detail": f"Provider '{pid}' unhealthy: {error}"}`.
   - else: **422**, body `{"detail": f"Unknown provider_id '{pid}'."}`.
4. If `body.reasoning_effort is not None`: run
   `_validate_reasoning_effort(engine, body.model_name, body.reasoning_effort,
   provider_id=pid)` — looks up `ModelDetails` filtered by `provider_id`. **422**
   on mismatch.
5. Existing orchestrator-resolution / mode-501 / SSE-vs-NDJSON branches
   continue unchanged, with `provider_id=pid` threaded through `engine.chat(...)`.

### `/api/v1/models` (response shape unchanged: `list[str]`)

Values: union across **healthy** providers. Duplicate model_names are
disambiguated as `"<provider_id>/<model_name>"`; unique names remain bare.
Order: providers iterated in declaration order from `model_registry`; within
each provider, in the order returned by `provider.list_models()`.

Rationale for prefix-on-collision (rather than precedence + de-dup):
clients selecting a model by string never silently get routed to a
different provider when configuration changes. The bare-when-unique rule
keeps single-provider deployments visually identical to today.

### `/api/v1/models/details` (response shape: `List[ModelDetails]`)

`ModelDetails` gains `provider_id: str` (see §5). NO de-dup: duplicates
appear as separate rows with different `provider_id` values. The CLI
disambiguates from `provider_id`; web clients can de-dup by
`(provider_id, id)`.

### `/api/v1/readyz`

Additive over the current shape. Full example:

```json
{
  "ready": true,
  "store": true,
  "provider": true,
  "hw_health": {
    "providers": [{"status": "healthy", "details": {...}}],
    "overall": "healthy"
  },
  "connectors": [],
  "connector_start_failures": [],
  "providers": {
    "mlc-local": {
      "healthy": true,
      "kind": "mlc",
      "source": "local",
      "error": null
    },
    "copilot-gpt5": {
      "healthy": true,
      "kind": "copilot",
      "source": "remote",
      "error": null
    },
    "copilot-aux": {
      "healthy": false,
      "kind": "unknown",
      "source": "unknown",
      "error": "AuthError: COPILOT_GITHUB_TOKEN unset"
    }
  },
  "default_provider_id": "mlc-local"
}
```

Ready computation:

```python
ready = (
    store_ok
    and (hw_health_overall != "error")
    and any(p["healthy"] for p in providers_block.values())
    and not connector_start_failures
)
```

The legacy top-level `provider: bool` continues to mean "≥1 provider is
healthy"; supervisors keying on it keep working.

## 5. `ModelDetails` extension (`src/tether/providers/types.py`)

```python
class ModelDetails(BaseModel):
    id: str
    provider_id: str          # NEW — required.
    provider_kind: str
    source: Literal["local", "remote"]
    context_window: int
    supports_thinking: bool
    supports_reasoning_effort: bool
    reasoning_efforts: Optional[List[str]] = None
    is_default: bool = False

    model_config = ConfigDict(frozen=True, extra="forbid")
```

**Who fills `provider_id`**: Engine wraps each provider's
`list_model_info()` output via `info.model_copy(update={"provider_id": pid})`
inside `Engine.list_model_info`. Providers themselves DO NOT populate
`provider_id` — they don't know their own registry id. The ABC default
implementation of `list_model_info` uses a sentinel
`provider_id="_unwrapped_"` so a direct test on a bare provider doesn't
raise; Engine always overwrites. Phase 2 sub-agent for the types module
MUST add this default + a comment explaining the wrap-at-Engine
contract.

## 6. CLI (`src/tether/cli/main.py`)

### Typer options

Root callback and `chat` subcommand both gain:

```python
provider: Optional[str] = typer.Option(
    None, "--provider", "-P",
    help=(
        "Provider id to route the request to. When omitted, the server "
        "uses its configured default. Use `\\providers` in the REPL to "
        "see available ids and health."
    ),
)
```

### POST body

When `provider` is non-None, include `"provider_id": provider` in the body
of POSTs to `/api/v1/chat/stream`. Otherwise omit the key (preserves the
legacy request shape on the wire).

### `\providers` slash command output

Table (Rich), columns: `id`, `kind`, `source`, `default?`, `health`, `error`.
Sourced from `GET /api/v1/readyz` (`providers` block + `default_provider_id`).
Example:

```
┌──────────────┬─────────┬────────┬─────────┬──────────┬──────────────────────────┐
│ id           │ kind    │ source │ default │ health   │ error                    │
├──────────────┼─────────┼────────┼─────────┼──────────┼──────────────────────────┤
│ mlc-local    │ mlc     │ local  │   ✓     │ healthy  │                          │
│ copilot-gpt5 │ copilot │ remote │         │ healthy  │                          │
│ copilot-aux  │ copilot │ remote │         │ DOWN     │ AuthError: token unset   │
└──────────────┴─────────┴────────┴─────────┴──────────┴──────────────────────────┘
```

### `\models` selector update

Grow a `provider_id` column sourced from `GET /api/v1/models/details`.
Unhealthy providers' rows are filtered out (already absent from the
endpoint response). Default selection: row with `is_default=True` AND
`provider_id == default_provider_id`.

### Ambiguous `--model` fallback

When the user runs `tether-cli chat --model X` without `--provider` and
`/models/details` returns multiple rows whose `id == X` (different
`provider_id`s), drop into the `\models` selector pre-filtered to those
rows. Justified: silently picking one would change behaviour when config
changes.

## 7. Default config (`src/tether/config/default.yml`)

Recommended shipped shape: keep the legacy `providers.model` example
working AND advertise the new registry as a commented example.

```yaml
providers:
  # Legacy single-provider shape. Synthesised into a one-entry
  # model_registry at load time with provider_id="default". Deprecated
  # in favour of `model_registry` below; will be removed one release
  # after ADR-0021 ships.
  model:
    impl: "tether.providers.mlc.provider.MLCProvider"
    args:
      device: "auto"
      max_tokens: 1024
      marker_only_tools: true

  # NEW: multi-provider registry. Uncomment to expose multiple providers
  # from one server. When this section is set, you MUST also remove the
  # `model:` block above and set `default_model_provider`.
  #
  # model_registry:
  #   mlc-local:
  #     impl: "tether.providers.mlc.provider.MLCProvider"
  #     args:
  #       device: "auto"
  #       max_tokens: 1024
  #       marker_only_tools: true
  #   copilot-gpt5:
  #     impl: "tether.providers.copilot.provider.CopilotProvider"
  #     args:
  #       model: "gpt-5"
  #       reasoning_efforts: ["minimal", "low", "medium", "high"]
  # default_model_provider: "mlc-local"

  parser:
    impl: "tether.protocol.parsers.sliding.SlidingParser"
    args:
      max_tool_chars: 32768
  session_store:
    impl: "tether.context.sqlite_store.SqliteSessionStore"
    args: {}
```

The `tests/unit/config/test_settings.py` golden assertions that today
read `settings.providers.model.impl` keep working unchanged because the
legacy field is still parsed; new tests assert `model_registry["default"]`
mirrors it.

## 8. Error / status code reference

| Condition                                          | Status | Origin                                                |
|----------------------------------------------------|--------|-------------------------------------------------------|
| `provider_id` unknown                              | 422    | `chat.py::stream`                                     |
| `provider_id` known but unhealthy                  | 503    | `chat.py::stream`                                     |
| `reasoning_effort` unsupported for (pid, model)    | 422    | `chat.py::_validate_reasoning_effort`                 |
| `reasoning_effort` validation can't fetch details  | 503    | existing path, now `provider_id`-scoped               |
| `/models/details` carries unwrapped sentinel       | 500    | Engine bug (must not happen) — assert in tests        |
| all providers failed to load                       | startup | `ConfigError` from `Engine.from_settings`             |
| `model_registry` empty AND `model` unset           | config | `ConfigError` from `ProvidersSettings._promote_legacy_and_validate` |
| both `model` and `model_registry` set              | config | `ConfigError` ditto                                   |
| `default_model_provider` not a key of registry     | config | `ConfigError` ditto                                   |

## 9. Out-of-scope reminders for Phase 2 sub-agents

- Do NOT change `runtime/daemon_call.py`.
- Do NOT add eager imports of `tether.providers.mlc.*` or
  `tether.providers.copilot.*` anywhere outside `factory.load`.
- Do NOT remove `Engine.provider` (singular) in this phase — it's the
  deprecation shim for the 1660 existing tests.
- Do NOT introduce a periodic provider re-probe loop.
- Do NOT change connector / inbox / store lifecycle ordering.