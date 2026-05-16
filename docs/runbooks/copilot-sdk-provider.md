# GitHub Copilot SDK provider

Tether can use the public-preview GitHub Copilot SDK as an experimental model
provider. Install it explicitly:

```powershell
C:\ProgramData\miniconda3\envs\mlc-venv2\python.exe -m pip install -e ".[server,cli,brave,dev,copilot]"
```

Example provider override:

```yaml
providers:
  model:
    impl: "tether.providers.copilot.provider.CopilotProvider"
    args:
      model: "gpt-5"
      models: ["gpt-5", "claude-sonnet-4.5"]
      github_token_env: "COPILOT_GITHUB_TOKEN"
      use_logged_in_user: true
      enable_builtin_tools: false
      # Optional: narrow which models advertise reasoning_effort support.
      # Defaults to "every configured model"; pass an explicit list when
      # the SDK adds non-reasoning models.
      reasoning_effort_models: ["gpt-5"]
      reasoning_efforts: ["minimal", "low", "medium", "high"]
```

## Model metadata and reasoning effort

`CopilotProvider` advertises its models via `GET /api/v1/models/details`
(companion to the back-compat `GET /api/v1/models` `list[str]` endpoint). The
details response carries `supports_reasoning_effort`, `reasoning_efforts`,
`context_window`, `source="remote"`, and `is_default`, so clients can render
the right selection UI without hard-coded knowledge of each provider.

To send a reasoning effort hint with a chat turn:

```json
POST /api/v1/chat/stream
{
  "session_id": "...",
  "prompt": "...",
  "model_name": "gpt-5",
  "reasoning_effort": "high"
}
```

When the chosen model does not advertise `supports_reasoning_effort=true`, or
the value is outside its `reasoning_efforts` list, the server returns **422
before any streaming starts**. The provider's `stream(...)` is never invoked.

In `tether-cli`:

```powershell
tether-cli --model gpt-5 --reasoning-effort high
```

Mid-chat:

- `\reasoning` prompts for a new value, scoped to the current model's
  `reasoning_efforts` list. Use `off` to clear the override.
- `\models` shows provider / source / context / reasoning columns when
  `GET /models/details` is reachable, and automatically clears a stale
  reasoning effort when you switch to a model that doesn't accept it.

By default, `CopilotProvider` denies Copilot SDK tool permission requests. This
keeps Tether's own orchestrator, tool audit, connector send-safety, and SQLite
session history authoritative. Treat full Copilot agent-runtime integration as a
future orchestrator mode, not as a model-provider responsibility.

Authentication follows the SDK's supported mechanisms: stored Copilot CLI login,
`COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, `GITHUB_TOKEN`, explicit OAuth user tokens, or
BYOK provider config. Do not put tokens in YAML or commit them to the repo.

---

## Running MLC and Copilot in the same server

Since ADR-0021 (Phase 12), `Engine` holds a `providers: Dict[str, ModelProvider]`
registry. Use the `model_registry` YAML shape to expose both providers from a single
server instance:

```yaml
providers:
  # NEW multi-provider shape (ADR-0021). Remove `model:` when using this.
  model_registry:
    mlc-local:
      impl: "tether.providers.mlc.provider.MLCProvider"
      args:
        device: "auto"
        max_tokens: 1024
        marker_only_tools: true
    copilot-gpt5:
      impl: "tether.providers.copilot.provider.CopilotProvider"
      args:
        model: "gpt-5"
        models: ["gpt-5", "claude-sonnet-4.5"]
        github_token_env: "COPILOT_GITHUB_TOKEN"
        use_logged_in_user: true
        enable_builtin_tools: false
        reasoning_effort_models: ["gpt-5"]
        reasoning_efforts: ["minimal", "low", "medium", "high"]
  default_model_provider: "mlc-local"
  parser:
    impl: "tether.protocol.parsers.sliding.SlidingParser"
    args:
      max_tool_chars: 32768
  session_store:
    impl: "tether.context.sqlite_store.SqliteSessionStore"
    args: {}
```

`default_model_provider` is required when `model_registry` is set.
Setting both `model:` (singular, deprecated) and `model_registry:` in the same
config is a `ConfigError`.

### Routing a request to Copilot via `provider_id`

**curl:**

```bash
curl -X POST http://localhost:8080/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "my-session",
    "prompt": "Explain async generators in Python.",
    "model_name": "gpt-5",
    "provider_id": "copilot-gpt5"
  }'
```

**Python `requests`:**

```python
import requests

resp = requests.post(
    "http://localhost:8080/api/v1/chat/stream",
    json={
        "session_id": "my-session",
        "prompt": "Explain async generators in Python.",
        "model_name": "gpt-5",
        "provider_id": "copilot-gpt5",   # omit to use default_model_provider
    },
    stream=True,
)
for line in resp.iter_lines():
    if line:
        print(line.decode())
```

When `provider_id` is omitted, the server uses `default_model_provider` (`"mlc-local"`
above). An unknown `provider_id` returns **422**; a known-but-unhealthy one returns **503**.

### Inspecting provider details and health

**`GET /api/v1/models/details`** — merged `ModelDetails` list across all healthy providers.
Each row carries a `provider_id` field so clients can tell which backend owns the model:

```bash
curl http://localhost:8080/api/v1/models/details | python -m json.tool
```

Example response excerpt:
```json
[
  {"id": "Qwen3-4B-q4f16_0-MLC", "provider_id": "mlc-local", "source": "local", "is_default": true, ...},
  {"id": "gpt-5",                 "provider_id": "copilot-gpt5", "source": "remote", "is_default": true, ...}
]
```

**`GET /api/v1/readyz`** — per-provider health block:

```bash
curl http://localhost:8080/api/v1/readyz | python -m json.tool
```

Key new fields in the response:
```json
{
  "providers": {
    "mlc-local":    {"healthy": true,  "kind": "mlc",     "source": "local",   "error": null},
    "copilot-gpt5": {"healthy": true,  "kind": "copilot", "source": "remote",  "error": null}
  },
  "default_provider_id": "mlc-local",
  "provider": true
}
```

`provider: true` means ≥1 provider is healthy (legacy supervisor key, unchanged).

### CLI usage

```powershell
# Route a single chat turn to Copilot
tether-cli --provider copilot-gpt5 --model gpt-5

# Inside the REPL, list all configured providers and their health
\providers

# See all available models with provider_id column
\models
```

The `--provider` / `-P` option is available on both the root `tether-cli` callback
and the `chat` subcommand. When `--model X` is ambiguous across providers (same
model name, different `provider_id`s) and `--provider` was not given, the CLI
drops into the `\models` selector pre-filtered to those rows.

### Deprecation alias

The singular `providers.model:` shape still works for one release cycle — it is
silently promoted into a one-entry `model_registry` with `provider_id="default"`.
A `DeprecationWarning` is emitted on every settings load. Migrate to
`model_registry` + `default_model_provider` before the next release. See ADR-0021.

