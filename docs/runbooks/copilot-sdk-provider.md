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

