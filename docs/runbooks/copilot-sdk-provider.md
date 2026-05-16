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
      github_token_env: "COPILOT_GITHUB_TOKEN"
      use_logged_in_user: true
      enable_builtin_tools: false
```

By default, `CopilotProvider` denies Copilot SDK tool permission requests. This
keeps Tether's own orchestrator, tool audit, connector send-safety, and SQLite
session history authoritative. Treat full Copilot agent-runtime integration as a
future orchestrator mode, not as a model-provider responsibility.

Authentication follows the SDK's supported mechanisms: stored Copilot CLI login,
`COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, `GITHUB_TOKEN`, explicit OAuth user tokens, or
BYOK provider config. Do not put tokens in YAML or commit them to the repo.

