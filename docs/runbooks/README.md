# Runbooks

Operational runbooks for incidents, recovery, and known-issue mitigations.

## Contents

- [`one-command-launch.md`](./one-command-launch.md) — `tether.ps1`: start GenieX, the Tether service and the CLI with one command
- [`fresh-env-setup.md`](./fresh-env-setup.md) — create the native ARM64 stdlib venv for GenieX or bootstrap the x64/Prism conda environment for MLC
- [`geniex-provider.md`](./geniex-provider.md) — operate the default external GenieX NPU provider and its out-of-repo model store
- [`ollama-provider.md`](./ollama-provider.md) — configure and validate local/LAN Ollama providers
- [`shutdown-hang-fix-summary.md`](./shutdown-hang-fix-summary.md) — root cause + fix for the OpenCL/Adreno shutdown hang on Snapdragon X Elite
- [`model-dependent-shutdown-fix.md`](./model-dependent-shutdown-fix.md) — model-specific shutdown behavior differences (Qwen2.5-7B vs Qwen3-4B `prefill_chunk_size`)
