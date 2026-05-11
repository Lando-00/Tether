# ADR-0013: `src/` layout + `tether_service` deprecation alias

- **Status**: Accepted (Phase 8 of refactor)
- **Date**: 2026-05 (Phase 8)
- **Synthesis digest**: [`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md)

## Context

Pre-refactor, the importable package was `tether_service`, sitting at the repo root next to
the legacy `llm_service/`. Two related issues motivated change. First, the **package name**:
`tether_service` is awkward, double-named, and predates the decision to brand the project
`tether`. Second, the **layout**: a flat repo-root package has no separation between
"installable thing" and "ancillary repo content" (scripts, docs, configs, examples). Tests
that imported the package while sitting in the same directory could accidentally pick up the
source tree instead of the installed wheel — a class of bugs the `src/`-layout convention
specifically prevents.

## Decision

Adopt **`src/` layout** with import name **`tether`** and a transitional **`tether_service`**
re-export package:

- **Layout**: `src/tether/` is the package. `pyproject.toml` (hatchling backend) declares
  `[tool.hatch.build.targets.wheel] packages = ["src/tether"]`. Distribution name stays
  `tether` (user-ratified §11.6).
- **Migration order** within Phase 8 step 80 (synthesis §4): `config → core → context →
  protocol → tools → providers → adapters → app/cli/server`. Each sub-package moves in
  one PR, dotted-config paths and imports updated together (step 82).
- **Console scripts**: `tether-server` (`src/tether/server/main.py`) and `tether-cli`
  (`src/tether/cli/main.py`).
- **Transitional alias**: `src/tether_service/` is created as a thin re-export package:
  ```python
  # src/tether_service/__init__.py
  import warnings
  from tether import *  # noqa: F401, F403
  warnings.warn(
      "tether_service is a deprecation alias; import from tether",
      DeprecationWarning, stacklevel=2,
  )
  ```
  Sub-modules (`tether_service.context.sqlite_store`, etc.) re-export their `tether.*`
  counterparts so existing user scripts and the Connectors Spec's `tether_service.*`
  paths keep working during the migration window.
- **Alias deletion deferred indefinitely** (§11 R22 + §11.6 #15): step 95 was originally
  "delete `tether_service` after one cycle." Deferred — keeping the alias is cheap;
  deleting it the day it lands breaks downstream `pip install tether-foo` packages that
  reference the old path. Any future deletion is a separate major-version event.
- **Other cleanup in Phase 8**: `tests/` lives at repo root, not in `src/`. `scripts/dev/`
  hosts the moved root utility scripts. `models/` replaces `dist/`. `legacy/`,
  `llm_service/`, `cli_chat.py`, `launch_*.{py,ps1}`, `run_debug.{py,bat,ps1}`,
  `tether_service.zip` retire (archived to `archive/pre-tether-src-layout-2026-05-09`
  branch, then deleted from `main`).
- **`pytest.ini` migrates** to `[tool.pytest.ini_options]` in `pyproject.toml` (Phase 1
  step 14; mentioned here for completeness of the layout story).

## Consequences

### Positive
- `from tether import Engine` reads naturally and matches the project name.
- `src/` layout prevents the "import from tree, not wheel" class of test bugs.
- Existing user scripts and the Connectors Spec's `tether_service.*` paths keep
  working through the transitional alias, with a clear deprecation warning.
- Hatchling + extras (`server`, `cli`, `brave`, `ollama`, `mlc`, `sqlcipher`, `dev`) make
  `pip install tether[mlc]` clean.

### Negative
- The transitional alias is technical debt that we are explicitly choosing not to delete.
  It costs ~50 LOC of re-export shims forever.
- Migration step 80 ordering matters; doing sub-packages out of order causes import
  cycles. Mitigated by the documented dependency order.

### Trade-offs accepted
- Indefinite alias retention prioritizes downstream stability over a clean tree. For a
  single-user app this is the right trade.

## Alternatives considered

- **Flat `tether/` at repo root, no `src/`** — rejected: loses the test-isolation
  benefit; `pytest` must be configured to ignore the source tree on `sys.path`.
- **Rename to `tether-llm`** — deferred per §11.6 question: PyPI publishing is not
  imminent; renaming the dist later is a non-event compared to renaming imports.
- **Hard-cut `tether_service` at the same release** — rejected per §11 R22; alignment
  cost exceeds the technical-debt cost.

## References

- Synthesis digest: [`../refactor/synthesis-2026-05.md`](../refactor/synthesis-2026-05.md)
- `src/tether/__init__.py`, `src/tether_service/__init__.py` (alias)
- `pyproject.toml` (hatchling, extras), `[tool.pytest.ini_options]`
- ADR-0001 (the public surface this layout exports)
