"""Console-script wrapper for ``tether-cli``.

Phase 8 RD Fix 3 (gpt-5.5 CONCERN #3): ``tether-cli`` ships in the base
wheel (``[project.scripts]``) but its underlying entry point in
``tether.cli.main`` requires the optional ``cli`` extras
(``typer`` / ``rich`` / ``prompt-toolkit``). A minimal
``pip install tether`` install would therefore expose a script that
crashes with a bare ``ModuleNotFoundError: typer``.

This wrapper catches that ``ImportError`` at script invocation time and
prints a clear remediation message before exiting non-zero, so users hit
``pip install tether[cli]`` instead of a Python traceback. The import is
performed inside :func:`main` (NOT at module load) so the gating remains
effective.
"""
from __future__ import annotations

import sys


def main() -> None:
    """Entry point for the ``tether-cli`` console script."""
    try:
        from tether.cli.main import app
    except ImportError as exc:
        print(
            "tether-cli requires the 'cli' optional dependencies.\n"
            "Install with:  pip install tether[cli]\n"
            f"\n(import failed: {exc})",
            file=sys.stderr,
        )
        sys.exit(1)
    app()
