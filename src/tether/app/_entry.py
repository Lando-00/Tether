"""Console-script wrapper for ``tether-server``.

Phase 8 RD Fix 3 (gpt-5.5 CONCERN #3): ``tether-server`` ships in the base
wheel (``[project.scripts]``) but its underlying entry point in
``tether.app.__main__`` requires the optional ``server`` extras
(``fastapi`` / ``uvicorn``). A minimal ``pip install tether`` install
would therefore expose a script that crashes with a bare
``ModuleNotFoundError: fastapi``.

This wrapper catches that ``ImportError`` at script invocation time and
prints a clear remediation message before exiting non-zero, so users hit
``pip install tether[server]`` instead of a Python traceback. The import
is performed inside :func:`main` (NOT at module load) so the gating
remains effective.
"""
from __future__ import annotations

import sys


def main() -> None:
    """Entry point for the ``tether-server`` console script."""
    try:
        from tether.app.__main__ import main as _real_main
    except ImportError as exc:
        print(
            "tether-server requires the 'server' optional dependencies.\n"
            "Install with:  pip install tether[server]\n"
            f"\n(import failed: {exc})",
            file=sys.stderr,
        )
        sys.exit(1)
    _real_main()
