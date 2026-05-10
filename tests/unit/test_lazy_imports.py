"""Library-first import surface tests.

``import tether`` must NOT pull in FastAPI, MLC, Brave, or the
orchestrator. Per _synthesis.md §4 Phase 2 step 22 (R8 lazy-import rule).

These tests run in subprocesses because pytest itself loads many modules,
so an in-process check would be polluted by the test runner's imports.
"""
from __future__ import annotations

import subprocess
import sys


PYTHON = sys.executable


def _module_loaded_after_import_tether(module_name: str) -> bool:
    """Run ``import tether`` in a fresh subprocess and check whether
    ``module_name`` ended up in ``sys.modules``.
    """
    code = (
        "import sys\n"
        "import tether  # noqa: F401\n"
        f"print({module_name!r} in sys.modules)\n"
    )
    res = subprocess.run(
        [PYTHON, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    out = res.stdout.strip().splitlines()[-1]
    assert out in ("True", "False"), f"unexpected output: {res.stdout!r}"
    return out == "True"


def test_import_tether_does_not_load_fastapi():
    assert not _module_loaded_after_import_tether("fastapi")


def test_import_tether_does_not_load_mlc():
    assert not _module_loaded_after_import_tether("mlc_llm")


def test_import_tether_does_not_load_brave_client():
    assert not _module_loaded_after_import_tether(
        "tether.tools.brave_client"
    )


def test_import_tether_does_not_load_orchestrator():
    assert not _module_loaded_after_import_tether(
        "tether.protocol.orchestration.orchestrator"
    )


def test_import_tether_does_not_load_connectors_router():
    """Phase 4.5 step 47e: the HTTP connectors router pulls in FastAPI
    and the connector spec types; ``import tether`` must NOT
    eagerly trigger that — only :func:`create_app` does."""
    assert not _module_loaded_after_import_tether(
        "tether.app.http.routers.connectors"
    )
