"""Library-first import surface tests.

``import tether_service`` must NOT pull in FastAPI, MLC, Brave, or the
orchestrator. Per _synthesis.md §4 Phase 2 step 22 (R8 lazy-import rule).

These tests run in subprocesses because pytest itself loads many modules,
so an in-process check would be polluted by the test runner's imports.
"""
from __future__ import annotations

import subprocess
import sys


PYTHON = sys.executable


def _module_loaded_after_import_tether_service(module_name: str) -> bool:
    """Run ``import tether_service`` in a fresh subprocess and check whether
    ``module_name`` ended up in ``sys.modules``.
    """
    code = (
        "import sys\n"
        "import tether_service  # noqa: F401\n"
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


def test_import_tether_service_does_not_load_fastapi():
    assert not _module_loaded_after_import_tether_service("fastapi")


def test_import_tether_service_does_not_load_mlc():
    assert not _module_loaded_after_import_tether_service("mlc_llm")


def test_import_tether_service_does_not_load_brave_client():
    assert not _module_loaded_after_import_tether_service(
        "tether_service.tools.brave_client"
    )


def test_import_tether_service_does_not_load_orchestrator():
    assert not _module_loaded_after_import_tether_service(
        "tether_service.protocol.orchestration.orchestrator"
    )
