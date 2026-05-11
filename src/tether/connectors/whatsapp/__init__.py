"""WhatsApp connector package (Phase 2b).

Applies the platform-machine patch (wa-D-neonize.md BLOCKER-1) at import time
so any subsequent ``import neonize`` (typically from
:mod:`tether.connectors.whatsapp.neonize_adapter`) loads the correct DLL on
Windows x64-under-Prism.

Public API (Wave 2 implementations land alongside these stubs):

* :class:`tether.connectors.whatsapp.connector.WhatsAppConnector`
* :class:`tether.connectors.whatsapp.adapter.WhatsAppClientAdapter`

See ADR-0018 for the design rationale + wa-D-synthesis §D1 / §D9 for the
adapter seam contract.
"""
from __future__ import annotations

import os
import platform

import structlog

logger = structlog.get_logger(__name__)


def _apply_platform_machine_patch_if_needed() -> None:
    """Force ``platform.machine() == 'AMD64'`` on Windows x64-under-Prism.

    Background: on Snapdragon X Elite, ``platform.machine()`` returns
    ``'ARM64'`` even when the Python process is x64 under Prism emulation.
    neonize's ``utils/platform.py:36`` uses this to pick a Go DLL, so it
    tries to load ``neonize-windows-arm64.dll`` into an x64 process and
    crashes with ``OSError: [WinError 193]``. Both DLLs ship in the
    ``win_amd64`` wheel; the patch makes neonize load the correct one.

    Conditional: only patches on Windows when the hardware-reported arch is
    not already AMD64/x86_64. A no-op on Linux/macOS or on native ARM
    Python (which Tether does not support anyway — CodeLinaro wheels are
    cp312-cp312-win_amd64 only).
    """
    if os.name != "nt":
        return
    current = platform.machine()
    if current.lower() in ("amd64", "x86_64"):
        return
    logger.info(
        "whatsapp.platform_machine_patched",
        current=current,
        patched_to="AMD64",
    )
    platform.machine = lambda: "AMD64"  # type: ignore[assignment]


_apply_platform_machine_patch_if_needed()

__all__: list[str] = []
