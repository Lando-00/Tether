"""Watchdog mode enum used by ``Engine.from_settings``.

Phase 2 ships the enum + the ``Engine.from_settings(watchdog_mode=...)`` plumbing.
Phase 3 wires ``HardwareWatchdog`` around these modes.

Cited in _synthesis.md §11.2 R14 (LIBRARY + SERVER only; SUPERVISED dropped).
"""
from enum import Enum


class WatchdogMode(Enum):
    """How the Engine should manage shutdown / signals.

    LIBRARY  — caller is responsible for signal handling. ``Engine.aclose()``
               returns control to the caller after a bounded shutdown
               (no ``os._exit``).
    SERVER   — Engine installs signal handlers / force-exit timer. Used by the
               uvicorn server lifespan.
    """

    LIBRARY = "library"
    SERVER = "server"
