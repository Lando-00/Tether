"""Tether runtime module — process / lifecycle / hardware coordination.

Phase 2 ships only ``WatchdogMode`` (used by ``Engine.from_settings``).
Phase 3 will populate this module with ``HardwareWatchdog`` and friends.
Cited in _synthesis.md §11.2 R14, §11.3 R22, §4 Phase 3 step 35.
"""
