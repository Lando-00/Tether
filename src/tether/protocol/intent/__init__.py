"""Intent classification strategy package (Phase 2b · ADR-0019).

Provides the :class:`ConfirmIntentClassifier` ABC consumed by
:mod:`tether.protocol.orchestration.chatty` to decide whether a user
turn is an explicit affirmation of a previously-drafted destructive
action (e.g. ``whatsapp_confirm_send``). The null default preserves
current behaviour for engines that do not opt in.

See ``classifier.py`` for the contract; concrete implementations
(``regex_classifier`` etc.) ship in Wave 2 / IMP-C per
wa-D-synthesis §D8.
"""

from __future__ import annotations

from tether.protocol.intent.classifier import (
    ConfirmIntentClassifier,
    NullConfirmIntentClassifier,
)

__all__ = ["ConfirmIntentClassifier", "NullConfirmIntentClassifier"]
