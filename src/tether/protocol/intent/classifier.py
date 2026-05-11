"""Confirm-intent classifier ABC (Phase 2b · ADR-0019).

The classifier flips
:attr:`tether.core.types.ToolExecutionContext.user_confirmed_send` in
:mod:`tether.protocol.orchestration.chatty` so draft+confirm send tools
(``whatsapp_confirm_send`` today, future ``gmail_confirm_send`` and any
other destructive send) can require explicit last-turn affirmation per
ADR-0015 / connector spec §4 footer.

Concrete implementations live in sibling modules
(``regex_classifier.py`` ships in Wave 2 / IMP-C; future
``llm_classifier.py`` for v2 multilingual support — wa-D-synthesis §D8).
The :class:`NullConfirmIntentClassifier` always returns ``False`` and is
the default if no impl is configured — preserves current behaviour for
callers who do not want the feature.
"""

from __future__ import annotations

import abc


class ConfirmIntentClassifier(abc.ABC):
    """Strategy interface for "is this message an explicit confirmation?".

    The orchestrator constructs one instance per Engine (chosen by
    settings); ``classify()`` is called per turn before any destructive
    tool is dispatched. Implementations MUST be cheap (the regex impl
    runs in microseconds; the future LLM impl will short-circuit on
    obviously-non-confirmation inputs before any model call).

    Per wa-D-synthesis §D8.
    """

    @abc.abstractmethod
    def classify(self, last_user_message: str) -> bool:
        """Return ``True`` iff ``last_user_message`` is an explicit
        affirmation of the previously-drafted action.

        Concrete impls SHOULD:

        * Normalize first (strip + lower + drop salutation prefix +
          strip leading/trailing punctuation).
        * Apply a deny-list early-exit before the affirmative match
          (catches ``"yes but cancel"`` / ``"yes, actually wait"`` —
          the key hardening over the locked-spec literal regex
          proposal per wa-D-synthesis §D8).
        * Classify on the WHOLE normalized message — the deny-list
          must see the entire message, not just a prefix.

        ``last_user_message`` may be empty; impls MUST return ``False``
        in that case.
        """


class NullConfirmIntentClassifier(ConfirmIntentClassifier):
    """No-op classifier: always returns ``False``.

    The default when no classifier is configured — preserves the
    pre-ADR-0019 behaviour where destructive send tools never see
    ``user_confirmed_send=True`` and therefore always refuse with a
    "draft requires explicit confirmation" response. Callers that DO
    want the draft+confirm pattern wire a real impl
    (``RegexConfirmIntentClassifier`` from Wave 2 / IMP-C) via
    settings.
    """

    def classify(self, last_user_message: str) -> bool:
        return False


__all__ = ["ConfirmIntentClassifier", "NullConfirmIntentClassifier"]
