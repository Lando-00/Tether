"""RegexConfirmIntentClassifier — v1 impl of the confirm-intent seam.

Per ADR-0019 (Appendix A pattern; Appendix B test corpus).
"""

from __future__ import annotations

import re
from typing import Final

from tether.core.logging import logger
from tether.protocol.intent.classifier import ConfirmIntentClassifier

_MAX_LEN: Final[int] = 2048

_CONFIRM_RE: Final[re.Pattern[str]] = re.compile(
    r"^("
    # multi-word
    r"that(?:'s| is) (?:right|correct)"
    r"|sounds good|looks (?:good|fine|great)"
    r"|go (?:ahead|for it|on)"
    r"|sure thing|yeah sure|yes please"
    r"|send (?:it|them|away)|fire (?:away|it off)|ship it"
    r"|do it|copy that|roger that|10-4"
    r"|never mind"
    # single-word
    r"|yes|yeah|yep|yup|yas|ya"
    r"|sure|ok|okay|okey|okie|kk|k"
    r"|send|go|proceed|continue|confirm(?:ed|ing)?"
    r"|correct|right|approved?|lgtm|sgtm"
    r"|affirmative|roger|copy"
    # bare emoji
    r"|\U0001F44D|\u2705|\U0001F44C|\U0001F197|\u2611\ufe0f?|\u2714\ufe0f?"
    r")(?=\W|$)"
)

_DENY_SUBSTRINGS: Final[tuple[str, ...]] = (
    "nope",
    "nah",
    "naw",
    "n't",
    "wait",
    "hold on",
    "stop",
    "cancel",
    "abort",
    "undo",
    "redo",
    "scrap",
    "scrub",
    "delete",
    "remove",
    "drop",
    "discard",
    "throw out",
    "wrong",
    "incorrect",
    "typo",
    "mistake",
    "ignore",
    "skip",
    "revert",
    "rollback",
    "don't",
    "do not",
    "never",
    "nvm",
    "never mind",
    "nevermind",
    "forget it",
    "let me think",
    "maybe",
    "actually",
    "hmm",
    "on second thought",
    "second thoughts",
    " but ",
    "but,",
    "instead",
    "except",
    "rephrase",
    "reword",
    "rewrite",
    "change",
    "edit",
    "modify",
    " fix",
    "tweak",
    "in spanish",
    "in french",
    "in german",
    " add ",
    " remove ",
    " replace ",
    " first",
)

_SALUTATION_RE: Final[re.Pattern[str]] = re.compile(
    r"^(tether|ai|bot|assistant|@\w+)[,:\s]+"
)
_LEADING_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"^[\"'`\(\[\s,\.!\?]+")
_WS_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_DENY_WORD_RE: Final[re.Pattern[str]] = re.compile(r"(?:^|\W)(?:no|not)(?=\W|$)")
_DIGIT_RUN_RE: Final[re.Pattern[str]] = re.compile(r"\d{8,}")


def _normalize(message: str) -> str:
    s = message.strip().lower()
    s = _SALUTATION_RE.sub("", s)
    s = _LEADING_PUNCT_RE.sub("", s)
    return _WS_RE.sub(" ", s)


class RegexConfirmIntentClassifier(ConfirmIntentClassifier):
    """Deterministic regex-based v1 classifier. English-only.

    Per ADR-0019. The ABC seam allows future LLM/multilingual impl
    without consumer changes.
    """

    def classify(self, last_user_message: str) -> bool:
        if not last_user_message:
            return False
        if len(last_user_message) > _MAX_LEN:
            return False
        normalized = _normalize(last_user_message)
        if not normalized:
            return False
        if _DENY_WORD_RE.search(normalized):
            return False
        padded = f" {normalized} "
        for token in _DENY_SUBSTRINGS:
            if token in padded:
                return False
        matched = bool(_CONFIRM_RE.match(normalized))
        if matched and _DIGIT_RUN_RE.search(normalized):
            logger.warning(
                "confirm_intent.digit_run_co_occurrence",
                normalized_len=len(normalized),
            )
        return matched


__all__ = ["RegexConfirmIntentClassifier"]
