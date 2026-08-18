"""Rules-first :class:`TurnTriage` — no model call, microsecond cost.

See :mod:`tether.protocol.intent.turn_triage` for why this is biased toward
``DIRECT``: a turn is routed to ``RESEARCH`` only on a positive evidence signal.
"""

from __future__ import annotations

import re

from tether.protocol.intent.turn_triage import TurnKind, TurnTriage

# Domains a locally-registered tool already covers. These must NOT go to web
# research: the research loop only knows `web_search`, so sending "what time is
# it in Europe/Dublin?" there produces a search (or, with no search backend, a
# failed turn) instead of a `time` tool call. Routing them DIRECT hands them to
# the chat orchestrator, which has the full tool loop.
_LOCAL_TOOL_DOMAINS = (
    "what time", "what's the time", "whats the time", "time is it",
    "current time", "time in ", "what date", "what day", "today's date",
    "weather", "forecast", "temperature", "how hot", "how cold",
    "raining", "sunny", "snowing",
)

# Phrases that clearly need *current or external* facts from the open web.
# These are the positive signals — without one of these (or an entity question,
# below) a turn stays DIRECT.
_EVIDENCE_MARKERS = (
    "latest", "current version", "currently", "today's news", "tonight",
    "this week", "this year", "right now", "recent", "recently", "news",
    "up to date", "up-to-date", "as of",
    "price", "cost of", "stock", "share price", "exchange rate",
    "who won", "who is winning", "score", "results of", "release date",
    "released", "announced", "launch date", "population of", "capital of",
    "search for", "look up", "google", "find out",
)

# Back-references to the conversation itself. These are answerable from history
# and must never trigger a web search.
_SELF_REFERENTIAL = (
    "you said", "i said", "you told me", "i told you", "i asked",
    "earlier", "previously", "before that", "last message", "above",
    "this conversation", "our conversation", "repeat that", "say that again",
    "what did i", "what did you", "summarise our", "summarize our",
)

# Generative / transformational work on text the user supplies or the model
# invents. No external evidence required.
_CREATIVE_OPENERS = (
    "write", "compose", "draft", "rewrite", "reword", "rephrase",
    "translate", "summarise", "summarize", "shorten", "expand",
    "brainstorm", "imagine", "pretend", "roleplay", "role play",
    "code", "implement", "refactor", "debug", "fix",
    "generate", "make up", "invent", "continue",
)

# Small talk and meta questions about the assistant.
_SMALL_TALK = {
    "hi", "hii", "hey", "hello", "yo", "sup", "hiya", "howdy",
    "thanks", "thank you", "ty", "cheers", "nice", "cool", "ok", "okay",
    "bye", "goodbye", "see you", "good morning", "good afternoon",
    "good evening", "good night", "how are you", "how are you?",
    "who are you", "who are you?", "what can you do", "what can you do?",
    "what are you", "help",
}

# "who/what/when/where is <proper noun>" style questions. Interrogative alone is
# not enough (it would catch "what should I write about?"), so a capitalised
# token or a four-digit year must also be present.
_INTERROGATIVE = re.compile(
    r"^\s*(who|what|when|where|which|how many|how much)\b", re.IGNORECASE
)
_PROPER_NOUN = re.compile(r"\b[A-Z][a-z]{2,}")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


class RulesTurnTriage(TurnTriage):
    """Keyword + shape heuristics; defaults to :attr:`TurnKind.DIRECT`."""

    def classify(self, prompt: str, *, has_history: bool = False) -> TurnKind:
        text = (prompt or "").strip()
        if not text:
            return TurnKind.DIRECT

        lowered = text.lower()
        stripped = lowered.rstrip("?!. ")

        # 1. Small talk is never research, regardless of anything else.
        if stripped in _SMALL_TALK:
            return TurnKind.DIRECT

        # 2. Back-references belong to the transcript, not the web. Checked
        #    before evidence markers so "what did you say about the latest
        #    release?" reads history instead of searching.
        if any(marker in lowered for marker in _SELF_REFERENTIAL):
            return TurnKind.DIRECT

        # 3. Questions a local tool already answers (time, weather, forecast).
        #    Checked before evidence markers so "what's the weather right now?"
        #    calls the weather tool instead of searching the web.
        if any(domain in lowered for domain in _LOCAL_TOOL_DOMAINS):
            return TurnKind.DIRECT

        # 4. Explicit evidence markers are the strongest positive signal.
        if any(marker in lowered for marker in _EVIDENCE_MARKERS):
            return TurnKind.RESEARCH

        # 5. Generative work. Checked after evidence markers so "write a summary
        #    of today's news" still researches.
        first_word = re.split(r"[^a-z]+", stripped, maxsplit=1)[0]
        if first_word in _CREATIVE_OPENERS:
            return TurnKind.DIRECT

        # 6. Entity questions: interrogative + a proper noun or a year.
        if _INTERROGATIVE.search(text) and (
            _PROPER_NOUN.search(text) or _YEAR.search(text)
        ):
            return TurnKind.RESEARCH

        # 7. No positive signal — answer directly.
        return TurnKind.DIRECT


__all__ = ["RulesTurnTriage"]
