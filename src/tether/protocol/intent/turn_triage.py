"""Turn triage — does this turn need external evidence? (design doc: §5 Option B)

The fact-based orchestrator (:class:`~tether.protocol.orchestration.notebook.NotebookOrchestrator`)
is a *web-research* loop: Plan → Explore → Extract → Refine → Synthesize. That is
the right shape for "who won the 2022 World Cup?" and completely the wrong shape
for "hello", "write me a poem", or "what did I just say?".

Triage is the cheap first decision that lets the research loop become the
**default** orchestrator without turning small talk into five web searches.

Design notes
------------

**Bias toward ``DIRECT``.** The two failure directions are not symmetric:

* Wrongly routing a research question to ``DIRECT`` → the model answers from
  parametric memory. Imperfect, but it is exactly what the previous chat-mode
  default did, so it is never a regression.
* Wrongly routing small talk to ``RESEARCH`` → several web searches, seconds of
  latency, and (with no search backend configured) an error where a greeting
  should have been.

So a turn is only routed to ``RESEARCH`` on a *positive* signal. Silence means
``DIRECT``.

**Cheap by contract.** :meth:`TurnTriage.classify` runs on every turn, before any
model call. The shipped implementation is pure string work (microseconds). An
LLM-backed implementation is a legal future variant, but it MUST short-circuit
obvious cases before spending a model call — on a 1.7 B NPU model, planning
already costs ~3 s and a triage call would be paid by every "hello".

Mirrors the :class:`~tether.protocol.intent.classifier.ConfirmIntentClassifier`
seam (ADR-0019): ABC here, concrete rules impl in
:mod:`tether.protocol.intent.rules_turn_triage`, selected by settings.
"""

from __future__ import annotations

import abc
import enum


class TurnKind(str, enum.Enum):
    """What kind of handling a turn needs."""

    DIRECT = "direct"
    """Answer conversationally from the model + session history. No search."""

    RESEARCH = "research"
    """Gather external evidence first: Plan → Explore → Extract → Synthesize."""


class TurnTriage(abc.ABC):
    """Strategy interface for "does this turn need external evidence?"."""

    @abc.abstractmethod
    def classify(self, prompt: str, *, has_history: bool = False) -> TurnKind:
        """Return the handling this turn needs.

        Args:
            prompt: The user's message, verbatim.
            has_history: Whether the session has prior turns. Lets an
                implementation treat back-references ("what did I just say?")
                as answerable rather than researchable.

        Implementations MUST be cheap, MUST tolerate an empty prompt (returning
        :attr:`TurnKind.DIRECT`), and MUST NOT raise.
        """


class AlwaysResearchTriage(TurnTriage):
    """Every turn is a research turn — the pre-triage behaviour.

    Kept so the previous semantics of ``mode="research"`` remain reachable
    verbatim, and as the honest null object for tests that want to exercise the
    research loop unconditionally.
    """

    def classify(self, prompt: str, *, has_history: bool = False) -> TurnKind:
        return TurnKind.RESEARCH


__all__ = ["TurnKind", "TurnTriage", "AlwaysResearchTriage"]
