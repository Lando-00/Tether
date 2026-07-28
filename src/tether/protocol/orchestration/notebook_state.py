"""NotebookOrchestrator state types (Phase 9 · ADR-0020).

Pure dataclasses. No I/O, no LLM, no orchestrator logic — those live in
``notebook.py``. Designed so a future research-mode persistent-notebook
follow-up (``fu-research-persisted-notebook``) can serialise these to
SQLite without touching the orchestrator.

Citations:
    - ADR-0020 §D1.bis (fact-text dedup).
    - ADR-0020 §D5 (settings/state split and loop bounds).
    - ``docs/research/06_context_strategies.md`` (Notebook algorithm spec).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
import string
from typing import Literal, Optional


@dataclass(frozen=True)
class AtomicFact:
    """A single atomic fact written into the Notebook by the Extractor.

    ``confidence`` is the extractor's self-rated confidence — kept as a
    string Literal rather than a float so the prompt can be deterministic
    (``"high"|"medium"|"low"``). See ADR-0020 Appendix A, Extractor
    rule 5.
    """

    text: str
    source_query: str
    confidence: Literal["high", "medium", "low"]
    source_kind: Literal["web_search", "local_deterministic"] = "web_search"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _dedup_key(text: str) -> str:
    """Normalize fact text for exact dedup comparison (ADR-0020 §D1.bis)."""
    lowered = text.strip().lower()
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", no_punct).strip()


def _conf_rank(level: Literal["low", "medium", "high"]) -> int:
    """Return confidence ordering for duplicate replacement."""
    return {"low": 0, "medium": 1, "high": 2}[level]


def _normalize_query(q: str) -> str:
    """Normalize a query for processed-query dedup. Punctuation is preserved."""
    return q.strip().lower()


_CONTRAST_TERMS = (
    " but ",
    " however ",
    " whereas ",
    " while ",
    " although ",
    " though ",
    " except ",
    " instead ",
)


def _has_contrast_scope(text: str) -> bool:
    """True when a fact contains contrast/scope language that changes meaning.

    Containment dedup is intentionally conservative around contrast terms:
    "X grew in Europe" and "X grew in Europe but fell in Asia" are related
    but not duplicates. Keep both.
    """
    padded = f" {text.strip().lower()} "
    return any(term in padded for term in _CONTRAST_TERMS)


@dataclass
class NotebookState:
    """Live state of an in-flight research-mode turn.

    Mutable; the orchestrator updates it as the loop runs. The
    ``iteration`` counter advances on each explore→extract→refine cycle.
    Bounds (``max_facts``, ``max_iterations``) are enforced by the
    orchestrator before each iteration via :meth:`should_continue`.

    The fields ``facts`` / ``queue`` / ``processed_queries`` are mutated
    in place by the orchestrator; ``iteration`` is incremented at the
    start of each loop pass.
    """

    facts: list[AtomicFact] = field(default_factory=list)
    queue: deque[str] = field(default_factory=deque)
    processed_queries: set[str] = field(default_factory=set)
    iteration: int = 0

    # Bounds (copied in from ResearchSettings at start of run).
    max_facts: int = 40
    max_iterations: int = 20
    max_facts_per_extract: int = 10

    def should_continue(self) -> bool:
        """True iff the loop should run another iteration.

        False when (a) the queue is empty, (b) we have hit ``max_facts``,
        or (c) we have hit ``max_iterations``. The orchestrator emits
        :class:`NotebookLimitReached` when one of (b)/(c) is the cause —
        see :meth:`limit_kind`.
        """
        return (
            bool(self.queue)
            and len(self.facts) < self.max_facts
            and self.iteration < self.max_iterations
        )

    def try_add_fact(self, fact: AtomicFact) -> bool:
        """Attempt to add a fact.

        Two-pass dedup:

        1. **Exact-match** on the normalized :func:`_dedup_key`. Replace
           the existing fact if the new one has strictly higher
           confidence; otherwise reject.
        2. **Substring-containment** (second pass). When both normalized
           keys are at least 20 characters long and one is a *strict*
           substring of the other, the pair is treated as a paraphrase
           duplicate. We keep the **longer / more specific** text
           regardless of confidence (a longer hedged claim beats a
           shorter absolute one). Equal-length containment is impossible
           here — equal length plus substring relation implies the keys
           are identical, which is already handled by Pass 1.

        Returns ``True`` if the fact was added or if it replaced an
        existing one. Returns ``False`` if it was rejected as a
        duplicate.
        """
        key = _dedup_key(fact.text)

        # Pass 1: exact normalized-key match.
        for i, existing in enumerate(self.facts):
            if _dedup_key(existing.text) == key:
                if _conf_rank(fact.confidence) > _conf_rank(existing.confidence):
                    self.facts[i] = fact
                    return True
                return False

        # Pass 2: substring containment (paraphrase dedup).
        if len(key) >= 20:
            for i, existing in enumerate(self.facts):
                existing_key = _dedup_key(existing.text)
                if len(existing_key) < 20:
                    continue
                if key in existing_key or existing_key in key:
                    if _has_contrast_scope(fact.text) or _has_contrast_scope(existing.text):
                        continue
                    if len(key) > len(existing_key):
                        self.facts[i] = fact
                        return True
                    # len(key) < len(existing_key): shorter loses.
                    # Equal length is unreachable: equal-length containment
                    # implies equal keys, already handled by Pass 1.
                    return False

        self.facts.append(fact)
        return True

    def limit_kind(self) -> Optional[Literal["max_facts", "max_iterations"]]:
        """Return which bound (if any) is currently saturated.

        May return non-None even if the loop is about to terminate
        naturally (e.g., last extract hit max_facts). Orchestrator uses
        this to emit NotebookLimitReached only when a bound is *strictly*
        what stopped the loop (i.e., should_continue() == False AND
        limit_kind() != None AND the natural exit condition is not also
        true).
        """
        if len(self.facts) >= self.max_facts:
            return "max_facts"
        if self.iteration >= self.max_iterations:
            return "max_iterations"
        return None


__all__ = ["AtomicFact", "NotebookState"]
