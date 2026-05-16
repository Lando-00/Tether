"""5-layer JSON output parser for NotebookOrchestrator (ADR-0020 D3).

LLMs do not reliably emit clean JSON. This parser falls through 5 layers
until it gets a result or gives up (returning an empty
:class:`ExtractResult`). It NEVER raises — the orchestrator must always
get a typed value back, even on total parser failure, so the loop can
continue.

Layer order (bullet from rs-D-prompts.md §2.1):

1. ``json.loads(raw)`` — happy path.
2. Strip ```` ```json ... ``` ```` code fences, retry ``json.loads``.
3. Regex-extract first balanced ``{...}`` block + repair trailing
   commas / single quotes / unquoted keys, retry ``json.loads``.
4. Line-by-line ``FACT: <text>`` / ``- <text>`` / numbered bullet
   extraction (no follow-up queries from this layer).
5. Log raw output at ``ERROR``, return empty :class:`ExtractResult`.

The 20-row acceptance corpus lives in
``tests/unit/protocol/orchestration/test_notebook_parser.py`` (Wave 3
rs-T-C); see rs-D-prompts.md §2.2 for the test cases verbatim.

Wave 1 deliverable: SIGNATURES ONLY. All bodies raise
``NotImplementedError`` with the Wave-2 owner tag. Wave 2 IMP-C fills
in the layer logic; rs-T-C exercises it against the 20-row corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Optional

from tether.protocol.orchestration.notebook_state import AtomicFact


_FENCE_RE = re.compile(r"```(?:[a-zA-Z]+)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ExtractResult:
    """Output of :func:`parse_extract_output`. Both lists may be empty.

    Empty ``facts`` + empty ``follow_up_queries`` is a valid answer
    (the extractor decided the snippets contained nothing new). The
    orchestrator treats it as "this iteration produced nothing"; the
    loop continues if the queue still has work.
    """

    facts: list[AtomicFact] = field(default_factory=list)
    follow_up_queries: list[str] = field(default_factory=list)
    parser_layer: int = 5
    raw_length: int = 0


def parse_plan_output(raw: str, *, max_queries: int = 5) -> list[str]:
    """Parse Planner output into a list of seed sub-queries.

    Falls through the 5 layers (see module docstring). On total
    failure returns an empty list — the orchestrator then has nothing
    to enqueue, emits an empty notebook, and the synthesiser falls
    back to its "empty notebook" disclaimer per rs-D-prompts.md §1.5
    rule 4.

    Args:
        raw: Raw model output (concatenated text deltas from one
            Planner stream).
        max_queries: Maximum seed sub-queries to return.

    Returns:
        Ordered list of seed sub-queries (2-5 strings per the
        Planner contract; may be shorter if the model under-emits or
        the parser fell through to layer 4 / 5).
    """
    raise NotImplementedError("Wave 2 IMP-C")


def parse_extract_output(
    raw: str,
    source_query: str,
    *,
    max_facts: int = 10,
) -> ExtractResult:
    """Parse Extractor output into atomic facts + follow-up queries.

    ``source_query`` is attached to every emitted :class:`AtomicFact`
    so we can trace provenance later (NotebookFactAdded event carries
    it on the wire). Confidence defaults to ``"medium"`` if the LLM
    omits or emits an invalid value (rs-D-prompts §2.1
    ``_coerce``).

    Args:
        raw: Raw model output (concatenated text deltas from one
            Extractor stream).
        source_query: The sub-query whose tool result the extractor
            was processing. Stamped onto every emitted fact.
        max_facts: Maximum facts to return from one Extract call.

    Returns:
        :class:`ExtractResult` with parsed facts (capped to
        ``max_facts_per_extract`` by Wave 2) and follow-up queries.
        Empty on total parser failure.
    """
    raise NotImplementedError("Wave 2 IMP-C")


# ---------------------------------------------------------------------------
# Layer-specific helpers (exposed for unit testing).
#
# Wave 3 rs-T-C exercises each layer directly against the 20-row
# corpus, asserting which layer parses each row. Keeping the helpers
# module-level (not nested) makes them addressable from tests.
# ---------------------------------------------------------------------------


def _layer_1_direct_json(raw: str) -> Optional[dict]:
    """Try ``json.loads(raw)`` straight up. Returns ``None`` on failure."""
    raise NotImplementedError("Wave 2 IMP-C")


def _layer_2_strip_fences(raw: str) -> Optional[dict]:
    """Strip ``` ```json ... ``` ``` code fences, retry ``json.loads``.

    Returns ``None`` if no fence found or inner content still
    unparseable.
    """
    raise NotImplementedError("Wave 2 IMP-C")


def _layer_3_balanced_brace_extract(raw: str) -> Optional[dict]:
    """Extract first balanced ``{...}`` block + repair loose JSON.

    Repairs include trailing commas before ``}`` / ``]``, single-quoted
    strings → double-quoted, and unquoted keys (where unambiguous).
    Returns ``None`` if no balanced block found or repair still fails.
    """
    raise NotImplementedError("Wave 2 IMP-C")


def _layer_4_bullet_fallback(raw: str) -> list[str]:
    """Line-by-line bullet / numbered / ``FACT:`` extraction.

    Returns a list of fact texts (no follow_up_queries from this
    layer — the bullet format does not distinguish facts from
    follow-ups). Wave 2 wraps each into an :class:`AtomicFact` with
    ``confidence="low"`` in :func:`parse_extract_output`. Returns an
    empty list if no bullet-shaped lines are found.
    """
    raise NotImplementedError("Wave 2 IMP-C")


__all__ = [
    "ExtractResult",
    "parse_plan_output",
    "parse_extract_output",
]
