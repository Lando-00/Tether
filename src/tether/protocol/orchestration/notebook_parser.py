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
from datetime import datetime, timezone
import json
import re
from typing import Any, Optional

import structlog

from tether.core.redact import redact_text
from tether.protocol.orchestration.notebook_state import AtomicFact


_FENCE_RE = re.compile(r"```(?:[a-zA-Z]+)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)
_FACT_LINE_RE = re.compile(
    r"^\s*(?:FACT[:\s\-]+|[-*\u2022]\s+|\d+[.)]\s+|\[\d+\]\s+)(.+?)\s*$",
    re.IGNORECASE,
)
_SNIPPET_META_RE = re.compile(
    r"^\s*the\s+snippet\s+(?:says|mentions|talks\s+about|indicates|states)\b",
    re.IGNORECASE,
)
_VALID_CONFIDENCE = {"low", "medium", "high"}
_MAX_PLAN_QUERY_CHARS = 160
_MAX_PLAN_QUERY_WORDS = 16
_PLAN_META_RE = re.compile(
    r"\b(?:the user|user is|i need|i should|but wait|however,?\s+the user|"
    r"let'?s|we need|that'?s a separate query)\b",
    re.IGNORECASE,
)

# Narrowly anchored extraction-process vocabulary. Each entry is a
# prefix (already lowercased, includes trailing space) that strongly
# indicates the LLM is describing the search results / its own
# reasoning rather than emitting a world-fact. We deliberately do NOT
# include generic English starters like "this is " or "i " — those
# would over-fire on valid facts ("This benchmark reports 45 TOPS",
# "The first iPhone shipped in 2007", "Confidence in the vaccine
# increased in 2025"). See plan.md §17.4 W1-A and the live-test
# failure mode addressed by I-2.
_REASONING_LEAK_PREFIXES: tuple[str, ...] = (
    "the first snippet ",
    "the second snippet ",
    "the third snippet ",
    "the fourth snippet ",
    "the fifth snippet ",
    "this snippet ",
    "confidence is ",
    "the confidence ",
    "this is a fact about ",
    "this fact ",
    "i think ",
    "the user is asking ",
)

logger = structlog.get_logger(__name__)


def _is_reasoning_leak(text: str) -> bool:
    """Return True iff ``text`` opens with extraction-process meta-prose.

    Comparison is case-insensitive and tolerant of leading whitespace.
    Only the narrow ``_REASONING_LEAK_PREFIXES`` vocabulary triggers;
    generic English openers are intentionally NOT included so that
    valid facts like "The first iPhone shipped in 2007" or "This
    benchmark reports 45 TOPS" are preserved.
    """
    if not text:
        return False
    needle = text.strip().lower()
    return (
        any(needle.startswith(prefix) for prefix in _REASONING_LEAK_PREFIXES)
        or _SNIPPET_META_RE.match(text) is not None
    )


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
    reasoning_leak_dropped: int = 0


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
    raw_length = len(raw or "")
    try:
        for layer_fn in (_layer_1_direct_json, _layer_2_strip_fences, _layer_3_balanced_brace_extract):
            result = layer_fn(raw)
            if result is not None:
                queries = result.get("key_elements", [])
                if isinstance(queries, list):
                    return _sanitize_plan_queries(queries, max_queries=max_queries)

        bullets = _layer_4_bullet_fallback(raw)
        if bullets:
            return _sanitize_plan_queries(bullets, max_queries=max_queries)

        logger.warning("notebook_parser.plan_total_fail", raw_length=raw_length)
        return []
    except Exception:
        logger.warning("notebook_parser.plan_total_fail", raw_length=raw_length, exc_info=True)
        return []


def _sanitize_plan_queries(items: list[Any], *, max_queries: int) -> list[str]:
    """Return short, search-shaped planner queries.

    Real Qwen planner runs can place chain-of-thought/meta commentary inside
    a ``key_elements`` string (for example, "The math problem 25 + 50. But
    wait, the user..."). Those strings are not useful Brave queries and can
    trigger 422s. Keep only concise, self-contained search-query candidates.
    """
    queries: list[str] = []
    seen: set[str] = set()
    for item in items:
        query = str(item).strip()
        if not query:
            continue
        if len(query) > _MAX_PLAN_QUERY_CHARS:
            continue
        if len(query.split()) > _MAX_PLAN_QUERY_WORDS:
            continue
        if _PLAN_META_RE.search(query):
            continue
        normalized = _normalize_plan_query(query)
        if normalized in seen:
            continue
        seen.add(normalized)
        queries.append(query)
        if len(queries) >= max_queries:
            break
    return queries


def _normalize_plan_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


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
    raw_length = len(raw or "")
    try:
        for layer_fn, layer_num in (
            (_layer_1_direct_json, 1),
            (_layer_2_strip_fences, 2),
            (_layer_3_balanced_brace_extract, 3),
        ):
            result = layer_fn(raw)
            if result is not None:
                facts_raw = result.get("facts", [])
                followups_raw = result.get("follow_up_queries", [])
                if not isinstance(followups_raw, list):
                    followups_raw = []
                coerced = _coerce_facts(facts_raw, source_query)
                kept_facts: list[AtomicFact] = []
                leak_dropped = 0
                for fact in coerced:
                    if _is_reasoning_leak(fact.text):
                        leak_dropped += 1
                        continue
                    kept_facts.append(fact)
                facts = kept_facts[:max_facts]
                followups = [q.strip() for q in followups_raw if isinstance(q, str) and q.strip()][
                    :max_facts
                ]
                return ExtractResult(
                    facts=facts,
                    follow_up_queries=followups,
                    parser_layer=layer_num,
                    raw_length=raw_length,
                    reasoning_leak_dropped=leak_dropped,
                )

        bullets = _layer_4_bullet_fallback(raw)
        if bullets:
            leak_dropped = 0
            kept_bullets: list[str] = []
            for bullet in bullets:
                if _is_reasoning_leak(bullet):
                    leak_dropped += 1
                    continue
                kept_bullets.append(bullet)
            facts = [
                AtomicFact(
                    text=b,
                    source_query=source_query,
                    confidence="low",
                    created_at=datetime.now(timezone.utc),
                )
                for b in kept_bullets[:max_facts]
            ]
            return ExtractResult(
                facts=facts,
                follow_up_queries=[],
                parser_layer=4,
                raw_length=raw_length,
                reasoning_leak_dropped=leak_dropped,
            )

        logger.warning(
            "notebook_parser.extract_total_fail",
            raw_length=raw_length,
            source_query=redact_text(source_query),
        )
        return ExtractResult(facts=[], follow_up_queries=[], parser_layer=5, raw_length=raw_length)
    except Exception:
        logger.warning(
            "notebook_parser.extract_total_fail",
            raw_length=raw_length,
            source_query=redact_text(source_query),
            exc_info=True,
        )
        return ExtractResult(facts=[], follow_up_queries=[], parser_layer=5, raw_length=raw_length)


# ---------------------------------------------------------------------------
# Layer-specific helpers (exposed for unit testing).
#
# Wave 3 rs-T-C exercises each layer directly against the 20-row
# corpus, asserting which layer parses each row. Keeping the helpers
# module-level (not nested) makes them addressable from tests.
# ---------------------------------------------------------------------------


def _layer_1_direct_json(raw: str) -> Optional[dict]:
    """Try ``json.loads(raw)`` straight up. Returns ``None`` on failure."""
    try:
        result = json.loads((raw or "").strip())
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _layer_2_strip_fences(raw: str) -> Optional[dict]:
    """Strip ``` ```json ... ``` ``` code fences, retry ``json.loads``.

    Returns ``None`` if no fence found or inner content still
    unparseable.
    """
    try:
        match = _FENCE_RE.search(raw or "")
        if not match:
            return None
        result = json.loads(match.group(1).strip())
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _layer_3_balanced_brace_extract(raw: str) -> Optional[dict]:
    """Extract first balanced ``{...}`` block + repair loose JSON.

    Repairs include trailing commas before ``}`` / ``]``, single-quoted
    strings → double-quoted, and unquoted keys (where unambiguous).
    Returns ``None`` if no balanced block found or repair still fails.
    """
    try:
        block = _first_balanced_object(raw or "")
        if block is None:
            return None
        result = _loads_dict(block)
        if result is not None:
            return result
        return _loads_dict(_repair_loose_json(block))
    except Exception:
        return None


def _layer_4_bullet_fallback(raw: str) -> list[str]:
    """Line-by-line bullet / numbered / ``FACT:`` extraction.

    Returns a list of fact texts (no follow_up_queries from this
    layer — the bullet format does not distinguish facts from
    follow-ups). Wave 2 wraps each into an :class:`AtomicFact` with
    ``confidence="low"`` in :func:`parse_extract_output`. Returns an
    empty list if no bullet-shaped lines are found.
    """
    facts: list[str] = []
    try:
        for line in (raw or "").splitlines():
            match = _FACT_LINE_RE.match(line)
            if not match:
                continue
            candidate = match.group(1).strip().rstrip(",;.")
            if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {'"', "'"}:
                candidate = candidate[1:-1].strip()
            if candidate:
                facts.append(candidate)
        return facts
    except Exception:
        return []


def _coerce_facts(items: Any, source_query: str) -> list[AtomicFact]:
    """Coerce extractor ``facts`` payload into ``AtomicFact`` instances."""
    if not isinstance(items, list):
        return []

    facts: list[AtomicFact] = []
    for item in items:
        try:
            if isinstance(item, str):
                text = item.strip()
                confidence = "medium"
            elif isinstance(item, dict):
                raw_text = item.get("text") or item.get("fact") or item.get("statement") or ""
                text = raw_text.strip() if isinstance(raw_text, str) else ""
                raw_confidence = item.get("confidence", "medium")
                confidence = raw_confidence.strip().lower() if isinstance(raw_confidence, str) else "medium"
                if confidence not in _VALID_CONFIDENCE:
                    confidence = "medium"
            else:
                continue
            if not text:
                continue
            facts.append(
                AtomicFact(
                    text=text,
                    source_query=source_query,
                    confidence=confidence,  # type: ignore[arg-type]
                    created_at=datetime.now(timezone.utc),
                )
            )
        except Exception:
            continue
    return facts


def _loads_dict(raw: str) -> Optional[dict]:
    try:
        result = json.loads(raw)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _first_balanced_object(raw: str) -> Optional[str]:
    start = raw.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    return None


def _repair_loose_json(raw: str) -> str:
    repaired = re.sub(r",(\s*[}\]])", r"\1", raw)
    repaired = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", _single_quoted_to_double, repaired)
    repaired = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)", r'\1"\2"\3', repaired)
    return repaired


def _single_quoted_to_double(match: re.Match[str]) -> str:
    inner = match.group(1).replace('"', '\\"')
    return f'"{inner}"'


__all__ = [
    "ExtractResult",
    "parse_plan_output",
    "parse_extract_output",
    "_is_reasoning_leak",
    "_REASONING_LEAK_PREFIXES",
]
