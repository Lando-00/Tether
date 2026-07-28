"""Pure input preparation for Notebook research turns.

The persisted user transcript is deliberately never changed here.  Callers use
``effective_question`` only for the current research turn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
import re
from typing import Any, Iterable, Literal, Mapping, Optional

from tether.protocol.orchestration.notebook_state import AtomicFact

_MAX_HISTORY = 12
_MAX_PROMPT = 2_048
_MAX_ARITHMETIC_CLAUSES = 4
_MAX_CANDIDATE_CHARS = 256
_CORRECTION = re.compile(r"^\s*([\w'-]+(?:\s+[\w'-]+){0,2})\*\s*$", re.UNICODE)
_TOKEN = re.compile(r"\w+", re.UNICODE)
_ARITHMETIC = re.compile(
    r"^(?:(?:what\s+is|calculate)\s+)?"
    r"([+-]?\d+(?:\.\d+)?)\s*([+\-*/x×÷])\s*([+-]?\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)
# Clause boundaries used to locate a self-contained arithmetic span. Only the
# span itself is removed from the question — surrounding text (including
# conjunctions inside names such as "Trinidad and Tobago") is left untouched.
_CLAUSE_SPLIT = re.compile(r"[,;!?]+|\b(?:and|then)\b", re.IGNORECASE)
_MULTIPLY_OPS = frozenset({"*", "x", "×"})
_DIVIDE_OPS = frozenset({"/", "÷"})
_SYNTHETIC_PREFIXES = ("<<tool_result", "<<function_call", "<<tool_call")
_STOP_WORDS = frozenset(
    {"what", "which", "where", "when", "why", "how", "tell", "about", "the", "and", "for", "with", "from", "capital", "population"}
)


@dataclass(frozen=True)
class ClarificationRequest:
    reason: Literal["ambiguous_correction", "ambiguous_entity", "unsearchable_input"]
    message: str
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedResearchInput:
    """Transient research input; ``original_prompt`` remains transcript truth.

    ``resolved_question`` is the full corrected question and is what the
    Extractor and Synthesizer see, so locally answered sub-questions are not
    hidden from the final answer. ``effective_question`` additionally has
    resolved arithmetic removed and drives planning and web search only.
    """

    original_prompt: str
    effective_question: str
    local_facts: tuple[AtomicFact, ...] = field(default_factory=tuple)
    clarification: Optional[ClarificationRequest] = None
    resolved_question: str = ""

    def __post_init__(self) -> None:
        if not self.resolved_question:
            object.__setattr__(self, "resolved_question", self.effective_question)

    @property
    def prompt(self) -> str:
        """Compatibility-friendly name for the unmodified user input."""
        return self.original_prompt


def _message_parts(row: Any) -> tuple[Optional[str], Optional[str]]:
    if isinstance(row, Mapping):
        role, content = row.get("role"), row.get("content")
    else:
        role, content = getattr(row, "role", None), getattr(row, "content", None)
    return (role if isinstance(role, str) else None, content if isinstance(content, str) else None)


def _is_correction(text: str) -> Optional[str]:
    match = _CORRECTION.fullmatch(text)
    return match.group(1) if match else None


def _distance(left: str, right: str) -> int:
    """Small deterministic Levenshtein implementation; inputs are bounded."""
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, 1):
        current = [index]
        for other_index, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[other_index] + 1,
                previous[other_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def _replace_correction(question: str, correction: str) -> tuple[Optional[str], tuple[str, ...]]:
    desired = correction.lower()
    words = list(_TOKEN.finditer(question))
    desired_words = _TOKEN.findall(correction)
    span_size = len(desired_words)
    scored: list[tuple[int, int, int, int, str]] = []
    for index in range(len(words) - span_size + 1):
        start, end = words[index].start(), words[index + span_size - 1].end()
        token = question[start:end]
        distance = _distance(" ".join(_TOKEN.findall(token)).lower(), desired)
        # One edit for short words, up to two for ordinary entity names.
        threshold = 1 if len(desired) <= 5 else 2
        # A case-only correction is still a correction: use the spelling and
        # casing supplied in this turn if exactly one prior span supports it.
        if distance <= threshold:
            scored.append((distance, -len(token), start, end, token))
    if not scored:
        return None, ()
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    best = scored[0]
    tied = [item for item in scored if item[:2] == best[:2]]
    if len(tied) != 1 or (len(scored) > 1 and scored[1][0] - best[0] < 1):
        return None, tuple(dict.fromkeys(item[4] for item in scored[:5]))
    return question[:best[2]] + correction + question[best[3]:], ()


def _user_history(prior_history: Iterable[Any]) -> list[str]:
    values: list[str] = []
    for row in prior_history:
        role, content = _message_parts(row)
        if role == "user" and content and not content.lstrip().lower().startswith(_SYNTHETIC_PREFIXES):
            values.append(content[:_MAX_PROMPT])
    return values[-_MAX_HISTORY:]


def _replayed_context(history: list[str]) -> str:
    """Replay old short corrections against the most recent substantive turn."""
    context = ""
    for message in history:
        correction = _is_correction(message)
        if correction is None:
            context = message
            continue
        if context:
            replacement, _ = _replace_correction(context, correction)
            if replacement:
                context = replacement
    return context


def _format_decimal(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _calculation(clause: str) -> Optional[AtomicFact]:
    match = _ARITHMETIC.fullmatch(clause.strip())
    if not match:
        return None
    left_text, raw_operator, right_text = match.groups()
    if max(len(left_text), len(right_text)) > 40:
        return None
    # ``X`` matches case-insensitively; normalize before dispatch so an
    # uppercase multiplication can never fall through to division.
    operator = raw_operator.lower()
    try:
        with localcontext() as context:
            context.prec = 80
            left, right = Decimal(left_text), Decimal(right_text)
            if not (left.is_finite() and right.is_finite()):
                return None
            if operator == "+":
                value = left + right
            elif operator == "-":
                value = left - right
            elif operator in _MULTIPLY_OPS:
                value = left * right
            elif operator in _DIVIDE_OPS:
                if not right:
                    return None
                value = left / right
            else:
                return None
            if not value.is_finite() or abs(value) > Decimal("1e60"):
                return None
    except (InvalidOperation, ZeroDivisionError):
        return None
    normalized = f"{_format_decimal(left)} {raw_operator} {_format_decimal(right)}"
    return AtomicFact(
        text=f"{normalized} = {_format_decimal(value)}",
        source_query=normalized,
        confidence="high",
        source_kind="local_deterministic",
    )


def _extract_calculations(prompt: str) -> tuple[str, tuple[AtomicFact, ...]]:
    """Remove self-contained arithmetic spans and return ``(residual, facts)``.

    Only the matched spans are cut out of the original text, so unrelated
    wording — including conjunctions inside names like "Trinidad and
    Tobago" — survives untouched.
    """
    facts: list[AtomicFact] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for separator in list(_CLAUSE_SPLIT.finditer(prompt)) + [None]:
        end = separator.start() if separator is not None else len(prompt)
        clause = prompt[cursor:end]
        stripped = clause.strip()
        if stripped and len(facts) < _MAX_ARITHMETIC_CLAUSES:
            fact = _calculation(stripped)
            if fact is not None:
                facts.append(fact)
                offset = cursor + (len(clause) - len(clause.lstrip()))
                spans.append((offset, offset + len(stripped)))
        cursor = separator.end() if separator is not None else len(prompt)

    if not facts:
        return prompt.strip(), ()

    residual_parts: list[str] = []
    previous = 0
    for start, end in spans:
        residual_parts.append(prompt[previous:start])
        previous = end
    residual_parts.append(prompt[previous:])
    residual = "".join(residual_parts)
    # Tidy separators orphaned by the removal without rewriting real words.
    residual = re.sub(r"\s*\b(?:and|then)\b\s*(?=[,;.!?]|$)", "", residual, flags=re.IGNORECASE)
    residual = re.sub(r"^[\s,;.!?]*(?:\b(?:and|then)\b)?\s*", "", residual, flags=re.IGNORECASE)
    residual = re.sub(r"([,;.!?])\s*[,;.!?]+", r"\1", residual)
    residual = re.sub(r"\s{2,}", " ", residual).strip(" ,;")
    return residual.strip(), tuple(facts)


def _entity_tokens(question: str) -> set[str]:
    tokens = list(_TOKEN.finditer(question))
    entities: set[str] = set()
    for index, match in enumerate(tokens):
        token = match.group(0)
        if token[:1].isupper() and token.lower() not in _STOP_WORDS:
            entities.add(token.lower())
        if index and tokens[index - 1].group(0).lower() in {"about", "of", "in", "for"}:
            if token.lower() not in _STOP_WORDS and len(token) >= 3:
                entities.add(token.lower())
    return entities


def has_entity_drift(planner_query: str, effective_question: str) -> bool:
    """True only for a near-spelling substitution of an explicit entity.

    This is a correction guard, not a generic planner relevance policy.
    Entities the query preserves exactly are removed from consideration
    first, so a question that legitimately names two similar entities
    (``Compare Ireland and Iceland``) is not treated as drift. Introducing
    a separate proper name is likewise allowed.
    """
    expected = _entity_tokens(effective_question)
    planned = _entity_tokens(planner_query)
    if not expected or not planned:
        return False
    unmatched_expected = expected - planned
    unmatched_planned = planned - expected
    for expected_token in unmatched_expected:
        threshold = 1 if len(expected_token) <= 5 else 2
        for planned_token in unmatched_planned:
            if _distance(expected_token, planned_token) <= threshold:
                return True
    return False


def prepare_research_input(prompt: str, prior_history: Iterable[Any]) -> PreparedResearchInput:
    """Prepare transient correction and deterministic-calculation context."""
    original = prompt
    if len(original) > _MAX_PROMPT:
        return PreparedResearchInput(
            original,
            "",
            clarification=ClarificationRequest(
                "unsearchable_input",
                "Please shorten your question before I research it.",
            ),
        )
    correction = _is_correction(original)
    if correction is not None:
        context = _replayed_context(_user_history(prior_history))
        replacement, candidates = _replace_correction(context, correction) if context else (None, ())
        if replacement is None:
            message = "I need the earlier question to apply that correction."
            if candidates:
                message = "Which earlier term should this correction replace?"
            return PreparedResearchInput(
                original,
                original,
                clarification=ClarificationRequest(
                    "ambiguous_correction",
                    message,
                    # Bound each candidate to the wire-event field limit so a
                    # pathological history span can never abort the stream.
                    tuple(item[:_MAX_CANDIDATE_CHARS] for item in candidates[:5]),
                ),
            )
        residual, facts = _extract_calculations(replacement)
        return PreparedResearchInput(
            original, residual, facts, resolved_question=replacement.strip()
        )

    residual, facts = _extract_calculations(original)
    if not residual and not facts:
        return PreparedResearchInput(
            original, "", clarification=ClarificationRequest(
                "unsearchable_input", "Please provide a question I can research."
            )
        )
    return PreparedResearchInput(
        original, residual, facts, resolved_question=original.strip()
    )


__all__ = [
    "ClarificationRequest",
    "PreparedResearchInput",
    "has_entity_drift",
    "prepare_research_input",
]
