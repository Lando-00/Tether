"""Fit a conversation into the model's context window.

Why this exists
---------------
Providers do not reject an over-long prompt — they silently evict the front of
the context. Measured against GenieX with the default ``--nctx 4096`` on
``Qwen3-8B:Q4_0``, using a sentinel instruction planted in the system prompt:

===============  ====================
``prompt_tokens``  System prompt
===============  ====================
277               honoured
2329              honoured
4489              **lost**
6757              **lost**
===============  ====================

The request still returns HTTP 200 and a fluent-looking answer, so nothing
upstream notices. That is the dangerous part.

For Tether specifically this is worse than losing conversational context.
Marker-only providers (GenieX, and MLC with ``marker_only_tools``) drop the
``tools=`` parameter, so the system prompt is the *only* place the model learns
the ``<<function_call>>`` calling convention. Once the front of the context is
evicted, the model still sees the per-turn tool roster appended at the end but
no longer knows the syntax for calling any of it — tool calling degrades
silently in exactly the long conversations where it is most useful.

Trimming in the middle, rather than letting the provider trim the front, keeps
the two things that must never be dropped: the system prompt and the most
recent turns.

Token estimation
----------------
There is deliberately no tokenizer dependency. Tether talks to several
providers whose tokenizers differ, and the wrong tokenizer would give false
precision. :func:`estimate_tokens` uses a conservative characters-per-token
ratio instead, calibrated against observed GenieX ``usage.prompt_tokens``
(2322 chars -> 458 tokens, 14370 -> 2714, 20394 -> 4489: roughly 4.5-5.0
chars/token for English prose). The ratio here is deliberately *lower* than
observed, so the estimate over-counts and the budget errs toward trimming a
little too early rather than overflowing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

# Deliberately below the observed 4.5-5.0 chars/token so estimates run high.
# Over-estimating costs a little usable context; under-estimating costs silent
# eviction, which is the failure this module exists to prevent.
_CHARS_PER_TOKEN = 3.5

# Per-message overhead for role markers and chat-template scaffolding.
_MESSAGE_OVERHEAD_TOKENS = 4

# Fraction of the window left free for the model's reply plus estimation error.
# The reply has to fit in the same window as the prompt.
_DEFAULT_RESERVE_RATIO = 0.35


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for *text* (over-counts by design)."""
    if not text:
        return 0
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def estimate_message_tokens(message: Dict[str, Any]) -> int:
    """Token estimate for one chat message, including role scaffolding."""
    content = message.get("content") or ""
    return estimate_tokens(str(content)) + _MESSAGE_OVERHEAD_TOKENS


def estimate_messages_tokens(messages: Sequence[Dict[str, Any]]) -> int:
    """Token estimate for a whole message list."""
    return sum(estimate_message_tokens(m) for m in messages)


def _is_system(message: Dict[str, Any]) -> bool:
    return message.get("role") == "system"


def fit_to_context(
    messages: List[Dict[str, Any]],
    *,
    context_window: int,
    reserve_ratio: float = _DEFAULT_RESERVE_RATIO,
) -> tuple[List[Dict[str, Any]], int]:
    """Drop oldest non-system messages until the prompt fits the window.

    Returns ``(messages, dropped_count)``.

    System messages are always kept: they carry the calling convention and the
    per-turn tool roster, and they are small. Everything else is dropped oldest
    first, which preserves recency — the part of a conversation a user is
    actually referring to.

    If the system messages alone exceed the budget the list is returned intact.
    Truncating a system prompt would corrupt the calling convention rather than
    merely lose context, and the provider's own eviction is no worse than a
    mangled instruction block.
    """
    if context_window <= 0 or not messages:
        return messages, 0

    budget = int(context_window * (1.0 - reserve_ratio))
    if estimate_messages_tokens(messages) <= budget:
        return messages, 0

    system_tokens = sum(
        estimate_message_tokens(m) for m in messages if _is_system(m)
    )
    if system_tokens >= budget:
        return messages, 0

    # Walk backwards keeping the newest messages that still fit. Positions are
    # tracked by index, not identity: the same dict object can legitimately
    # appear twice in a history (a repeated prompt), and `id()` would then keep
    # or drop both together.
    keep_indices: set[int] = set()
    used = system_tokens
    dropped = 0
    for index in range(len(messages) - 1, -1, -1):
        if _is_system(messages[index]):
            continue  # Already counted; re-inserted in order below.
        cost = estimate_message_tokens(messages[index])
        if used + cost > budget:
            dropped += 1
            continue
        used += cost
        keep_indices.add(index)

    fitted = [
        m for i, m in enumerate(messages) if _is_system(m) or i in keep_indices
    ]
    return fitted, dropped
