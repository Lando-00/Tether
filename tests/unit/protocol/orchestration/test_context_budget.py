"""Unit tests for the context-window budget.

The numbers here come from measurement against GenieX (`--nctx 4096`,
`Qwen3-8B:Q4_0`), using a sentinel instruction planted in the system prompt to
detect front-eviction:

    prompt_tokens=277   -> system prompt honoured
    prompt_tokens=2329  -> system prompt honoured
    prompt_tokens=4489  -> system prompt LOST
    prompt_tokens=6757  -> system prompt LOST

The request still returned HTTP 200 and fluent text in the failing cases, which
is why nothing upstream noticed.
"""

from __future__ import annotations

from tether.protocol.orchestration.context_budget import (
    estimate_messages_tokens,
    estimate_tokens,
    fit_to_context,
)


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _filler(chars: int) -> str:
    return "word " * (chars // 5)


class TestEstimateTokens:
    def test_empty_text(self):
        assert estimate_tokens("") == 0

    def test_estimate_grows_with_length(self):
        assert estimate_tokens("a" * 400) > estimate_tokens("a" * 100)

    def test_estimate_is_conservative_versus_observed(self):
        """Must over-count: under-counting causes the silent eviction.

        Observed on GenieX: 2322 chars -> 458 tokens, 14370 -> 2714,
        20394 -> 4489 (roughly 4.5-5.0 chars/token).
        """
        for chars, actual in ((2322, 458), (14370, 2714), (20394, 4489)):
            assert estimate_tokens("x" * chars) >= actual

    def test_message_list_includes_role_overhead(self):
        one = estimate_messages_tokens([_msg("user", "hi")])
        two = estimate_messages_tokens([_msg("user", "hi"), _msg("user", "hi")])
        assert two > one * 1.5


class TestFitToContext:
    def test_short_conversation_is_untouched(self):
        messages = [
            _msg("system", "You are helpful."),
            _msg("user", "hello"),
            _msg("assistant", "hi there"),
        ]
        fitted, dropped = fit_to_context(messages, context_window=4096)
        assert fitted == messages
        assert dropped == 0

    def test_long_conversation_is_trimmed(self):
        messages = [_msg("system", "You are helpful.")]
        for i in range(40):
            messages.append(_msg("user", _filler(1000) + f" turn {i}"))
            messages.append(_msg("assistant", _filler(1000) + f" reply {i}"))

        fitted, dropped = fit_to_context(messages, context_window=4096)
        assert dropped > 0
        assert len(fitted) < len(messages)

    def test_system_prompt_always_survives(self):
        """The regression that motivated this module.

        The system prompt carries the ``<<function_call>>`` convention for
        marker-only providers, so losing it silently disables tool calling.
        """
        system = _msg("system", "CALLING CONVENTION")
        messages = [system]
        for i in range(60):
            messages.append(_msg("user", _filler(2000) + f" {i}"))

        fitted, _ = fit_to_context(messages, context_window=4096)
        assert system in fitted
        assert fitted[0]["role"] == "system"

    def test_tool_roster_system_message_survives(self):
        """The per-turn roster is a system message appended at the end."""
        roster = _msg("system", "Tools available to you right now: time, weather")
        messages = [_msg("system", "You are helpful.")]
        for i in range(60):
            messages.append(_msg("user", _filler(2000) + f" {i}"))
        messages.append(roster)

        fitted, _ = fit_to_context(messages, context_window=4096)
        assert roster in fitted

    def test_most_recent_turn_is_kept(self):
        """Recency matters most: the user's actual question must survive."""
        messages = [_msg("system", "You are helpful.")]
        for i in range(60):
            messages.append(_msg("user", _filler(2000) + f" {i}"))
        newest = _msg("user", "What is 2+2?")
        messages.append(newest)

        fitted, _ = fit_to_context(messages, context_window=4096)
        assert fitted[-1] == newest

    def test_chronological_order_is_preserved(self):
        messages = [_msg("system", "sys")]
        for i in range(40):
            messages.append(_msg("user", _filler(1500) + f" {i}"))

        fitted, _ = fit_to_context(messages, context_window=4096)
        kept = [m for m in fitted if m["role"] == "user"]
        indices = [int(m["content"].rsplit(" ", 1)[1]) for m in kept]
        assert indices == sorted(indices)

    def test_result_fits_the_budget(self):
        messages = [_msg("system", "sys")]
        for i in range(80):
            messages.append(_msg("user", _filler(1000) + f" {i}"))

        fitted, _ = fit_to_context(messages, context_window=4096)
        # Must leave room for the reply, which shares the same window.
        assert estimate_messages_tokens(fitted) <= 4096

    def test_oversized_system_prompt_is_left_intact(self):
        """Truncating a system prompt would corrupt the calling convention."""
        messages = [
            _msg("system", _filler(40000)),
            _msg("user", "hello"),
        ]
        fitted, dropped = fit_to_context(messages, context_window=4096)
        assert fitted == messages
        assert dropped == 0

    def test_empty_input(self):
        assert fit_to_context([], context_window=4096) == ([], 0)

    def test_unknown_window_is_a_no_op(self):
        messages = [_msg("user", _filler(50000))]
        fitted, dropped = fit_to_context(messages, context_window=0)
        assert fitted == messages
        assert dropped == 0

    def test_duplicate_messages_are_handled_independently(self):
        """Identical content must not be kept or dropped as a pair.

        An identity-based implementation would treat equal dicts as one.
        """
        repeated = _msg("user", _filler(3000))
        messages = [_msg("system", "sys")] + [repeated] * 20
        fitted, dropped = fit_to_context(messages, context_window=4096)
        kept = [m for m in fitted if m["role"] == "user"]
        assert 0 < len(kept) < 20
        assert dropped == 20 - len(kept)

    def test_larger_window_keeps_more(self):
        messages = [_msg("system", "sys")]
        for i in range(60):
            messages.append(_msg("user", _filler(1000) + f" {i}"))

        small, _ = fit_to_context(messages, context_window=4096)
        large, _ = fit_to_context(messages, context_window=16384)
        assert len(large) > len(small)
