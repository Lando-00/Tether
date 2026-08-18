"""Unit tests for collapsed-output detection.

The positive cases are modelled on real degenerate GenieX generations —
including the run lengths actually observed, which are short (20-80 chars per
unit before it switches). The negative cases defend the property that matters
most: the guard must never fire on legitimate output, in any language.
"""

from __future__ import annotations

from tether.providers.geniex.degeneracy import (
    DegenerateOutputGuard,
    find_periodic_suffix,
)


def _feed(text: str, chunk: int = 5) -> str | None:
    """Stream *text* through a guard in small deltas, as SSE would."""
    guard = DegenerateOutputGuard()
    for i in range(0, len(text), chunk):
        reason = guard.observe(text[i : i + chunk])
        if reason is not None:
            return reason
    return None


class TestFindPeriodicSuffix:
    def test_empty_text(self):
        assert find_periodic_suffix("") is None

    def test_non_repeating_text(self):
        assert find_periodic_suffix("the quick brown fox") is None

    def test_single_character_run(self):
        found = find_periodic_suffix("\n" * 40)
        assert found is not None
        assert found.period == 1
        assert found.run_chars == 40

    def test_prefers_smallest_period_on_tie(self):
        found = find_periodic_suffix("ab" * 40)
        assert found is not None
        assert found.period == 2
        assert found.run_chars == 80

    def test_measures_only_the_suffix_run(self):
        found = find_periodic_suffix("Here is the answer. " + "loop" * 10)
        assert found is not None
        assert found.unit == "loop"
        assert found.run_chars == 40

    def test_wordlike_classification(self):
        wordy = find_periodic_suffix("loop" * 10)
        dashes = find_periodic_suffix("-" * 40)
        assert wordy is not None and wordy.is_wordlike
        assert dashes is not None and not dashes.is_wordlike


class TestDegenerateOutputGuard:
    def test_real_collapse_is_detected(self):
        """Shaped after the reported failure: prose, then a repetition lock.

        Run lengths here match what was actually observed — each unit holds for
        only a few dozen characters before the model switches to another.
        """
        text = (
            "Yes, using an alias like `cddev"
            + "\u7ec8\u7aef\u5168\u7aef"
            + "loop" * 20
            + "\u7f06" * 3
            + "\u6e2f\u53e3" * 11
            + "\u8863" * 70
        )
        reason = _feed(text)
        assert reason is not None
        assert "collapsed" in reason

    def test_single_repeated_word_trips(self):
        assert _feed("Sure! " + "loop" * 20) is not None

    def test_repeated_cjk_character_trips(self):
        assert _feed("\u8863" * 70) is not None

    def test_newline_flood_trips_only_when_extreme(self):
        """Separators need a much longer run than word-like units."""
        assert _feed("Here is the answer:" + "\n" * 100) is None
        assert _feed("Here is the answer:" + "\n" * 400) is not None

    def test_normal_reply_does_not_trip(self):
        reply = (
            "Certainly! Using an alias like `cddev` can be more user-friendly "
            "and concise. Here's how you can define it:\n\n"
            "```powershell\n"
            "function cddev { Set-Location -Path 'C:\\Dev' }\n"
            "```\n\n"
            "Add that to your PowerShell profile so it is available in every "
            "new session. You can find the profile path with `$PROFILE`.\n"
        )
        assert _feed(reply) is None

    def test_non_latin_reply_does_not_trip(self):
        """A user chatting in another language is normal, not a collapse.

        The guard must key on periodicity, never on script.
        """
        chinese = (
            "\u4f60\u597d\uff01\u4f7f\u7528 cddev \u8fd9\u6837\u7684\u522b\u540d"
            "\u786e\u5b9e\u6bd4 cd-dev \u66f4\u65b9\u4fbf\uff0c\u56e0\u4e3a"
            "\u8fde\u5b57\u7b26\u9700\u8981\u6309\u4f4f\u4fee\u9970\u952e\u3002"
            "\u4f60\u53ef\u4ee5\u628a\u5b83\u5199\u5165\u914d\u7f6e\u6587\u4ef6"
            "\uff0c\u8fd9\u6837\u6bcf\u6b21\u542f\u52a8\u90fd\u4f1a\u52a0\u8f7d\u3002"
        ) * 4
        assert _feed(chinese) is None

    def test_markdown_rule_does_not_trip(self):
        """`-----` separators are legitimate periodic output."""
        assert _feed("Summary\n" + "-" * 80 + "\nDone.\n") is None

    def test_indented_code_block_does_not_trip(self):
        block = "".join(" " * 8 + f"call_step_{i}()\n" for i in range(30))
        assert _feed(block) is None

    def test_repetitive_but_legitimate_structure_does_not_trip(self):
        """Structured output repeats a shape, not an identical short unit."""
        table = "".join(f"| item-{i:03d} | value-{i:03d} | ok |\n" for i in range(60))
        assert _feed(table) is None

    def test_guard_reports_characters_seen(self):
        guard = DegenerateOutputGuard()
        guard.observe("hello ")
        guard.observe("world")
        assert guard.chars_seen == 11

    def test_empty_deltas_are_ignored(self):
        guard = DegenerateOutputGuard()
        assert guard.observe("") is None
        assert guard.chars_seen == 0
