"""Detection of collapsed (degenerate) model output.

Why this exists
---------------
The GenieX/Hexagon-NPU stack intermittently produces a generation that has
collapsed: the model stops producing language and instead locks onto a short
sequence, emitting it over and over — often cycling between a few such
sequences::

    ... an alias like `cddev终端全端looplooploop…缆缆缆港口港口港口…衣衣衣衣…注意注意…

Measured on ``Qwen3-8B:Q4_0`` by replaying one fixed conversation 69 times
across three request variants, this occurred in roughly 8% of turns. It is
**not** caused by the request: varying ``temperature`` / ``top_p`` / ``top_k``
/ ``min_p`` / ``presence_penalty``, or toggling ``enable_think``, does not move
the rate (2/23 vs 1/23 across sampling variants — indistinguishable), and it
reproduces on prompts of only a few hundred tokens, so it is not context
overflow either. It appears to be nondeterministic corruption inside the
inference stack, which Tether cannot prevent.

What Tether *can* do is stop streaming the garbage. Left alone the collapse
runs until the model emits a stop token, which in practice means the operator
watches nonsense scroll past and interrupts the client by hand.

Design constraints
------------------
The detector must never fire on legitimate output, so it deliberately does
**not** look at which script or language the text is in — a user chatting in
Chinese, Korean or Russian is normal, and an earlier draft that keyed on
non-Latin characters would have punished them for it. It keys only on
*pathological periodicity*: a run of text that is one short unit repeated
end to end.

Two thresholds, because not all repetition is equal:

* A unit containing a letter or digit (``loop``, ``港口``, ``衣``) repeated for
  dozens of characters is not something language does. Threshold is low.
* A unit of pure punctuation or whitespace is a *separator* — ``-----``,
  ``=====``, a blank-line run, an indented block. Those are legitimate, so the
  threshold is much higher and only a true flood trips it.

The run lengths that matter were taken from the real failure: its individual
repeated runs were 20-80 characters before it switched to a different unit, so
a detector calibrated on longer runs would have missed the actual bug.
"""

from __future__ import annotations

from typing import NamedTuple

# Minimum repeated run for a unit that contains a letter or digit. The observed
# collapse held single units for 20-80 characters, so this has to sit well below
# the "hundreds of characters" a naive design would pick.
_MIN_RUN_WORDLIKE = 64

# Minimum repeated run for a unit made only of punctuation/whitespace. Markdown
# rules, ASCII separators and indented code blocks are legitimately periodic, so
# only a flood well past any sane separator width counts.
_MIN_RUN_SEPARATOR = 240

# Longest repeating unit considered. Collapses lock onto a token or two; longer
# periods risk matching legitimately repetitive structured output.
_MAX_PERIOD_CHARS = 40

# Tail retained for analysis. Must exceed the largest threshold so a qualifying
# run is always fully visible.
_TAIL_CHARS = _MIN_RUN_SEPARATOR + _MAX_PERIOD_CHARS


class PeriodicSuffix(NamedTuple):
    """A maximal run of one repeated unit at the end of some text."""

    period: int
    run_chars: int
    unit: str

    @property
    def is_wordlike(self) -> bool:
        """True when the repeated unit contains a letter or digit.

        Drives threshold selection: word-like repetition is always
        pathological, separator repetition usually is not.
        """
        return any(ch.isalnum() for ch in self.unit)

    @property
    def exceeds_threshold(self) -> bool:
        limit = _MIN_RUN_WORDLIKE if self.is_wordlike else _MIN_RUN_SEPARATOR
        return self.run_chars >= limit


def find_periodic_suffix(text: str) -> PeriodicSuffix | None:
    """Return the longest single-unit repetition ending *text*, if any.

    Ties are resolved toward the smallest period, so ``"abab..."`` reports
    period 2 rather than 4.
    """
    if not text:
        return None

    best: PeriodicSuffix | None = None
    for period in range(1, min(_MAX_PERIOD_CHARS, len(text)) + 1):
        unit = text[-period:]
        run = period
        # Walk backwards while the preceding `period` characters repeat the unit.
        while run + period <= len(text) and text[-(run + period) : -run] == unit:
            run += period
        if run <= period:
            continue  # Unit appears once; not a repetition.
        if best is None or run > best.run_chars:
            best = PeriodicSuffix(period=period, run_chars=run, unit=unit)
    return best


class DegenerateOutputGuard:
    """Incrementally watch a token stream for collapse.

    Feed each content delta to :meth:`observe`. It returns a description of the
    collapse once one is detected, and ``None`` otherwise. Only a bounded tail
    is inspected, so memory does not grow with the response, and the scan is
    bounded at roughly ``_MAX_PERIOD_CHARS * _TAIL_CHARS`` character
    comparisons — cheap enough to run on every delta, which keeps detection
    immediate once the threshold is crossed.
    """

    def __init__(self) -> None:
        self._tail = ""
        self._chars_seen = 0

    @property
    def chars_seen(self) -> int:
        return self._chars_seen

    def observe(self, delta: str) -> str | None:
        """Record a content delta; return a reason string once collapsed."""
        if not delta:
            return None

        self._chars_seen += len(delta)
        self._tail = (self._tail + delta)[-_TAIL_CHARS:]

        found = find_periodic_suffix(self._tail)
        if found is None or not found.exceeds_threshold:
            return None

        return (
            f"output collapsed into a repeating sequence {found.unit!r} "
            f"({found.run_chars} characters) after {self._chars_seen} "
            "characters of output"
        )
