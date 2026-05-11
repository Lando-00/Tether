from __future__ import annotations

import pytest

from tether.protocol.intent.regex_classifier import RegexConfirmIntentClassifier

CORPUS = [
    ("yes", True, "Canonical, spec seed"),
    ("Yes", True, "Case-insensitive after normalization"),
    ("YES", True, "Case-insensitive"),
    ("  yes  ", True, "Whitespace stripped"),
    ('"yes"', True, "Wrapping punctuation stripped"),
    ("yes!", True, "Trailing punctuation does not block `\\b`"),
    ("yes.", True, "Same"),
    ("yes please", True, "Affirmative + polite suffix, no deny tokens"),
    ("yes please send", True, "Same-action reinforcement"),
    (
        "please yes",
        False,
        "Not anchored at start; v1 chooses strict-anchor (justification: avoids "
        "accepting `please yes do not send` and similar embedded affirmatives)",
    ),
    (
        "yesterday i said yes",
        False,
        "`^yes\\b` fails on `yesterday` (`\\b` between `s`/`t` is not a "
        "boundary); even if it slipped, deny-list catches nothing — but the "
        "message simply doesn't start with an affirmative",
    ),
    ("yeah", True, "Yes-family"),
    ("yep", True, "Yes-family"),
    ("yup", True, "Yes-family"),
    ("ya", True, "Yes-family"),
    ("sure", True, "Sure-family"),
    ("sure thing", True, "Sure-family"),
    ("ok", True, "OK-family"),
    ("okay", True, "OK-family"),
    ("k", True, "Terse OK"),
    ("kk", True, "Terse OK"),
    ("send", True, "Spec seed"),
    ("send it", True, "Imperative send"),
    ("ship it", True, "Imperative send"),
    ("go", True, "Spec seed (extended)"),
    ("go ahead", True, "Spec seed"),
    ("go for it", True, "Imperative go"),
    ("proceed", True, "Imperative go"),
    ("confirm", True, "Spec seed"),
    ("confirmed", True, "Confirm-family"),
    (
        "confirmation bias",
        False,
        "`\\b` between `m`/`a` fails — does not match `^confirm\\b`",
    ),
    ("correct", True, "Correctness"),
    ("that's right", True, "Correctness"),
    ("looks good", True, "Approval"),
    ("lgtm", True, "Approval"),
    ("approved", True, "Approval"),
    ("affirmative", True, "Military / radio"),
    ("roger that", True, "Military / radio"),
    ("do it", True, "Spec seed"),
    ("👍", True, "Bare emoji"),
    ("✅", True, "Bare emoji"),
    ("👌", True, "Bare emoji"),
    ("sure 👍", True, "Affirmative + emoji, no deny tokens"),
    ("no", False, "Refusal"),
    ("nope", False, "Refusal"),
    ("nah", False, "Refusal"),
    ("not yet", False, "Hesitation (`not` deny token)"),
    ("wait", False, "Hesitation"),
    ("hold on", False, "Hesitation"),
    ("stop", False, "Hesitation"),
    ("cancel", False, "Refusal"),
    ("nvm", False, "Refusal"),
    ("never mind", False, "Refusal"),
    ("actually", False, "`actually` deny token, no affirmative head anyway"),
    ("i guess", False, "Not in affirmative list"),
    ("whatever", False, "Not in affirmative list"),
    ("if you say so", False, "Not in affirmative list"),
    (
        "yes, but change the wording",
        False,
        "`but` + `change` deny tokens — KEY false-positive defense",
    ),
    ("yes, in Spanish though", False, "`in spanish` deny token"),
    ("yes please rephrase first", False, "`rephrase` + `first` deny tokens"),
    (
        "yes I want you to NOT send",
        False,
        "`not` deny token — KEY false-positive defense",
    ),
    ("yes please cancel", False, "`cancel` deny token"),
    ("yes, abort that", False, "destructive verb after yes"),
    ("yes please undo", False, "destructive verb after yes"),
    ("yes - delete that", False, "destructive verb after yes"),
    ("yes that was wrong", False, "incorrect after yes"),
    ("ok, scrub it", False, "scrub deny token"),
    ("confirm, but discard the body", False, "discard deny token"),
    ("yep, ignore that draft", False, "ignore deny token"),
    ("yes but no", False, "`but` + `no` deny tokens"),
    ("confirm — no hold on", False, "`no` + `hold on` deny tokens"),
    ("send to John instead", False, "`instead` deny token"),
    ("Hello John, lunch tomorrow?", False, "No affirmative head; safe hold"),
    ("tether, yes", True, "Salutation stripped, then `yes` matches"),
    ("@tether send it", True, "Salutation stripped"),
    ("ai: confirm", True, "Salutation stripped"),
    (
        "Yes. Now send the second one too",
        True,
        "Affirmative head, no deny tokens — accepted (scope-creep tolerated; §5)",
    ),
    ("Yes! Also change the time", False, "`change` deny token"),
    ("sí", False, "Non-English; v1 English-only"),
    ("oui", False, "Non-English"),
    ("はい", False, "Non-English"),
    ("", False, "Guard clause"),
    ("   ", False, "Guard clause"),
    (
        "confirm 12345678901234",
        True,
        "Adversarial digit-run heuristic logs warning but does not block; tool "
        "layer validates draft_id",
    ),
]


@pytest.mark.parametrize(("message", "expected", "description"), CORPUS)
def test_adr_0019_appendix_b_corpus(
    message: str,
    expected: bool,
    description: str,
) -> None:
    classifier = RegexConfirmIntentClassifier()

    assert classifier.classify(message) is expected, description


def test_empty_string_returns_false() -> None:
    assert RegexConfirmIntentClassifier().classify("") is False


def test_max_length_exceeded_returns_false() -> None:
    assert RegexConfirmIntentClassifier().classify("yes" * 1000) is False


def test_unicode_emoji_confirmation_each_variant() -> None:
    classifier = RegexConfirmIntentClassifier()

    for emoji in ("👍", "✅", "👌", "🆗", "☑️", "✔️"):
        assert classifier.classify(emoji) is True


def test_yesterday_does_not_match() -> None:
    assert RegexConfirmIntentClassifier().classify("yesterday") is False


def test_classifier_is_pure() -> None:
    classifier = RegexConfirmIntentClassifier()

    assert [classifier.classify("yes please") for _ in range(5)] == [True] * 5
    assert [classifier.classify("yes but cancel") for _ in range(5)] == [False] * 5
