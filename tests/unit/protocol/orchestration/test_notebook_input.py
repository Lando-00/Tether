from __future__ import annotations

import pytest

from tether.protocol.orchestration.notebook_input import (
    has_entity_drift,
    prepare_research_input,
)


def test_short_correction_reconstructs_recent_user_question() -> None:
    prepared = prepare_research_input(
        "Ireland*",
        [{"role": "user", "content": "Tell me about Irelend's capital."}],
    )

    assert prepared.original_prompt == "Ireland*"
    assert prepared.effective_question == "Tell me about Ireland's capital."
    assert prepared.clarification is None


def test_case_only_correction_replaces_a_unique_prior_span() -> None:
    prepared = prepare_research_input(
        "Ireland*",
        [{"role": "user", "content": "Tell me about IReland."}],
    )

    assert prepared.effective_question == "Tell me about Ireland."
    assert prepared.clarification is None


def test_correction_without_history_requests_clarification() -> None:
    prepared = prepare_research_input("Ireland*", [])

    assert prepared.clarification is not None
    assert prepared.clarification.reason == "ambiguous_correction"


def test_ambiguous_correction_requests_clarification() -> None:
    prepared = prepare_research_input(
        "Ireland*",
        [{"role": "user", "content": "Compare Irelend and Irelend."}],
    )

    assert prepared.clarification is not None
    assert prepared.clarification.candidates == ("Irelend",)


def test_correction_replays_prior_correction_before_current_one() -> None:
    prepared = prepare_research_input(
        "Ireland*",
        [
            {"role": "user", "content": "Tell me about Irelamd."},
            {"role": "user", "content": "Irelend*"},
        ],
    )

    assert prepared.effective_question == "Tell me about Ireland."


def test_synthetic_tool_result_is_not_correction_context() -> None:
    prepared = prepare_research_input(
        "Ireland*",
        [{"role": "user", "content": "<<tool_result>> Tell me about Irelend."}],
    )

    assert prepared.clarification is not None


def test_entity_drift_flags_only_near_spelling_substitution() -> None:
    assert has_entity_drift("Iceland population", "Tell me about Ireland")
    assert not has_entity_drift("Ireland population", "Tell me about Ireland")
    assert not has_entity_drift(
        "Catherine Connolly date of birth", "Tell me about Ireland"
    )


def test_over_limit_prompt_remains_exact_transcript_truth() -> None:
    prompt = "x" * 2_049
    prepared = prepare_research_input(prompt, [])

    assert prepared.original_prompt == prompt
    assert prepared.effective_question == ""
    assert prepared.clarification is not None
    assert prepared.clarification.reason == "unsearchable_input"


def test_mixed_arithmetic_is_local_and_leaves_research_question() -> None:
    prepared = prepare_research_input("What is 2.5 x 4? Tell me about Ireland.", [])

    assert prepared.effective_question == "Tell me about Ireland."
    assert len(prepared.local_facts) == 1
    fact = prepared.local_facts[0]
    assert fact.text == "2.5 x 4 = 10"
    assert fact.source_query == "2.5 x 4"
    assert fact.source_kind == "local_deterministic"


def test_pure_arithmetic_has_no_residual_question() -> None:
    prepared = prepare_research_input("calculate -3 ÷ 2", [])

    assert prepared.effective_question == ""
    assert prepared.local_facts[0].text == "-3 ÷ 2 = -1.5"


def test_unsafe_arithmetic_is_preserved_as_research_input() -> None:
    for expression in ("2 + 2%", "2e3 + 1", "sum(2, 2)", "2(3)", "2 / 0"):
        prepared = prepare_research_input(expression, [])
        assert prepared.local_facts == ()
        assert prepared.effective_question == expression


# ---------------------------------------------------------------------------
# Phase 9.8 W4 reconcile — review findings
# ---------------------------------------------------------------------------


def test_conjunctions_inside_names_survive_arithmetic_removal() -> None:
    prepared = prepare_research_input("What is 2 + 2? Tell me about Trinidad and Tobago.", [])

    assert prepared.effective_question == "Tell me about Trinidad and Tobago."
    assert prepared.local_facts[0].text == "2 + 2 = 4"


def test_full_question_is_preserved_for_synthesis() -> None:
    prepared = prepare_research_input("Who is the president of Ireland? What is 25 + 50?", [])

    assert prepared.effective_question == "Who is the president of Ireland?"
    assert prepared.resolved_question == "Who is the president of Ireland? What is 25 + 50?"


def test_pure_arithmetic_still_exposes_the_original_question() -> None:
    prepared = prepare_research_input("what is 25 + 50", [])

    assert prepared.effective_question == ""
    assert prepared.resolved_question == "what is 25 + 50"


def test_uppercase_multiplication_is_not_division() -> None:
    prepared = prepare_research_input("What is 6 X 3?", [])

    assert prepared.local_facts[0].text == "6 X 3 = 18"


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("what is 6 x 3", "6 x 3 = 18"), ("what is 6 ÷ 3", "6 ÷ 3 = 2"), ("what is 6 / 3", "6 / 3 = 2")],
)
def test_operator_dispatch_is_explicit(expression: str, expected: str) -> None:
    prepared = prepare_research_input(expression, [])

    assert prepared.local_facts[0].text == expected


def test_drift_guard_allows_two_similar_entities_named_by_the_user() -> None:
    assert not has_entity_drift("Compare Ireland and Iceland", "Compare Ireland and Iceland")
    assert not has_entity_drift("Ireland and Iceland populations", "Compare Ireland and Iceland")
    # A genuine substitution is still caught.
    assert has_entity_drift("Iceland population", "Tell me about Ireland")


def test_ambiguous_correction_candidates_are_bounded() -> None:
    token = "Irelend" + "z" * 400
    prepared = prepare_research_input(
        "Ireland*",
        [{"role": "user", "content": f"Compare {token} and {token}."}],
    )

    assert prepared.clarification is not None
    assert all(len(item) <= 256 for item in prepared.clarification.candidates)
