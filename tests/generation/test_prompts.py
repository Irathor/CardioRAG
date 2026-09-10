from cardiorag.generation.prompts import SYSTEM_PROMPT, build_user_prompt


def test_system_prompt_states_evidence_only_rule():
    assert "ONLY" in SYSTEM_PROMPT
    assert "[Source N]" in SYSTEM_PROMPT


def test_system_prompt_forbids_inventing_citations():
    assert "invent" in SYSTEM_PROMPT.lower()


def test_system_prompt_requires_explicit_refusal_on_insufficient_evidence():
    assert "say so explicitly" in SYSTEM_PROMPT.lower()


def test_system_prompt_includes_non_diagnostic_disclaimer():
    assert "not a medical diagnostic system" in SYSTEM_PROMPT.lower()


def test_build_user_prompt_includes_question_and_context():
    prompt = build_user_prompt("What is LVEF?", "[Source 1]\nText: ...")

    assert "What is LVEF?" in prompt
    assert "[Source 1]" in prompt


def test_build_user_prompt_handles_empty_context():
    prompt = build_user_prompt("Some question", "")

    assert "no sources were retrieved" in prompt
