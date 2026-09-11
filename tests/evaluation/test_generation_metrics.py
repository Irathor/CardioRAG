from types import SimpleNamespace

import numpy as np
import pytest

from cardiorag.evaluation.generation_metrics import (
    evaluate_answer_relevance,
    evaluate_context_relevance,
    evaluate_faithfulness,
    extract_statements,
    generate_hypothetical_questions,
    looks_like_refusal,
    verify_statements,
)


class _ScriptedProvider:
    """Returns canned responses in order, one per call - lets a test control
    a multi-step LLM-as-judge flow (e.g. extract-then-verify) deterministically."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self._responses.pop(0)


class _FakeEmbedder:
    """Deterministic stand-in for Embedder: maps known texts to fixed unit vectors."""

    def __init__(self, vector_for_text: dict[str, list[float]]):
        self._vector_for_text = vector_for_text

    def embed(self, texts: list[str]):
        vectors = np.array([self._vector_for_text[t] for t in texts], dtype=np.float32)
        return SimpleNamespace(vectors=vectors)


# --- extract_statements ---


def test_extract_statements_parses_json_array():
    provider = _ScriptedProvider(['["Statement one.", "Statement two."]'])

    result = extract_statements(provider, "a question", "an answer")

    assert result == ["Statement one.", "Statement two."]


def test_extract_statements_tolerates_surrounding_text():
    provider = _ScriptedProvider(['Here they are:\n["a fact", "another fact"]\nDone.'])

    result = extract_statements(provider, "q", "a")

    assert result == ["a fact", "another fact"]


def test_extract_statements_returns_empty_for_blank_answer():
    provider = _ScriptedProvider([])  # must not be called at all

    result = extract_statements(provider, "q", "   ")

    assert result == []
    assert provider.calls == []


# --- statement_is_supported ---


def test_verify_statements_parses_boolean_array():
    provider = _ScriptedProvider(["[true, false, true]"])

    result = verify_statements(provider, "context", ["s1", "s2", "s3"])

    assert result == [True, False, True]


def test_verify_statements_empty_for_no_statements():
    provider = _ScriptedProvider([])  # must not be called

    assert verify_statements(provider, "context", []) == []
    assert provider.calls == []


def test_verify_statements_pads_short_response_as_unsupported():
    provider = _ScriptedProvider(["[true]"])  # model only returned one flag for three statements

    result = verify_statements(provider, "context", ["s1", "s2", "s3"])

    assert result == [True, False, False]


def test_verify_statements_falls_back_to_all_unsupported_on_unparseable_response():
    provider = _ScriptedProvider(["I refuse to answer in JSON."])

    result = verify_statements(provider, "context", ["s1", "s2"])

    assert result == [False, False]


# --- evaluate_faithfulness ---


def test_evaluate_faithfulness_computes_correct_score():
    provider = _ScriptedProvider(['["s1", "s2", "s3"]', "[true, false, true]"])

    result = evaluate_faithfulness(provider, "q", "an answer with three claims", "context")

    assert result.score == pytest.approx(2 / 3)
    assert result.statements == ["s1", "s2", "s3"]
    assert result.unsupported_statements == ["s2"]


def test_evaluate_faithfulness_vacuously_one_when_no_statements():
    provider = _ScriptedProvider(["[]"])

    result = evaluate_faithfulness(provider, "q", "an answer", "context")

    assert result.score == 1.0
    assert result.unsupported_statements == []


# --- generate_hypothetical_questions / evaluate_answer_relevance ---


def test_generate_hypothetical_questions_parses_array():
    provider = _ScriptedProvider(['["Q1?", "Q2?", "Q3?"]'])

    result = generate_hypothetical_questions(provider, "an answer", n=3)

    assert result == ["Q1?", "Q2?", "Q3?"]


def test_evaluate_answer_relevance_uses_cosine_similarity():
    provider = _ScriptedProvider(['["hyp question"]'])
    embedder = _FakeEmbedder({"real question": [1.0, 0.0], "hyp question": [0.0, 1.0]})

    score = evaluate_answer_relevance(provider, embedder, "real question", "an answer", n=1)

    assert score == pytest.approx(0.0, abs=1e-6)  # orthogonal vectors -> cosine similarity 0


def test_evaluate_answer_relevance_perfect_when_questions_match():
    provider = _ScriptedProvider(['["real question"]'])
    embedder = _FakeEmbedder({"real question": [1.0, 0.0]})

    score = evaluate_answer_relevance(provider, embedder, "real question", "an answer", n=1)

    assert score == pytest.approx(1.0, abs=1e-6)


def test_evaluate_answer_relevance_zero_when_no_hypothetical_questions_generated():
    provider = _ScriptedProvider(["not valid json"])
    embedder = _FakeEmbedder({})  # must not be called

    score = evaluate_answer_relevance(provider, embedder, "q", "an answer", n=1)

    assert score == 0.0


# --- evaluate_context_relevance ---


def test_evaluate_context_relevance_parses_score():
    provider = _ScriptedProvider(["0.75"])
    assert evaluate_context_relevance(provider, "q", "some context") == pytest.approx(0.75)


def test_evaluate_context_relevance_clamps_values_above_one():
    provider = _ScriptedProvider(["1.5"])
    assert evaluate_context_relevance(provider, "q", "some context") == 1.0


def test_evaluate_context_relevance_returns_zero_for_unparseable_response():
    provider = _ScriptedProvider(["I cannot rate this."])
    assert evaluate_context_relevance(provider, "q", "some context") == 0.0


def test_evaluate_context_relevance_zero_for_empty_context():
    provider = _ScriptedProvider([])  # must not be called
    assert evaluate_context_relevance(provider, "q", "") == 0.0
    assert provider.calls == []


# --- looks_like_refusal ---


@pytest.mark.parametrize(
    "answer",
    [
        "The available corpus does not contain enough evidence to answer this question.",
        "There is insufficient evidence in the provided sources.",
        "I cannot answer this based on the given context.",
        "This topic is not covered by the sources provided.",
    ],
)
def test_looks_like_refusal_detects_known_phrases(answer):
    assert looks_like_refusal(answer) is True


def test_looks_like_refusal_false_for_a_normal_answer():
    assert looks_like_refusal("LVEF measures the heart's pumping efficiency [Source 1].") is False


def test_looks_like_refusal_detects_real_world_partial_refusal_phrasing():
    # Real text captured during Phase 13 manual inspection (model: allam-2-7b),
    # which substantively declines to answer but used phrasing missing from
    # the original keyword list entirely.
    answer = (
        "Based on the provided sources, there is no direct mention of an FDA-approved "
        "AI algorithm for automated coronary artery calcium scoring... The sources do "
        "not mention any FDA-approved AI algorithm for automated coronary artery "
        "calcium scoring in CMR."
    )
    assert looks_like_refusal(answer) is True
