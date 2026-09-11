"""LLM-based generation quality metrics.

Inspired directly by the Ragas paper - which happens to already be in this
project's own corpus (doc7) - but implemented from scratch rather than
depending on the `ragas` library: the goal is a fully visible, testable
evaluation core, consistent with this project's principle of not hiding
the pipeline behind a framework. Every metric below asks the SAME kind of
LLM used for generation to act as judge, which is the standard
reference-free approach when there's no human-annotated gold answer to
compare against (only reference key-facts, in our evaluation dataset).
"""

import json
import logging
import re
from dataclasses import dataclass

import numpy as np

from cardiorag.generation.providers import LLMProvider

logger = logging.getLogger(__name__)


def _extract_json_array(text: str) -> list[str]:
    """Best-effort extraction of a JSON string array from an LLM response,
    tolerating minor formatting noise (a code fence, a leading sentence)."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed if isinstance(item, str) and item.strip()]


# --- Faithfulness: are the answer's claims actually supported by the context? ---

_STATEMENT_EXTRACTION_PROMPT = """Given a question and an answer, break the answer down into a list of \
distinct factual statements it makes. Each statement should be short and self-contained (understandable \
without the rest of the answer).

Question: {question}
Answer: {answer}

Return ONLY a JSON array of strings, one per statement. No other text, no code fences."""

_BATCH_VERIFICATION_PROMPT = """Given a context and a numbered list of statements, judge for EACH \
statement whether it can be directly inferred from the context alone, without outside knowledge.

Context:
{context}

Statements:
{numbered_statements}

Return ONLY a JSON array of {n} booleans (true if supported by the context, false if not), in the same \
order as the statements. No other text, no code fences."""


def extract_statements(provider: LLMProvider, question: str, answer: str) -> list[str]:
    if not answer.strip():
        return []
    response = provider.generate(
        "You are a precise fact-extraction assistant.",
        _STATEMENT_EXTRACTION_PROMPT.format(question=question, answer=answer),
    )
    return _extract_json_array(response)


def verify_statements(provider: LLMProvider, context: str, statements: list[str]) -> list[bool]:
    """Judge every statement against the context in ONE call rather than one
    call per statement - a single question can otherwise repeat a large
    context block once per statement, which is what actually exhausted a
    free-tier daily token budget after only 13 questions in practice."""
    if not statements:
        return []
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
    response = provider.generate(
        "You are a strict fact-checking assistant.",
        _BATCH_VERIFICATION_PROMPT.format(
            context=context, numbered_statements=numbered, n=len(statements)
        ),
    )
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if not match:
        logger.warning("Could not parse verification response, treating all as unsupported: %r", response)
        return [False] * len(statements)
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [False] * len(statements)

    flags = [bool(v) for v in parsed][: len(statements)]
    flags += [False] * (len(statements) - len(flags))  # defensively pad a short/malformed response
    return flags


@dataclass(frozen=True)
class FaithfulnessResult:
    score: float  # (statements - unsupported) / statements; 1.0 vacuously if there were no statements
    statements: list[str]
    unsupported_statements: list[str]


def evaluate_faithfulness(
    provider: LLMProvider, question: str, answer: str, context: str
) -> FaithfulnessResult:
    statements = extract_statements(provider, question, answer)
    if not statements:
        return FaithfulnessResult(score=1.0, statements=[], unsupported_statements=[])

    supported_flags = verify_statements(provider, context, statements)
    unsupported = [s for s, supported in zip(statements, supported_flags) if not supported]
    score = (len(statements) - len(unsupported)) / len(statements)
    return FaithfulnessResult(score=score, statements=statements, unsupported_statements=unsupported)


# --- Answer relevance: does the answer actually address the question asked? ---

_HYPOTHETICAL_QUESTIONS_PROMPT = """Given the following answer, generate {n} different questions that this \
answer would be a good, direct response to.

Answer: {answer}

Return ONLY a JSON array of {n} question strings. No other text, no code fences."""


def generate_hypothetical_questions(provider: LLMProvider, answer: str, n: int = 3) -> list[str]:
    if not answer.strip():
        return []
    response = provider.generate(
        "You generate questions that a given answer would appropriately respond to.",
        _HYPOTHETICAL_QUESTIONS_PROMPT.format(n=n, answer=answer),
    )
    return _extract_json_array(response)


def evaluate_answer_relevance(provider: LLMProvider, embedder, question: str, answer: str, n: int = 3) -> float:
    """Cosine similarity between the real question and questions an LLM
    guesses the answer was responding to. If the answer stayed on-topic,
    a reverse-engineered question should closely resemble the real one;
    a rambling or off-topic answer produces mismatched hypothetical
    questions and a lower score."""
    hypothetical = generate_hypothetical_questions(provider, answer, n=n)
    if not hypothetical:
        return 0.0
    question_vec = embedder.embed([question]).vectors[0]
    hyp_vecs = embedder.embed(hypothetical).vectors
    similarities = hyp_vecs @ question_vec  # unit-normalized vectors -> dot product == cosine similarity
    return float(np.mean(similarities))


# --- Context relevance: is the retrieved context focused, or mostly noise? ---

_CONTEXT_RELEVANCE_PROMPT = """Given a question and a block of retrieved context, rate what fraction of \
the context is actually relevant to answering the question, from 0.0 (none of it is relevant) to 1.0 \
(all of it is relevant).

Question: {question}

Context:
{context}

Respond with ONLY a number between 0.0 and 1.0."""


def evaluate_context_relevance(provider: LLMProvider, question: str, context: str) -> float:
    if not context.strip():
        return 0.0
    response = provider.generate(
        "You are a strict relevance-judging assistant.",
        _CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context),
    )
    match = re.search(r"([01](?:\.\d+)?)", response)
    if not match:
        logger.warning("Could not parse a context relevance score from: %r", response)
        return 0.0
    return max(0.0, min(1.0, float(match.group(1))))


# --- Answerability / refusal behavior: does the system refuse when it should (and not otherwise)? ---

_REFUSAL_PHRASES = (
    "does not contain enough evidence",
    "not enough evidence",
    "insufficient evidence",
    "cannot answer",
    "can't answer",
    "no evidence",
    "not covered",
    "unable to answer",
    # Added after Phase 13 manual inspection found a weaker model (allam-2-7b)
    # substantively declining to answer using phrasing the original list
    # missed entirely - these are real observed phrasings, not guesses.
    "no direct mention",
    "does not mention",
    "not directly",
    "has not been evaluated for",
    "cannot provide information on",
    "not explicitly mentioned",
)


def looks_like_refusal(answer_text: str) -> bool:
    """Heuristic, phrase-based detector anchored to the exact language the
    system prompt instructs the model to use when evidence is
    insufficient. Imperfect (a model could refuse in different words this
    doesn't catch), but directly checks compliance with our own prompt's
    instructions rather than assuming it.

    Known limitation, confirmed by manual inspection (Phase 13): a model
    that doesn't follow the "refuse as the first thing in your answer"
    instruction can bury a real refusal after paragraphs of tangential
    content, or partially refuse then continue with unsupported material
    anyway - this function only detects the presence of refusal language
    (a proxy signal), not whether the whole answer stayed properly refused.
    Manual review remains necessary for that judgment.
    """
    lowered = answer_text.lower()
    return any(phrase in lowered for phrase in _REFUSAL_PHRASES)
