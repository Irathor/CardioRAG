"""Grounded answer generation: retrieved evidence + a strict prompt -> an
LLM call -> an answer with exactly the sources it was given attached.

If retrieval found nothing at all, we refuse before ever calling the LLM -
there's no reason to spend latency/cost on a call whose only correct
answer is "insufficient evidence", and it guarantees that refusal happens
even if the model wouldn't have followed the prompt's instruction to say
so itself.
"""

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

from cardiorag.generation.context_builder import build_context
from cardiorag.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from cardiorag.generation.providers import LLMProvider
from cardiorag.models import RetrievedChunk

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The available corpus does not contain enough evidence to answer this question."
)


@dataclass(frozen=True)
class GroundedAnswer:
    question: str
    answer: str
    sources: list[RetrievedChunk]  # exactly what was fed as context, not re-derived from the answer text
    generation_seconds: float


def _build_user_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str | None:
    """The prompt to send the LLM, or None if there's no evidence to ground
    an answer in - shared by generate_answer() and stream_answer() so the
    refusal-before-calling-the-LLM logic exists in exactly one place."""
    if not retrieved_chunks:
        logger.info("No chunks retrieved for question - refusing without calling the LLM")
        return None
    context = build_context(retrieved_chunks)
    return build_user_prompt(question, context)


def generate_answer(
    provider: LLMProvider,
    question: str,
    retrieved_chunks: list[RetrievedChunk],
) -> GroundedAnswer:
    user_prompt = _build_user_prompt(question, retrieved_chunks)
    if user_prompt is None:
        return GroundedAnswer(
            question=question,
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            sources=[],
            generation_seconds=0.0,
        )

    start = time.perf_counter()
    answer_text = provider.generate(SYSTEM_PROMPT, user_prompt)
    elapsed = time.perf_counter() - start

    logger.info(
        "Generated answer in %.2fs using %d source(s)", elapsed, len(retrieved_chunks)
    )

    return GroundedAnswer(
        question=question,
        answer=answer_text,
        sources=retrieved_chunks,
        generation_seconds=elapsed,
    )


def stream_answer(
    provider: LLMProvider,
    question: str,
    retrieved_chunks: list[RetrievedChunk],
) -> Iterator[str]:
    """Same grounding logic as generate_answer(), yielding the answer
    incrementally instead of returning it all at once - for the streaming
    API endpoint (project improvement round). The no-evidence refusal still
    short-circuits before any LLM call, yielded as a single piece so a
    caller consuming either function's output sees text either way."""
    user_prompt = _build_user_prompt(question, retrieved_chunks)
    if user_prompt is None:
        yield INSUFFICIENT_EVIDENCE_MESSAGE
        return
    yield from provider.stream(SYSTEM_PROMPT, user_prompt)
