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


def generate_answer(
    provider: LLMProvider,
    question: str,
    retrieved_chunks: list[RetrievedChunk],
) -> GroundedAnswer:
    if not retrieved_chunks:
        logger.info("No chunks retrieved for question - refusing without calling the LLM")
        return GroundedAnswer(
            question=question,
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            sources=[],
            generation_seconds=0.0,
        )

    context = build_context(retrieved_chunks)
    user_prompt = build_user_prompt(question, context)

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
