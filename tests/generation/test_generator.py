from cardiorag.generation.generator import INSUFFICIENT_EVIDENCE_MESSAGE, generate_answer
from cardiorag.models import Chunk, RetrievedChunk


class _FakeProvider:
    def __init__(self, response: str = "a grounded answer [Source 1]"):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


def _make_retrieved(chunk_id: str = "c0") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        text="Cardiac MRI assesses myocardial function.",
        token_count=6,
        page_numbers=[3],
        title="A Test Paper",
        doi="10.1/test",
        source_filename="synthetic.pdf",
    )
    return RetrievedChunk(chunk=chunk, score=0.8)


def test_generate_answer_refuses_without_calling_provider_when_no_chunks():
    provider = _FakeProvider()

    result = generate_answer(provider, "unanswerable question", [])

    assert result.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert result.sources == []
    assert result.generation_seconds == 0.0
    assert provider.calls == []  # never called - no point spending latency/cost


def test_generate_answer_calls_provider_with_question_and_context():
    provider = _FakeProvider(response="LVEF measures pumping efficiency [Source 1].")
    retrieved = [_make_retrieved()]

    result = generate_answer(provider, "What does LVEF measure?", retrieved)

    assert result.answer == "LVEF measures pumping efficiency [Source 1]."
    assert result.sources == retrieved
    system_prompt, user_prompt = provider.calls[0]
    assert "What does LVEF measure?" in user_prompt
    assert "[Source 1]" in user_prompt
    assert "ONLY" in system_prompt


def test_generate_answer_records_generation_seconds():
    provider = _FakeProvider()

    result = generate_answer(provider, "a question", [_make_retrieved()])

    assert result.generation_seconds >= 0.0
