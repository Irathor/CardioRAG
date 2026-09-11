import queue
import sys
from types import SimpleNamespace

import httpx
import pytest
import torch
from openai import RateLimitError

from cardiorag.generation.providers import (
    HuggingFaceLocalProvider,
    OpenAIProvider,
    RetryingProvider,
    load_provider_from_settings,
)


def _rate_limit_error() -> RateLimitError:
    response = httpx.Response(status_code=429, request=httpx.Request("POST", "http://test"))
    return RateLimitError("rate limited", response=response, body=None)


class _FlakyProvider:
    """Raises RateLimitError a fixed number of times, then succeeds."""

    def __init__(self, fail_times: int, response: str = "ok"):
        self._fail_times = fail_times
        self._response = response
        self.call_count = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        if self.call_count <= self._fail_times:
            raise _rate_limit_error()
        return self._response


def test_retrying_provider_retries_and_eventually_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    flaky = _FlakyProvider(fail_times=2)
    provider = RetryingProvider(flaky, max_retries=5, base_delay_seconds=0.01)

    result = provider.generate("sys", "user")

    assert result == "ok"
    assert flaky.call_count == 3


def test_retrying_provider_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    flaky = _FlakyProvider(fail_times=10)
    provider = RetryingProvider(flaky, max_retries=2, base_delay_seconds=0.01)

    with pytest.raises(RateLimitError):
        provider.generate("sys", "user")


class _FlakyStreamingProvider:
    """Raises RateLimitError a fixed number of times *before yielding
    anything*, then succeeds - simulates the real failure mode this covers:
    a rate limit hit on the request itself, before any output exists."""

    def __init__(self, fail_times: int, pieces: list[str] | None = None):
        self._fail_times = fail_times
        self._pieces = pieces if pieces is not None else ["ok"]
        self.call_count = 0

    def stream(self, system_prompt: str, user_prompt: str):
        self.call_count += 1
        if self.call_count <= self._fail_times:
            raise _rate_limit_error()
        yield from self._pieces


class _MidStreamFailureProvider:
    """Yields one real piece, then raises - simulates a rate limit (or any
    error) that only surfaces after output has already reached the caller."""

    def __init__(self):
        self.call_count = 0

    def stream(self, system_prompt: str, user_prompt: str):
        self.call_count += 1
        yield "partial answer "
        raise _rate_limit_error()


def test_retrying_provider_stream_retries_before_any_output_and_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    flaky = _FlakyStreamingProvider(fail_times=2, pieces=["hel", "lo"])
    provider = RetryingProvider(flaky, max_retries=5, base_delay_seconds=0.01)

    result = list(provider.stream("sys", "user"))

    assert result == ["hel", "lo"]
    assert flaky.call_count == 3


def test_retrying_provider_stream_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    flaky = _FlakyStreamingProvider(fail_times=10)
    provider = RetryingProvider(flaky, max_retries=2, base_delay_seconds=0.01)

    with pytest.raises(RateLimitError):
        list(provider.stream("sys", "user"))


def test_retrying_provider_stream_does_not_retry_once_a_piece_was_already_yielded():
    """The core correctness fix: retrying after output has already reached
    the caller would re-send already-streamed text with no way for the
    caller to undo it - so a failure here must propagate, not retry."""
    mid_stream_failure = _MidStreamFailureProvider()
    provider = RetryingProvider(mid_stream_failure, max_retries=5, base_delay_seconds=0.01)

    # A for-loop, not list(...): the generator raises partway through, and
    # list() would discard whatever it had already collected internally
    # once the exception propagates - this needs the partial result kept.
    collected = []
    with pytest.raises(RateLimitError):
        for piece in provider.stream("sys", "user"):
            collected.append(piece)  # noqa: PERF402

    assert collected == ["partial answer "]
    assert mid_stream_failure.call_count == 1  # never retried


def test_openai_provider_sends_correct_messages_and_returns_content(monkeypatch):
    captured_calls = []

    class _FakeCompletions:
        def create(self, **kwargs):
            captured_calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="mocked answer"))]
            )

    class _FakeOpenAIClient:
        def __init__(self, api_key=None, base_url=None):
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAIClient)

    provider = OpenAIProvider(model="gpt-4o-mini", api_key="fake-key")
    result = provider.generate("system prompt", "user prompt")

    assert result == "mocked answer"
    assert len(captured_calls) == 1
    call = captured_calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 800
    assert call["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]


def test_openai_provider_handles_empty_content(monkeypatch):
    class _FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])

    class _FakeOpenAIClient:
        def __init__(self, api_key=None, base_url=None):
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAIClient)

    provider = OpenAIProvider(model="gpt-4o-mini", api_key="fake-key")

    assert provider.generate("sys", "user") == ""


def test_openai_provider_stream_yields_each_deltas_content(monkeypatch):
    def _delta_chunk(content):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])

    captured_calls = []

    class _FakeCompletions:
        def create(self, **kwargs):
            captured_calls.append(kwargs)
            # A real streaming response also includes an empty/None final
            # delta chunk - must be skipped, not yielded as an empty piece.
            return iter([_delta_chunk("Hel"), _delta_chunk("lo"), _delta_chunk(None)])

    class _FakeOpenAIClient:
        def __init__(self, api_key=None, base_url=None):
            self.chat = SimpleNamespace(completions=_FakeCompletions())

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAIClient)

    provider = OpenAIProvider(model="gpt-4o-mini", api_key="fake-key")
    result = list(provider.stream("system prompt", "user prompt"))

    assert result == ["Hel", "lo"]
    assert captured_calls[0]["stream"] is True


def test_load_provider_from_settings_raises_without_api_key():
    from cardiorag.config import Settings

    settings = Settings(_env_file=None, llm_provider="openai", openai_api_key=None)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        load_provider_from_settings(settings)


def test_load_provider_from_settings_raises_without_groq_api_key():
    from cardiorag.config import Settings

    settings = Settings(_env_file=None, llm_provider="groq", groq_api_key=None)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        load_provider_from_settings(settings)


def test_load_provider_from_settings_builds_groq_via_openai_provider(monkeypatch):
    from cardiorag.config import Settings
    from cardiorag.generation.providers import GROQ_BASE_URL

    captured = {}

    class _FakeOpenAIClient:
        def __init__(self, api_key=None, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = None

    monkeypatch.setattr("openai.OpenAI", _FakeOpenAIClient)

    settings = Settings(
        _env_file=None, llm_provider="groq", groq_api_key="fake-groq-key", groq_model="llama-3.3-70b-versatile"
    )
    provider = load_provider_from_settings(settings)

    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "llama-3.3-70b-versatile"
    assert captured["api_key"] == "fake-groq-key"
    assert captured["base_url"] == GROQ_BASE_URL


def test_load_provider_from_settings_raises_for_unimplemented_provider():
    from cardiorag.config import Settings

    settings = Settings(_env_file=None, llm_provider="ollama")

    with pytest.raises(NotImplementedError):
        load_provider_from_settings(settings)


def test_retrying_provider_still_works_when_openai_package_is_absent(monkeypatch):
    """HuggingFaceLocalProvider never raises openai.RateLimitError and
    shouldn't need the `openai` package installed at all - RetryingProvider
    wraps every provider unconditionally (see api/dependencies.py), so it
    must degrade gracefully rather than crash on the missing import."""
    monkeypatch.setitem(sys.modules, "openai", None)

    class _AlwaysSucceeds:
        def generate(self, system_prompt: str, user_prompt: str) -> str:
            return "ok"

    provider = RetryingProvider(_AlwaysSucceeds())

    assert provider.generate("sys", "user") == "ok"


# --- HuggingFaceLocalProvider (Phase 20 fix #7) ---


class _FakeLocalInputs(dict):
    def to(self, device):
        return self


class _FakeLocalTokenizer:
    eos_token_id = 999

    def __init__(self):
        self.last_messages = None
        self.decoded_tokens = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.last_messages = messages
        return "PROMPT_TEXT"

    def __call__(self, text, return_tensors="pt"):
        return _FakeLocalInputs(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, tokens, skip_special_tokens=True):
        self.decoded_tokens = tokens
        return " a fake generated answer "


class _FakeLocalModel:
    def __init__(self):
        self.generate_kwargs = None
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        streamer = kwargs.get("streamer")
        if streamer is not None:
            # Real transformers streaming generate() pushes text into the
            # streamer and its return value goes unused (it's invoked via a
            # background Thread) - mirror that instead of returning a tensor.
            for piece in ["Hel", "lo"]:
                streamer.put_text(piece)
            streamer.end()
            return None
        # Simulates real behavior: generate() returns the prompt tokens
        # followed by newly generated ones (here, two: 10 and 11).
        return torch.cat([kwargs["input_ids"], torch.tensor([[10, 11]])], dim=1)


class _FakeTextIteratorStreamer:
    """Mirrors real TextIteratorStreamer's actual interface (also a
    thread-safe queue-backed iterator) so it behaves correctly regardless
    of whether the producer thread runs before or after iteration starts."""

    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self._queue: queue.Queue = queue.Queue()

    def put_text(self, text):
        self._queue.put(text)

    def end(self):
        self._queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self._queue.get()
        if value is None:
            raise StopIteration
        return value


@pytest.fixture
def fake_local_model(monkeypatch):
    tokenizer = _FakeLocalTokenizer()
    model = _FakeLocalModel()
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *a, **k: model)
    monkeypatch.setattr("transformers.TextIteratorStreamer", _FakeTextIteratorStreamer)
    return tokenizer, model


def test_hf_local_provider_uses_chat_template_with_system_and_user_roles(fake_local_model):
    tokenizer, _ = fake_local_model
    provider = HuggingFaceLocalProvider("fake-model", device="cpu")

    provider.generate("a system prompt", "a user question")

    assert tokenizer.last_messages == [
        {"role": "system", "content": "a system prompt"},
        {"role": "user", "content": "a user question"},
    ]


def test_hf_local_provider_decodes_only_newly_generated_tokens(fake_local_model):
    tokenizer, _ = fake_local_model
    provider = HuggingFaceLocalProvider("fake-model", device="cpu")

    result = provider.generate("sys", "usr")

    # 3 prompt tokens were fed in; only the 2 tokens generated after them
    # (10, 11) should reach decode() - never an echo of the prompt.
    assert tokenizer.decoded_tokens.tolist() == [10, 11]
    assert result == "a fake generated answer"  # stripped of surrounding whitespace


def test_hf_local_provider_decodes_greedily_with_configured_max_new_tokens(fake_local_model):
    _, model = fake_local_model
    provider = HuggingFaceLocalProvider("fake-model", device="cpu", max_new_tokens=128)

    provider.generate("sys", "usr")

    assert model.generate_kwargs["do_sample"] is False  # matches temperature=0.0 elsewhere
    assert model.generate_kwargs["max_new_tokens"] == 128
    assert model.generate_kwargs["pad_token_id"] == 999


def test_hf_local_provider_resolves_auto_device_to_cpu_without_cuda(fake_local_model, monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    _, model = fake_local_model

    HuggingFaceLocalProvider("fake-model", device="auto")

    assert model.device == "cpu"


def test_hf_local_provider_stream_yields_pieces_from_the_streamer(fake_local_model):
    provider = HuggingFaceLocalProvider("fake-model", device="cpu")

    result = list(provider.stream("sys", "usr"))

    assert result == ["Hel", "lo"]


def test_hf_local_provider_stream_requests_prompt_and_special_tokens_skipped(fake_local_model, monkeypatch):
    captured = {}
    real_streamer_cls = _FakeTextIteratorStreamer

    class _CapturingStreamer(real_streamer_cls):
        def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
            captured["skip_prompt"] = skip_prompt
            captured["skip_special_tokens"] = skip_special_tokens
            super().__init__(tokenizer, skip_prompt, skip_special_tokens)

    monkeypatch.setattr("transformers.TextIteratorStreamer", _CapturingStreamer)
    provider = HuggingFaceLocalProvider("fake-model", device="cpu")

    list(provider.stream("sys", "usr"))

    assert captured == {"skip_prompt": True, "skip_special_tokens": True}


def test_load_provider_from_settings_builds_hf_local_provider(monkeypatch):
    from cardiorag.config import Settings

    captured = {}

    class _FakeProvider:
        def __init__(self, model_name, device, max_new_tokens):
            captured["model_name"] = model_name
            captured["device"] = device
            captured["max_new_tokens"] = max_new_tokens

    monkeypatch.setattr("cardiorag.generation.providers.HuggingFaceLocalProvider", _FakeProvider)

    settings = Settings(
        _env_file=None,
        llm_provider="huggingface_local",
        huggingface_local_model="fake/model",
        huggingface_local_device="cpu",
        huggingface_local_max_new_tokens=256,
    )
    provider = load_provider_from_settings(settings)

    assert isinstance(provider, _FakeProvider)
    assert captured == {"model_name": "fake/model", "device": "cpu", "max_new_tokens": 256}
