from types import SimpleNamespace

import httpx
import pytest
from openai import RateLimitError

from cardiorag.generation.providers import OpenAIProvider, RetryingProvider, load_provider_from_settings


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

    assert flaky.call_count == 3  # initial attempt + 2 retries


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
