from types import SimpleNamespace

import pytest

from cardiorag.generation.providers import OpenAIProvider, load_provider_from_settings


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
