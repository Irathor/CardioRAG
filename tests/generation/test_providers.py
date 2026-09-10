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


def test_load_provider_from_settings_raises_for_unimplemented_provider():
    from cardiorag.config import Settings

    settings = Settings(_env_file=None, llm_provider="ollama")

    with pytest.raises(NotImplementedError):
        load_provider_from_settings(settings)
