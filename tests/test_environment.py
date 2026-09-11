"""Phase 0 smoke tests: package imports and configuration loads with sane defaults."""

import cardiorag
from cardiorag.config import Settings


def test_package_version_is_set():
    assert cardiorag.__version__ == "0.1.0"


def test_settings_load_with_defaults(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "openai"
    assert settings.retrieval_top_k == 5
    assert settings.embedding_model.startswith("sentence-transformers/")


def test_settings_are_overridable_via_env(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_TOP_K", "10")
    settings = Settings(_env_file=None)

    assert settings.retrieval_top_k == 10


def test_masked_settings_never_expose_real_api_key_values():
    settings = Settings(_env_file=None, openai_api_key="sk-super-secret", groq_api_key=None)

    masked = settings.masked()

    assert masked["openai_api_key"] is True
    assert masked["groq_api_key"] is False
    assert "sk-super-secret" not in str(masked)
