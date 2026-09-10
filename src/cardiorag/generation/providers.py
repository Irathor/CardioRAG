"""LLM provider abstraction (Phase 9): interchangeable text-generation
backends behind one interface, so generator.py never depends on which
concrete provider (OpenAI, a local Hugging Face model, Ollama...) is
actually answering the question.

Only OpenAIProvider is implemented so far. The others in
Settings.llm_provider's type are declared but intentionally raise
NotImplementedError - we don't fabricate support that doesn't exist yet.
"""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class OpenAIProvider:
    """Calls an OpenAI-compatible chat completions endpoint. `base_url` can
    point this at any OpenAI-API-compatible server (e.g. a local vLLM or
    Ollama instance exposing that shape), which is why this one class
    covers more than just OpenAI's own API despite the name.
    """

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        # Imported lazily so the rest of the app doesn't require the `openai`
        # package installed unless this specific provider is actually used.
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # 0.0 minimizes creative variance: for grounded scientific
            # answering we want reproducible, evidence-bound output, not
            # creative rephrasing.
            temperature=0.0,
        )
        return response.choices[0].message.content or ""


def load_provider_from_settings(settings) -> LLMProvider:
    """Build the configured provider from typed Settings (config.py)."""
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set; cannot use the 'openai' LLM provider. "
                "Set it in .env or choose a different llm_provider."
            )
        return OpenAIProvider(model=settings.openai_model, api_key=settings.openai_api_key)

    raise NotImplementedError(
        f"LLM provider {settings.llm_provider!r} is not implemented yet. "
        "Only 'openai' is currently supported."
    )
