"""LLM provider abstraction (Phase 9): interchangeable text-generation
backends behind one interface, so generator.py never depends on which
concrete provider (OpenAI, Groq, a local Hugging Face model, Ollama...) is
actually answering the question.

OpenAIProvider also serves Groq (and would serve a local vLLM/Ollama
server) since they all expose the same OpenAI-shaped chat completions API -
proof the abstraction works across genuinely different backends without
new provider code, just different configuration.

HuggingFaceLocalProvider (Phase 20 fix #7) runs a small instruction-tuned
model directly in-process via `transformers`, for a genuinely
API-key-free, offline generation path. A native Ollama client (its own,
non-OpenAI-compatible API) still intentionally raises NotImplementedError -
we don't fabricate support that doesn't exist yet.
"""

import logging
import time
from typing import Protocol

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class LLMProvider(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class OpenAIProvider:
    """Calls an OpenAI-compatible chat completions endpoint. `base_url` can
    point this at any OpenAI-API-compatible server (e.g. a local vLLM or
    Ollama instance exposing that shape), which is why this one class
    covers more than just OpenAI's own API despite the name.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 800,
    ):
        # Imported lazily so the rest of the app doesn't require the `openai`
        # package installed unless this specific provider is actually used.
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        # Explicit, since some providers default to a much higher value than
        # their own enforced per-minute output-token ceiling (observed: a
        # free-tier Groq model rejected an unset-max_tokens request outright
        # for requesting more output than its limit allowed - not a request
        # we were anywhere near actually needing).
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
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


class HuggingFaceLocalProvider:
    """Runs a small instruction-tuned causal LM locally via `transformers` -
    no API key, no per-request network call, at a real quality cost
    compared to a much larger hosted model (see README's honest comparison
    against the Groq default). Loads and holds the model/tokenizer once at
    construction (like Embedder/Reranker), not per call - re-loading a
    model's weights on every generate() would dominate latency.
    """

    def __init__(self, model_name: str, device: str = "auto", max_new_tokens: int = 512):
        # Imported lazily, same reasoning as OpenAIProvider's `openai` import:
        # torch/transformers shouldn't be required unless this provider is
        # actually selected.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved_device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        if resolved_device == "auto":
            resolved_device = "cpu"

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._model.to(resolved_device)
        self._model.eval()
        self._device = resolved_device
        self.max_new_tokens = max_new_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._device)

        with self._torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                # Greedy decoding, matching OpenAIProvider's temperature=0.0 -
                # reproducible, evidence-bound output rather than creative
                # rephrasing.
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # generate() returns the prompt tokens followed by the new ones -
        # slice them off so the caller gets only the model's actual answer,
        # not an echo of the prompt it was fed.
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class RetryingProvider:
    """Wraps another LLMProvider, retrying with exponential backoff on rate
    limit errors.

    Free-tier APIs (e.g. Groq) enforce requests-per-minute limits that a
    multi-call-per-question evaluation loop (Phase 13) can realistically
    hit - failing an entire evaluation run on the first 429 would be
    needlessly fragile when the fix is just "wait and retry."
    """

    def __init__(self, wrapped: LLMProvider, max_retries: int = 5, base_delay_seconds: float = 2.0):
        self._wrapped = wrapped
        self._max_retries = max_retries
        self._base_delay_seconds = base_delay_seconds

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import RateLimitError
        except ImportError:
            # RetryingProvider wraps whatever provider load_provider_from_settings
            # built, including HuggingFaceLocalProvider (Phase 20 fix #7), which
            # never raises this and doesn't need the `openai` package installed
            # at all. Falling back to a placeholder that nothing ever raises
            # keeps the except clause below harmless instead of crashing every
            # generate() call in a local-only deployment.
            class RateLimitError(Exception):
                pass

        for attempt in range(self._max_retries + 1):
            try:
                return self._wrapped.generate(system_prompt, user_prompt)
            except RateLimitError:
                if attempt == self._max_retries:
                    raise
                delay = self._base_delay_seconds * (2**attempt)
                logger.warning(
                    "Rate limited, retrying in %.1fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    self._max_retries,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable")  # loop always returns or raises


def load_provider_from_settings(settings) -> LLMProvider:
    """Build the configured provider from typed Settings (config.py)."""
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set; cannot use the 'openai' LLM provider. "
                "Set it in .env or choose a different llm_provider."
            )
        return OpenAIProvider(model=settings.openai_model, api_key=settings.openai_api_key)

    if settings.llm_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set; cannot use the 'groq' LLM provider. "
                "Set it in .env or choose a different llm_provider."
            )
        return OpenAIProvider(
            model=settings.groq_model, api_key=settings.groq_api_key, base_url=GROQ_BASE_URL
        )

    if settings.llm_provider == "huggingface_local":
        return HuggingFaceLocalProvider(
            model_name=settings.huggingface_local_model,
            device=settings.huggingface_local_device,
            max_new_tokens=settings.huggingface_local_max_new_tokens,
        )

    raise NotImplementedError(
        f"LLM provider {settings.llm_provider!r} is not implemented yet. "
        "Only 'openai', 'groq' and 'huggingface_local' are currently supported."
    )
