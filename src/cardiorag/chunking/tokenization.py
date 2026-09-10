"""Tokenizer abstraction so chunking always counts tokens the way the
configured embedding model actually will.

We deliberately use the embedding model's own Hugging Face tokenizer rather
than a generic approximation like tiktoken: a "512-token chunk" is only a
meaningful, enforceable unit if it's measured in the units the embedding
model itself truncates on. A mismatch here would let chunks silently exceed
the model's max sequence length without us ever seeing it.
"""

from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerFast


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
    """Load (and cache) the fast tokenizer for `model_name`.

    Must be a "fast" (Rust-backed) tokenizer: chunking relies on
    `return_offsets_mapping` to map token spans back to exact character
    ranges in the source text, which slow (pure-Python) tokenizers don't
    support.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.is_fast:
        raise ValueError(
            f"Tokenizer for {model_name!r} is not a fast tokenizer; "
            "offset-mapping-based chunking requires one."
        )
    return tokenizer


def count_tokens(text: str, model_name: str) -> int:
    """Number of tokens `text` would occupy for `model_name`, excluding
    special tokens (CLS/SEP) since those aren't part of the chunk content."""
    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(text, add_special_tokens=False))
