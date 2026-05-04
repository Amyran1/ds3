"""Provider pricing tables and cost estimation functions.

Prices are approximate but in the right ballpark. Updated for 2025 rates.
All per-token prices are in USD.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4  # rough average for English text


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length (~4 chars/token)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_tokens_from_messages(
    messages: list[dict[str, str]],
) -> int:
    """Estimate total tokens across chat messages."""
    return sum(estimate_tokens(m.get("content", "")) for m in messages)


# ---------------------------------------------------------------------------
# Model matching
# ---------------------------------------------------------------------------

_PriceT = float | tuple[float, float]


def _match_price(
    model: str,
    prices: Mapping[str, _PriceT],
) -> _PriceT | None:
    """Match a model string to a pricing entry.

    Tries exact match first, then longest prefix match.
    """
    if model in prices:
        return prices[model]
    matches = [(k, v) for k, v in prices.items() if model.startswith(k)]
    if matches:
        return max(matches, key=lambda x: len(x[0]))[1]
    return None


# ===========================================================================
# OpenAI
# ===========================================================================

_OPENAI_EMBED_PER_TOKEN: dict[str, _PriceT] = {
    "text-embedding-3-small": 0.02e-6,
    "text-embedding-3-large": 0.13e-6,
    "text-embedding-ada-002": 0.10e-6,
}

_OPENAI_COMPLETION_PER_TOKEN: dict[str, _PriceT] = {
    "gpt-4o": (2.50e-6, 10.0e-6),
    "gpt-4o-mini": (0.15e-6, 0.60e-6),
    "gpt-4.1": (2.00e-6, 8.00e-6),
    "gpt-4.1-mini": (0.40e-6, 1.60e-6),
    "gpt-4.1-nano": (0.10e-6, 0.40e-6),
    "o3": (2.00e-6, 8.00e-6),
    "o3-mini": (1.10e-6, 4.40e-6),
    "o4-mini": (1.10e-6, 4.40e-6),
}

_OPENAI_EMBED_FALLBACK = 0.13e-6
_OPENAI_COMPLETION_FALLBACK = (2.50e-6, 10.0e-6)


def estimate_openai_embedding(model: str, texts: list[str]) -> float:
    """Estimate cost for OpenAI embedding call."""
    per_token = _match_price(model, _OPENAI_EMBED_PER_TOKEN)
    if not isinstance(per_token, float):
        per_token = _OPENAI_EMBED_FALLBACK
    tokens = sum(estimate_tokens(t) for t in texts)
    return tokens * per_token


def estimate_openai_completion(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate cost for OpenAI completion (actual token counts)."""
    prices = _match_price(model, _OPENAI_COMPLETION_PER_TOKEN)
    if not isinstance(prices, tuple):
        prices = _OPENAI_COMPLETION_FALLBACK
    return input_tokens * prices[0] + output_tokens * prices[1]


def estimate_openai_completion_from_messages(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> float:
    """Estimate cost for OpenAI completion from message text."""
    input_tokens = estimate_tokens_from_messages(messages)
    output_estimate = min(max_tokens, 500)
    return estimate_openai_completion(model, input_tokens, output_estimate)


# ===========================================================================
# Anthropic
# ===========================================================================

# Longer/more-specific prefixes must come before shorter ones so that
# _match_price's longest-prefix rule resolves correctly.
_ANTHROPIC_COMPLETION_PER_TOKEN: dict[str, _PriceT] = {
    # Claude 4 family — exact versioned IDs come first so they match before
    # the shorter prefix entries below.
    "claude-opus-4-7": (15.0e-6, 75.0e-6),
    "claude-opus-4-6": (15.0e-6, 75.0e-6),
    "claude-opus-4-5": (15.0e-6, 75.0e-6),
    "claude-opus-4": (15.0e-6, 75.0e-6),
    "claude-sonnet-4-6": (3.0e-6, 15.0e-6),
    "claude-sonnet-4-5": (3.0e-6, 15.0e-6),
    "claude-sonnet-4": (3.0e-6, 15.0e-6),
    "claude-haiku-4-5": (1.0e-6, 5.0e-6),
    "claude-haiku-4": (1.0e-6, 5.0e-6),
    # Claude 3.x family
    "claude-haiku-3.5": (0.80e-6, 4.0e-6),
    "claude-3-5-sonnet": (3.0e-6, 15.0e-6),
    "claude-3-5-haiku": (0.80e-6, 4.0e-6),
    "claude-3-opus": (15.0e-6, 75.0e-6),
    "claude-3-sonnet": (3.0e-6, 15.0e-6),
    "claude-3-haiku": (0.25e-6, 1.25e-6),
}

_ANTHROPIC_COMPLETION_FALLBACK = (3.0e-6, 15.0e-6)

# NOTE: The pricing module does not yet differentiate cached-input token costs
# (Anthropic charges $0.10/MTok for cache reads, $1.25/MTok for cache writes on
# Haiku 4.5) at the per-model level in a separate table. The
# estimate_anthropic_completion_cached() function below applies the multipliers
# (0.10x and 1.25x) against the base input price for whichever model is matched.
# This is correct in relative terms but relies on the base input price being
# right. The separate "cached-input pricing table" is a follow-up gap.


def estimate_anthropic_completion(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate cost for Anthropic completion (actual token counts)."""
    prices = _match_price(model, _ANTHROPIC_COMPLETION_PER_TOKEN)
    if not isinstance(prices, tuple):
        prices = _ANTHROPIC_COMPLETION_FALLBACK
    return input_tokens * prices[0] + output_tokens * prices[1]


def estimate_anthropic_completion_cached(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Estimate cost for Anthropic completion with prompt caching.

    Cache creation tokens are billed at 1.25x the base input price.
    Cache read tokens are billed at 0.1x the base input price.
    """
    prices = _match_price(model, _ANTHROPIC_COMPLETION_PER_TOKEN)
    if not isinstance(prices, tuple):
        prices = _ANTHROPIC_COMPLETION_FALLBACK
    input_price, output_price = prices
    return (
        input_tokens * input_price
        + output_tokens * output_price
        + cache_creation_tokens * input_price * 1.25
        + cache_read_tokens * input_price * 0.10
    )


def estimate_anthropic_completion_from_messages(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> float:
    """Estimate cost for Anthropic completion from message text."""
    input_tokens = estimate_tokens_from_messages(messages)
    output_estimate = min(max_tokens, 500)
    return estimate_anthropic_completion(
        model,
        input_tokens,
        output_estimate,
    )


# ===========================================================================
# Cohere
# ===========================================================================

_COHERE_RERANK_PER_SEARCH: float = 2.0 / 1000

_COHERE_EMBED_PER_TOKEN: dict[str, _PriceT] = {
    "embed-english-v3.0": 0.10e-6,
    "embed-multilingual-v3.0": 0.10e-6,
    "embed-english-v2.0": 0.10e-6,
}

_COHERE_EMBED_FALLBACK = 0.10e-6


def estimate_cohere_rerank(n_documents: int) -> float:
    """Estimate cost for Cohere rerank (1 search unit per doc)."""
    return n_documents * _COHERE_RERANK_PER_SEARCH


def estimate_cohere_embed(model: str, texts: list[str]) -> float:
    """Estimate cost for Cohere embedding."""
    per_token = _match_price(model, _COHERE_EMBED_PER_TOKEN)
    if not isinstance(per_token, float):
        per_token = _COHERE_EMBED_FALLBACK
    tokens = sum(estimate_tokens(t) for t in texts)
    return tokens * per_token


# ===========================================================================
# Pinecone (serverless pricing)
# ===========================================================================

_PINECONE_READ_PER_UNIT: float = 8.25e-6
_PINECONE_WRITE_PER_UNIT: float = 2.00e-6


def estimate_pinecone_read(n_queries: int = 1) -> float:
    """Estimate cost for Pinecone read operations."""
    return n_queries * _PINECONE_READ_PER_UNIT


def estimate_pinecone_write(n_vectors: int) -> float:
    """Estimate cost for Pinecone write operations."""
    return n_vectors * _PINECONE_WRITE_PER_UNIT


# ===========================================================================
# S3
# ===========================================================================

_S3_PUT_PER_REQ: float = 0.005 / 1000
_S3_GET_PER_REQ: float = 0.0004 / 1000
_S3_LIST_PER_REQ: float = 0.005 / 1000
_S3_DATA_OUT_PER_BYTE: float = 0.09 / (1024**3)


def estimate_s3_put(data_bytes: int = 0) -> float:
    """Estimate cost for S3 PUT + data transfer."""
    return _S3_PUT_PER_REQ + data_bytes * _S3_DATA_OUT_PER_BYTE


def estimate_s3_get(data_bytes: int = 0) -> float:
    """Estimate cost for S3 GET + data transfer."""
    return _S3_GET_PER_REQ + data_bytes * _S3_DATA_OUT_PER_BYTE


def estimate_s3_list() -> float:
    """Estimate cost for S3 LIST operation."""
    return _S3_LIST_PER_REQ


# ===========================================================================
# MongoDB Atlas (rough per-operation estimates)
# ===========================================================================

_MONGO_READ_PER_OP: float = 0.10 / 1_000_000
_MONGO_WRITE_PER_OP: float = 0.10 / 1_000_000


def estimate_mongo_read(n_documents: int = 1) -> float:
    """Estimate cost for MongoDB read operations."""
    return n_documents * _MONGO_READ_PER_OP


def estimate_mongo_write(n_documents: int = 1) -> float:
    """Estimate cost for MongoDB write operations."""
    return n_documents * _MONGO_WRITE_PER_OP
