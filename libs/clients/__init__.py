"""Thin async client wrappers for external services."""

from __future__ import annotations

from libs.clients.anthropic import AnthropicClient
from libs.clients.anthropic import CompletionOptions as AnthropicCompletionOptions
from libs.clients.cohere import CohereClient, RerankOptions, RerankResult
from libs.clients.mongo import FindOptions, MongoClient
from libs.clients.openai import CompletionOptions, OpenAIClient
from libs.clients.pinecone import (
    PINECONE_SPARSE_ENGLISH_V0,
    PineconeClient,
    QueryMatch,
    QueryOptions,
    SparseEmbedding,
    SparseValues,
    UpsertItem,
)
from libs.clients.s3 import S3Client
from libs.settings import Settings

__all__ = [
    "PINECONE_SPARSE_ENGLISH_V0",
    "AnthropicClient",
    "AnthropicCompletionOptions",
    "CohereClient",
    "CompletionOptions",
    "FindOptions",
    "MongoClient",
    "OpenAIClient",
    "PineconeClient",
    "QueryMatch",
    "QueryOptions",
    "RerankOptions",
    "RerankResult",
    "S3Client",
    "Settings",
    "SparseEmbedding",
    "SparseValues",
    "UpsertItem",
]
