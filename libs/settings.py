"""Pydantic Settings for loading client credentials from .env."""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Loads client credentials from .env file.

    API keys use SecretStr to prevent accidental logging/printing.
    Access the raw value with `settings.openai_api_key.get_secret_value()`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: SecretStr = SecretStr("")

    # Anthropic
    anthropic_api_key: SecretStr = SecretStr("")

    # Cohere
    cohere_api_key: SecretStr = SecretStr("")

    # Pinecone
    pinecone_api_key: SecretStr = SecretStr("")
    pinecone_index_host: str = ""

    # MongoDB
    mongo_uri: SecretStr = SecretStr("")
    mongo_database: str = ""

    # AWS S3
    aws_access_key_id: SecretStr = SecretStr("")
    aws_secret_access_key: SecretStr = SecretStr("")
    aws_region: str = "us-east-1"
    aws_s3_bucket: str = ""

    # Redshift (Civic Shout mirror)
    redshift_host: str = ""
    redshift_port: int = 5439
    redshift_database: str = ""
    redshift_user: str = ""
    redshift_password: SecretStr = SecretStr("")
    redshift_schema: str = ""

    # Budget
    budget_dollars: float = 50.0
    cost_ledger_path: str = ".costs/ledger.jsonl"
