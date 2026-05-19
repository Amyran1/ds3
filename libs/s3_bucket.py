"""Resolve the default S3 bucket for DS cache artifacts."""

from __future__ import annotations

from libs.settings import Settings

_FALLBACK = "chorus-content-assets"


def default_bucket() -> str:
    """Return AWS_S3_BUCKET from Settings, or the legacy shared bucket."""
    configured = Settings().aws_s3_bucket.strip()
    return configured or _FALLBACK
