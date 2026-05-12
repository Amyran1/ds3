from __future__ import annotations

_V1_FIELDS: dict[str, str] = {
    "email_id": "Primary key; integer email ID from content_routing.emails.email_id",
    "organization_id": "Organization that sent the email; sourced from content_routing.emails.organization_id",
    "date_sent": "UTC datetime when the email was sent; sourced from content_routing.emails.sent_at",
    "body_plaintext": "Plain-text body of the email; sourced from content_routing.emails.body_plaintext",
    "dense_vector": (
        "1536-dimensional dense embedding (float32 list) for the email content. "
        "Fetched from Pinecone index 'production-content-routing-dense', "
        "namespace 'civic_shout_vv1', keyed by email_id."
    ),
    "sparse_vector": (
        "BM25-style sparse embedding computed fresh via Pinecone Inference API "
        "'pinecone-sparse-english-v0'. NOT fetched from any stored Pinecone index. "
        "Struct with fields: indices (list[i64]) and values (list[f32])."
    ),
}

VERSIONS: dict[int, dict[str, str]] = {
    1: _V1_FIELDS,
}
