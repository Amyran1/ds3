"""Data dictionary for scout_emails cache versions.

Ported verbatim from
`datascience/projects/scout/extractions/emails/data_dictionary.py`.
Each version maps column names to human-readable descriptions.
"""

from __future__ import annotations

_V1_FIELDS: dict[str, str] = {
    "id": "Unique identifier for the email record (MongoDB _id)",
    "date_sent": "Datetime when the email was sent",
    "subject_line": "Email subject line text",
    "topic_summary": ("JSON-serialized dict of AI-generated topic summary"),
    "client_organization_id": "Organization that sent the email",
    "dense_vector_summary": ("Dense embedding vector for the email content"),
    "sparse_vector_summary": ("Sparse embedding vector for the email content"),
    "email_rubric": ("JSON-serialized dict of rubric scoring results"),
    "personal_pronouns": ("JSON-serialized analysis of pronoun usage"),
    "image_position": ("JSON-serialized image placement metadata"),
    "copy_tone": "JSON-serialized tone analysis results",
    "has_match_language": ("JSON-serialized match/donation language detection"),
    "sender_analysis": ("JSON-serialized sender name/address analysis"),
    "personalization": ("JSON-serialized personalization feature detection"),
    "personal_story": ("JSON-serialized personal story presence analysis"),
    "end_of_year_proximity": ("JSON-serialized end-of-year timing features"),
}

VERSIONS: dict[int, dict[str, str]] = {
    1: _V1_FIELDS,
    2: {
        "_note": (
            "v2 backfills 7,321 non-PETA emails with HTML from "
            "MongoDB (analysis.emails). v1 had incomplete "
            "extraction: 75-100% of non-PETA orgs had empty "
            "html fields. Created by investigation 02 "
            "(02_relevance-labels/03_backfill_html.py) on "
            "2025-03-18."
        ),
        "_coverage": (
            "HTML coverage by org after v2 backfill: "
            "peta=100%, hias=100%, fbotr=99%, shh=100%, "
            "fa=98%, ajws=97%, cfbnj=93%, acs=90%, ff=16%. "
            "Body extraction gaps: shh=0% (no <body> tag in "
            "HTML), fbotr=29% (no <body> tag). ff=16% (HTML "
            "missing from MongoDB). All other fields identical "
            "to v1."
        ),
        **_V1_FIELDS,
    },
}
