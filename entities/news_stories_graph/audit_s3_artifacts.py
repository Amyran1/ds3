"""Audit S3 artifacts for news_stories_graph v1 and publish manifest_v1.json.

Walks the expected S3 key list (Addendum 6 "Expected S3 inventory"),
``head_object`` on each, aggregates into ``manifest_v1.json`` written both
locally under this entity directory and uploaded to S3 at
``.../news_stories_graph/manifest_v1.json``.

Usage:
    python -m entities\
        .news_stories_graph.audit_s3_artifacts
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import boto3
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from collections.abc import Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_BUCKET = "chorus-content-assets"
_ENTITY_PREFIX = (
    "autonomous-data-scientist/civic_shout_news_environment/entities/news_stories_graph"
)

_ENTITY_DIR = Path(
    "entities/news_stories_graph",
)
_FINDINGS_DIR = _ENTITY_DIR / "findings"
_MANIFEST_LOCAL = _ENTITY_DIR / "manifest_v1.json"
_MANIFEST_S3_KEY = f"{_ENTITY_PREFIX}/manifest_v1.json"
_REPORT_PATH = _FINDINGS_DIR / "s3_audit.md"

# (relative_key, (min_bytes, max_bytes))
EXPECTED_KEYS: list[tuple[str, tuple[int, int]]] = [
    ("bipartite_edges/bipartite_edges_v1.parquet", (700_000_000, 800_000_000)),
    (
        "dense_embeddings/embeddings_v1.npy",
        (21_000_000_000, 22_500_000_000),
    ),
    ("dense_embeddings/story_ids_v1.json", (100_000_000, 140_000_000)),
    (
        "dense_faiss_index/dense_faiss_index_v1.faiss",
        (20_000_000_000, 22_000_000_000),
    ),
    ("dense_edges/dense_edges_v1.parquet", (3_000_000_000, 4_000_000_000)),
    ("sparse_tfidf/tfidf_v1.npz", (1_800_000_000, 2_500_000_000)),
    ("sparse_tfidf/vocab_v1.txt", (300_000, 700_000)),
    ("sparse_tfidf/story_ids_v1.json", (100_000_000, 140_000_000)),
    ("bm25_vectors/bm25_v1.npz", (400_000_000, 800_000_000)),
    ("bm25_vectors/params_v1.json", (100, 2_000)),
    ("bm25_vectors/story_ids_v1.json", (100_000_000, 140_000_000)),
]


def _head_object(
    s3_client: object,
    bucket: str,
    key: str,
) -> tuple[int | None, datetime | None, str | None]:
    """Return (size, last_modified, error) for an S3 key; None on 404."""
    try:
        resp = s3_client.head_object(Bucket=bucket, Key=key)  # type: ignore[attr-defined]
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None, None, "missing"
        return None, None, f"error:{code}"
    size = int(resp["ContentLength"])
    lm = resp["LastModified"]
    if isinstance(lm, datetime) and lm.tzinfo is None:
        lm = lm.replace(tzinfo=UTC)
    return size, lm, None


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    step = 1024
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < step or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= step
    return f"{size} B"


def _write_report(
    rows: Sequence[dict[str, object]],
    *,
    all_present: bool,
    all_in_band: bool,
) -> None:
    _FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# news_stories_graph v1 — S3 Artifact Audit",
        "",
        f"Run at: {datetime.now(UTC).isoformat()}",
        f"Bucket: `{_BUCKET}`",
        f"Prefix: `{_ENTITY_PREFIX}/`",
        "",
        f"- all_present: **{all_present}**",
        f"- all_in_band: **{all_in_band}**",
        "",
        "| Key | Size | Expected band | In band | Last modified |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        "| `{key}` | {size} | {band} | {in_band} | {lm} |".format(
            key=row["key"],
            size=_fmt_bytes(cast("int | None", row["size_bytes"])),
            band="{lo}-{hi}".format(
                lo=_fmt_bytes(cast("int", row["band_lo"])),
                hi=_fmt_bytes(cast("int", row["band_hi"])),
            ),
            in_band=row["size_in_band"],
            lm=row["last_modified"] or "-",
        )
        for row in rows
    )
    _REPORT_PATH.write_text("\n".join(lines) + "\n")
    logger.info("Wrote report: %s", _REPORT_PATH)


def main() -> int:
    s3 = boto3.client("s3")
    logger.info("Auditing %d expected S3 keys", len(EXPECTED_KEYS))

    rows: list[dict[str, object]] = []
    manifest_artifacts: list[dict[str, object]] = []
    total_bytes = 0
    all_present = True
    all_in_band = True

    for rel_key, (lo, hi) in EXPECTED_KEYS:
        full_key = f"{_ENTITY_PREFIX}/{rel_key}"
        size, lm, err = _head_object(s3, _BUCKET, full_key)
        present = err is None and size is not None
        in_band = present and size is not None and lo <= size <= hi
        if not present:
            all_present = False
        if not in_band:
            all_in_band = False
        if size is not None:
            total_bytes += size

        lm_str = lm.isoformat() if lm is not None else None
        logger.info(
            "  %-55s size=%s in_band=%s present=%s",
            rel_key,
            _fmt_bytes(size),
            in_band,
            present,
        )
        rows.append(
            {
                "key": rel_key,
                "size_bytes": size,
                "band_lo": lo,
                "band_hi": hi,
                "size_in_band": in_band,
                "last_modified": lm_str,
                "error": err,
            },
        )
        manifest_artifacts.append(
            {
                "key": rel_key,
                "s3_key": full_key,
                "size_bytes": size,
                "last_modified": lm_str,
                "size_in_band": in_band,
            },
        )

    manifest: dict[str, object] = {
        "version": 1,
        "built_at": datetime.now(UTC).isoformat(),
        "entity": "civic_shout_news_environment/news_stories_graph",
        "bucket": _BUCKET,
        "prefix": _ENTITY_PREFIX,
        "total_bytes": total_bytes,
        "n_artifacts": len(EXPECTED_KEYS),
        "artifacts": manifest_artifacts,
        "all_present": all_present,
        "all_in_band": all_in_band,
    }

    _ENTITY_DIR.mkdir(parents=True, exist_ok=True)
    _MANIFEST_LOCAL.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Wrote local manifest: %s", _MANIFEST_LOCAL)

    # Upload to S3
    try:
        s3.put_object(  # type: ignore[attr-defined]
            Bucket=_BUCKET,
            Key=_MANIFEST_S3_KEY,
            Body=_MANIFEST_LOCAL.read_bytes(),
            ContentType="application/json",
        )
        logger.info("Uploaded manifest to s3://%s/%s", _BUCKET, _MANIFEST_S3_KEY)
    except ClientError:
        logger.exception("Failed to upload manifest")
        return 2

    _write_report(rows, all_present=all_present, all_in_band=all_in_band)

    logger.info(
        "total_bytes=%s n=%d all_present=%s all_in_band=%s",
        _fmt_bytes(total_bytes),
        len(EXPECTED_KEYS),
        all_present,
        all_in_band,
    )
    if not all_present:
        logger.error("FAIL: missing S3 keys")
        return 1
    if not all_in_band:
        logger.warning("WARN: all present but some sizes outside expected band")
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
