"""Per-run YAML manifest emitter (reproducibility safety net)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


def _dict_fingerprint(dict_path: Path) -> str:
    content = dict_path.read_bytes()
    return "sha256:" + hashlib.sha256(content).hexdigest()[:16]


def _data_fingerprint(df: Any) -> str:
    sorted_triples = (
        df.select(["user_id", "email_id", "date_sent"])
        .sort(["user_id", "email_id", "date_sent"])
        .to_pandas()
    )
    content = sorted_triples.to_csv(index=False).encode()
    return "sha256:" + hashlib.sha256(content).hexdigest()[:16]


def emit_manifest(
    metadata: Any,
    harness_response: Any,
    run_dir: Path,
    feature_dictionaries: dict[str, Path] | None = None,
    data_for_fingerprint: Any | None = None,
) -> Path:
    features: dict[str, Any] = {}
    if feature_dictionaries:
        for family, dict_path in feature_dictionaries.items():
            raw = yaml.safe_load(dict_path.read_text())
            features[family] = {
                "version": raw.get("version", "v1"),
                "dict_fingerprint": _dict_fingerprint(dict_path),
            }

    data_fp = "null"
    if data_for_fingerprint is not None:
        data_fp = _data_fingerprint(data_for_fingerprint)

    outcome_version = getattr(metadata, "outcome_version", None) or "v1"

    recorded_at = metadata.recorded_at_utc
    if hasattr(recorded_at, "isoformat"):
        recorded_at = recorded_at.isoformat()

    perf = harness_response.performance
    repro = harness_response.reproducibility
    data_profile = harness_response.data

    manifest: dict[str, Any] = {
        "run_id": metadata.run_id,
        "project": metadata.project,
        "comparison_group": metadata.comparison_group,
        "scope": metadata.scope,
        "git_sha": metadata.git_sha,
        "recorded_at_utc": recorded_at,
        "outcome": {
            "version": outcome_version,
            "name": "actioned_24h_per_send",
        },
        "features": features,
        "data": {
            "source_cache": "civic_shout_user_emails",
            "source_version": 3,
            "n_rows": data_profile.n_rows,
            "fingerprint": data_fp,
        },
        "model": {
            "type": repro.ml_model_type,
            "args": repro.ml_model_args,
        },
        "sample": {
            "frac": getattr(data_profile, "sample_frac", None),
            "seed": getattr(data_profile, "sample_seed", None),
        },
        "performance": {
            "elapsed_seconds": perf.total_seconds,
            "bottleneck_stage": getattr(perf, "bottleneck_stage", None),
        },
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "manifest.yaml"
    out_path.write_text(yaml.dump(manifest, default_flow_style=False, sort_keys=False))
    return out_path.resolve()
