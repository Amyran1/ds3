"""Toy fixture tests for civic_shout_action_rate_increase discovery harness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from libs.responses import (
    DiscoveryConfig,
    DiscoveryResultRow,
    RunResultMetadata,
)
from projects.civic_shout_action_rate_increase.discovery import (
    CivicShoutDiscoveryResponse,
    CivicShoutDiscoveryResultRow,
    discover_gaps,
)

# ---------------------------------------------------------------------------
# Fixture constants
# ---------------------------------------------------------------------------

_N_ROWS = 5000
_N_USERS = 200
_N_EMAILS = 50
_N_FOLDS = 8
_FEATURE_COLS = [
    "actioned_last_5",
    "sends_since_last_action",
    "action_in_last_5",
    "lifetime_actions_prior",
]


def _make_predictions_df(seed: int = 42) -> pl.DataFrame:
    """Build a toy predictions dataframe with engineered slice and noise."""
    rng = np.random.default_rng(seed)

    n = _N_ROWS
    user_ids = rng.integers(0, _N_USERS, size=n)
    email_ids = rng.integers(0, _N_EMAILS, size=n)

    # 8 fold quarters (one per month bucket for simplicity)
    folds = rng.integers(0, _N_FOLDS, size=n)

    # Base score ~ uniform
    score = rng.uniform(0.05, 0.50, size=n)
    label = (rng.uniform(size=n) < score).astype(np.int8)

    # Base row_loss (binary cross-entropy approximation)
    eps = 1e-7
    row_loss = -(
        label * np.log(np.clip(score, eps, 1 - eps))
        + (1 - label) * np.log(np.clip(1 - score, eps, 1 - eps))
    )

    # Feature columns
    actioned_last_5 = rng.integers(0, 6, size=n).astype(np.int32)
    sends_since_last_action = rng.integers(0, 20, size=n).astype(np.int32)
    action_in_last_5 = (actioned_last_5 > 0).astype(np.int8)
    lifetime_actions_prior = rng.integers(0, 100, size=n).astype(np.int32)

    # Engineered high-loss slice: actioned_last_5 >= 3 AND sends_since_last_action <= 2
    slice_mask = (actioned_last_5 >= 3) & (sends_since_last_action <= 2)
    row_loss = row_loss.copy()
    row_loss[slice_mask] += 2.5  # large uplift

    # Engineered label noise: 30 rows with label=0 but score=0.95
    noise_indices = rng.choice(np.where(~slice_mask)[0], size=30, replace=False)
    label = label.copy()
    score = score.copy()
    label[noise_indices] = 0
    score[noise_indices] = 0.95
    # Recompute row_loss for noise rows
    row_loss[noise_indices] = -(
        0 * np.log(np.clip(0.95, eps, 1 - eps)) + 1 * np.log(np.clip(1 - 0.95, eps, 1 - eps))
    )

    # Build a fake date_sent as int ms timestamps (8 quarters)
    quarter_offsets_ms = np.array([i * 90 * 24 * 3600 * 1000 for i in range(8)], dtype=np.int64)
    base_ms = np.int64(1704067200000)  # 2024-01-01
    date_sent_ms = base_ms + quarter_offsets_ms[folds]

    return pl.DataFrame(
        {
            "user_id": pl.Series(user_ids.astype(np.int32)),
            "email_id": pl.Series(email_ids.astype(np.int32)),
            "date_sent": pl.Series(date_sent_ms, dtype=pl.Int64).cast(pl.Datetime("ms")),
            "score": pl.Series(score.astype(np.float32)),
            "score_residualized": pl.Series(score.astype(np.float32)),
            "label": pl.Series(label.astype(np.int8)),
            "row_loss": pl.Series(row_loss.astype(np.float32)),
            "fold": pl.Series(folds.astype(np.int32)),
            "actioned_last_5": pl.Series(actioned_last_5),
            "sends_since_last_action": pl.Series(sends_since_last_action),
            "action_in_last_5": pl.Series(action_in_last_5),
            "lifetime_actions_prior": pl.Series(lifetime_actions_prior),
        }
    )


def _make_data_df(predictions_df: pl.DataFrame) -> pl.DataFrame:
    """Return the feature portion of predictions_df as the 'data' input."""
    return predictions_df.select(["user_id", "email_id", "date_sent"] + _FEATURE_COLS)


def _make_harness_response(parquet_path: str) -> object:
    """Build a minimal HarnessResponse-like object with a predictions artifact."""
    from libs.responses import (
        HarnessArtifact,
        HarnessDataProfile,
        HarnessParallelism,
        HarnessPerformance,
        HarnessResponse,
        HarnessSummary,
    )

    return HarnessResponse(
        summary=HarnessSummary(
            primary_metric_name="roc_auc",
            primary_metric_value=0.70,
        ),
        performance=HarnessPerformance(
            total_seconds=10.0,
            parallelism=HarnessParallelism(),
        ),
        data_profile=HarnessDataProfile(
            n_rows=_N_ROWS,
            n_rows_source=_N_ROWS,
            sample_frac=1.0,
            is_full_run=False,
            n_features=len(_FEATURE_COLS),
        ),
        artifacts=[
            HarnessArtifact(
                kind="predictions",
                path=parquet_path,
                name="predictions",
            )
        ],
        fingerprint="abc123",
    )


@pytest.fixture()
def tmp_parquet(tmp_path: Path) -> tuple[Path, pl.DataFrame]:
    predictions_df = _make_predictions_df(seed=42)
    parquet_path = tmp_path / "predictions.parquet"
    predictions_df.write_parquet(parquet_path)
    return parquet_path, predictions_df


@pytest.fixture()
def discovery_inputs(tmp_parquet: tuple[Path, pl.DataFrame]) -> dict:
    parquet_path, predictions_df = tmp_parquet
    data_df = _make_data_df(predictions_df)
    harness_resp = _make_harness_response(str(parquet_path))
    cfg = DiscoveryConfig(enabled_methods=[], top_k_findings=50)
    return {
        "data": data_df,
        "harness_response": harness_resp,
        "config": cfg,
        "predictions_df": predictions_df,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_discover_response_shape(
    discovery_inputs: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    from projects.civic_shout_action_rate_increase import discovery as disc_mod

    monkeypatch.setattr(disc_mod, "_PROJECT_ROOT", tmp_path)

    resp = discover_gaps(
        data=discovery_inputs["data"],
        feature_cols=_FEATURE_COLS,
        outcome_variable="actioned_24h",
        prediction_cols={
            "score": "score",
            "label": "label",
            "fold": "fold",
            "row_loss": "row_loss",
        },
        harness_response=discovery_inputs["harness_response"],
        discovery_config=discovery_inputs["config"],
    )

    assert isinstance(resp, CivicShoutDiscoveryResponse)
    assert resp.scores is not None
    assert resp.performance is not None
    assert resp.performance.total_seconds > 0

    method_names = {t.stage for t in resp.performance.stage_timings}
    assert "residual_slice_tree" in method_names
    assert "label_noise_scan" in method_names


def test_residual_slice_tree_finds_engineered_slice(
    discovery_inputs: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    from projects.civic_shout_action_rate_increase import discovery as disc_mod

    monkeypatch.setattr(disc_mod, "_PROJECT_ROOT", tmp_path)

    cfg = DiscoveryConfig(enabled_methods=["residual_slice_tree"], top_k_findings=50)
    resp = discover_gaps(
        data=discovery_inputs["data"],
        feature_cols=_FEATURE_COLS,
        outcome_variable="actioned_24h",
        prediction_cols={
            "score": "score",
            "label": "label",
            "fold": "fold",
            "row_loss": "row_loss",
        },
        harness_response=discovery_inputs["harness_response"],
        discovery_config=cfg,
    )

    slice_findings = [f for f in resp.findings if f.kind == "residual_slice"]
    assert len(slice_findings) >= 1, "Expected at least one residual_slice finding"

    feature_names_in_slices = set()
    for f in slice_findings:
        slice_def = f.slice_definition
        for feat in _FEATURE_COLS:
            if feat in slice_def:
                feature_names_in_slices.add(feat)

    assert (
        "actioned_last_5" in feature_names_in_slices
        or "sends_since_last_action" in feature_names_in_slices
    ), (
        f"Expected engineered features in slice definitions. Found: {feature_names_in_slices}. "
        f"Slices: {[f.slice_definition for f in slice_findings]}"
    )


def test_label_noise_scan_finds_engineered_noise(
    discovery_inputs: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    from projects.civic_shout_action_rate_increase import discovery as disc_mod

    monkeypatch.setattr(disc_mod, "_PROJECT_ROOT", tmp_path)

    cfg = DiscoveryConfig(enabled_methods=["label_noise_scan"], top_k_findings=200)
    resp = discover_gaps(
        data=discovery_inputs["data"],
        feature_cols=_FEATURE_COLS,
        outcome_variable="actioned_24h",
        prediction_cols={
            "score": "score",
            "label": "label",
            "fold": "fold",
            "row_loss": "row_loss",
        },
        harness_response=discovery_inputs["harness_response"],
        discovery_config=cfg,
    )

    noise_findings = [f for f in resp.findings if f.kind == "label_noise"]
    assert resp.n_label_noise_candidates == len(noise_findings)

    import json as _json

    high_score_label0 = []
    for f in noise_findings:
        try:
            info = _json.loads(f.slice_definition)
        except Exception:
            continue
        if info.get("label") == 0 and float(info.get("score", 0)) >= 0.94:
            high_score_label0.append(f)

    # The 30 engineered noise rows have score≈0.95 and label=0; expect to find at least 25
    assert len(high_score_label0) >= 25, (
        f"Expected >= 25 engineered noise rows in findings, got {len(high_score_label0)}. "
        f"Total noise findings: {len(noise_findings)}"
    )


def test_to_discovery_summary_row_round_trip(
    discovery_inputs: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    from projects.civic_shout_action_rate_increase import discovery as disc_mod

    monkeypatch.setattr(disc_mod, "_PROJECT_ROOT", tmp_path)

    resp = discover_gaps(
        data=discovery_inputs["data"],
        feature_cols=_FEATURE_COLS,
        outcome_variable="actioned_24h",
        prediction_cols={
            "score": "score",
            "label": "label",
            "fold": "fold",
            "row_loss": "row_loss",
        },
        harness_response=discovery_inputs["harness_response"],
        discovery_config=discovery_inputs["config"],
    )

    metadata = RunResultMetadata(
        run_id="run_01",
        project="civic_shout_action_rate_increase",
        comparison_group="action_rate_increase_temporal_holdout_residualized_auc_v1",
        scope="comparison",
        recorded_at_utc="2026-05-13T00:00:00Z",
        git_sha="abc123",
    )

    row = resp.to_discovery_summary_row(metadata)
    assert isinstance(row, CivicShoutDiscoveryResultRow)
    assert isinstance(row, DiscoveryResultRow)
    assert row.run_id == "run_01"
    assert row.project == "civic_shout_action_rate_increase"
    assert row.scope == "comparison"
    assert row.discovery_total_seconds > 0
    assert row.n_label_noise_candidates >= 0
    assert row.top_slice_loss_uplift_pct >= 0.0

    json_str = row.model_dump_json()
    recovered = CivicShoutDiscoveryResultRow.model_validate_json(json_str)
    assert recovered.run_id == row.run_id
    assert recovered.n_label_noise_candidates == row.n_label_noise_candidates


def test_predictions_parquet_columns_assertion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    from projects.civic_shout_action_rate_increase import discovery as disc_mod

    monkeypatch.setattr(disc_mod, "_PROJECT_ROOT", tmp_path)

    bad_df = pl.DataFrame(
        {
            "user_id": [1, 2, 3],
            "email_id": [1, 2, 3],
            "score": [0.1, 0.2, 0.3],
        }
    )
    bad_parquet = tmp_path / "bad_predictions.parquet"
    bad_df.write_parquet(bad_parquet)

    from libs.responses import (
        HarnessArtifact,
        HarnessDataProfile,
        HarnessParallelism,
        HarnessPerformance,
        HarnessResponse,
        HarnessSummary,
    )

    bad_harness = HarnessResponse(
        summary=HarnessSummary(primary_metric_name="roc_auc", primary_metric_value=0.7),
        performance=HarnessPerformance(total_seconds=1.0, parallelism=HarnessParallelism()),
        data_profile=HarnessDataProfile(
            n_rows=3, n_rows_source=3, sample_frac=1.0, is_full_run=False, n_features=1
        ),
        artifacts=[HarnessArtifact(kind="predictions", path=str(bad_parquet), name="predictions")],
    )

    dummy_data = pl.DataFrame({"user_id": [1], "email_id": [1], "date_sent": [0]})
    cfg = DiscoveryConfig()

    with pytest.raises(ValueError, match="missing required columns"):
        discover_gaps(
            data=dummy_data,
            feature_cols=[],
            outcome_variable="actioned_24h",
            prediction_cols={"score": "score"},
            harness_response=bad_harness,
            discovery_config=cfg,
        )


def test_deterministic_same_seed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    from projects.civic_shout_action_rate_increase import discovery as disc_mod

    monkeypatch.setattr(disc_mod, "_PROJECT_ROOT", tmp_path)

    predictions_df = _make_predictions_df(seed=42)
    parquet_path = tmp_path / "predictions.parquet"
    predictions_df.write_parquet(parquet_path)

    data_df = _make_data_df(predictions_df)
    harness_resp = _make_harness_response(str(parquet_path))
    cfg = DiscoveryConfig(enabled_methods=["residual_slice_tree"], top_k_findings=50)

    resp1 = discover_gaps(
        data=data_df,
        feature_cols=_FEATURE_COLS,
        outcome_variable="actioned_24h",
        prediction_cols={
            "score": "score",
            "label": "label",
            "fold": "fold",
            "row_loss": "row_loss",
        },
        harness_response=harness_resp,
        discovery_config=cfg,
    )
    resp2 = discover_gaps(
        data=data_df,
        feature_cols=_FEATURE_COLS,
        outcome_variable="actioned_24h",
        prediction_cols={
            "score": "score",
            "label": "label",
            "fold": "fold",
            "row_loss": "row_loss",
        },
        harness_response=harness_resp,
        discovery_config=cfg,
    )

    assert len(resp1.findings) == len(resp2.findings), (
        f"Finding counts differ: {len(resp1.findings)} vs {len(resp2.findings)}"
    )
    for f1, f2 in zip(resp1.findings, resp2.findings):
        assert f1.id == f2.id
        assert f1.slice_definition == f2.slice_definition
        assert abs(f1.residual_lift - f2.residual_lift) < 1e-6
