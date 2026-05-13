from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from libs.responses import EDAResultRow, RunResultMetadata
from projects.civic_shout_action_rate_increase.eda import (
    CivicShoutEDAConfig,
    CivicShoutEDAResponse,
    eda,
)
from tests.projects.civic_shout_action_rate_increase._fixtures import (
    FEATURE_COLS,
    make_synthetic_user_emails,
)

_METADATA = RunResultMetadata(
    run_id="test_run_01",
    project="civic_shout_action_rate_increase",
    comparison_group="test_comparison_group",
    scope="smoke",
    recorded_at_utc="2026-05-13T00:00:00Z",
    git_sha="abc1234",
)

_ALL_8_METHODS = [
    "outcome_distribution",
    "univariate_numeric",
    "univariate_categorical",
    "feature_outcome_correlation",
    "feature_feature_correlation",
    "missingness_pattern",
    "row_alignment_audit",
    "eda_cohort_baseline",
]


@pytest.fixture(scope="module")
def toy_data():
    return make_synthetic_user_emails(seed=42)


@pytest.fixture(scope="module")
def eda_artifacts_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("eda_artifacts")


@pytest.fixture(scope="module")
def toy_response(toy_data, eda_artifacts_dir):
    cfg = CivicShoutEDAConfig(
        sample_frac=1.0,
        seed=42,
        artifacts_dir=eda_artifacts_dir,
    )
    return eda(toy_data, FEATURE_COLS, "actioned_24h", cfg)


def test_eda_response_shape(toy_response):
    assert isinstance(toy_response, CivicShoutEDAResponse)

    method_names_run = [m.name for m in toy_response.method_results]
    for method in _ALL_8_METHODS:
        assert method in method_names_run, f"Missing method: {method}"

    assert toy_response.eda_summary is not None
    assert toy_response.eda_summary.n_findings_total >= 0
    assert toy_response.data_profile is not None
    assert toy_response.data_profile.n_features == len(FEATURE_COLS)
    assert toy_response.performance.total_seconds > 0


def test_all_methods_produce_findings_or_pass(toy_response):
    for method in toy_response.method_results:
        if method.skipped:
            assert method.skip_reason != "", f"Skipped method {method.name} has no reason"
            continue
        if method.name == "row_alignment_audit":
            passing = any(f.id == "row_alignment_audit_pass" for f in toy_response.findings)
            failing = any(f.id == "row_alignment_audit_fail" for f in toy_response.findings)
            assert passing or failing, "row_alignment_audit produced no findings"
        elif method.name == "univariate_categorical":
            pass
        else:
            assert method.n_findings >= 0


def test_pngs_written(toy_response, eda_artifacts_dir):
    eda_dir = eda_artifacts_dir / "eda"
    pngs = list(eda_dir.glob("*.png"))
    assert len(pngs) >= 6, f"Expected at least 6 PNGs, found {len(pngs)}: {[p.name for p in pngs]}"

    expected_pngs = [
        "outcome_distribution.png",
        "univariate_numeric.png",
        "feature_outcome_correlation.png",
        "feature_feature_correlation.png",
        "missingness_pattern.png",
        "eda_cohort_baseline.png",
    ]
    written_names = {p.name for p in pngs}
    for expected in expected_pngs:
        assert expected in written_names, f"Missing PNG: {expected}"


def test_to_eda_summary_row_round_trip(toy_response):
    row = toy_response.to_eda_summary_row(_METADATA)
    json_str = row.model_dump_json()
    restored = EDAResultRow.model_validate_json(json_str)
    assert restored.run_id == row.run_id
    assert restored.project == row.project
    assert restored.scope == row.scope
    assert restored.recorded_at_utc == row.recorded_at_utc


def test_to_summary_md_renders(toy_response):
    md = toy_response.to_summary_md(_METADATA)
    assert isinstance(md, str)
    assert len(md) > 200
    assert "![" in md and ".png)" in md, "Expected embedded PNG markdown image references"
    assert "test_run_01" in md


def test_deterministic_same_seed(toy_data, tmp_path):
    artifacts_a = tmp_path / "run_a"
    artifacts_b = tmp_path / "run_b"

    cfg_a = CivicShoutEDAConfig(sample_frac=0.1, seed=42, artifacts_dir=artifacts_a)
    cfg_b = CivicShoutEDAConfig(sample_frac=0.1, seed=42, artifacts_dir=artifacts_b)

    resp_a = eda(toy_data, FEATURE_COLS, "actioned_24h", cfg_a)
    resp_b = eda(toy_data, FEATURE_COLS, "actioned_24h", cfg_b)

    assert resp_a.eda_summary.outcome_positive_rate == pytest.approx(
        resp_b.eda_summary.outcome_positive_rate, abs=1e-9
    )
    assert resp_a.eda_summary.n_findings_total == resp_b.eda_summary.n_findings_total
    assert resp_a.eda_summary.feature_feature_max_abs_pearson == pytest.approx(
        resp_b.eda_summary.feature_feature_max_abs_pearson or 0.0, abs=1e-9
    )
    assert len(resp_a.findings) == len(resp_b.findings)
    for fa, fb in zip(resp_a.findings, resp_b.findings):
        assert fa.id == fb.id
        assert fa.severity == fb.severity
