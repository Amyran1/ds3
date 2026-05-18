from __future__ import annotations

import polars as pl
import pytest

from libs.responses import RunResultMetadata
from libs.run_record import run_record

_MINIMAL_METADATA = RunResultMetadata(
    run_id="run_kwtest",
    project="test_project",
    comparison_group="g",
    scope="smoke",
    recorded_at_utc="2026-01-01T00:00:00+00:00",
)

_MINIMAL_DF = pl.DataFrame({"y": [0, 1]})

_MINIMAL_MODEL_CFG: dict = {}


def _make_fake_harness(captured: dict, sentinel: str = "captured-and-stop"):
    def fake_harness(**kwargs):
        captured.update(kwargs)
        raise RuntimeError(sentinel)

    return fake_harness


def _patch_project_modules(monkeypatch, fake_harness):
    import types

    import libs.run_record as rr_mod

    original_import = rr_mod.importlib.import_module

    def patched_import(name, *args, **kwargs):
        if name == "projects.test_project.harness":
            mod = types.ModuleType(name)
            mod.harness = fake_harness  # type: ignore[attr-defined]
            return mod
        if name == "projects.test_project.eda":
            from libs.responses import EDAConfig, EDAPerformance, EDAResponse, EDAScores

            def fake_eda(**kwargs):
                return EDAResponse(
                    findings=[],
                    scores=EDAScores(drift_score=0.0, n_warnings=0, n_errors=0),
                    performance=EDAPerformance(total_seconds=0.0, method_seconds={}),
                    outcome_prevalence=None,
                )

            mod = types.ModuleType(name)
            mod.eda = fake_eda  # type: ignore[attr-defined]
            return mod
        if name == "projects.test_project.discovery":
            mod = types.ModuleType(name)
            return mod
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(rr_mod.importlib, "import_module", patched_import)


def test_harness_kwargs_forwards_through(monkeypatch):
    captured: dict = {}
    _patch_project_modules(monkeypatch, _make_fake_harness(captured))

    with pytest.raises(RuntimeError, match="captured-and-stop"):
        run_record(
            metadata=_MINIMAL_METADATA,
            data=_MINIMAL_DF,
            feature_cols=[],
            outcome_variable="y",
            model_cfg=_MINIMAL_MODEL_CFG,
            harness_kwargs={"scope": "comparison_fast", "bootstrap_n_resamples": 100},
        )

    assert captured.get("scope") == "comparison_fast"
    assert captured.get("bootstrap_n_resamples") == 100


@pytest.mark.parametrize(
    "reserved_key",
    [
        "data",
        "feature_cols",
        "outcome_variable",
        "ml_model_config",
        "sample_frac",
        "sample_seed",
        "predictions_dir",
    ],
)
def test_harness_kwargs_rejects_reserved_keys(reserved_key):
    with pytest.raises(ValueError, match="reserved"):
        run_record(
            metadata=_MINIMAL_METADATA,
            data=_MINIMAL_DF,
            feature_cols=[],
            outcome_variable="y",
            model_cfg=_MINIMAL_MODEL_CFG,
            harness_kwargs={reserved_key: "should-be-blocked"},
        )


def test_harness_kwargs_none_default_no_extra_kwargs(monkeypatch):
    captured: dict = {}
    _patch_project_modules(monkeypatch, _make_fake_harness(captured))

    with pytest.raises(RuntimeError, match="captured-and-stop"):
        run_record(
            metadata=_MINIMAL_METADATA,
            data=_MINIMAL_DF,
            feature_cols=[],
            outcome_variable="y",
            model_cfg=_MINIMAL_MODEL_CFG,
        )

    assert "scope" not in captured
    assert "bootstrap_n_resamples" not in captured
    assert "data" in captured
    assert "feature_cols" in captured
    assert "sample_frac" not in captured
    assert "sample_seed" not in captured


def test_harness_kwargs_empty_dict_treated_as_none(monkeypatch):
    """harness_kwargs={} (explicit empty) behaves identically to harness_kwargs=None."""
    captured: dict = {}
    _patch_project_modules(monkeypatch, _make_fake_harness(captured))

    with pytest.raises(RuntimeError, match="captured-and-stop"):
        run_record(
            metadata=_MINIMAL_METADATA,
            data=_MINIMAL_DF,
            feature_cols=[],
            outcome_variable="y",
            model_cfg=_MINIMAL_MODEL_CFG,
            harness_kwargs={},
        )

    assert "scope" not in captured
    assert "bootstrap_n_resamples" not in captured


def test_harness_kwargs_unknown_kwarg_propagates_as_error(monkeypatch):
    """Unknown kwarg in harness_kwargs reaches harness and causes TypeError (not silent drop)."""

    def real_harness(
        *,
        data,
        feature_cols,
        outcome_variable,
        ml_model_config,
        sample_frac=None,
        predictions_dir=None,
        scope="comparison",
    ):
        raise RuntimeError("should not reach")

    _patch_project_modules(monkeypatch, real_harness)

    with pytest.raises(TypeError):
        run_record(
            metadata=_MINIMAL_METADATA,
            data=_MINIMAL_DF,
            feature_cols=[],
            outcome_variable="y",
            model_cfg=_MINIMAL_MODEL_CFG,
            harness_kwargs={"nonexistent_arg": 1},
        )


def test_sample_frac_forwarded_to_harness(monkeypatch):
    captured: dict = {}
    _patch_project_modules(monkeypatch, _make_fake_harness(captured))

    with pytest.raises(RuntimeError, match="captured-and-stop"):
        run_record(
            metadata=_MINIMAL_METADATA,
            data=_MINIMAL_DF,
            feature_cols=[],
            outcome_variable="y",
            model_cfg=_MINIMAL_MODEL_CFG,
            sample_frac=0.1,
            sample_seed=42,
        )

    assert captured.get("sample_frac") == 0.1
    assert captured.get("sample_seed") == 42


def test_sample_frac_alone_no_seed_omits_seed_kwarg(monkeypatch):
    captured: dict = {}
    _patch_project_modules(monkeypatch, _make_fake_harness(captured))

    with pytest.raises(RuntimeError, match="captured-and-stop"):
        run_record(
            metadata=_MINIMAL_METADATA,
            data=_MINIMAL_DF,
            feature_cols=[],
            outcome_variable="y",
            model_cfg=_MINIMAL_MODEL_CFG,
            sample_frac=0.1,
            sample_seed=None,
        )

    assert captured.get("sample_frac") == 0.1
    assert "sample_seed" not in captured


def test_sample_seed_alone_no_frac_omits_frac_kwarg(monkeypatch):
    captured: dict = {}
    _patch_project_modules(monkeypatch, _make_fake_harness(captured))

    with pytest.raises(RuntimeError, match="captured-and-stop"):
        run_record(
            metadata=_MINIMAL_METADATA,
            data=_MINIMAL_DF,
            feature_cols=[],
            outcome_variable="y",
            model_cfg=_MINIMAL_MODEL_CFG,
            sample_frac=None,
            sample_seed=42,
        )

    assert "sample_frac" not in captured
    assert captured.get("sample_seed") == 42


@pytest.mark.parametrize(
    "collision_key",
    ["sample_frac", "sample_seed"],
)
def test_sample_frac_collision_with_harness_kwargs_raises(collision_key):
    with pytest.raises(ValueError, match="reserved"):
        run_record(
            metadata=_MINIMAL_METADATA,
            data=_MINIMAL_DF,
            feature_cols=[],
            outcome_variable="y",
            model_cfg=_MINIMAL_MODEL_CFG,
            sample_frac=0.1,
            harness_kwargs={collision_key: 0.2},
        )


def test_sample_seed_none_safe_for_harness_without_sample_seed(monkeypatch):
    # Regression: california_housing_demo's harness lacks sample_seed in its
    # signature. Unconditional sample_seed=None forwarding broke it with
    # TypeError. Conditional forwarding must reach harness without sample_seed.
    def strict_harness(
        *,
        data,
        feature_cols,
        outcome_variable,
        ml_model_config,
        sample_frac=None,
        predictions_dir=None,
    ):
        raise RuntimeError("reached-harness")

    _patch_project_modules(monkeypatch, strict_harness)

    with pytest.raises(RuntimeError, match="reached-harness"):
        run_record(
            metadata=_MINIMAL_METADATA,
            data=_MINIMAL_DF,
            feature_cols=[],
            outcome_variable="y",
            model_cfg=_MINIMAL_MODEL_CFG,
            sample_seed=None,
        )
