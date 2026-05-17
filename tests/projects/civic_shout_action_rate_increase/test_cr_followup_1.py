import pytest

from projects.civic_shout_action_rate_increase.harness import (
    _LGBM_ONLY_ARGS,
    HarnessArtifact,
    _build_lr_args,
)


def test_harness_artifact_accepts_byte_size():
    a = HarnessArtifact(
        name="predictions", kind="predictions", uri="file:///tmp/foo.parquet", byte_size=12345
    )
    assert a.byte_size == 12345


def test_harness_artifact_byte_size_defaults_none():
    a = HarnessArtifact(name="x", kind="other", uri="file:///tmp/x")
    assert a.byte_size is None


def test_lgbm_only_args_is_frozenset():
    assert isinstance(_LGBM_ONLY_ARGS, frozenset)
    assert "num_leaves" in _LGBM_ONLY_ARGS
    assert "C" not in _LGBM_ONLY_ARGS  # LR-compatible


def test_build_lr_args_strips_lgbm_keys():
    base = {"penalty": "l2", "C": 1.0, "num_leaves": 31, "force_col_wise": True}
    out = _build_lr_args(base, seed=42)
    assert "num_leaves" not in out
    assert "force_col_wise" not in out
    assert out["C"] == 1.0
    assert out["random_state"] == 42


def test_build_lr_args_resolves_l2_to_newton_cholesky():
    out = _build_lr_args({"penalty": "l2", "C": 1.0}, seed=42)
    # On sklearn >=1.4 should be newton-cholesky; otherwise lbfgs
    assert out["solver"] in {"newton-cholesky", "lbfgs"}


def test_build_lr_args_resolves_l1_to_liblinear():
    out = _build_lr_args({"penalty": "l1", "C": 0.1}, seed=42)
    assert out["solver"] == "liblinear"


def test_build_lr_args_resolves_elasticnet_to_saga():
    out = _build_lr_args({"penalty": "elasticnet", "C": 1.0}, seed=42)
    assert out["solver"] == "saga"
    assert out["l1_ratio"] == 0.5
