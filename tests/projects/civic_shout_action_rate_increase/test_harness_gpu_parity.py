from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from projects.civic_shout_action_rate_increase._gpu_bootstrap import (
    gpu_auc_batch,
    gpu_auc_single,
    gpu_user_block_bootstrap,
)
from projects.civic_shout_action_rate_increase.harness import (
    ML_Model_Config,
    ML_Model_Type,
    harness,
)
from projects.civic_shout_action_rate_increase.harness_gpu import harness as harness_gpu
from tests.projects.civic_shout_action_rate_increase._fixtures import (
    FEATURE_COLS,
    make_synthetic_with_interaction,
)

_MODEL_CONFIG = ML_Model_Config(
    ml_model_type=ML_Model_Type.LIGHTGBM_CLASSIFIER,
    args={"n_estimators": 30, "num_leaves": 15, "random_state": 42, "verbose": -1},
)


def test_gpu_auc_matches_sklearn() -> None:
    rng = np.random.default_rng(7)
    for n in (200, 1000, 5000):
        scores = rng.random(n).astype(np.float32)
        labels = rng.integers(0, 2, n)
        expected = roc_auc_score(labels, scores)
        got = gpu_auc_single(mx.array(scores), mx.array(labels.astype(np.float32)))
        assert abs(got - expected) < 1e-6, f"N={n}: sklearn={expected:.8f} gpu={got:.8f}"


def test_batched_matches_per_call() -> None:
    B, N = 50, 500
    rng = np.random.default_rng(13)
    scores = rng.random((B, N)).astype(np.float32)
    labels = rng.integers(0, 2, (B, N)).astype(np.float32)

    batch_result = np.array(gpu_auc_batch(mx.array(scores), mx.array(labels)).tolist())
    per_call = np.array(
        [gpu_auc_single(mx.array(scores[i]), mx.array(labels[i])) for i in range(B)]
    )
    assert np.abs(batch_result - per_call).max() < 1e-5


def test_gpu_bootstrap_determinism() -> None:
    rng = np.random.default_rng(99)
    n = 3000
    y = rng.integers(0, 2, n).astype(np.float32)
    s = rng.random(n).astype(np.float32)
    user_ids = rng.integers(0, 300, n)

    run1 = gpu_user_block_bootstrap(y, s, user_ids, n_boot=50, seed=42)
    run2 = gpu_user_block_bootstrap(y, s, user_ids, n_boot=50, seed=42)
    assert np.abs(run1 - run2).max() < 1e-3


@pytest.fixture(scope="module")
def cpu_response():
    df = make_synthetic_with_interaction(seed=42)
    return harness(df, FEATURE_COLS, _MODEL_CONFIG, sample_seed=42)


@pytest.fixture(scope="module")
def gpu_response():
    df = make_synthetic_with_interaction(seed=42)
    return harness_gpu(df, FEATURE_COLS, _MODEL_CONFIG, sample_seed=42)


def test_gpu_cpu_harness_parity(cpu_response, gpu_response) -> None:
    cpu_primary = cpu_response.summary.primary_metric_value
    gpu_primary = gpu_response.summary.primary_metric_value
    delta = abs(cpu_primary - gpu_primary)
    assert delta < 1e-3, (
        f"CPU primary={cpu_primary:.6f} GPU primary={gpu_primary:.6f} delta={delta:.2e} > 1e-3"
    )
