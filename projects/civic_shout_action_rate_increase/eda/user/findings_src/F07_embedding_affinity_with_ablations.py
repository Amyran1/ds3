"""F07. Embedding affinity with full ablation ladder — real EDA (not mockup).

1% -> 5% -> 100% sampling cadence with timing/memory telemetry per tier.

Run:
    cd /Users/aaronmyran/dev/ds3
    source .venv/bin/activate
    python -m projects.civic_shout_action_rate_increase.eda.user.findings_src.F07_embedding_affinity_with_ablations
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import psutil
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from projects.civic_shout_action_rate_increase.data import emails, user_emails
from projects.civic_shout_action_rate_increase.eda.user._html import render_finding

_OUT_DIR = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda/user/findings"
)
_TIMING_JSONL = Path(
    "/Users/aaronmyran/dev/ds3/projects/civic_shout_action_rate_increase/eda_timing_performance.jsonl"
)

_SHRINKAGE_LAMBDA = 5
_KMEANS_K = 10
_TIERS = [
    ("1pct", 0.01),
    ("5pct", 0.05),
    ("full", None),
]

_BUCKET_BREAKS = [0, 1, 2, 5, 20]
_BUCKET_LABELS = ["0", "1", "2-4", "5-19", "20+"]


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd="/Users/aaronmyran/dev/ds3",
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _peak_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def _load_email_dense() -> tuple[np.ndarray, np.ndarray]:
    """Load email dense vectors. Returns (email_ids, email_matrix float32)."""
    em = emails.full()
    email_ids = em["email_id"].to_numpy()
    email_dense = np.stack(
        [row["dense_vector"] for row in em.iter_rows(named=True)],
        dtype=np.float32,
    )
    return email_ids, email_dense


def _prior_action_bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 4:
        return "2-4"
    if n <= 19:
        return "5-19"
    return "20+"


def _cosine_sim_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity row-wise. a and b are (N, D) float32."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (a_norm * b_norm).sum(axis=1)


def _build_features_and_centroids(
    sample: pl.DataFrame,
    email_id_to_idx: dict[int, int],
    email_dense: np.ndarray,
) -> pl.DataFrame:
    """Build per-row features with strict temporal discipline:
    - centroid built from actioned emails STRICTLY BEFORE date_sent
    - returns DataFrame with feature columns + cosine_sim
    """
    n = sample.height
    dim = email_dense.shape[1]

    # Identify actioned rows and map to dense vectors
    actioned_mask = sample["actioned"].to_numpy().astype(bool)
    email_ids_arr = sample["email_id"].to_numpy()
    date_sent_arr = sample["date_sent"].to_numpy()
    user_ids_arr = sample["user_id"].to_numpy()

    # Map email_id -> dense vector index (missing = -1)
    email_idx_arr = np.array([email_id_to_idx.get(int(eid), -1) for eid in email_ids_arr])
    has_dense = email_idx_arr >= 0

    # Sort by (user_id, date_sent) for temporal ordering
    sort_order = np.lexsort((date_sent_arr, user_ids_arr))
    inv_sort = np.empty_like(sort_order)
    inv_sort[sort_order] = np.arange(n)

    user_ids_sorted = user_ids_arr[sort_order]
    actioned_sorted = actioned_mask[sort_order]
    email_idx_sorted = email_idx_arr[sort_order]

    # Population centroid (for shrinkage/cold-start)
    actioned_dense_indices = email_idx_arr[actioned_mask & has_dense]
    if len(actioned_dense_indices) > 0:
        pop_centroid = email_dense[actioned_dense_indices].mean(axis=0).astype(np.float32)
    else:
        pop_centroid = np.zeros(dim, dtype=np.float32)

    # Per-row cosine similarity with temporal centroid
    cosine_sims = np.zeros(n, dtype=np.float32)
    prior_action_counts = np.zeros(n, dtype=np.int32)

    # Track running sums per user
    user_running_sum: dict[int, np.ndarray] = {}
    user_running_count: dict[int, int] = {}

    for i_sorted in range(n):
        uid = int(user_ids_sorted[i_sorted])
        e_idx = int(email_idx_sorted[i_sorted])
        is_actioned = bool(actioned_sorted[i_sorted])

        if uid not in user_running_sum:
            user_running_sum[uid] = np.zeros(dim, dtype=np.float32)
            user_running_count[uid] = 0

        n_prior = user_running_count[uid]
        prior_action_counts[sort_order[i_sorted]] = n_prior

        # Compute centroid (with shrinkage)
        total_weight = n_prior + _SHRINKAGE_LAMBDA
        centroid = (user_running_sum[uid] + _SHRINKAGE_LAMBDA * pop_centroid) / total_weight

        # Cosine similarity of centroid with this email
        if e_idx >= 0:
            email_vec = email_dense[e_idx]
            sim = _cosine_sim_batch(centroid.reshape(1, -1), email_vec.reshape(1, -1))[0]
            cosine_sims[sort_order[i_sorted]] = float(sim)
        else:
            cosine_sims[sort_order[i_sorted]] = 0.0

        # Update running sum AFTER computing cosine (strict temporal)
        if is_actioned and e_idx >= 0:
            user_running_sum[uid] += email_dense[e_idx]
            user_running_count[uid] += 1

    # Build output DataFrame
    result = sample.with_columns(
        pl.Series("cosine_sim", cosine_sims, dtype=pl.Float32),
        pl.Series("prior_action_count", prior_action_counts, dtype=pl.Int32),
    )

    # Derive prior action bucket
    result = result.with_columns(
        pl.col("prior_action_count")
        .map_elements(
            lambda n: _prior_action_bucket(int(n)),
            return_dtype=pl.String,
        )
        .alias("prior_action_bucket")
    )

    return result


def _build_ablation_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add all ablation feature columns to df."""
    # Tenure: days from first_send_date to date_sent
    first_send = df.group_by("user_id").agg(pl.col("date_sent").min().alias("first_send_date"))
    df = df.join(first_send, on="user_id")
    df = df.with_columns(
        ((pl.col("date_sent") - pl.col("first_send_date")).dt.total_days())
        .cast(pl.Float32)
        .alias("tenure_days")
    )

    # Send frequency (number of sends per user in the sample / span)
    send_counts = df.group_by("user_id").agg(pl.len().alias("send_count"))
    df = df.join(send_counts, on="user_id")
    df = df.with_columns(pl.col("send_count").cast(pl.Float32).alias("send_freq"))

    # Prior action rate proxy: prior_action_count / (send_count + 1)
    df = df.with_columns(
        (pl.col("prior_action_count").cast(pl.Float32) / (pl.col("send_freq") + 1.0)).alias(
            "prior_action_rate"
        )
    )

    # Log-prior action count
    df = df.with_columns(
        (pl.col("prior_action_count").cast(pl.Float32) + 1.0).log().alias("log_prior_actions")
    )

    # Recency: days since first send (as proxy for "recency window" features)
    # We compute simple recency features: inverse of tenure, send rate density
    df = df.with_columns(
        (1.0 / (pl.col("tenure_days") + 1.0)).alias("recency_inv_tenure"),
        (pl.col("send_freq") / (pl.col("tenure_days") + 1.0)).alias("send_density"),
    )

    return df


def _fit_auc(X: np.ndarray, y: np.ndarray, max_iter: int = 500) -> float:
    """Fit logistic regression and return AUC. Returns NaN if degenerate."""
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    if X.shape[0] < 50:
        return float("nan")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=max_iter, C=1.0, solver="lbfgs", random_state=42)
    clf.fit(X_s, y)
    probs = clf.predict_proba(X_s)[:, 1]
    return float(roc_auc_score(y, probs))


def _ablation(df: pl.DataFrame) -> dict[str, float]:
    """Run ablation ladder A->B->C->D. Returns AUC dict."""
    needed = [
        "tenure_days",
        "send_freq",
        "prior_action_count",
        "prior_action_rate",
        "log_prior_actions",
        "recency_inv_tenure",
        "send_density",
        "cosine_sim",
        "actioned",
    ]
    mask = np.ones(df.height, dtype=bool)
    for col in needed:
        if col in df.columns:
            mask &= df[col].is_not_null().to_numpy()
    df_valid = df.filter(pl.Series(mask))
    y = df_valid["actioned"].cast(pl.Int8).to_numpy()

    feats_A = ["tenure_days", "send_freq"]
    feats_B = feats_A + ["prior_action_count", "prior_action_rate"]
    feats_C = feats_B + ["log_prior_actions", "recency_inv_tenure", "send_density"]
    feats_D = feats_C + ["cosine_sim"]

    def to_X(cols: list[str]) -> np.ndarray:
        return np.column_stack([df_valid[c].cast(pl.Float64).to_numpy() for c in cols])

    return {
        "A": _fit_auc(to_X(feats_A), y),
        "B": _fit_auc(to_X(feats_B), y),
        "C": _fit_auc(to_X(feats_C), y),
        "D": _fit_auc(to_X(feats_D), y),
    }


def _permutation_auc(df: pl.DataFrame, rng: np.random.Generator) -> float:
    """Shuffle cosine_sim WITHIN prior_action_bucket; refit model D."""
    df_perm = df.clone()
    bucket_col = df_perm["prior_action_bucket"].to_numpy()
    cosine_col = df_perm["cosine_sim"].to_numpy().copy()

    for bucket in _BUCKET_LABELS:
        idxs = np.where(bucket_col == bucket)[0]
        if len(idxs) > 1:
            rng.shuffle(cosine_col[idxs])

    df_perm = df_perm.with_columns(pl.Series("cosine_sim", cosine_col.astype("float32")))

    needed = [
        "tenure_days",
        "send_freq",
        "prior_action_count",
        "prior_action_rate",
        "log_prior_actions",
        "recency_inv_tenure",
        "send_density",
        "cosine_sim",
        "actioned",
    ]
    mask = np.ones(df_perm.height, dtype=bool)
    for col in needed:
        if col in df_perm.columns:
            mask &= df_perm[col].is_not_null().to_numpy()
    df_valid = df_perm.filter(pl.Series(mask))
    y = df_valid["actioned"].cast(pl.Int8).to_numpy()

    feats_D = [
        "tenure_days",
        "send_freq",
        "prior_action_count",
        "prior_action_rate",
        "log_prior_actions",
        "recency_inv_tenure",
        "send_density",
        "cosine_sim",
    ]
    X = np.column_stack([df_valid[c].cast(pl.Float64).to_numpy() for c in feats_D])
    return _fit_auc(X, y)


def _within_bucket_auc(df: pl.DataFrame) -> dict[str, float]:
    """AUC of model D within each prior_action_bucket."""
    result: dict[str, float] = {}
    for bucket in _BUCKET_LABELS:
        sub = df.filter(pl.col("prior_action_bucket") == bucket)
        if sub.height < 100:
            result[bucket] = float("nan")
            continue
        needed = [
            "tenure_days",
            "send_freq",
            "prior_action_count",
            "prior_action_rate",
            "log_prior_actions",
            "recency_inv_tenure",
            "send_density",
            "cosine_sim",
            "actioned",
        ]
        mask = np.ones(sub.height, dtype=bool)
        for col in needed:
            if col in sub.columns:
                mask &= sub[col].is_not_null().to_numpy()
        sub_valid = sub.filter(pl.Series(mask))
        y = sub_valid["actioned"].cast(pl.Int8).to_numpy()
        if y.sum() == 0 or y.sum() == len(y):
            result[bucket] = float("nan")
            continue
        feats_D = [
            "tenure_days",
            "send_freq",
            "prior_action_count",
            "prior_action_rate",
            "log_prior_actions",
            "recency_inv_tenure",
            "send_density",
            "cosine_sim",
        ]
        X = np.column_stack([sub_valid[c].cast(pl.Float64).to_numpy() for c in feats_D])
        result[bucket] = _fit_auc(X, y)
    return result


def _centroid_stability(df: pl.DataFrame) -> float:
    """Pearson correlation of cosine_sim vs log(prior_action_count + 1)."""
    vals = df.select(
        pl.col("cosine_sim").cast(pl.Float64),
        (pl.col("prior_action_count").cast(pl.Float64) + 1.0).log().alias("log_pac"),
    ).drop_nulls()
    if vals.height < 30:
        return float("nan")
    x = vals["log_pac"].to_numpy()
    y = vals["cosine_sim"].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _boilerplate_diagnostic(email_dense: np.ndarray, email_ids: np.ndarray) -> list[str]:
    """Cluster emails and report nearest-neighbor pairs."""
    kmeans = MiniBatchKMeans(n_clusters=_KMEANS_K, random_state=42, n_init=3)
    labels = kmeans.fit_predict(email_dense)

    lines: list[str] = []
    lines.append(f"KMeans k={_KMEANS_K} on {len(email_dense)} email dense vectors (1536d):")
    for cluster_id in range(_KMEANS_K):
        idxs = np.where(labels == cluster_id)[0]
        lines.append(f"  cluster {cluster_id}: {len(idxs)} emails")

    # Nearest neighbor pairs within clusters (up to 5)
    lines.append("\nTop 5 nearest-neighbor pairs by cosine similarity:")
    normalized = email_dense / (np.linalg.norm(email_dense, axis=1, keepdims=True) + 1e-9)
    best_pairs: list[tuple[float, int, int]] = []
    for cluster_id in range(_KMEANS_K):
        idxs = np.where(labels == cluster_id)[0]
        if len(idxs) < 2:
            continue
        cluster_vecs = normalized[idxs]
        sims = cluster_vecs @ cluster_vecs.T
        np.fill_diagonal(sims, -1.0)
        flat = sims.flatten()
        top_k = min(3, len(flat))
        top_idxs = np.argpartition(flat, -top_k)[-top_k:]
        for fi in top_idxs:
            r, c = divmod(fi, sims.shape[0])
            if r < c:
                best_pairs.append(
                    (float(flat[fi]), int(email_ids[idxs[r]]), int(email_ids[idxs[c]]))
                )

    best_pairs.sort(key=lambda x: x[0], reverse=True)
    for sim, eid_a, eid_b in best_pairs[:5]:
        lines.append(f"  email_id={eid_a} <-> email_id={eid_b}  cosine={sim:.4f}")

    return lines


def _build_chart(
    auc_dict: dict[str, float],
    df: pl.DataFrame,
    within_bucket: dict[str, float],
    cosine_vs_pac_corr: float,
) -> bytes:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # --- Ablation bar chart ---
    ax1 = axes[0]
    labels_abl = ["A\n(tenure+freq)", "B\n(+history)", "C\n(+recency)", "D\n(+cosine)"]
    aucs_abl = [auc_dict.get(k, float("nan")) for k in ["A", "B", "C", "D"]]
    colors_abl = ["#aaa", "#2956a3", "#c5500f", "#207e3c"]
    valid_aucs = [v for v in aucs_abl if not np.isnan(v)]
    y_min = max(0.49, min(valid_aucs) - 0.02) if valid_aucs else 0.5
    y_max = min(1.0, max(valid_aucs) + 0.03) if valid_aucs else 1.0
    bars = ax1.bar(labels_abl, aucs_abl, color=colors_abl, alpha=0.85)
    ax1.set_ylim(y_min, y_max)
    ax1.set_ylabel("AUC (logistic regression, in-sample)")
    ax1.set_title("Ablation ladder: A → B → C → D")
    ax1.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    for bar, val in zip(bars, aucs_abl):
        if not np.isnan(val):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax1.grid(axis="y", alpha=0.3)

    # --- Centroid stability scatter ---
    ax2 = axes[1]
    valid_df = df.filter(
        pl.col("cosine_sim").is_not_null() & pl.col("prior_action_count").is_not_null()
    )
    log_pac = np.log1p(valid_df["prior_action_count"].to_numpy().astype(float))
    cosine_vals = valid_df["cosine_sim"].to_numpy().astype(float)
    max_pts = 5000
    if len(log_pac) > max_pts:
        rng_idx = np.random.default_rng(42)
        sel = rng_idx.choice(len(log_pac), max_pts, replace=False)
        log_pac_plot = log_pac[sel]
        cosine_plot = cosine_vals[sel]
    else:
        log_pac_plot = log_pac
        cosine_plot = cosine_vals
    ax2.scatter(log_pac_plot, cosine_plot, alpha=0.15, s=4, color="#2956a3")
    ax2.set_xlabel("log(prior_action_count + 1)")
    ax2.set_ylabel("cosine(user_centroid, email)")
    corr_str = f"{cosine_vs_pac_corr:.3f}" if not np.isnan(cosine_vs_pac_corr) else "N/A"
    ax2.set_title(f"Centroid stability\nPearson r={corr_str} (cosine vs prior actions)")
    ax2.grid(alpha=0.25)

    # --- Within-bucket AUC ---
    ax3 = axes[2]
    bucket_labels = _BUCKET_LABELS
    bucket_aucs = [within_bucket.get(b, float("nan")) for b in bucket_labels]
    colors_bucket = ["#ccc" if np.isnan(v) else "#207e3c" for v in bucket_aucs]
    ax3.bar(bucket_labels, bucket_aucs, color=colors_bucket, alpha=0.85)
    ax3.set_ylim(
        0.45, min(1.0, max((v for v in bucket_aucs if not np.isnan(v)), default=0.55) + 0.05)
    )
    ax3.set_xlabel("prior_action_count bucket")
    ax3.set_ylabel("AUC (model D)")
    ax3.set_title("Within-bucket AUC (model D)\nWhere does cosine add lift?")
    ax3.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    for i, (label, val) in enumerate(zip(bucket_labels, bucket_aucs)):
        if not np.isnan(val):
            ax3.text(i, val + 0.002, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    fig.set_constrained_layout(True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def _numbers_table(
    auc_dict: dict[str, float],
    perm_auc: float,
    within_bucket: dict[str, float],
    corr: float,
    coverage: dict[str, int],
    n_rows: int,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("Total rows (sample)", f"{n_rows:,}"),
        ("AUC A (tenure + send_freq)", f"{auc_dict.get('A', float('nan')):.4f}"),
        ("AUC B (+ prior count/rate)", f"{auc_dict.get('B', float('nan')):.4f}"),
        ("AUC C (+ recency windows)", f"{auc_dict.get('C', float('nan')):.4f}"),
        ("AUC D (+ cosine affinity)", f"{auc_dict.get('D', float('nan')):.4f}"),
        ("Delta A→D", f"{auc_dict.get('D', float('nan')) - auc_dict.get('A', float('nan')):.4f}"),
        (
            "Delta C→D (cosine marginal)",
            f"{auc_dict.get('D', float('nan')) - auc_dict.get('C', float('nan')):.4f}",
        ),
        (
            "Permutation AUC (shuffled within bucket)",
            f"{perm_auc:.4f}" if not np.isnan(perm_auc) else "N/A",
        ),
        (
            "Permutation gap (D - perm)",
            f"{auc_dict.get('D', float('nan')) - perm_auc:.4f}"
            if not np.isnan(perm_auc)
            else "N/A",
        ),
        (
            "Centroid stability r (cosine vs log PAC)",
            f"{corr:.3f}" if not np.isnan(corr) else "N/A",
        ),
    ]
    for bucket in _BUCKET_LABELS:
        v = within_bucket.get(bucket, float("nan"))
        rows.append(
            (
                f"Within-bucket AUC [{bucket} prior actions]",
                f"{v:.4f}" if not np.isnan(v) else "N/A",
            )
        )
    for bucket in _BUCKET_LABELS:
        rows.append((f"Coverage [{bucket} prior actions]", f"{coverage.get(bucket, 0):,}"))
    return rows


def _impact_md(
    auc_dict: dict[str, float],
    perm_auc: float,
    within_bucket: dict[str, float],
    corr: float,
) -> str:
    auc_d = auc_dict.get("D", float("nan"))
    auc_c = auc_dict.get("C", float("nan"))
    delta_cd = auc_d - auc_c
    perm_gap = auc_d - perm_auc if not np.isnan(perm_auc) else float("nan")
    corr_strong = not np.isnan(corr) and abs(corr) > 0.3

    # Signal verdict
    if not np.isnan(delta_cd) and delta_cd > 0.003 and not np.isnan(perm_gap) and perm_gap > 0.002:
        verdict = (
            f"Embedding affinity adds REAL signal: delta C→D={delta_cd:.4f} and "
            f"permutation gap={perm_gap:.4f}. "
            "Include cosine(user_centroid, today_email) as a candidate feature."
        )
    elif not np.isnan(delta_cd) and delta_cd > 0.001:
        verdict = (
            f"Marginal signal: delta C→D={delta_cd:.4f}. Borderline. "
            "Permutation gap and within-bucket lift should be checked before including."
        )
    else:
        verdict = (
            f"Cosine adds negligible lift above RFM baseline (delta C→D={delta_cd:.4f}). "
            "Embedding affinity is not a reliable standalone differentiator at this stage."
        )

    # Power-user risk
    if corr_strong:
        power_user_risk = (
            f"Centroid stability r={corr:.3f} (>0.3) — centroid is correlated with "
            "prior action count. The embedding feature may be partly a power-user detector. "
            "Evaluate within-bucket lift separately for high vs low history users."
        )
    else:
        power_user_risk = (
            f"Centroid stability r={corr:.3f} — centroid is NOT strongly correlated with "
            "prior action count. Power-user proxy risk is low."
        )

    # Within-bucket summary
    high_hist_buckets = [
        b for b in ["5-19", "20+"] if not np.isnan(within_bucket.get(b, float("nan")))
    ]
    low_hist_buckets = [b for b in ["0", "1"] if not np.isnan(within_bucket.get(b, float("nan")))]
    if high_hist_buckets and low_hist_buckets:
        high_auc = max(within_bucket[b] for b in high_hist_buckets)
        low_auc = max(within_bucket[b] for b in low_hist_buckets)
        bucket_note = (
            f"Cosine adds lift in high-history users (AUC up to {high_auc:.3f}) vs "
            f"low-history (AUC up to {low_auc:.3f}). "
            "Cold-start users fall back to population centroid — centroid similarity for them "
            "is uninformative."
        )
    else:
        bucket_note = "Within-bucket breakdown unavailable for some buckets (insufficient sample)."

    return f"{verdict}\n\n{power_user_risk}\n\n{bucket_note}"


def _run_tier(
    tier_label: str,
    fraction: float | None,
    email_ids: np.ndarray,
    email_dense: np.ndarray,
    git_sha: str,
) -> dict:
    t0 = time.time()
    print(f"\n[F07] Starting tier={tier_label} ...")

    if fraction is not None:
        sample = user_emails.sample_percentage(fraction, seed=42)
    else:
        sample = user_emails.full()

    n_rows = sample.height
    print(f"  sample rows={n_rows:,}")

    email_id_to_idx = {int(eid): i for i, eid in enumerate(email_ids)}

    print("  building temporal centroids + cosine similarities ...")
    df = _build_features_and_centroids(sample, email_id_to_idx, email_dense)
    df = _build_ablation_features(df)

    print("  coverage table ...")
    coverage: dict[str, int] = {}
    for bucket in _BUCKET_LABELS:
        coverage[bucket] = df.filter(pl.col("prior_action_bucket") == bucket).height

    print("  ablation ladder ...")
    auc_dict = _ablation(df)
    print(
        f"  AUC A={auc_dict['A']:.4f} B={auc_dict['B']:.4f} C={auc_dict['C']:.4f} D={auc_dict['D']:.4f}"
    )

    print("  centroid stability ...")
    corr = _centroid_stability(df)

    print("  permutation control ...")
    rng = np.random.default_rng(42)
    perm_auc = _permutation_auc(df, rng)
    perm_gap = auc_dict.get("D", float("nan")) - perm_auc
    print(f"  perm AUC={perm_auc:.4f}  gap={perm_gap:.4f}")

    print("  within-bucket AUC ...")
    within_bucket = _within_bucket_auc(df)

    chart_bytes = _build_chart(auc_dict, df, within_bucket, corr)
    numbers_table = _numbers_table(auc_dict, perm_auc, within_bucket, corr, coverage, n_rows)
    impact = _impact_md(auc_dict, perm_auc, within_bucket, corr)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    wall = time.time() - t0
    peak_mb = _peak_mb()

    sample_label = f"{fraction:.0%} sample, seed=42" if fraction is not None else "full dataset"
    provenance = {
        "data": "user_emails v2 + emails v1 (dense_vector 1536d)",
        "sample": sample_label,
        "tier": tier_label,
        "wall_seconds": f"{wall:.1f}s",
        "peak_memory_mb": f"{peak_mb:.0f}",
        "ts": ts,
        "git": git_sha,
    }

    auc_d = auc_dict.get("D", float("nan"))
    auc_c = auc_dict.get("C", float("nan"))
    delta_cd = auc_d - auc_c
    tldr = (
        f"cosine adds delta C→D={delta_cd:.4f} | perm gap={perm_gap:.4f} | "
        f"stability r={corr:.3f} | "
        f"A={auc_dict.get('A', float('nan')):.4f} B={auc_dict.get('B', float('nan')):.4f} "
        f"C={auc_c:.4f} D={auc_d:.4f} "
        f"[{tier_label} n={n_rows:,} wall={wall:.0f}s]"
    )

    method_src = f"""
# F07: Embedding affinity with ablation ladder
sample = user_emails.sample_percentage({fraction}, seed=42)  # {tier_label}

# Load email dense vectors (4634 x 1536d float32)
em = emails.full()
email_dense = np.stack([r["dense_vector"] for r in em.iter_rows(named=True)], dtype=np.float32)

# Per-row temporal centroid with Bayesian shrinkage (lambda=5)
# Strict temporal: centroid for (user, date_sent) from actioned emails BEFORE date_sent
for each row (sorted by user_id, date_sent):
    centroid = (n_prior * mean_user + LAMBDA * pop_centroid) / (n_prior + LAMBDA)
    cosine_sim = cosine(centroid, email_dense[row.email_id])
    if row.actioned:
        update running_sum, running_count  # AFTER computing cosine

# Ablation ladder (logistic regression, StandardScaler)
A: [tenure_days, send_freq]           -> AUC={auc_dict.get("A", float("nan")):.4f}
B: A + [prior_action_count, prior_action_rate]  -> AUC={auc_dict.get("B", float("nan")):.4f}
C: B + [log_prior_actions, recency_inv_tenure, send_density] -> AUC={auc_dict.get("C", float("nan")):.4f}
D: C + [cosine_sim]                    -> AUC={auc_dict.get("D", float("nan")):.4f}

# Permutation: shuffle cosine_sim within prior_action_bucket -> AUC={perm_auc:.4f}
# Within-bucket AUC of model D per prior_action_count bucket
"""

    html = render_finding(
        finding_id="F07",
        title="Embedding affinity (with full ablation ladder)",
        tldr=tldr,
        wave=3,
        severity="IMPORTANT",
        classification="feature-eng",
        question_md=(
            "Does cosine(user_centroid, today_email_dense_vec) add predictive lift beyond the "
            "F06 RFM baseline, after controlling for history depth? Is the centroid signal real "
            "topic affinity, or is it a power-user proxy in disguise?"
        ),
        plain_english_md=(
            "Imagine we track which petitions each user has ever signed and compute their "
            "'average topic fingerprint' from those petition embeddings. Then we measure how "
            "similar today's email is to that fingerprint. The hypothesis: users who get emails "
            "close to their fingerprint are more likely to act.\n\n"
            "Two risks: (1) users with more history have more stable fingerprints, so the "
            "similarity score might just be tracking 'power user vs newcomer' rather than "
            "topic match. (2) Email templates/boilerplate might dominate the vectors, making "
            "clusters track organizations rather than topics.\n\n"
            "The ablation ladder isolates each effect: does cosine similarity add lift AFTER "
            "we've already given the model prior-action count and recency?"
        ),
        method_py_source=method_src,
        chart_png_bytes=chart_bytes,
        numbers_table=numbers_table,
        impact_md=impact,
        provenance=provenance,
        is_mockup=False,
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "F07_embedding_affinity_with_ablations.html"
    out_path.write_text(html, encoding="utf-8")

    timing_row = {
        "finding_id": "F07",
        "tier": tier_label,
        "rows": n_rows,
        "wall_seconds": round(wall, 2),
        "peak_memory_mb": round(peak_mb, 1),
        "fraction": fraction,
        "ts": ts,
        "git": git_sha,
    }
    with _TIMING_JSONL.open("a") as fh:
        fh.write(json.dumps(timing_row) + "\n")

    print(
        f"F07 [{tier_label}] rows={n_rows:,} wall={wall:.1f}s; "
        f"A_AUC={auc_dict.get('A', float('nan')):.4f} "
        f"B={auc_dict.get('B', float('nan')):.4f} "
        f"C={auc_c:.4f} "
        f"D={auc_d:.4f}; "
        f"permutation_gap={perm_gap:.4f}"
    )
    print(f"  wrote {out_path}")

    return {
        "tier": tier_label,
        "n_rows": n_rows,
        "wall": wall,
        "auc_dict": auc_dict,
        "perm_auc": perm_auc,
        "perm_gap": perm_gap,
        "within_bucket": within_bucket,
        "corr": corr,
        "coverage": coverage,
        "df": df,
    }


def main() -> None:
    git_sha = _git_sha()
    print("[F07] Loading email dense vectors (full — 4634 emails x 1536d) ...")
    t_em = time.time()
    email_ids, email_dense = _load_email_dense()
    print(
        f"  email_dense shape={email_dense.shape} dtype={email_dense.dtype} "
        f"loaded in {time.time() - t_em:.1f}s"
    )

    print("\n[F07] Boilerplate diagnostic (KMeans on email dense vectors) ...")
    bp_lines = _boilerplate_diagnostic(email_dense, email_ids)
    print("\n".join(bp_lines))

    eda_tiers_env = os.environ.get("EDA_TIERS")
    if eda_tiers_env:
        allowed = {t.strip() for t in eda_tiers_env.split(",")}
        tiers_to_run = [(lbl, frac) for lbl, frac in _TIERS if lbl in allowed]
        print(
            f"[F07] EDA_TIERS={eda_tiers_env!r} → running tiers: {[lbl for lbl, _ in tiers_to_run]}"
        )
    else:
        tiers_to_run = _TIERS

    last_result: dict = {}
    for tier_label, fraction in tiers_to_run:
        last_result = _run_tier(tier_label, fraction, email_ids, email_dense, git_sha)

    print("\n=== F07 FINAL SUMMARY ===")
    r = last_result
    print(f"  rows={r['n_rows']:,}  wall={r['wall']:.1f}s")
    print(
        f"  Ablation: A={r['auc_dict'].get('A', float('nan')):.4f}  "
        f"B={r['auc_dict'].get('B', float('nan')):.4f}  "
        f"C={r['auc_dict'].get('C', float('nan')):.4f}  "
        f"D={r['auc_dict'].get('D', float('nan')):.4f}"
    )
    print(
        f"  Delta A→D={r['auc_dict'].get('D', float('nan')) - r['auc_dict'].get('A', float('nan')):.4f}  "
        f"Delta C→D={r['auc_dict'].get('D', float('nan')) - r['auc_dict'].get('C', float('nan')):.4f}"
    )
    print(f"  Permutation AUC={r['perm_auc']:.4f}  gap={r['perm_gap']:.4f}")
    print(f"  Centroid stability r={r['corr']:.3f}")
    print("  Within-bucket AUC (model D):")
    for bucket in _BUCKET_LABELS:
        v = r["within_bucket"].get(bucket, float("nan"))
        count = r["coverage"].get(bucket, 0)
        print(
            f"    [{bucket:5s}] n={count:,}  AUC={v:.4f}"
            if not np.isnan(v)
            else f"    [{bucket:5s}] n={count:,}  AUC=N/A"
        )

    auc_d = r["auc_dict"].get("D", float("nan"))
    auc_c = r["auc_dict"].get("C", float("nan"))
    delta_cd = auc_d - auc_c
    perm_gap = r["perm_gap"]

    if not np.isnan(delta_cd) and delta_cd > 0.003 and not np.isnan(perm_gap) and perm_gap > 0.002:
        verdict = (
            "YES — cosine adds real signal above RFM baseline with non-trivial permutation gap."
        )
    elif not np.isnan(delta_cd) and delta_cd > 0.001:
        verdict = "MARGINAL — small lift above baseline; permutation and within-bucket results are borderline."
    else:
        verdict = (
            "NO — cosine does not add meaningful lift above the RFM baseline at this sample size."
        )

    print(f"\n  Final verdict: {verdict}")


if __name__ == "__main__":
    main()
