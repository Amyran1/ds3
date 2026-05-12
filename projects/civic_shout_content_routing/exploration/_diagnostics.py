"""Print numeric diagnostics for Q10-Q14 so we can quantify spread.

Run: python -m projects.civic_shout_content_routing.exploration._diagnostics
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from projects.civic_shout_content_routing.exploration.explore_users import (
    build_action_matrix,
    build_user_frame,
)

WIDE = Path("data/projects/civic_shout_content_routing/exploration/wide_engagement.parquet")


def q10_diagnostics(users: pl.DataFrame) -> None:
    print("\n=== Q10. Heterogeneity at fixed exposure (actioners only) ===")
    df = users.filter((pl.col("n_actioned") > 0) & (pl.col("n_sent") > 0))
    n_sent = df["n_sent"].to_numpy().astype(float)
    n_act = df["n_actioned"].to_numpy().astype(float)
    rate = n_act / n_sent

    edges = np.unique(np.quantile(n_sent, np.linspace(0, 1, 7)).astype(int))
    print(f"actioners with >=1 send: {len(rate):,}")
    print(f"  bins: {list(edges)}")
    print(f"  {'bucket':>14} {'n':>8} {'med':>8} {'iqr':>10} {'p10':>8} {'p90':>8} {'cv':>6}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (n_sent >= lo) & (n_sent <= hi)
        if m.sum() < 200:
            continue
        r = rate[m]
        print(
            f"  [{lo:>5}-{hi:<6}] {m.sum():>8,} {np.median(r):>8.4f} "
            f"{np.percentile(r,75)-np.percentile(r,25):>10.4f} "
            f"{np.percentile(r,10):>8.4f} {np.percentile(r,90):>8.4f} "
            f"{(np.std(r)/np.mean(r) if np.mean(r) else 0):>6.2f}"
        )


def q11_diagnostics(M: sp.csr_matrix) -> None:
    print("\n=== Q11. Pairwise user Jaccard vs popularity-weighted null ===")
    rng = np.random.default_rng(0)
    counts = np.asarray(M.sum(axis=1)).flatten()
    eligible = np.where(counts >= 2)[0]
    sample_size = min(5000, len(eligible))
    sample = rng.choice(eligible, size=sample_size, replace=False)
    sums = counts[sample]

    Ms = M[sample]
    inter = (Ms @ Ms.T).toarray().astype(np.float32)
    union = sums[:, None] + sums[None, :] - inter
    J_obs = np.where(union > 0, inter / union, 0)
    iu = np.triu_indices(sample_size, k=1)
    obs = J_obs[iu]

    pop = np.asarray(M.sum(axis=0)).flatten()
    p = pop / pop.sum()
    n_emails = M.shape[1]
    rows: list[int] = []
    cols: list[int] = []
    for i, u in enumerate(sample):
        k = int(counts[u])
        chosen = rng.choice(n_emails, size=k, replace=False, p=p)
        rows.extend([i] * k)
        cols.extend(chosen.tolist())
    Mn = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(sample_size, n_emails),
    )
    inter_n = (Mn @ Mn.T).toarray().astype(np.float32)
    union_n = sums[:, None] + sums[None, :] - inter_n
    J_null = np.where(union_n > 0, inter_n / union_n, 0)
    null = J_null[iu]

    print(f"sample size: {sample_size:,} actioners with >=2 actions")
    print(f"pairs:       {len(obs):,}")
    for label, arr in [("observed", obs), ("null", null)]:
        print(
            f"  {label:>9}: mean={arr.mean():.5f}  median={np.median(arr):.5f}  "
            f"p90={np.percentile(arr, 90):.5f}  p99={np.percentile(arr, 99):.5f}  "
            f"max={arr.max():.5f}  frac_zero={(arr == 0).mean():.3f}"
        )
    print(f"  lift (obs/null mean): {obs.mean()/max(null.mean(),1e-9):.2f}x")
    print(f"  KS-style: frac obs > null p99: {(obs > np.percentile(null, 99)).mean():.3f}")


def q12_diagnostics(M: sp.csr_matrix) -> None:
    print("\n=== Q12. Email x email Jaccard ===")
    co = (M.T @ M).toarray().astype(np.float32)
    pop = np.diag(co).copy()
    union = pop[:, None] + pop[None, :] - co
    J = np.where(union > 0, co / union, 0).astype(np.float32)
    np.fill_diagonal(J, 0)
    iu = np.triu_indices(J.shape[0], k=1)
    off = J[iu]

    print(f"emails: {J.shape[0]:,}  pairs: {len(off):,}")
    print(
        f"  mean={off.mean():.5f}  median={np.median(off):.5f}  "
        f"p90={np.percentile(off, 90):.5f}  p99={np.percentile(off, 99):.5f}  "
        f"p99.9={np.percentile(off, 99.9):.5f}  max={off.max():.5f}"
    )
    for thr in (0.05, 0.10, 0.20, 0.30, 0.50):
        n = int((off >= thr).sum())
        print(f"  pairs with Jaccard >= {thr:.2f}: {n:,}  ({100*n/len(off):.4f}%)")


def q13_diagnostics(M: sp.csr_matrix) -> None:
    print("\n=== Q13. SVD ===")
    counts = np.asarray(M.sum(axis=1)).flatten()
    svd = TruncatedSVD(n_components=5, random_state=0)
    U = svd.fit_transform(M)
    ve = svd.explained_variance_ratio_
    print(f"variance ratio per component: {[f'{v:.4f}' for v in ve]}")
    print(f"cumulative: {[f'{c:.4f}' for c in np.cumsum(ve)]}")
    # Correlation of each component with action volume.
    for i in range(U.shape[1]):
        r = np.corrcoef(U[:, i], counts)[0, 1]
        print(f"  comp {i}: corr with n_actioned = {r:+.3f}")
    # Spread on content axes (1,2): RMS distance of points from mean.
    for ax_label, idx in [("comp 0 (volume)", 0), ("comp 1 (content)", 1), ("comp 2 (content)", 2)]:
        v = U[:, idx]
        print(
            f"  {ax_label:>20}: mean={v.mean():+.4f} std={v.std():.4f} "
            f"p1={np.percentile(v, 1):+.4f} p99={np.percentile(v, 99):+.4f}"
        )


def q14_diagnostics(M: sp.csr_matrix, k: int = 5) -> None:
    print(f"\n=== Q14. Top-{k} cohort retention vs random ===")
    co = (M.T @ M).toarray().astype(np.float32)
    pop = np.diag(co).copy()
    union = pop[:, None] + pop[None, :] - co
    J = np.where(union > 0, co / union, 0).astype(np.float32)
    np.fill_diagonal(J, -1.0)

    n_emails = J.shape[0]
    topk = np.argpartition(-J, k, axis=1)[:, :k]
    adj = sp.lil_matrix((n_emails, n_emails), dtype=np.float32)
    for e in range(n_emails):
        adj[e, topk[e]] = 1.0
    adj = adj.tocsr()
    G = M @ adj.T
    G_b = (G > 0).astype(np.float32)
    denom = np.asarray(M.sum(axis=0)).flatten()
    numer = np.asarray(M.multiply(G_b).sum(axis=0)).flatten()
    retention = np.where(denom > 0, numer / denom, np.nan)

    rng = np.random.default_rng(0)
    rand = rng.integers(0, n_emails - 1, size=(n_emails, k))
    self_mask = rand >= np.arange(n_emails)[:, None]
    rand = rand + self_mask.astype(int)
    adj_r = sp.lil_matrix((n_emails, n_emails), dtype=np.float32)
    for e in range(n_emails):
        adj_r[e, rand[e]] = 1.0
    adj_r = adj_r.tocsr()
    G_r = M @ adj_r.T
    G_rb = (G_r > 0).astype(np.float32)
    numer_r = np.asarray(M.multiply(G_rb).sum(axis=0)).flatten()
    rand_ret = np.where(denom > 0, numer_r / denom, np.nan)

    for label, arr in [("top-K", retention), ("random", rand_ret)]:
        a = arr[~np.isnan(arr)]
        print(
            f"  {label:>7}: mean={a.mean():.4f}  median={np.median(a):.4f}  "
            f"p10={np.percentile(a, 10):.4f}  p90={np.percentile(a, 90):.4f}  "
            f"max={a.max():.4f}  frac>=0.5={(a>=0.5).mean():.3f}"
        )
    a = retention[~np.isnan(retention)]
    b = rand_ret[~np.isnan(rand_ret)]
    print(f"  lift (top-K / random, mean): {a.mean()/max(b.mean(),1e-9):.2f}x")


def main() -> None:
    print(f"loading {WIDE}")
    wide = pl.read_parquet(WIDE)
    print(f"  rows={wide.height:,}  emails={wide['email_id'].n_unique():,}  "
          f"users={wide['user_id'].n_unique():,}")
    users = build_user_frame(wide)
    print("building action matrix")
    M, _, _ = build_action_matrix(wide)
    print(f"  M.shape={M.shape}  nnz={M.nnz:,}")

    q10_diagnostics(users)
    q11_diagnostics(M)
    q12_diagnostics(M)
    q13_diagnostics(M)
    q14_diagnostics(M)


if __name__ == "__main__":
    main()
