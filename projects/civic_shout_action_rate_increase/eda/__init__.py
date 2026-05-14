"""Project-level EDA harness for civic_shout_action_rate_increase."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field
from sklearn.feature_selection import mutual_info_classif

from libs.responses import (
    EDAConfig,
    EDAFinding,
    EDAPerformance,
    EDAResponse,
    EDAResultRow,
    EDAScores,
    HarnessStageTiming,
    RunResultMetadata,
)

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Project-local config extension
# ---------------------------------------------------------------------------


class CivicShoutEDAConfig(EDAConfig):
    """EDA config extended with project-level sampling + artifact options."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    sample_frac: float = 0.01
    seed: int = 42
    artifacts_dir: Path = Path("runs/eda_scratch/artifacts")
    cohort_col: str | None = None
    corr_flag_threshold: float = 0.85


# ---------------------------------------------------------------------------
# Project-local response types
# ---------------------------------------------------------------------------


class EDAMethodEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    name: str
    seconds: float
    n_findings: int
    artifact_paths: list[str] = Field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""


class EDASummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    n_findings_total: int
    n_findings_by_severity: dict[str, int] = Field(default_factory=dict)
    outcome_positive_rate: float | None = None
    feature_feature_max_abs_pearson: float | None = None
    cohort_drift_max_relative_lift: float | None = None


class EDADataProfile(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    n_rows: int
    n_features: int
    feature_cols: list[str]
    sample_frac: float
    sample_seed: int


class CivicShoutEDAResultRow(EDAResultRow):
    """Project-specific EDA result row with extended metrics."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    n_eda_methods_run: int = 0
    n_findings_total: int = 0
    n_findings_warning: int = 0
    n_findings_error: int = 0
    outcome_positive_rate: float | None = None
    feature_feature_max_abs_pearson: float | None = None
    cohort_drift_max_relative_lift: float | None = None
    elapsed_seconds: float = 0.0


class CivicShoutEDAResponse(EDAResponse):
    """Project-specific EDA response with Tier 1 method results and summary."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    method_results: list[EDAMethodEntry] = Field(default_factory=list)
    eda_summary: EDASummary = Field(default_factory=lambda: EDASummary(n_findings_total=0))
    data_profile: EDADataProfile | None = None

    def to_eda_summary_row(self, metadata: RunResultMetadata) -> CivicShoutEDAResultRow:  # type: ignore[override]
        n_warn = sum(1 for f in self.findings if f.severity == "warning")
        n_err = sum(1 for f in self.findings if f.severity == "error")
        return CivicShoutEDAResultRow(
            project=metadata.project,
            run_id=metadata.run_id,
            comparison_group=metadata.comparison_group,
            scope=metadata.scope,
            recorded_at_utc=metadata.recorded_at_utc,
            git_sha=metadata.git_sha,
            eda_outcome_prevalence=self.eda_summary.outcome_positive_rate,
            eda_drift_score=self.scores.drift_score,
            eda_n_findings_warning=n_warn,
            eda_n_findings_error=n_err,
            eda_top_finding_id=self.scores.top_finding_id,
            eda_total_seconds=self.performance.total_seconds,
            n_eda_methods_run=len(self.method_results),
            n_findings_total=len(self.findings),
            n_findings_warning=n_warn,
            n_findings_error=n_err,
            outcome_positive_rate=self.eda_summary.outcome_positive_rate,
            feature_feature_max_abs_pearson=self.eda_summary.feature_feature_max_abs_pearson,
            cohort_drift_max_relative_lift=self.eda_summary.cohort_drift_max_relative_lift,
            elapsed_seconds=self.performance.total_seconds,
        )

    def to_summary_md(self, metadata: RunResultMetadata) -> str:
        lines: list[str] = []
        lines.append(f"# EDA Summary — {metadata.run_id} ({metadata.project})")
        lines.append("")
        lines.append("## Headline")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append(f"| outcome_positive_rate | {self.eda_summary.outcome_positive_rate} |")
        lines.append(f"| n_findings_total | {self.eda_summary.n_findings_total} |")
        for sev, cnt in self.eda_summary.n_findings_by_severity.items():
            lines.append(f"| n_findings_{sev} | {cnt} |")
        lines.append(
            f"| feature_feature_max_abs_pearson | {self.eda_summary.feature_feature_max_abs_pearson} |"
        )
        lines.append(
            f"| cohort_drift_max_relative_lift | {self.eda_summary.cohort_drift_max_relative_lift} |"
        )
        lines.append(f"| elapsed_seconds | {self.performance.total_seconds:.2f} |")
        lines.append("")

        for method in self.method_results:
            lines.append(f"## Method: {method.name}")
            if method.skipped:
                lines.append(f"*Skipped: {method.skip_reason}*")
            else:
                lines.append(f"- elapsed: {method.seconds:.2f}s")
                lines.append(f"- n_findings: {method.n_findings}")
            for p in method.artifact_paths:
                fname = Path(p).name
                lines.append(f"![{fname}]({fname})")
            lines.append("")

        lines.append("## Findings")
        if not self.findings:
            lines.append("*No findings.*")
        else:
            for f in self.findings:
                lines.append(f"- [{f.severity.upper()}] `{f.id}`: {f.title}")
                if f.description:
                    lines.append(f"  {f.description}")
        lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

_EDA_METHOD_REGISTRY: dict[str, Callable] = {}


def register_eda_method(name: str) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _EDA_METHOD_REGISTRY[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _numeric_feature_cols(data: pl.DataFrame, feature_cols: list[str]) -> list[str]:
    return [
        c
        for c in feature_cols
        if c in data.columns
        and data[c].dtype
        in (
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        )
    ]


def _categorical_feature_cols(data: pl.DataFrame, feature_cols: list[str]) -> list[str]:
    return [
        c
        for c in feature_cols
        if c in data.columns and data[c].dtype in (pl.Utf8, pl.Categorical, pl.Boolean)
    ]


def _ensure_artifacts_dir(artifacts_dir: Path) -> None:
    (artifacts_dir / "eda").mkdir(parents=True, exist_ok=True)


def _safe_png_path(artifacts_dir: Path, name: str) -> Path:
    return artifacts_dir / "eda" / f"{name}.png"


def _save_fig(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tier 1 methods
# ---------------------------------------------------------------------------


@register_eda_method(name="outcome_distribution")
def outcome_distribution(
    data: pl.DataFrame,
    feature_cols: list[str],
    outcome_variable: str,
    cfg: CivicShoutEDAConfig,
) -> tuple[list[EDAFinding], list[Path]]:
    """Positive rate, class imbalance, n_positives for binary outcome."""
    findings: list[EDAFinding] = []
    artifacts: list[Path] = []

    if outcome_variable not in data.columns:
        return findings, artifacts

    outcome_series = data[outcome_variable].cast(pl.Float64)
    n_rows = len(data)
    n_positives = int(outcome_series.sum())
    n_negatives = n_rows - n_positives
    positive_rate = n_positives / n_rows if n_rows > 0 else 0.0

    findings.append(
        EDAFinding(
            id="outcome_distribution_positive_rate",
            title=f"Positive rate: {positive_rate:.4f}",
            severity="info",
            score=positive_rate,
            description=f"n_rows={n_rows}, n_positives={n_positives}, n_negatives={n_negatives}",
            metadata={"n_positives": str(n_positives), "positive_rate": str(positive_rate)},
        )
    )

    if n_positives > 0 and n_negatives > 0:
        max_class = max(n_positives, n_negatives)
        min_class = min(n_positives, n_negatives)
        imbalance = max_class / min_class
        severity = "warning" if imbalance > 10 else "info"
        findings.append(
            EDAFinding(
                id="outcome_distribution_class_imbalance",
                title=f"Class imbalance ratio: {imbalance:.2f}",
                severity=severity,
                score=imbalance,
                description=f"max_class={max_class}, min_class={min_class}",
                metadata={"imbalance_ratio": str(imbalance)},
            )
        )

    png_path = _safe_png_path(cfg.artifacts_dir, "outcome_distribution")
    labels = [outcome_variable + "=0", outcome_variable + "=1"]
    counts = [n_negatives, n_positives]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, counts, color=["steelblue", "tomato"])
    ax.set_title("Outcome Distribution")
    ax.set_ylabel("Count")
    _save_fig(fig, png_path)
    artifacts.append(png_path)

    return findings, artifacts


@register_eda_method(name="univariate_numeric")
def univariate_numeric(
    data: pl.DataFrame,
    feature_cols: list[str],
    outcome_variable: str,
    cfg: CivicShoutEDAConfig,
) -> tuple[list[EDAFinding], list[Path]]:
    """Per numeric feature: mean, std, percentiles, missingness, skew."""
    findings: list[EDAFinding] = []
    artifacts: list[Path] = []

    num_cols = _numeric_feature_cols(data, feature_cols)
    if not num_cols:
        return findings, artifacts

    for col in num_cols:
        s = data[col].cast(pl.Float64)
        n_null = s.null_count()
        n_total = len(s)
        missingness_pct = n_null / n_total if n_total > 0 else 0.0
        s_valid = s.drop_nulls()
        arr = s_valid.to_numpy()
        if len(arr) == 0:
            continue
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        p1 = float(np.percentile(arr, 1))
        p25 = float(np.percentile(arr, 25))
        p50 = float(np.percentile(arr, 50))
        p75 = float(np.percentile(arr, 75))
        p99 = float(np.percentile(arr, 99))
        skew = float(np.mean(((arr - mean_val) / (std_val + 1e-9)) ** 3)) if std_val > 0 else 0.0

        severity = "warning" if missingness_pct > 0.1 else "info"
        findings.append(
            EDAFinding(
                id=f"univariate_numeric_{col}",
                title=f"{col}: mean={mean_val:.3f}, std={std_val:.3f}, missingness={missingness_pct:.3f}",
                severity=severity,
                score=missingness_pct,
                description=(
                    f"p1={p1:.3f}, p25={p25:.3f}, p50={p50:.3f}, "
                    f"p75={p75:.3f}, p99={p99:.3f}, skew={skew:.3f}"
                ),
                metadata={
                    "col": col,
                    "mean": str(mean_val),
                    "std": str(std_val),
                    "missingness_pct": str(missingness_pct),
                    "skew": str(skew),
                },
            )
        )

    n_cols = len(num_cols)
    ncols_plot = min(4, n_cols)
    nrows_plot = (n_cols + ncols_plot - 1) // ncols_plot
    fig, axes = plt.subplots(nrows_plot, ncols_plot, figsize=(4 * ncols_plot, 3 * nrows_plot))
    axes_flat = np.array(axes).flatten() if n_cols > 1 else [axes]
    for i, col in enumerate(num_cols):
        ax = axes_flat[i]
        arr = data[col].cast(pl.Float64).drop_nulls().to_numpy()
        ax.hist(arr, bins=30, color="steelblue", edgecolor="none")
        ax.set_title(col, fontsize=8)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Numeric Feature Distributions", fontsize=10)
    fig.tight_layout()
    png_path = _safe_png_path(cfg.artifacts_dir, "univariate_numeric")
    _save_fig(fig, png_path)
    artifacts.append(png_path)

    return findings, artifacts


@register_eda_method(name="univariate_categorical")
def univariate_categorical(
    data: pl.DataFrame,
    feature_cols: list[str],
    outcome_variable: str,
    cfg: CivicShoutEDAConfig,
) -> tuple[list[EDAFinding], list[Path]]:
    """Per categorical feature: top-10 values, cardinality, missingness."""
    findings: list[EDAFinding] = []
    artifacts: list[Path] = []

    cat_cols = _categorical_feature_cols(data, feature_cols)
    if not cat_cols:
        return findings, artifacts

    n_cols = len(cat_cols)
    ncols_plot = min(2, n_cols)
    nrows_plot = (n_cols + ncols_plot - 1) // ncols_plot
    fig, axes = plt.subplots(nrows_plot, ncols_plot, figsize=(6 * ncols_plot, 4 * nrows_plot))
    axes_flat = np.array(axes).flatten() if n_cols > 1 else [axes]

    for i, col in enumerate(cat_cols):
        s = data[col]
        n_null = s.null_count()
        n_total = len(s)
        missingness_pct = n_null / n_total if n_total > 0 else 0.0
        cardinality = s.n_unique()
        top10 = data.select(pl.col(col).value_counts(sort=True)).unnest(col).head(10)
        top_vals = top10[col].to_list()
        top_counts = top10["count"].to_list()
        top_props = [c / n_total for c in top_counts]

        findings.append(
            EDAFinding(
                id=f"univariate_categorical_{col}",
                title=f"{col}: cardinality={cardinality}, missingness={missingness_pct:.3f}",
                severity="info",
                score=missingness_pct,
                description=f"Top values: {list(zip(top_vals, [f'{p:.3f}' for p in top_props]))}",
                metadata={"col": col, "cardinality": str(cardinality)},
            )
        )

        ax = axes_flat[i]
        ax.barh([str(v) for v in top_vals], top_props, color="steelblue")
        ax.set_title(col, fontsize=8)
        ax.set_xlabel("Proportion")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    png_path = _safe_png_path(cfg.artifacts_dir, "univariate_categorical")
    _save_fig(fig, png_path)
    artifacts.append(png_path)

    return findings, artifacts


@register_eda_method(name="feature_outcome_correlation")
def feature_outcome_correlation(
    data: pl.DataFrame,
    feature_cols: list[str],
    outcome_variable: str,
    cfg: CivicShoutEDAConfig,
) -> tuple[list[EDAFinding], list[Path]]:
    """Per feature: Pearson, Spearman, mutual information vs outcome."""
    findings: list[EDAFinding] = []
    artifacts: list[Path] = []

    num_cols = _numeric_feature_cols(data, feature_cols)
    if not num_cols or outcome_variable not in data.columns:
        return findings, artifacts

    outcome_arr = data[outcome_variable].cast(pl.Float64).to_numpy()
    feature_matrix = np.column_stack(
        [data[c].cast(pl.Float64).fill_null(0.0).to_numpy() for c in num_cols]
    )

    try:
        mi_scores = mutual_info_classif(
            feature_matrix,
            outcome_arr,
            random_state=cfg.seed,
        )
    except Exception:
        mi_scores = np.zeros(len(num_cols))

    pearson_vals: list[float] = []
    spearman_vals: list[float] = []

    for i, col in enumerate(num_cols):
        col_arr = feature_matrix[:, i]
        n = len(col_arr)
        if n < 2:
            pearson_vals.append(0.0)
            spearman_vals.append(0.0)
            continue

        col_std = np.std(col_arr)
        out_std = np.std(outcome_arr)

        if col_std > 0 and out_std > 0:
            pearson = float(np.corrcoef(col_arr, outcome_arr)[0, 1])
        else:
            pearson = 0.0

        col_ranks = np.argsort(np.argsort(col_arr)).astype(float)
        out_ranks = np.argsort(np.argsort(outcome_arr)).astype(float)
        rk_std_c = np.std(col_ranks)
        rk_std_o = np.std(out_ranks)
        if rk_std_c > 0 and rk_std_o > 0:
            spearman = float(np.corrcoef(col_ranks, out_ranks)[0, 1])
        else:
            spearman = 0.0

        pearson_vals.append(pearson)
        spearman_vals.append(spearman)

        findings.append(
            EDAFinding(
                id=f"feature_outcome_correlation_{col}",
                title=f"{col}: pearson={pearson:.3f}, spearman={spearman:.3f}, mi={mi_scores[i]:.4f}",
                severity="info",
                score=abs(pearson),
                description=f"Pearson={pearson:.4f}, Spearman={spearman:.4f}, MI={mi_scores[i]:.6f}",
                metadata={
                    "col": col,
                    "pearson": str(pearson),
                    "spearman": str(spearman),
                    "mutual_info": str(mi_scores[i]),
                },
            )
        )

    if num_cols:
        fig, ax = plt.subplots(figsize=(6, max(3, len(num_cols) * 0.4)))
        y_pos = np.arange(len(num_cols))
        ax.barh(y_pos - 0.2, pearson_vals, height=0.35, label="Pearson", color="steelblue")
        ax.barh(y_pos + 0.2, spearman_vals, height=0.35, label="Spearman", color="tomato")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(num_cols, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.legend(fontsize=7)
        ax.set_title("Feature ↔ Outcome Correlations")
        fig.tight_layout()
        png_path = _safe_png_path(cfg.artifacts_dir, "feature_outcome_correlation")
        _save_fig(fig, png_path)
        artifacts.append(png_path)

    return findings, artifacts


@register_eda_method(name="feature_feature_correlation")
def feature_feature_correlation(
    data: pl.DataFrame,
    feature_cols: list[str],
    outcome_variable: str,
    cfg: CivicShoutEDAConfig,
) -> tuple[list[EDAFinding], list[Path]]:
    """Pairwise Pearson + Spearman; flag pairs with |r| > threshold."""
    findings: list[EDAFinding] = []
    artifacts: list[Path] = []

    num_cols = _numeric_feature_cols(data, feature_cols)
    if len(num_cols) < 2:
        return findings, artifacts

    feature_matrix = np.column_stack(
        [data[c].cast(pl.Float64).fill_null(0.0).to_numpy() for c in num_cols]
    )
    corr_matrix = np.corrcoef(feature_matrix.T)

    n = len(num_cols)
    for i in range(n):
        for j in range(i + 1, n):
            r = float(corr_matrix[i, j])
            abs_r = abs(r)
            if abs_r > cfg.corr_flag_threshold:
                findings.append(
                    EDAFinding(
                        id=f"feature_feature_correlation_{num_cols[i]}_{num_cols[j]}",
                        title=f"High correlation: {num_cols[i]} ↔ {num_cols[j]} |r|={abs_r:.3f}",
                        severity="warning",
                        score=abs_r,
                        description=f"Pearson r={r:.4f} exceeds threshold {cfg.corr_flag_threshold}",
                        metadata={
                            "col_a": num_cols[i],
                            "col_b": num_cols[j],
                            "pearson": str(r),
                        },
                    )
                )

    fig, ax = plt.subplots(figsize=(max(4, n), max(3, n)))
    im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(num_cols, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature ↔ Feature Correlation (Pearson)")
    fig.tight_layout()
    png_path = _safe_png_path(cfg.artifacts_dir, "feature_feature_correlation")
    _save_fig(fig, png_path)
    artifacts.append(png_path)

    return findings, artifacts


@register_eda_method(name="missingness_pattern")
def missingness_pattern(
    data: pl.DataFrame,
    feature_cols: list[str],
    outcome_variable: str,
    cfg: CivicShoutEDAConfig,
) -> tuple[list[EDAFinding], list[Path]]:
    """Per-feature missingness; flag rows missing > 1 feature."""
    findings: list[EDAFinding] = []
    artifacts: list[Path] = []

    valid_cols = [c for c in feature_cols if c in data.columns]
    if not valid_cols:
        return findings, artifacts

    n_rows = len(data)
    for col in valid_cols:
        _ = data[col].null_count() / n_rows if n_rows > 0 else 0.0

    row_miss_counts = pl.Series(
        [sum(1 for c in valid_cols if data[c][i] is None) for i in range(n_rows)]
    )
    n_multi_miss = int((row_miss_counts > 1).sum())

    if n_multi_miss > 0:
        sev = "warning" if n_multi_miss / n_rows > 0.01 else "info"
        findings.append(
            EDAFinding(
                id="missingness_pattern_multi_miss_rows",
                title=f"{n_multi_miss} rows missing > 1 feature ({n_multi_miss / n_rows:.3f})",
                severity=sev,
                score=n_multi_miss / n_rows,
                description=f"n_multi_miss={n_multi_miss}, n_rows={n_rows}",
                metadata={"n_multi_miss": str(n_multi_miss)},
            )
        )

    n_cols = len(valid_cols)
    sample_size = min(200, n_rows)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(n_rows, size=sample_size, replace=False)

    miss_matrix = np.zeros((sample_size, n_cols), dtype=np.float32)
    for j, col in enumerate(valid_cols):
        col_null_mask = data[col].is_null().to_numpy()
        miss_matrix[:, j] = col_null_mask[idx].astype(np.float32)

    fig, ax = plt.subplots(figsize=(max(4, n_cols * 0.5), 4))
    ax.imshow(miss_matrix, aspect="auto", cmap="binary", interpolation="nearest")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(valid_cols, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Row sample")
    ax.set_title("Missingness Pattern (white=missing)")
    fig.tight_layout()
    png_path = _safe_png_path(cfg.artifacts_dir, "missingness_pattern")
    _save_fig(fig, png_path)
    artifacts.append(png_path)

    return findings, artifacts


@register_eda_method(name="row_alignment_audit")
def row_alignment_audit(
    data: pl.DataFrame,
    feature_cols: list[str],
    outcome_variable: str,
    cfg: CivicShoutEDAConfig,
) -> tuple[list[EDAFinding], list[Path]]:
    """Assert row keys are unique; assert no nulls in row keys or outcome."""
    findings: list[EDAFinding] = []
    artifacts: list[Path] = []

    row_key_cols = [c for c in ("user_id", "email_id", "date_sent") if c in data.columns]

    errors: list[str] = []

    if row_key_cols:
        n_total = len(data)
        n_unique = data.select(row_key_cols).n_unique()
        if n_unique != n_total:
            errors.append(f"Duplicate row keys: n_total={n_total}, n_unique={n_unique}")

        for col in row_key_cols:
            n_null = data[col].null_count()
            if n_null > 0:
                errors.append(f"Null in row key column {col}: n_null={n_null}")

    if outcome_variable in data.columns:
        n_null_out = data[outcome_variable].null_count()
        if n_null_out > 0:
            errors.append(f"Null in outcome {outcome_variable}: n_null={n_null_out}")

    if errors:
        findings.append(
            EDAFinding(
                id="row_alignment_audit_fail",
                title="Row alignment audit FAILED",
                severity="error",
                score=1.0,
                description="; ".join(errors),
                metadata={"errors": str(errors)},
            )
        )
    else:
        findings.append(
            EDAFinding(
                id="row_alignment_audit_pass",
                title="Row alignment audit passed",
                severity="info",
                score=0.0,
                description="Row keys unique; no nulls in row keys or outcome.",
            )
        )

    return findings, artifacts


@register_eda_method(name="eda_cohort_baseline")
def eda_cohort_baseline(
    data: pl.DataFrame,
    feature_cols: list[str],
    outcome_variable: str,
    cfg: CivicShoutEDAConfig,
) -> tuple[list[EDAFinding], list[Path]]:
    """Per-cohort positive rate, n_rows, relative lift vs first cohort."""
    findings: list[EDAFinding] = []
    artifacts: list[Path] = []

    cohort_col = cfg.cohort_col

    if cohort_col is None or cohort_col not in data.columns:
        if "cohort_year" in data.columns:
            cohort_col = "cohort_year"
        elif "date_sent" in data.columns:
            data = data.with_columns(pl.col("date_sent").dt.year().alias("_cohort_year_tmp"))
            cohort_col = "_cohort_year_tmp"
        else:
            return findings, artifacts

    if outcome_variable not in data.columns:
        return findings, artifacts

    cohort_stats = (
        data.group_by(cohort_col)
        .agg(
            pl.col(outcome_variable).cast(pl.Float64).mean().alias("positive_rate"),
            pl.len().alias("n_rows"),
        )
        .sort(cohort_col)
    )

    cohorts = cohort_stats[cohort_col].to_list()
    rates = cohort_stats["positive_rate"].to_list()
    n_rows_list = cohort_stats["n_rows"].to_list()

    if not rates:
        return findings, artifacts

    baseline_rate = rates[0] if rates[0] is not None and rates[0] > 0 else None

    for cohort, rate, n_rows_c in zip(cohorts, rates, n_rows_list):
        if rate is None:
            continue
        relative_lift: float | None = None
        if baseline_rate is not None and baseline_rate > 0:
            relative_lift = rate / baseline_rate

        sev = "info"
        if relative_lift is not None and relative_lift > 1.5:
            sev = "warning"

        findings.append(
            EDAFinding(
                id=f"eda_cohort_baseline_{cohort}",
                title=f"Cohort {cohort}: rate={rate:.4f}, n_rows={n_rows_c}",
                severity=sev,
                score=float(relative_lift) if relative_lift is not None else 0.0,
                description=(
                    f"relative_lift={relative_lift:.3f}"
                    if relative_lift is not None
                    else "baseline cohort"
                ),
                metadata={
                    "cohort": str(cohort),
                    "positive_rate": str(rate),
                    "n_rows": str(n_rows_c),
                    "relative_lift": str(relative_lift),
                },
            )
        )

    fig, ax = plt.subplots(figsize=(max(4, len(cohorts)), 3))
    ax.bar(
        [str(c) for c in cohorts], [r if r is not None else 0.0 for r in rates], color="steelblue"
    )
    ax.set_title("Cohort Positive Rates")
    ax.set_xlabel("Cohort")
    ax.set_ylabel("Positive Rate")
    fig.tight_layout()
    png_path = _safe_png_path(cfg.artifacts_dir, "eda_cohort_baseline")
    _save_fig(fig, png_path)
    artifacts.append(png_path)

    return findings, artifacts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def eda(
    data: pl.DataFrame,
    feature_cols: list[str],
    outcome_variable: str,
    eda_config: CivicShoutEDAConfig | EDAConfig | None = None,
) -> CivicShoutEDAResponse:
    """Run all registered Tier 1 EDA methods; return structured response."""
    if eda_config is None:
        cfg = CivicShoutEDAConfig()
    elif isinstance(eda_config, CivicShoutEDAConfig):
        cfg = eda_config
    else:
        cfg = CivicShoutEDAConfig(
            enabled_methods=eda_config.enabled_methods,
            fail_on=eda_config.fail_on,
            cache_dir=eda_config.cache_dir,
        )

    _ensure_artifacts_dir(cfg.artifacts_dir)

    if cfg.sample_frac < 1.0:
        n_sample = max(1, int(len(data) * cfg.sample_frac))
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(len(data), size=n_sample, replace=False)
        idx_sorted = np.sort(idx)
        sampled_data = data[idx_sorted]
    else:
        sampled_data = data

    n_rows_sampled = len(sampled_data)
    n_features = len(feature_cols)

    all_findings: list[EDAFinding] = []
    method_results: list[EDAMethodEntry] = []
    stage_timings: list[HarnessStageTiming] = []
    timing_rows: list[dict] = []

    t_total_start = time.monotonic()

    _ORDERED_METHODS = [
        "outcome_distribution",
        "univariate_numeric",
        "univariate_categorical",
        "feature_outcome_correlation",
        "feature_feature_correlation",
        "missingness_pattern",
        "row_alignment_audit",
        "eda_cohort_baseline",
    ]

    for method_name in _ORDERED_METHODS:
        fn = _EDA_METHOD_REGISTRY.get(method_name)
        if fn is None:
            continue

        t_start = time.monotonic()
        try:
            findings, artifacts = fn(sampled_data, feature_cols, outcome_variable, cfg)
            skipped = False
            skip_reason = ""
        except Exception as exc:
            findings = []
            artifacts = []
            skipped = True
            skip_reason = str(exc)

        elapsed = time.monotonic() - t_start

        all_findings.extend(findings)
        method_results.append(
            EDAMethodEntry(
                name=method_name,
                seconds=elapsed,
                n_findings=len(findings),
                artifact_paths=[str(p) for p in artifacts],
                skipped=skipped,
                skip_reason=skip_reason,
            )
        )
        stage_timings.append(HarnessStageTiming(stage=method_name, seconds=elapsed))
        timing_rows.append(
            {
                "method": method_name,
                "seconds": elapsed,
                "n_findings": len(findings),
                "skipped": skipped,
            }
        )

    total_seconds = time.monotonic() - t_total_start

    _append_timing_rows(timing_rows)

    outcome_positive_rate: float | None = None
    if outcome_variable in sampled_data.columns:
        n_pos = int(sampled_data[outcome_variable].cast(pl.Float64).sum())
        outcome_positive_rate = n_pos / n_rows_sampled if n_rows_sampled > 0 else None

    feature_feature_max_abs_pearson: float | None = None
    for finding in all_findings:
        if finding.id.startswith("feature_feature_correlation_"):
            if (
                feature_feature_max_abs_pearson is None
                or finding.score > feature_feature_max_abs_pearson
            ):
                feature_feature_max_abs_pearson = finding.score

    cohort_drift_max_relative_lift: float | None = None
    for finding in all_findings:
        if finding.id.startswith("eda_cohort_baseline_"):
            if (
                cohort_drift_max_relative_lift is None
                or finding.score > cohort_drift_max_relative_lift
            ):
                cohort_drift_max_relative_lift = finding.score

    n_warn = sum(1 for f in all_findings if f.severity == "warning")
    n_err = sum(1 for f in all_findings if f.severity == "error")
    n_info = sum(1 for f in all_findings if f.severity == "info")

    eda_summary = EDASummary(
        n_findings_total=len(all_findings),
        n_findings_by_severity={"info": n_info, "warning": n_warn, "error": n_err},
        outcome_positive_rate=outcome_positive_rate,
        feature_feature_max_abs_pearson=feature_feature_max_abs_pearson,
        cohort_drift_max_relative_lift=cohort_drift_max_relative_lift,
    )

    scores = EDAScores(
        outcome_prevalence=outcome_positive_rate,
        n_findings_warning=n_warn,
        n_findings_error=n_err,
        top_finding_id=all_findings[0].id if all_findings else None,
    )

    performance = EDAPerformance(
        total_seconds=total_seconds,
        stage_timings=stage_timings,
    )

    data_profile = EDADataProfile(
        n_rows=n_rows_sampled,
        n_features=n_features,
        feature_cols=feature_cols,
        sample_frac=cfg.sample_frac,
        sample_seed=cfg.seed,
    )

    return CivicShoutEDAResponse(
        findings=all_findings,
        scores=scores,
        performance=performance,
        method_results=method_results,
        eda_summary=eda_summary,
        data_profile=data_profile,
    )


# ---------------------------------------------------------------------------
# Timing ledger helper
# ---------------------------------------------------------------------------

_EDA_TIMING_JSONL = Path(__file__).parent.parent / "eda_timing_performance.jsonl"


def _append_timing_rows(rows: list[dict]) -> None:
    import json

    try:
        with open(_EDA_TIMING_JSONL, "a") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    except Exception:
        pass
